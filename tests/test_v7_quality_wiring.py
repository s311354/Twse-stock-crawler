"""SYNTHETIC ONLY: policy/source boundaries, not production readiness."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from application.daily_work import DailyWorkContext
from application.market_history import MarketHistoryAssembler
from application.low_entry_v7_daily import daily_report
from config.settings import AppSettings
from domain.fundamental_derivation import FundamentalBundle
from domain.les_v7.quality_policy import default_quality_policy, r2_quality_policy
from domain.les_v7.report import validate_report
from infrastructure.runtime.fundamental_refresh import FundamentalRefreshCoordinator
from infrastructure.storage.les_v7_repository import LesV7SignalRepository
from interface.cli import build_preflight_summary
from application.contracts import StockAnalysisRequest
from domain.models import Stocktype
from tests.strategies.test_low_entry_score_v7 import POLICY, CUTOFF, bars
from tests.strategies.test_daily_v7_pipeline import synthetic_market
from tests.fundamentals.test_v7_financial_sources import original
from domain.les_v7.financial_sources import ixbrl_record_r2
from application.les_v7 import analyze_request


def test_preflight_exposes_independent_registered_quality_policy():
    request = StockAnalysisRequest(["2330"], Stocktype.ELEC, "synthetic", 700)
    summary = build_preflight_summary(request, AppSettings(daily_entry_model="v7"))
    assert summary["les_v7_bundle_configured"] is False
    assert summary["les_v7_quality_policy"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R4"
    assert summary["les_v7_quality_policy"]["independent_of_model_bundle"] is True


def test_invalid_registered_policy_is_a_startup_error_not_issuer_data_failure(monkeypatch):
    from config.settings import SettingsError
    import domain.les_v7.quality_policy as module
    policy = default_quality_policy()
    policy["policy_sha256"] = "wrong"
    monkeypatch.setattr(module, "default_quality_policy", lambda: policy)
    with pytest.raises(SettingsError, match="Quality policy is invalid"):
        AppSettings.from_env({"TWSE_DAILY_ENTRY_MODEL": "v7"}, validate_paths=False)


@pytest.mark.parametrize("model,policy", [("v6", None), ("v7", POLICY)])
def test_native_source_companion_does_not_run_on_legacy_paths(model, policy):
    service = SimpleNamespace(load_v7_financial_sources=Mock(side_effect=AssertionError("legacy path")))
    bundle = FundamentalBundle(pd.DataFrame(), dict(les_v7_context={"quality_policy": policy} if policy else {}))
    crawler = SimpleNamespace(app_settings=SimpleNamespace(daily_entry_model=model, les_v7_bundle_json=None))
    result = FundamentalRefreshCoordinator(service)._attach_v7_sources(bundle, crawler=crawler,
        issuer="2330", cutoff=CUTOFF, market="TWSE")
    assert result is bundle
    service.load_v7_financial_sources.assert_not_called()


def test_source_prep_reuse_is_work_cutoff_and_market_bound():
    rows = [dict(source="MOPS_IXBRL", period_end="2026-06-30", content_sha256="a" * 64,
                 revision_id="a" * 64, available_at="2026-08-10T05:00:00Z")]
    service = SimpleNamespace(load_v7_financial_sources=Mock(return_value=dict(financial_audits={}, financial_supplements={})))
    bundle = FundamentalBundle(pd.DataFrame(), dict(les_v6_mops_filings=rows))
    crawler = SimpleNamespace(app_settings=SimpleNamespace(daily_entry_model="v7", les_v7_bundle_json=None))
    coordinator = FundamentalRefreshCoordinator(service)
    work = DailyWorkContext("SYNTHETIC_ONLY")
    try:
        with work.activate():
            for _ in range(2): coordinator._attach_v7_sources(bundle, crawler=crawler, issuer="2330", cutoff=CUTOFF, market="TWSE")
            assert service.load_v7_financial_sources.call_count == 1
            coordinator._attach_v7_sources(bundle, crawler=crawler, issuer="2330", cutoff="2026-10-13T13:30:00+08:00", market="TWSE")
            coordinator._attach_v7_sources(bundle, crawler=crawler, issuer="2330", cutoff=CUTOFF, market="TPEX")
        assert service.load_v7_financial_sources.call_count == 3
        assert bundle.snapshot == dict(les_v6_mops_filings=rows)
    finally: work.close()


def test_r2_daily_uses_canonical_original_not_batch_identity_with_nested_proof():
    row = ixbrl_record_r2("SYNTHETIC_ONLY", original(), original_verified=True)
    legacy = dict(row, source="TWSE_OPENAPI", content_sha256="b" * 64, revenue=1.)
    market = synthetic_market()
    market = replace(market, les_v7_bundle_json=None, fundamental_snapshots={"SYNTHETIC_ONLY": {
        "sector": "24", "les_v6_filings": [legacy], "les_v6_mops_filings": [row],
        "les_v7_context": {"quality_policy": r2_quality_policy()}}})
    report = daily_report(MarketHistoryAssembler().build(market, 0))
    assert report["diagnostics"]["quality"]["core_ttm_pass"] is False
    fields = report["diagnostics"]["quality"]["field_checks"]
    assert fields["2026Q2:revenue"]["content_sha256"] == row["content_sha256"]
    assert fields["2026Q2:revenue"]["source"] == "MOPS_IXBRL"


def test_r1_r2_same_cutoff_store_and_readiness_tamper_detection(tmp_path):
    request = dict(bars=bars().to_dict("records"), context=dict(symbol="SYNTHETIC_ONLY", market="TWSE",
                   as_of=CUTOFF, filings=[], quality_policy=r2_quality_policy()))
    r2 = analyze_request(request)["reports"][0]
    request["context"]["quality_policy"] = POLICY
    r1 = analyze_request(request)["reports"][0]
    repository = LesV7SignalRepository(tmp_path / "research")
    a, b = repository.publish(r1), repository.publish(r2)
    assert a["signal_id"] != b["signal_id"]
    assert repository.publish(r2) == b and repository.load(a["signal_id"]) == r1
    assert repository.load(b["signal_id"]) == r2
    changed = deepcopy(r2)
    changed["diagnostics"]["readiness"]["execution"]["state"] = "READY_FOR_ACCOUNT_CHECK"
    with pytest.raises(ValueError, match="V7_READINESS_BINDING_MISMATCH"): validate_report(changed)

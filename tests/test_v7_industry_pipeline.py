"""SYNTHETIC ONLY: source-bound industry integration, not real buy evidence."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.low_entry_v7_daily import daily_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import (
    INDUSTRY_BASELINE_POLICY, INDUSTRY_AVAILABLE_AT, RECOVERY_BASELINE_POLICY,
    baseline_policy, current_source_baseline_version, require_baseline_policy,
)
from tests.test_industry_benchmark import packet
from application.industry_benchmark import prepare_industry_benchmark
from tests.test_v7_r10_pipeline import market as r10_market


def market(*, with_industry=True):
    old = r10_market()
    at = (timestamp(INDUSTRY_AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    benchmark = None
    if with_industry:
        days = pd.to_datetime(old.dates[0][-61:])
        raw = packet(days, np.linspace(100., 101., len(days)))
        # Explicit synthetic observation/review occurs after these fixture bars.
        for document in raw["documents"].values():
            document["retrieved_at"] = at
        for row in [*raw["memberships"], *raw["bars"]]:
            row["publication_proof"]["verified_at"] = at
            row["publication_proof"]["document"]["retrieved_at"] = at
        benchmark = prepare_industry_benchmark(raw)
    from infrastructure.crawler.security_master import select_security_master
    from tests.test_security_master import packet as security_packet
    sources = deepcopy(old.native_source_selections)
    sources["2330"]["security"] = select_security_master(security_packet(), "2330", at)
    return replace(old, signal_dates={"2330": at},
        industry_benchmarks={"24": benchmark} if benchmark else {},
        native_source_selections=sources,
        les_v6_portfolio=dict(old.les_v6_portfolio, as_of=at))


def native(m=None):
    return daily_report(MarketHistoryAssembler().build(capture_comparison_contexts(m or market()), 0))


def test_r10_hash_and_cutoff_vintage_are_unchanged():
    assert identity(baseline_policy(version=RECOVERY_BASELINE_POLICY)) == (
        "5b94f7fef0fb757681cebb5b2c3233e0f14aca393f2b74b268c79ba6cec2809d")
    before = timestamp(INDUSTRY_AVAILABLE_AT) - timedelta(microseconds=1)
    assert current_source_baseline_version(before) == RECOVERY_BASELINE_POLICY
    assert current_source_baseline_version(INDUSTRY_AVAILABLE_AT) == INDUSTRY_BASELINE_POLICY
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(version=INDUSTRY_BASELINE_POLICY), market="TWSE", cutoff=before)


def test_only_industry_integration_changes_not_investment_parameters():
    old, new = (baseline_policy(version=v) for v in (RECOVERY_BASELINE_POLICY, INDUSTRY_BASELINE_POLICY))
    for key in ("config", "model_spec", "native_risk_limits", "valuation_policy", "valuation_input_policy",
                "entry_admission_policy", "score_policy", "score_adjustment_policy", "quality_policy"):
        assert old[key] == new[key]
    assert old["reference_policy"]["structural_rules"] == new["reference_policy"]["structural_rules"]


def test_daily_uses_actual_benchmark_and_keeps_missing_industry_unknown():
    report, missing = native(), native(market(with_industry=False))
    validate_report(report); validate_report(missing)
    baseline = report["diagnostics"]["rule_baseline"]
    assert baseline["policy"]["policy_version"] == INDUSTRY_BASELINE_POLICY
    assert baseline["features"]["industry_status"] == "PASS"
    assert baseline["features"]["industry_rs20"] is not None
    assert baseline["features"]["industry_rs60"] is not None
    assert missing["diagnostics"]["rule_baseline"]["features"]["industry_status"] == "UNKNOWN"
    assert missing["diagnostics"]["rule_baseline"]["features"]["industry_rs20"] is None
    assert report["diagnostics"]["native_evidence"]["source_packet"]["industry_input"]["symbol"] == "2330"
    assert report["diagnostics"]["entry_admission"]["policy_id"] == missing["diagnostics"]["entry_admission"]["policy_id"]
    assert report["pae_authorization"]["live_order_authorized"] is False


def test_validator_recomputes_industry_feature_and_source_binding():
    original = native()
    changed = deepcopy(original)
    changed["diagnostics"]["rule_baseline"]["features"]["industry_rs20"] = 999.
    with pytest.raises(ValueError, match="RECOMPUTATION_MISMATCH"):
        validate_report(changed)
    changed = deepcopy(original)
    changed["diagnostics"]["rule_baseline"]["industry_score"]["status"] = "FAKE"
    with pytest.raises(ValueError, match="RECOMPUTATION_MISMATCH"):
        validate_report(changed)
    changed = deepcopy(original)
    changed["diagnostics"]["native_evidence"]["source_packet"]["industry_input"]["symbol"] = "OTHER"
    with pytest.raises(ValueError):
        validate_report(changed)


def test_cached_evidence_and_attrs_modes_and_spawn_preserve_native_result():
    from infrastructure.daily_parallel import DailyProcessExecutor
    m = market(); profits = [[1, 1, 1, 1, .01]]
    serial = AnalysisDatasetBuilder().build_decisions(m, profits)
    assert serial == AnalysisDatasetBuilder(optimized_daily=False).build_decisions(m, profits)
    assert serial == AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2)).build_decisions(m, profits)
    from infrastructure.report.low_entry_v7_renderer import LowEntryV7Renderer
    html = LowEntryV7Renderer().render(serial)
    assert 'data-entry-policy="R11"' in html and "沿用 R10 估值／風控" in html


def test_zero_cash_changes_only_authorization_not_score_or_gates():
    m = market(); full = native(m)
    empty = native(replace(m, les_v6_portfolio=dict(m.les_v6_portfolio, available_cash=0.)))
    assert full["envelope"] == empty["envelope"]
    assert full["diagnostics"] == empty["diagnostics"]
    assert empty["pae_authorization"]["authorized"] is False

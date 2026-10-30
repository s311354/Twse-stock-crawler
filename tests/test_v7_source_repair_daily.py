"""SYNTHETIC ONLY: R6 daily ports and old-policy compatibility, no live orders."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from application.low_entry_v7_daily import daily_report
from application.market_history import MarketHistoryAssembler
from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.les_v7.contracts import identity
from domain.les_v7.financial_sources_r5 import audit_proof_r5
from domain.les_v7.official_pe import OfficialPeIndex, build_snapshot
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import BASELINE_POLICY, SOURCE_BASELINE_POLICY, baseline_policy
from domain.les_v7.source_projection import join_official_pe
from infrastructure.crawler.security_master import select_security_master
from infrastructure.runtime.native_source_capture import NativeSourceCapture
from tests.test_security_master import packet
from tests.test_official_pe import snapshot, payload
from tests.test_v7_eps_price_daily import rule_market_with_earnings
from tests.strategies.test_daily_v7_rule_baseline import CUTOFF


def repaired_market(*, master=True):
    market = rule_market_with_earnings()
    original = deepcopy(market.fundamental_snapshots)
    for row in original["2330"]["les_v6_mops_filings"]:
        import pandas as pd
        period = pd.Period(row["period_end"], freq="Q")
        proof = row["audit_evidence"]
        facts = tuple(FiledFact(**dict(fact, dimensions=tuple(tuple(d) for d in fact.get("dimensions", ()))))
                      for fact in proof["contexts"])
        filing = FiledFinancials(period.year, period.quarter, row["available_at"], row["revision_number"], facts,
            content_sha256=row["content_sha256"], source_url=row["source_url"], statement_scope=row["scope"],
            period_end=row["period_end"], audit_status=proof.get("audit_status"))
        row["audit_evidence_r5"] = audit_proof_r5(filing, symbol="2330", original_verified=True)
    security = select_security_master(packet(), "2330", CUTOFF) if master else None
    return replace(market, fundamental_snapshots=original, les_v7_source_repair=True,
        native_source_selections={"2330": dict(security=security, official_pe=[])})


def test_r6_daily_quality_and_identity_use_new_version_without_mutating_sources():
    market = repaired_market(); before = deepcopy(market.fundamental_snapshots)
    report = daily_report(MarketHistoryAssembler().build(market, 0))
    validate_report(report)
    assert report["diagnostics"]["rule_baseline"]["policy"]["policy_version"] == SOURCE_BASELINE_POLICY
    assert report["diagnostics"]["quality"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R5"
    assert report["envelope"]["quality"]["status"] == "PASS"
    gates = {item["gate_id"]: item["state"] for item in report["diagnostics"]["gate_receipts"]}
    assert gates["DATA_INTEGRITY"] == "PASS"
    assert gates["VALUATION"] == "FAIL"  # Same unchanged annual economics.
    assert not report["pae_authorization"]["authorized"]
    assert market.fundamental_snapshots == before


@pytest.mark.parametrize("mode", ["reported", "not_reported", "conflict", "unchanged"])
def test_r6_shared_features_follow_pe_join_and_match_uncached_result(mode, monkeypatch):
    import pandas as pd
    from domain.features import SharedFeaturePipeline
    from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
    market = repaired_market()
    history = MarketHistoryAssembler().build(market, 0)
    day = str(history.Date.iloc[-1])[:10]
    value = pd.to_numeric(history.PE, errors="coerce").iloc[-1]
    if mode == "reported":
        history.loc[history.index[-1], "PE"] = float("nan")
        pe = 12.5
    elif mode == "not_reported":
        history.loc[history.index[-1], "PE"] = 12.5
        pe = "-"
    elif mode == "conflict":
        history.loc[history.index[-1], "PE"] = 99.
        pe = 12.5
    else:
        pe = float(value)
    body = payload(symbol="2330", pe=str(pe))
    body["date"] = day.replace("-", "")
    body["data"][0][2] = str(history.Close.iloc[-1] / history.attrs[PRICE_SCALE_ATTR])
    observed = build_snapshot(json.dumps(body).encode(), market="TWSE", trade_date=day,
        fetched_at=CUTOFF)
    selected = OfficialPeIndex([observed]).select(market="TWSE", symbol="2330",
        trade_date=day, cutoff=CUTOFF)
    history.attrs["les_v7_native_sources"] = dict(history.attrs["les_v7_native_sources"], official_pe=[selected])
    pipeline = SharedFeaturePipeline()
    _, shared = pipeline.build_price_pair(history, divisor=history.attrs[PRICE_SCALE_ATTR])
    original = history.copy(deep=True)
    import application.low_entry_v7_daily as module
    strategy = module.get_strategy("low_entry_score_v7")
    observed_features = []
    class ObservedStrategy:
        def evaluate(self, data, **kwargs):
            observed_features.append(kwargs["features"])
            return strategy.evaluate(data, **kwargs)
    monkeypatch.setattr(module, "get_strategy", lambda name: ObservedStrategy())
    result = daily_report(history, features=shared)
    assert (observed_features[0] is shared) == (mode == "unchanged")
    assert result == daily_report(history)
    validate_report(result)
    pd.testing.assert_frame_equal(history, original)
    assert shared.source_fingerprint == pipeline.build_price_pair(history,
        divisor=history.attrs[PRICE_SCALE_ATTR])[1].source_fingerprint


def test_r6_pe_join_does_not_hide_unrelated_stale_feature_input():
    from domain.features import SharedFeaturePipeline
    history = MarketHistoryAssembler().build(repaired_market(), 0)
    _, shared = SharedFeaturePipeline().build_price_pair(history)
    history.loc[history.index[-1], "Open"] -= 1.
    with pytest.raises(ValueError, match="FeatureFrame source data does not match"):
        daily_report(history, features=shared)


def test_r6_official_pe_join_matches_optimized_unoptimized_and_spawn_daily_pipeline():
    from application.analysis_dataset import AnalysisDatasetBuilder
    from infrastructure.daily_parallel import DailyProcessExecutor
    market = repaired_market()
    ratios = list(market.pe_ratios[0]); ratios[-1] = float("nan")
    day = market.dates[0][-1]
    body = payload(symbol="2330", pe="12.5")
    body["date"] = day.replace("-", "")
    body["data"][0][2] = str(market.closes[0][-1] / 1000.)
    observed = build_snapshot(json.dumps(body).encode(), market="TWSE", trade_date=day,
        fetched_at=CUTOFF)
    selection = OfficialPeIndex([observed]).select(market="TWSE", symbol="2330",
        trade_date=day, cutoff=CUTOFF)
    sources = deepcopy(market.native_source_selections)
    sources["2330"]["official_pe"] = [selection]
    market = replace(market, pe_ratios=[ratios], native_source_selections=sources)
    profits = [[1, 1, 1, 1, .01]]
    optimized = AnalysisDatasetBuilder().build_decisions(market, profits)
    assert optimized == AnalysisDatasetBuilder(optimized_daily=False).build_decisions(market, profits)
    assert optimized == AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2)).build_decisions(market, profits)
    assert __import__("math").isnan(market.pe_ratios[0][-1])


def test_missing_master_cannot_silently_use_old_four_digit_regex():
    report = daily_report(MarketHistoryAssembler().build(repaired_market(master=False), 0))
    validate_report(report)
    states = {item["gate_id"]: item["state"] for item in report["diagnostics"]["gate_receipts"]}
    assert states["DATA_INTEGRITY"] == "UNKNOWN"
    assert not report["pae_authorization"]["authorized"]


def test_r6_cannot_fall_back_to_legacy_prebound_gate_receipts_or_external_producer():
    from domain.les_v7.engine import evaluate
    from tests.strategies.test_v7_rule_baseline import legal_bars
    context = dict(symbol="2330", market="TWSE", as_of=CUTOFF, rule_baseline=True,
        rule_baseline_policy_version=SOURCE_BASELINE_POLICY, filings=[])
    with pytest.raises(ValueError, match="R6_NATIVE_SOURCE_PRODUCER_REQUIRED"):
        evaluate(legal_bars(), context)
    with pytest.raises(ValueError, match="R6_NATIVE_SOURCE_PRODUCER_REQUIRED"):
        evaluate(legal_bars(), context, receipt_producer=lambda **kwargs: {})


def test_r6_report_cannot_drop_native_sources_to_use_legacy_receipt_validation():
    report = daily_report(MarketHistoryAssembler().build(repaired_market(), 0))
    report["diagnostics"].pop("native_evidence")
    with pytest.raises(ValueError, match="R6_NATIVE_SOURCE_EVIDENCE_REQUIRED"):
        validate_report(report)


def test_old_default_policy_identity_and_economics_are_unchanged():
    old, new = baseline_policy(), baseline_policy(version=SOURCE_BASELINE_POLICY)
    assert old["policy_version"] == BASELINE_POLICY
    assert identity(old) == "2401d1f1185716e7f7b5b456b310c73d8136ac73b8860963f3dd017755c4fd2e"
    for key in ("config", "model_spec", "valuation_policy", "valuation_input_policy", "score_policy", "context_policy", "native_risk_limits"):
        assert new[key] == old[key]
    assert "source_policy" not in old


def test_exact_date_pe_join_preserves_original_rows_and_detects_conflicts():
    original = [dict(Date="2026-10-13", Close=100., PE=None)]
    index = OfficialPeIndex([snapshot(symbol="2330")])
    selected = index.select(market="TWSE", symbol="2330", trade_date="2026-10-13", cutoff=CUTOFF)
    joined, diagnostics = join_official_pe(original, [selected], symbol="2330", market="TWSE", cutoff=CUTOFF)
    assert joined[0]["PE"] == 12.5 and original[0]["PE"] is None and not diagnostics
    joined, diagnostics = join_official_pe([dict(original[0], PE=10.)], [selected],
        symbol="2330", market="TWSE", cutoff=CUTOFF)
    assert joined[0]["PE"] is None and diagnostics[0]["reason"] == "OFFICIAL_PE_QUOTE_CONFLICT"


def test_tpex_quote_without_dated_proof_does_not_create_pe():
    joined, diagnostics = join_official_pe([dict(Date="2026-10-13", Close=100., PE=50.)], [],
        symbol="2330", market="TPEX", cutoff=CUTOFF)
    assert joined[0]["PE"] is None and not diagnostics


def test_uncached_older_pe_observation_is_missing_not_a_crash_or_positive_fallback():
    selection = OfficialPeIndex([]).select(market="TPEX", symbol="2330",
        trade_date="2026-10-13", cutoff=CUTOFF)
    joined, diagnostics = join_official_pe([dict(Date="2026-10-13", Close=100., PE=50.)], [selection],
        symbol="2330", market="TPEX", cutoff=CUTOFF)
    assert joined[0]["PE"] is None and diagnostics == []
    selection["pe"] = 99.
    with pytest.raises(ValueError):
        join_official_pe([dict(Date="2026-10-13", Close=100., PE=50.)], [selection],
            symbol="2330", market="TPEX", cutoff=CUTOFF)


@pytest.mark.parametrize("damage", ["source_policy", "master", "quote_rows", "pe_fields"])
def test_r6_source_rehashes_cannot_bypass_full_report_validation(damage):
    report = daily_report(MarketHistoryAssembler().build(repaired_market(), 0))
    native = report["diagnostics"]["native_evidence"]
    if damage == "source_policy": native["source_policy"]["missing"] = "PASS"
    elif damage == "master": native["source_packet"]["security_evidence"]["issuer_symbol"] = "OTHER"
    elif damage == "quote_rows": native["source_packet"]["quote_market_rows"][-1]["Close"] += 1
    else: native["source_packet"]["market_rows"][-1]["PE"] = 999
    with pytest.raises(ValueError): validate_report(report)


def test_capture_is_three_bounded_batch_calls_and_retries_do_not_reuse_failed_master(tmp_path):
    from unittest.mock import Mock, patch
    collector = Mock(side_effect=ValueError("SYNTHETIC_UNAVAILABLE"))
    pe_client = Mock(); pe_client.fetch.return_value = snapshot()
    repository = Mock(); repository.load_as_of.return_value = [snapshot()]
    clock = lambda: __import__("datetime").datetime.fromisoformat(CUTOFF)
    crawler = SimpleNamespace(iso_scheduled_times=["2026-10-13"], stocknumbers=["2330"],
                              daily_dates=[["2026-10-13"]])
    capture = NativeSourceCapture(tmp_path, collector=collector, pe_client=pe_client,
                                  pe_repository=repository, clock=clock)
    capture.master = object()
    capture.collect(crawler, market="TWSE")
    assert capture.master is None
    assert collector.call_count == pe_client.fetch.call_count == 1
    assert repository.record.call_count == repository.load_as_of.call_count == 1
    selections = capture.selections(crawler, cutoff=CUTOFF)
    assert selections["2330"]["security"] is None
    assert len(selections["2330"]["official_pe"]) == 1

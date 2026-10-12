"""Causal research hypotheses; synthetic prices/proofs are not performance evidence."""
from copy import deepcopy
from dataclasses import replace
import base64
import json

import numpy as np
import pandas as pd
import pytest

from application.industry_benchmark import prepare_industry_benchmark
from application.market_history import MarketHistoryAssembler
from application.low_entry_v6_daily import _daily_report
from application.replay_contracts import digest
from application.replay_inputs import PinnedReplayEvidence
from application.research_audit import coverage_audit
from domain.strategies.low_entry_score_v6 import LowEntryScoreV6Strategy, attach_low_entry_v6_context
from domain.strategies.low_entry_v6_leadership import leadership_features, recognition
from domain.strategies.low_entry_v6_signals import LowEntryV6Config, decide, score_setups, trade_plan
from domain.strategies.low_entry_v6_report import report_failure
from infrastructure.report.low_entry_v6_renderer import setup_score_summary
from infrastructure.replay_runtime import build_replay_engine, current_versions
from infrastructure.research_archive import create_dataset
from infrastructure.research_job import ResearchJob
from interface.les_v6 import analyze_request, main
from tests.test_industry_benchmark import packet, add_row
from tests.test_strict_replay import payload_fixture, request
from tests.test_research_archive import raw_fixture
from tests.strategies.test_low_entry_score_v6 import bars, features, financials
from tests.strategies.test_low_entry_v6_unified_decision import verified_quality
from tests.strategies.test_daily_v6_pipeline import daily_market


def base():
    f = pd.DataFrame(dict(open=99.9, high=101., low=99.4, close=100., volume=1_000_000.,
        amount=100_000_000., atr14=2., support=99.4, hh20=103., clv=.9, breakout_extension=.05,
        taiex=100., market_ready=True), index=range(40))
    f.loc[29:33, "high"] = 103.
    f.loc[34:38, ["volume", "amount"]] = [500_000., 50_000_000.]
    f.loc[19:39, "taiex"] = np.linspace(100, 94, 21)
    f.loc[39, ["open", "high", "low", "close", "volume", "amount"]] = [100.5, 103.2, 100., 103.1, 1_500_000., 154_650_000.]
    return f


def advanced(**changes):
    return features(industry_rs20=.03, industry_rs60=.05, downside_resilience=True,
                    base_compression=True, clv_breakout_quality=1., breakout_volume_ratio=1.3,
                    breakout_amount_ratio=1.3, **changes)


def test_prior_base_compresses_on_volume_and_repeated_causal_support_tests():
    source = base()
    row = leadership_features(source).iloc[-1].to_dict()
    assert row["base_compression"] is True and row["base_support"] == 99.4
    assert row["base_support_tests"] == 10
    assert row["base_range_atr10"] == pytest.approx(1.8)
    assert row["base_range_ratio5_10"] < .8 and row["base_volume_ratio5_20"] < .8
    assert row["breakout_quality"] is True and row["breakout_amount_ratio"] == pytest.approx(154.65 / 87.5)
    assert row["breakout_volume_ratio"] == pytest.approx(1.5 / .875)
    assert row["downside_capture20"] == 0 and row["downside_resilience"] is True
    assert recognition(dict(row, industry_rs20=.03))["leading_breakout"] is True


@pytest.mark.parametrize("column,value", [("low", 98.), ("high", 120.), ("volume", 2_000_000.)])
def test_broken_support_wide_base_or_expanding_volume_are_not_compression(column, value):
    source = base(); source.loc[34:38, column] = value
    assert leadership_features(source).iloc[-1]["base_compression"] is False


@pytest.mark.parametrize("column,value", [("clv", .49), ("volume", 900_000.),
    ("amount", 80_000_000.), ("breakout_extension", .751), ("close", 103.)])
def test_breakout_quality_does_not_reward_weak_close_turnover_or_overextension(column, value):
    source = base(); source.loc[39, column] = value
    assert leadership_features(source).iloc[-1]["breakout_quality"] is False


@pytest.mark.parametrize("column,value", [("atr14", 0.), ("volume", 0.), ("open", np.nan), ("low", 110.)])
def test_invalid_prior_bar_does_not_turn_missing_base_into_false_or_zero(column, value):
    source = base(); source.loc[34, column] = value
    row = leadership_features(source).iloc[-1]
    assert row["base_compression"] is None and pd.isna(row["base_range_atr10"])
    assert row["breakout_quality"] is None and row["downside_resilience"] is None


def test_no_material_market_drawdown_missing_market_and_stock_underperformance_are_distinct():
    source = base(); source["taiex"] = 100.
    assert leadership_features(source).iloc[-1]["downside_resilience"] is None
    source = base(); source.loc[25, "taiex"] = np.nan
    assert leadership_features(source).iloc[-1]["downside_resilience"] is None
    source = base()
    source.loc[19:39, "close"] = np.linspace(100., 90., 21)
    source.loc[19:39, "open"] = source.loc[19:39, "close"]
    source.loc[19:39, "low"] = source.loc[19:39, "close"] - .1
    row = leadership_features(source).iloc[-1]
    assert row["downside_resilience"] is False and row["downside_capture20"] > 1
    source = base(); source.loc[38, "low"] = 99.
    assert leadership_features(source).iloc[-1]["downside_resilience"] is False


def test_feature_prefix_invariance_no_future_support_or_volume_backfill():
    source = base()
    expected = leadership_features(source)
    extra = source.iloc[-1:].copy(); extra.index = [40]
    extra.loc[40, ["low", "support", "volume", "taiex"]] = [1., 1., 100_000_000., 1.]
    pd.testing.assert_frame_equal(leadership_features(pd.concat([source, extra])).iloc[:40], expected)
    assert recognition({})["resilient_base"] is None


@pytest.mark.parametrize("row,field,value", [(39, "amount", np.nan), (29, "amount", 0.),
                                           (39, "breakout_extension", np.nan)])
def test_missing_turnover_or_extension_is_unknown_not_bad_breakout(row, field, value):
    source = base(); source.loc[row, field] = value
    assert leadership_features(source).iloc[-1]["breakout_quality"] is None


def test_replaced_slots_keep_budget_one_hundred_and_other_setup_ledgers_unchanged():
    f = advanced()
    config = LowEntryV6Config()
    old = replace(config, leadership_policy_version=1)
    prior = score_setups(f, trade_plan(f, config=old), config=old)
    current = score_setups(f, trade_plan(f, config=config), config=config)
    for name in ("REVERSAL", "TREND_PULLBACK"):
        assert current[name] == prior[name]
    ledger = current["LEADERSHIP_CONTINUATION"]["score_evidence"]
    assert ledger["total_weight"] == ledger["available_weight"] == 100
    assert sum(c["weight"] for c in ledger["conditions"]) == 100
    assert sum(c["weight"] for c in ledger["conditions"] if c["group"] == "execution") == 10
    assert "industry_rs20" in ledger["conditions"][0]["dependencies"]
    missing = score_setups(features(), trade_plan(features(), config=config), config=config)["LEADERSHIP_CONTINUATION"]["score_evidence"]
    assert {"industry_rs20", "industry_rs60", "downside_resilience", "base_compression"} <= set(missing["missing_fields"])
    assert missing["available_weight"] < 100 and missing["earned_points"] < ledger["earned_points"]
    for v in (0, 3, True, "2"):
        with pytest.raises(ValueError): LowEntryV6Config(leadership_policy_version=v)


@pytest.mark.parametrize("changes,q", [({"restricted": None}, None), ({"restricted": True}, None),
    ({"support": 80.}, None), ({"market_ready": False}, None),
    ({}, verified_quality(status="REJECT")), ({}, verified_quality(status="DATA_INSUFFICIENT"))])
def test_new_high_quality_pattern_never_bypasses_original_hard_gates(changes, q):
    result = decide(q or verified_quality(), advanced(**changes), config=LowEntryV6Config())
    assert result["signal"] not in {"BUY", "BUY_SMALL"} and result["risk_fraction"] == 0


def native(source, benchmark):
    data = attach_low_entry_v6_context(source, symbol="2330", sector="SEMI", filings=financials(),
        as_of="2026-09-11T13:30:00+08:00", industry_benchmark=prepare_industry_benchmark(benchmark))
    return LowEntryScoreV6Strategy().run_latest(data).iloc[0]["LES v6 Report"]


def test_stock_leader_vs_entire_industry_rally_and_exact_date_gap():
    source = bars(61)
    matched = packet(source["Date"], source["Close"])
    broad_rally = native(source, matched)
    assert broad_rally["technical"]["industry_rs20"] == pytest.approx(0.)
    assert broad_rally["technical"]["industry_rs60"] == pytest.approx(0.)
    own_leader = native(source, packet(source["Date"], np.full(61, 100.)))
    assert own_leader["technical"]["industry_rs20"] > 0 and own_leader["technical"]["industry_rs60"] > 0
    # Missing one intermediate index session, not just endpoints: no compress/ffill.
    missing = packet(source["Date"], np.full(61, 100.)); missing["bars"].pop(-10)
    report = native(source, missing)
    assert report["technical"]["industry_rs20"] is report["technical"]["industry_rs60"] is None
    assert report["industry_benchmark"]["status"] == "UNKNOWN"


def test_offline_cli_native_api_and_future_revision_have_identical_results(tmp_path, capsys):
    source = bars(61); source["Date"] = source["Date"].dt.strftime("%Y-%m-%d")
    evidence = packet(source["Date"])
    payload = dict(bars=source.to_dict("records"), context=dict(symbol="2330", sector="SEMI",
        as_of="2026-09-11T13:30:00+08:00", filings=financials(), industry_benchmark=evidence))
    original = deepcopy(payload)
    expected = analyze_request(payload)
    assert expected["reports"][0] == native(source, evidence)
    assert payload == original
    # The ordinary fixture is not an executable trade; new diagnostics cannot authorize it.
    assert report_failure(expected["reports"][0]) == "HARD_VETO_OR_RISK_FAIL"
    path = tmp_path / "TEST_ONLY.json"; path.write_text(json.dumps(payload))
    assert main([str(path)]) == 0 and json.loads(capsys.readouterr().out) == expected
    row = evidence["bars"][-1]
    add_row(evidence, "bar", dict(trade_date=row["trade_date"], close=999., membership_version=row["membership_version"],
        revision=1, series_version=row["series_version"], published_at="2026-09-12T15:00:00+08:00"))
    assert analyze_request(payload) == expected
    html = setup_score_summary(expected["reports"][0])
    assert "產業基準 PASS" in html and "抗跌盤整 UNKNOWN" in html


def test_daily_and_pinned_replay_import_same_verified_sector_evidence(tmp_path):
    market = daily_market()
    evidence = packet(market.dates[0])
    market = replace(market, industry_benchmarks={"24": prepare_industry_benchmark(evidence)})
    assembled = MarketHistoryAssembler().build(market, 0)
    daily = _daily_report(assembled)
    assert daily["industry_benchmark"]["status"] == "PASS"
    payload = payload_fixture(); payload["industry_benchmarks"] = {"24": evidence}
    pinned = PinnedReplayEvidence(digest(payload), payload)
    selected = pinned.select(request(tmp_path, payload))
    replay = _daily_report(MarketHistoryAssembler().build(selected.market, 0))
    assert replay["technical"]["industry_rs60"] == daily["technical"]["industry_rs60"]
    assert replay["leadership"] == daily["leadership"]
    assert replay["industry_benchmark"] == daily["industry_benchmark"]
    damaged = deepcopy(payload); damaged["industry_benchmarks"]["24"]["bars"][0]["close"] = 999.
    invalid = PinnedReplayEvidence(digest(damaged), damaged).select(request(tmp_path, damaged))
    assert invalid.market.industry_benchmarks == {}
    assert invalid.diagnostics["industry_benchmark_import_errors"]["24"] == "INDUSTRY_EXTRACTION_NOT_VERIFIED"


def test_industry_raw_references_cache_on_off_full_research_and_coverage_parity(tmp_path):
    payload = raw_fixture()
    source = packet(daily_market().dates[0][-61:])
    for doc in source["documents"].values():
        doc["body_base64"] = base64.b64encode(doc.pop("body_utf8").encode()).decode()
    payload["industry_benchmarks"] = {"24": source}
    frozen = deepcopy(payload)
    file = create_dataset(payload, tmp_path / "TEST_ONLY_datasets", purpose="TEST_ONLY")
    assert payload == frozen
    assert "body_base64" not in file.read_text()
    req = replace(request(tmp_path, payload), evidence_id=file.parent.name, versions=current_versions())
    engine, _ = build_replay_engine(file)
    expected = engine.run(req)
    coverage = coverage_audit(req, engine.evidence.select(req), expected)
    with ResearchJob(file, tmp_path / "on") as on, ResearchJob(file, tmp_path / "off", cache_enabled=False) as off:
        a, b = on.run_day(req), off.run_day(req)
        assert a.result == b.result == expected
        assert a.coverage == b.coverage == coverage and a.state_id == b.state_id
        # Cutoff selection is re-evaluated; immutable import alone is cached.
        on.evidence.select(req)
        assert on.cache_stats["industry_benchmark_hit"] >= 1

"""Synthetic score-only ablations; no empirical alpha or historical proof claims."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from domain.strategies.low_entry_v6_correlation import CHANNELS, correlation_adjustment
from domain.strategies.low_entry_v6_decision import decide_unified
from domain.strategies.low_entry_v6_research import feature_ablation_audit
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from domain.strategies.low_entry_v6_signals import LowEntryV6Config, decide
from infrastructure.report.low_entry_v6_renderer import setup_score_summary
from interface.les_v6_research import analyze_scores, main
from tests.strategies.test_low_entry_score_v6 import features, bars, financials, raw_execution_report
from tests.strategies.test_low_entry_v6_unified_decision import verified_quality, unified_report

# Isolate source-family pooling from the subsequent industry-evidence policy.
CONFIG = LowEntryV6Config(leadership_policy_version=1)


def evaluate(f=None, *, config=CONFIG, q=None):
    return decide(q or verified_quality(), f if f is not None else features(), config=config)


def report(day="2026-09-11", **changes):
    result = unified_report()
    result["parameters"] = vars(CONFIG)
    result["technical"] = features(**changes)
    result["signal_date"] = day
    result["decision_time"] = day + "T13:30:00+08:00"
    decision = evaluate(result["technical"])
    result["decision"] = decision
    result["setups"] = decision["setups"]
    result["entry"] = dict(decision["plan"], trigger_price=result["technical"]["high"],
        starter_trigger=decision["starter_trigger"], confirmed_trigger=decision["confirmed_trigger"],
        rule="NEXT_SESSION_BREAKS_SIGNAL_HIGH", is_fill=False)
    return json.loads(json.dumps(result))


def test_default_source_family_pooling_really_changes_native_final_not_raw_ledger():
    current = evaluate()
    additive = evaluate(config=replace(CONFIG, correlation_policy_version=1))
    assert additive["score"] == 100 and current["score"] == 87
    assert current["raw_score"] == 100 and current["penalty"] == 13
    assert current["signal"] == "BUY" and current["position_multiplier"] == 1
    for name, removed in (("REVERSAL", 23), ("TREND_PULLBACK", 13), ("LEADERSHIP_CONTINUATION", 7)):
        actual, before = current["setup_decisions"][name], additive["setup_decisions"][name]
        assert actual["adjusted_score"] == before["adjusted_score"] - removed
        assert actual["raw_score"] == before["raw_score"]
        assert actual["correlation_adjustment"]["removed_points"] == removed
        assert actual["execution_score"] == before["execution_score"]
        assert actual["safety_cap"] == before["safety_cap"]
        assert actual["entry_checks"]["BUY"]["minimum_rr"] == before["entry_checks"]["BUY"]["minimum_rr"]
        evidence = current["setups"][name]["score_evidence"]
        assert evidence == additive["setups"][name]["score_evidence"]
        assert evidence["total_weight"] == 100
    assert current["plan"] == additive["plan"]


def test_macd_and_breakout_do_not_stack_when_ema_already_provides_the_stronger_axis():
    all_on = evaluate()
    macd_off = evaluate(features(macd_improving=False, macd_strict=-2.))
    breakout_off = evaluate(features(break_previous_high=False, reclaim10=False, reclaim20=False))
    for name in ("REVERSAL", "TREND_PULLBACK"):
        assert all_on["setup_decisions"][name]["adjusted_score"] == macd_off["setup_decisions"][name]["adjusted_score"]
        assert all_on["setup_decisions"][name]["adjusted_score"] == breakout_off["setup_decisions"][name]["adjusted_score"]
    assert macd_off["raw_score"] < all_on["raw_score"]  # Only the redundant additive contribution changed.


def test_flow_votes_compete_instead_of_rewarding_the_same_move_three_times():
    full = evaluate()
    clv_only = evaluate(features(cmf_improving=False, cmf20=-.1, obv_slope5=-1.))
    for name in CHANNELS:
        assert full["setup_decisions"][name]["adjusted_score"] == clv_only["setup_decisions"][name]["adjusted_score"]
        pool = full["setup_decisions"][name]["correlation_adjustment"]["pools"]["BUYING_FLOW"]
        assert pool["retained_points"] == max(
            full["setup_decisions"][name]["correlation_adjustment"]["channels"][c]["retained_points"]
            for c in ("CLV", "CMF", "OBV"))


def test_missing_channels_do_not_earn_points_and_all_missing_keeps_null_final():
    f = features(score_availability=dict(macd_improving=False, macd_strict=False,
                                       macd_signal_strict=False, cmf_improving=False, obv_slope5=False))
    result = evaluate(f)
    for setup in result["setup_decisions"].values():
        channels = setup["correlation_adjustment"]["channels"]
        assert channels["MACD"]["retained_points"] == 0
        assert channels["OBV"]["retained_points"] == 0
    empty = evaluate({key: None for key in features()})
    assert empty["score"] is None and empty["raw_score"] is None
    assert empty["signal"] == "WATCH" and empty["risk_fraction"] == 0


def test_penalties_do_not_double_count_conditions_or_change_risk_off_sizing():
    result = evaluate(features(market_regime="RISK_OFF"))
    assert result["penalty"] == 18
    assert [p["reason"] for p in result["penalties"]].count("RISK_OFF") == 1
    assert result["position_multiplier"] == .5
    for setup in result["setup_decisions"].values():
        pools = setup["correlation_adjustment"]["pools"]
        keys = [key for pool in pools.values() for key in pool["condition_ids"]]
        assert len(keys) == len(set(keys))
        assert all(0 <= p["retained_points"] <= p["additive_points"] for p in pools.values())


@pytest.mark.parametrize("changes,q", [({"restricted": None}, None), ({"restricted": True}, None),
    ({"support": 80.}, None), ({"market_ready": False}, None),
    ({}, verified_quality(status="REJECT")), ({}, verified_quality(status="DATA_INSUFFICIENT"))])
def test_dedup_and_feature_ablation_cannot_bypass_hard_gates(changes, q):
    f = features(**changes)
    native = evaluate(f, q=q)
    removed = decide_unified(q or verified_quality(), f, valuation="NOT_AVAILABLE", config=CONFIG,
                             score_exclusions=frozenset({"MACD", "EMA", "BREAKOUT"}))
    assert native["signal"] not in {"BUY", "BUY_SMALL"}
    assert removed["signal"] not in {"BUY", "BUY_SMALL"}
    assert removed["risk_fraction"] == 0 and removed["executable"] is False


def test_single_channel_can_inform_the_score_without_becoming_three_votes():
    f = features()
    result = evaluate(f)
    evidence = deepcopy(result["setups"]["REVERSAL"]["score_evidence"])
    for condition in evidence["conditions"]:
        if condition["id"] in CHANNELS["REVERSAL"]["EMA"] + CHANNELS["REVERSAL"]["BREAKOUT"]:
            condition.update(fraction=0., earned_points=0.)
    adjustment = correlation_adjustment("REVERSAL", evidence)
    assert adjustment["channels"]["MACD"]["retained_points"] == 5
    assert adjustment["pools"]["TREND_CONFIRMATION"]["retained_points"] == 5
    assert adjustment["pools"]["TREND_CONFIRMATION"]["removed_points"] == 4


def test_condition_identity_and_source_semantics_are_checked_before_pooling():
    evidence = deepcopy(evaluate()["setups"]["REVERSAL"]["score_evidence"])
    evidence["conditions"][0]["earned_points"] = -1
    with pytest.raises(ValueError, match="INVALID_CORRELATION_CONDITION"):
        correlation_adjustment("REVERSAL", evidence)
    evidence = deepcopy(evaluate()["setups"]["REVERSAL"]["score_evidence"])
    next(c for c in evidence["conditions"] if c["id"] == "momentum:1")["dependencies"] = ["rsi_cross40"]
    with pytest.raises(ValueError, match="CONDITION_CONTRACT_MISMATCH"):
        correlation_adjustment("REVERSAL", evidence)


def test_native_report_and_html_validate_and_reveal_exact_dedup_penalties():
    source = report()
    assert report_failure(source) is None
    assert "相關訊號去重：-13.00" in setup_score_summary(source)
    changed = deepcopy(source)
    changed["decision"]["correlation_adjustment"]["removed_points"] = 0
    assert entry_gate_checks(changed)["score"] == "FAIL"
    assert report_failure(changed) == "V6_SCORE_OR_SETUP_MISMATCH"
    counterfactual = deepcopy(source)
    counterfactual["decision"]["research_only"] = True
    assert report_failure(counterfactual) == "RESEARCH_ONLY_SCORE_INTERVENTION"


def test_research_recalculates_native_decisions_separates_partitions_and_preserves_inputs():
    sources = [report(f"2026-09-{i:02}", clv=.2 + i * .1) for i in range(1, 7)]
    original = deepcopy(sources)
    result = feature_ablation_audit(sources, validation_start="2026-09-04")
    assert sources == original
    assert result == feature_ablation_audit(sources, validation_start="2026-09-04")
    assert not result["executable"] and not result["performance_validated"]
    assert result["split_status"] == "BOTH_PARTITIONS"
    assert all(s["samples"] == 3 for s in result["summary"])
    macd = [s for s in result["summary"] if s["variant"] == "NO_MACD"]
    assert all(s["retention_ratio"] == 1 for s in macd)
    assert all(row["final_delta"] == 0 for row in result["rows"] if row["variant"] == "NO_MACD")
    # Policy-defined pooling alone is not empirical evidence of redundancy:
    # also remove the same source under the former additive decision policy.
    additive_macd = [r for r in result["rows"] if r["variant"] == "ADDITIVE_NO_MACD"]
    assert len(additive_macd) == 6 and all(row["final_delta"] == -2 for row in additive_macd)
    assert all(row["reference"]["final"] == 100 for row in additive_macd)
    assert all(s["reference_policy"] == "additive-source-confirmations/v1"
               for s in result["summary"] if s["variant"] == "ADDITIVE_NO_MACD")
    assert all(row["final_delta"] >= 0 for row in result["rows"] if row["variant"] == "ADDITIVE_BASELINE")
    assert all(c["pearson"] is None for c in result["comparisons"] if c["complete_samples"] < 3)


def test_no_entry_samples_retention_is_unknown_not_zero_or_success():
    result = feature_ablation_audit([report(restricted=None)], validation_start="2026-09-11")
    assert all(s["retention_ratio"] is None for s in result["summary"])
    assert result["split_status"] == "INSUFFICIENT_DATE_SPAN"


@pytest.mark.usefixtures("active_industry_profiles")
def test_raw_plugin_report_is_recomputed_under_both_score_policies():
    source = raw_execution_report()
    original = deepcopy(source)
    result = feature_ablation_audit([source], validation_start="2026-09-11")
    baseline = next(row for row in result["rows"] if row["variant"] == "ADDITIVE_BASELINE")
    assert baseline["reference"]["final"] == 71 and baseline["reference"]["signal"] == "BUY_SMALL"
    assert baseline["result"]["raw"] == baseline["result"]["final"] == 89
    assert baseline["result"]["signal"] == "BUY"
    assert baseline["final_delta"] == 18
    assert all(
        source["entry"][key] == value for key, value in baseline["result"]["plan"].items())
    assert source == original


@pytest.mark.parametrize("field,value", [("final_score", 100), ("penalty", 0), ("research_only", True)])
def test_research_rejects_inconsistent_or_already_intervened_source_reports(field, value):
    source = report()
    source["decision"][field] = value
    with pytest.raises(ValueError, match="NATIVE_RESULT_MISMATCH"):
        feature_ablation_audit([source], validation_start="2026-09-11")


def test_ablation_does_not_erase_ema_veto_evidence_or_invent_missing_values():
    f = features(ema5_slope=-1., ema20_slope=-1., atr_expansion=2., r5=-.2)
    before = deepcopy(f)
    value = decide_unified(verified_quality(), f, valuation="NOT_AVAILABLE", config=CONFIG,
                           score_exclusions=frozenset({"EMA"}))
    assert f == before
    assert "ACCELERATING_DECLINE" in value["hard_vetoes"]
    assert value["score"] == 0 and value["signal"] == "AVOID"
    assert value["research_only"] and not value["executable"]
    with pytest.raises(ValueError, match="UNKNOWN_SCORE_ABLATION_CHANNEL"):
        decide_unified(verified_quality(), f, valuation="NOT_AVAILABLE", config=CONFIG,
                       score_exclusions=frozenset({"NOT_A_CHANNEL"}))


def test_future_reports_cannot_change_existing_rows_and_mixed_versions_rejected():
    first = feature_ablation_audit([report()], validation_start="2026-09-12")
    appended = feature_ablation_audit([report(), report("2026-09-12")], validation_start="2026-09-12")
    assert first["rows"] == appended["rows"][:len(first["rows"])]
    source = report(); source["parameter_version"] = "previous"
    with pytest.raises(ValueError, match="CURRENT_DEDUP_REPORTS"):
        feature_ablation_audit([source], validation_start="2026-09-12")
    with pytest.raises(ValueError, match="DUPLICATE"):
        feature_ablation_audit([report(), report()], validation_start="2026-09-12")


def test_feature_audit_cli_api_json_parity_and_missing_split_rejection(tmp_path, capsys):
    frame = bars(6); frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    packet = dict(bars=frame.to_dict("records"), context=dict(symbol="2330", sector="ELEC",
        as_of="2026-09-11T13:30+08:00", filings=financials()))
    source = tmp_path / "synthetic.json"; source.write_text(json.dumps(packet))
    expected = analyze_scores(packet, validation_start="2026-09-09", feature_audit=True)
    assert main([str(source), "--feature-audit", "--validation-start", "2026-09-09"]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert expected["source_posture"] == "USER_SOURCE_PACKET_NOT_VERIFIED_REPLAY_DATASET"
    with pytest.raises(SystemExit):
        main([str(source), "--feature-audit"])


@pytest.mark.parametrize("version", [0, 3, True, "2"])
def test_bad_correlation_policy_versions_rejected(version):
    with pytest.raises(ValueError, match="correlation policy"):
        LowEntryV6Config(correlation_policy_version=version)

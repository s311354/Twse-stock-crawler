"""Synthetic contract tests; not market returns, fills or calibration evidence."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from application.low_entry_v6_daily import project_report
from domain.strategies.low_entry_score_v6 import PARAMETER_VERSION
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from domain.strategies.low_entry_v6_research import ablation, score_audit, _correlation
from domain.strategies.low_entry_v6_signals import (
    SETUPS, decide, score_setups, setup_comparison, setup_rank, trade_plan,
)
from domain.strategies.low_entry_v6_valuation import valuation_context
from infrastructure.report.low_entry_v6_renderer import setup_score_summary
from interface.les_v6_research import analyze_scores, main
from tests.strategies.test_low_entry_score_v6 import (
    CONFIG, bars, execution_report, raw_execution_report, features, financials, quality, setup_features,
)


def scored(f):
    return score_setups(f, trade_plan(f, config=CONFIG), config=CONFIG)


@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("changes", [{}, {"clv": None}, {"rs60": None},
                                    {"cmf_improving": None}, {"obv_slope5": None}])
def test_single_ledger_ties_both_scores_to_observed_conditions(setup, changes):
    f = features(**changes)
    before = deepcopy(f)
    item = scored(f)[setup]
    ledger = item["score_evidence"]
    earned = sum(c["earned_points"] for c in ledger["conditions"])
    available = sum(c["weight"] for c in ledger["conditions"] if c["observed"])
    total = sum(c["weight"] for c in ledger["conditions"])
    assert total == 100
    assert item["score"] == item["formal_score"] == earned
    assert item["estimated_score"] == pytest.approx(100 * earned / available)
    assert item["score_coverage"] == available
    assert f == before


def test_missing_clv_cannot_inflate_formal_score_or_pass_veto_evidence():
    full = scored(setup_features("REVERSAL"))["REVERSAL"]
    f = setup_features("REVERSAL", clv=None)
    item = scored(f)["REVERSAL"]
    assert item["formal_score"] == full["formal_score"] - 6
    assert item["estimated_score"] > item["formal_score"]
    assert item["score_coverage"] == 94
    decision = decide(quality(), f, config=CONFIG)
    assert decision["signal"] == "DATA_INSUFFICIENT"
    assert decision["reasons"] == ["INCOMPLETE_VETO_EVIDENCE"]
    assert decision["risk_fraction"] == 0


def test_warmup_mask_cannot_be_relabelled_as_measured_or_no_veto():
    f = features(score_availability={"clv": False})
    assert scored(f)["REVERSAL"]["score_coverage"] == 94
    assert decide(quality(), f, config=CONFIG)["signal"] == "DATA_INSUFFICIENT"


@pytest.mark.parametrize("key", ["formal_score", "estimated_score", "score_coverage", "score_evidence"])
def test_serialized_score_observability_is_recomputed_not_trusted(key):
    report = execution_report()
    assert report_failure(report) is None
    report["decision"][key] = {} if key == "score_evidence" else 1.
    assert entry_gate_checks(report)["score"] == "FAIL"
    assert report_failure(report) == "V6_SCORE_OR_SETUP_MISMATCH"


@pytest.mark.parametrize("key,value", [("high", None), ("high", 1), ("low", 0),
                                     ("open", None), ("close", None), ("volume", 0)])
def test_invalid_serialized_price_anchor_fails_without_numeric_exception(key, value):
    report = execution_report()
    report["technical"][key] = value
    assert entry_gate_checks(report)["score"] == "FAIL"
    assert report_failure(report) is not None


def test_coverage_aware_selection_prefers_fully_observed_eligible_setup():
    result = decide(quality(), features(macd_improving=None, macd_strict=None,
                                        macd_signal_strict=None), config=CONFIG)
    assert result["signal"] == "BUY"
    assert result["setup"] == "LEADERSHIP_CONTINUATION"
    assert result["score_coverage"] == 100
    assert result["setup_decisions"]["REVERSAL"]["score_coverage"] < 100
    assert result["comparison"]["allocation_value_status"] == "NOT_EVALUATED_USE_PAE"


def test_cross_setup_rank_uses_risk_axes_not_non_equivalent_bare_scores():
    a = decide(quality(), setup_features("REVERSAL"), config=CONFIG)
    b = deepcopy(a)
    b.update(score=100, formal_score=100)
    b["score_evidence"]["coverage_percent"] = 90
    b["comparison"] = setup_comparison(b)
    assert setup_rank(a) > setup_rank(b)
    b["score_evidence"]["coverage_percent"] = 100
    b["plan"]["net_reward_risk"] = a["plan"]["net_reward_risk"] + 1
    b["comparison"] = setup_comparison(b)
    assert setup_rank(b) > setup_rank(a)
    b["entry_checks"]["BUY"]["minimum_rr"] = None
    b["comparison"] = setup_comparison(b)
    assert b["comparison"]["net_rr_surplus"] is None
    assert b["comparison"]["rr_status"] == "PRICE_DISCOVERY_OR_UNKNOWN"


def test_comparison_cannot_override_quality_veto_or_risk():
    for q, f in [(quality(status="REJECT"), features()),
                 (quality(), features(ema5_slope=-1, ema20_slope=-1, atr_expansion=2, r5=-.2)),
                 (quality(), features(support=80))]:
        assert decide(q, f, config=CONFIG)["signal"] not in {"BUY", "BUY_SMALL"}


def reports():
    result = []
    for i in range(6):
        result.append({"symbol": "2330", "signal_date": f"2026-09-{i+1:02}",
                       "model_version": "6.0.0", "parameter_version": PARAMETER_VERSION,
                       "parameters": vars(CONFIG), "setups": scored(features(clv=.1 * i))})
    return result


def test_ablation_is_fixed_budget_read_only_and_split_is_chronological():
    source = reports()
    before = deepcopy(source)
    audit = score_audit(source, validation_start="2026-09-04")
    assert audit["split_status"] == "BOTH_PARTITIONS"
    assert audit["executable"] is False and audit["performance_validated"] is False
    assert source == before
    assert {r["date"] for r in audit["rows"] if r["partition"] == "discovery"} == {
        "2026-09-01", "2026-09-02", "2026-09-03"}
    for report in source:
        for setup in SETUPS:
            evidence = report["setups"][setup]["score_evidence"]
            for removed in ablation(evidence).values():
                assert removed["fixed_budget_score"] == evidence["formal_score"] - removed["removed_points"]
    assert all(c["complete_samples"] == 3 for c in audit["comparisons"])


def test_correlation_is_not_independent_evidence_or_small_sample_probability():
    assert _correlation([1, 2], [2, 4]) is None
    assert _correlation([1, 1, 1], [2, 3, 4]) is None
    assert _correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1)
    assert score_audit(reports()[:2], validation_start="2026-10-01")["split_status"] == "INSUFFICIENT_DATE_SPAN"


def test_score_audit_rejects_duplicates_and_mixed_parameters():
    source = reports()
    with pytest.raises(ValueError, match="DUPLICATE"):
        score_audit(source + source[:1], validation_start="2026-09-04")
    source[-1]["parameter_version"] = "other-policy"
    with pytest.raises(ValueError, match="MIXED_SCORE_RESEARCH_VERSIONS"):
        score_audit(source, validation_start="2026-09-04")


def valuation(records, q=None, cutoff="2026-09-11T13:30+08:00"):
    return valuation_context(records, symbol="2330", market="TWSE", as_of=cutoff,
                             quality=q or quality(metrics={"basic_eps_ttm": 20., "roe": .12}))


def test_fundamental_references_are_separate_from_pe_timing_and_cagr():
    source = financials()
    source[-1].update(parent_equity=6000, shares_outstanding=100_000, shares_unit="shares")
    before = deepcopy(source)
    result = valuation(source)
    assert result["reference_metrics"]["book_value_per_share"] == 60
    assert result["reference_metrics"]["median_operating_margin_8q"] == pytest.approx(.3)
    assert result["reference_metrics"]["median_annualized_quarter_eps_8q"] > 0
    assert result["status"] == "UNKNOWN" and result["affects_les_score"] is False
    assert result["fair_value"] is result["expected_cagr"] is None
    assert source == before
    assert len(result["sources"]) == 8
    assert pd.Timestamp(result["references_available_at"]) <= pd.Timestamp("2026-09-11T13:30+08:00")


@pytest.mark.parametrize("diagnostic_roe", [.24, None])
def test_ttm_valuation_reference_never_substitutes_reported_period_roe(diagnostic_roe):
    q = quality(metrics={"roe": .12, "roe_ttm": diagnostic_roe})
    result = valuation(financials(), q)
    assert result["reference_metrics"]["roe_ttm"] == diagnostic_roe


def test_future_restatement_does_not_change_cutoff_valuation_references():
    source = financials()
    original = valuation(source)
    source.append(dict(source[-1], available_at="2026-09-12T08:00+08:00", revision_id="future",
                       operating_income=-1, basic_eps=1000))
    assert valuation(source) == original
    assert valuation(source, cutoff="2026-09-13T13:30+08:00") != original


def test_ytd_eps_is_not_differenced_and_unknown_share_units_are_not_assumed():
    source = financials()
    for row in source:
        row["flow_basis"] = "YTD"
    source[-1]["shares_outstanding"] = 100_000
    result = valuation(source)
    assert result["reference_metrics"]["median_annualized_quarter_eps_8q"] is None
    assert result["reference_metrics"]["book_value_per_share"] is None
    assert "VERIFIED_PARENT_EQUITY_AND_SHARE_UNITS" in result["missing_evidence"]


def test_total_consolidated_equity_cannot_substitute_for_parent_book_value():
    source = financials()
    source[-1].update(shares_outstanding=100_000, shares_unit="shares")
    assert valuation(source)["reference_metrics"]["book_value_per_share"] is None


@pytest.mark.parametrize("change", [{"scope": "individual"}, {"unit": "dollar"}, {"currency": "USD"}])
def test_incomparable_fundamentals_do_not_manufacture_valuation(change):
    source = financials()
    source[-1].update(change)
    result = valuation(source)
    assert result["reference_metrics"] == {}
    assert result["status"] == "UNKNOWN"


def test_unverified_quality_and_missing_sources_stay_unknown():
    assert valuation(financials(), quality(status="PROVISIONAL"))["reason"] == "QUALITY_NOT_VERIFIED"
    assert valuation([])["reason"] == "INCOMPARABLE_OR_MISSING_VALUATION_SOURCES"


def test_projection_and_renderer_do_not_call_estimate_an_actual_score():
    report = raw_execution_report()
    values = project_report(report)
    for key, field in [("Formal Score", "formal_score"), ("Estimated Score", "estimated_score"),
                       ("Score Coverage", "score_coverage")]:
        assert values["LES v6 " + key] == report["decision"][field]
    assert "Final" in setup_score_summary(report)
    assert "Penalty" in setup_score_summary(report)


def packet(count=65):
    data = bars(count)
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    return {"bars": data.to_dict("records"), "context": {
        "symbol": "2330", "sector": "ELEC", "as_of": "2026-09-11T13:30+08:00", "filings": financials()}}


def test_offline_score_audit_uses_native_plugin_and_explicit_source_digest():
    source = packet()
    before = deepcopy(source)
    first = analyze_scores(source, validation_start="2026-09-01")
    assert first == analyze_scores(source, validation_start="2026-09-01")
    assert len(first["rows"]) == 65 * 3
    assert len(first["input_sha256"]) == 64
    assert first["source_posture"] == "USER_SOURCE_PACKET_NOT_VERIFIED_REPLAY_DATASET"
    assert source == before
    json.dumps(first, allow_nan=False)
    # Later closes must not change earlier score diagnostics.
    prefix = deepcopy(source)
    prefix["bars"] = prefix["bars"][:-5]
    prefix["context"]["as_of"] = prefix["bars"][-1]["Date"] + "T13:30+08:00"
    earlier = analyze_scores(prefix, validation_start="2026-09-01")
    assert earlier["rows"] == first["rows"][:-15]


def test_score_audit_cli_requires_explicit_holdout_boundary(tmp_path, capsys):
    input_file = tmp_path / "synthetic.json"
    input_file.write_text(json.dumps(packet(5)))
    assert main([str(input_file), "--score-audit", "--validation-start", "2026-09-01"]) == 0
    assert json.loads(capsys.readouterr().out)["research_only"] is True
    with pytest.raises(SystemExit):
        main([str(input_file), "--score-audit"])

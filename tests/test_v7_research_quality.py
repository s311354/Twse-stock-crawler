"""SYNTHETIC mechanics only; no real model efficacy or deployment evidence."""
from copy import deepcopy
from datetime import date, timedelta
from math import log

import pytest

from domain.les_v7.research_quality import compare_reference_quality, evaluate_reference_quality


def prepared(index, *, fill=1, rebound=1, joint=1, day=None, weight=1.):
    return dict(row_id=f"SYNTHETIC_{index}", cutoff=(day or "2026-01-02") + "T13:30:00+08:00",
        label_available_at="2026-12-01T13:30:00+08:00", features=dict(close=100., atr=5.),
        bottom=99., observation_weight=weight, utility=0.,
        labels=dict(fill=fill, rebound=rebound if fill else None, joint=joint if fill else None))


def prediction(fill, *, rebound=.8, joint=.5):
    return dict(p_fill=fill, p_rebound=rebound, p_entry=joint)


def execution(*, pnl="106.14"):
    from decimal import Decimal
    debit = Decimal("1001.43")
    return dict(filled=True, entry=100., initial_stop=95., cutoff_atr=5., shares=10,
        path_order_verified=True, path_evidence_ids=["SYNTHETIC_PATH"],
        cost_policy_version="SYNTHETIC_DECLARED_COST", cost_policy_id="SYNTHETIC_COST_ID",
        realized_net_pnl=float(pnl), cost_arithmetic=dict(realized_net_pnl=pnl,
            native_gross_reference_risk="50.0", buy_debit=str(debit), sell_proceeds=str(debit + Decimal(pnl))))


def fixture():
    rows = [prepared(0), prepared(1, fill=0, weight=2.), prepared(2, rebound=0, joint=0)]
    predictions = {rows[0]["row_id"]:prediction(.9), rows[1]["row_id"]:prediction(.2), rows[2]["row_id"]:prediction(.6)}
    return rows, predictions


def test_weighted_forecast_metrics_reliability_and_conditional_joint_semantics():
    rows, predictions = fixture()
    result = evaluate_reference_quality(rows, predictions)
    metrics = result["metrics"]
    assert metrics["fill_brier"] == pytest.approx((.01 + 2 * .04 + .16) / 4)
    assert metrics["fill_nll"] == pytest.approx((-log(.9) - 2 * log(.8) - log(.6)) / 4)
    assert metrics["weighted_fill_auc"] == 1
    assert metrics["observed_fill_fraction"] == .5
    assert metrics["conditional_rebound_brier"] == pytest.approx((.04 + .64) / 2)
    assert metrics["conditional_near_and_rebound_brier"] == .25
    assert metrics["unconditional_near_and_rebound_brier"] == pytest.approx(((.9 * .5 - 1) ** 2 + 2 * (.2 * .5) ** 2 + (.6 * .5) ** 2) / 4)
    bins = metrics["reliability_bins"]
    assert len(bins) == 10 and sum(item["weight_fraction"] for item in bins) == 1
    assert bins[2]["rows"] == 1 and bins[2]["weight_fraction"] == .5
    assert bins[0]["rows"] == 0 and bins[0]["observed_fill_fraction"] is None
    assert result["reference_quality"]["reference_paths"]["mean_unclipped_net_r"] is None
    assert result["portfolio_after_cost"]["status"] == "NOT_ASSESSABLE"
    assert result["execution_action"] == "NO_ACTION"


def test_unclipped_net_r_requires_exact_execution_cost_evidence_never_utility():
    rows, predictions = fixture()
    rows[2]["bottom"] = 90.
    lookup = {rows[0]["row_id"]:execution(), rows[2]["row_id"]:execution(pnl="-60.00")}
    result = evaluate_reference_quality(rows, predictions, lookup)
    quality = result["reference_quality"]
    path = quality["reference_paths"]
    assert quality["selected_events"] == quality["selected_filled_reference_events"] == 2
    assert quality["observed_near_and_rebound_given_fill"] == .5
    assert path["status"] == "COMPLETE_REFERENCE_DIAGNOSTICS"
    assert path["near_bottom_rate"] == .5
    assert path["mean_entry_bottom_atr"] == pytest.approx((.2 + 2.) / 2)
    assert path["mean_unclipped_net_r"] == pytest.approx((106.14 / 50 - 60 / 50) / 2)
    assert path["mean_reference_net_pnl_twd"] == pytest.approx((106.14 - 60) / 2)
    assert path["net_positive_reference_trade_rate"] == .5
    # utility=0 on both rows would imply -1 clipped R. It is never used.
    assert path["mean_unclipped_net_r"] != -1
    lookup[rows[2]["row_id"]]["cost_arithmetic"].pop("sell_proceeds")
    partial = evaluate_reference_quality(rows, predictions, lookup)["reference_quality"]["reference_paths"]
    assert partial["verified_reference_paths"] == 1
    assert partial["status"] == "UNKNOWN" and partial["mean_unclipped_net_r"] is None


def test_empty_selection_no_filled_subset_and_single_class_auc_remain_unknown():
    row = prepared(0, fill=0)
    report = evaluate_reference_quality([row], {row["row_id"]:prediction(.1)})
    assert report["metrics"]["weighted_fill_auc"] is None
    assert report["reference_quality"]["observed_selected_fill_rate"] is None
    assert report["reference_quality"]["reference_paths"]["reasons"] == ["NO_SELECTED_EVENTS"]
    selected = evaluate_reference_quality([row], {row["row_id"]:prediction(.8)})
    assert selected["reference_quality"]["observed_selected_fill_rate"] == 0
    assert selected["reference_quality"]["reference_paths"]["mean_reference_net_pnl_twd"] is None


def test_weighted_auc_ties_and_large_finite_weights():
    rows = [prepared(0, weight=1e308), prepared(1, fill=0, weight=1e308)]
    report = evaluate_reference_quality(rows, {row["row_id"]:prediction(.5) for row in rows})
    assert report["metrics"]["weighted_fill_auc"] == .5
    assert report["metrics"]["fill_brier"] == .25


@pytest.mark.parametrize("change,match", [
    (lambda rows, predictions:predictions.pop(rows[0]["row_id"]), "COVERAGE"),
    (lambda rows, predictions:predictions.update(extra=prediction(.5)), "COVERAGE"),
    (lambda rows, predictions:predictions[rows[0]["row_id"]].update(p_fill=float("nan")), "FINITE|finite"),
    (lambda rows, predictions:predictions[rows[0]["row_id"]].update(p_entry=.9), "PROJECTION"),
    (lambda rows, predictions:rows[1]["labels"].update(joint=0), "CONDITIONAL_LABELS"),
    (lambda rows, predictions:rows[0].update(observation_weight=0), "WEIGHT"),
    (lambda rows, predictions:rows[0].update(label_available_at=rows[0]["cutoff"]), "LABEL_MUST_FOLLOW"),
])
def test_invalid_or_incomplete_cases_never_get_dropped(change, match):
    rows, predictions = fixture();change(rows, predictions)
    with pytest.raises(ValueError, match=match):
        evaluate_reference_quality(rows, predictions)


def test_unmatured_labels_and_execution_label_conflict_fail_closed():
    rows, predictions = fixture()
    with pytest.raises(ValueError, match="UNMATURED"):
        evaluate_reference_quality(rows, predictions, evaluation_cutoff="2026-02-01T13:30:00+08:00")
    lookup = {rows[0]["row_id"]:dict(execution(), filled=False)}
    with pytest.raises(ValueError, match="EXECUTION_LABEL_MISMATCH"):
        evaluate_reference_quality(rows, predictions, lookup)


@pytest.mark.parametrize("change", [
    lambda value:value.update(path_order_verified=False),
    lambda value:value.update(path_evidence_ids=[1]),
    lambda value:value.update(cost_policy_id=[]),
    lambda value:value["cost_arithmetic"].update(sell_proceeds="1000.00"),
])
def test_incomplete_or_inconsistent_execution_cannot_produce_after_cost_quality(change):
    rows, predictions = fixture();value = execution();change(value)
    result = evaluate_reference_quality(rows, predictions, {rows[0]["row_id"]:value})
    path = result["reference_quality"]["reference_paths"]
    assert path["status"] == "UNKNOWN" and path["mean_unclipped_net_r"] is None
    assert result["metrics"]["fill_brier"] == pytest.approx(.0625)


def test_execution_path_must_agree_with_near_bottom_joint_label():
    rows, predictions = fixture();rows[0]["bottom"] = 80.
    with pytest.raises(ValueError, match="JOINT_LABEL_MISMATCH"):
        evaluate_reference_quality(rows, predictions, {rows[0]["row_id"]:execution()})


def test_paired_differences_keep_all_rows_and_small_windows_have_no_fake_ci():
    rows, baseline = fixture()
    candidate = {row["row_id"]:prediction(.9 if row["labels"]["fill"] else .1) for row in rows}
    result = compare_reference_quality(rows, baseline, candidate)
    assert result["paired_events"] == len(rows)
    assert result["fill_brier_difference_candidate_minus_baseline"] == pytest.approx(.01 - .0625)
    assert result["paired_interval"]["status"] == "UNKNOWN"
    assert result["paired_interval"]["interval"] is None


def test_block_interval_requires_contiguous_frozen_official_sessions_and_is_deterministic():
    sessions = [(date(2026, 1, 1) + timedelta(days=index)).isoformat() for index in range(43)]
    rows = [prepared(index, day=day, fill=index % 2) for index, day in enumerate(sessions[:42])]
    baseline = {row["row_id"]:prediction(.5) for row in rows}
    candidate = {row["row_id"]:prediction(.9 if row["labels"]["fill"] else .1) for row in rows}
    before = deepcopy((rows, baseline, candidate))
    missing = compare_reference_quality(rows, baseline, candidate)
    assert missing["paired_interval"]["reason"] == "FROZEN_OFFICIAL_SESSION_ORDER_REQUIRED"
    a = compare_reference_quality(rows, baseline, candidate, frozen_sessions=sessions)
    b = compare_reference_quality(rows, baseline, candidate, frozen_sessions=sessions)
    assert a == b and before == (rows, baseline, candidate)
    assert a["paired_interval"]["interval"]["upper"] == pytest.approx(-.24)
    assert a["paired_interval"]["bootstrap_replicates"] == 1999
    assert a["paired_interval"]["block_sessions"] == 21
    rows[-1]["cutoff"] = sessions[42] + "T13:30:00+08:00"
    gap = compare_reference_quality(rows, baseline, candidate, frozen_sessions=sessions)
    assert gap["paired_interval"]["reason"] == "DISCONTIGUOUS_OBSERVED_SESSION_SUPPORT"


def test_caller_mutation_of_results_cannot_change_source_or_global_inference():
    rows, baseline = fixture();before = deepcopy((rows, baseline))
    first = compare_reference_quality(rows, baseline, baseline)
    second = compare_reference_quality(rows, baseline, baseline)
    second["paired_interval"]["seed"] = 0
    second["baseline"]["metrics"]["reliability_bins"][0]["rows"] = 999
    assert before == (rows, baseline)
    assert compare_reference_quality(rows, baseline, baseline) == first

"""SYNTHETIC calibration controls; not fitted-market performance evidence."""
from copy import deepcopy
import math

import numpy as np
import pytest

from domain.les_v7.calibration import (_inputs, _loss_gradient, fit_calibration,
                                      score_calibration)
from domain.les_v7.contracts import identity
from domain.les_v7.localization import project_outputs


def rows():
    result = []
    for i in range(24):
        fill = int(i % 4 != 0)
        rebound = int(i % 3 == 0) if fill else None
        joint = int(i % 6 == 3) if rebound else 0 if fill else None
        logits = dict(fill=(2 * fill - 1) * .3 + .8,
                      rebound=(2 * (rebound or 0) - 1) * .2 - .8,
                      joint=(2 * (joint or 0) - 1) * .2 + .8)
        timing = [0.] * 21
        timing[i % 3] = .6
        result.append(dict(row_id=f"SYNTHETIC_{i}", raw_logits=logits,
                           timing_logits=timing, log_quantiles=[-.03, -.02, -.01],
                           labels=dict(fill=fill, rebound=rebound, joint=joint, timing=i % 3),
                           observation_weight=float(i % 5 + 1)))
    return result


def identity_calibration():
    return dict(version="SYNTHETIC_ONLY", timing_temperature=1.,
                fill=[1., 0.], rebound=[1., 0.], joint=[1., 0.])


def test_deterministic_actual_fit_metrics_and_bound_report():
    source = rows()
    original = deepcopy(source)
    fitted = fit_calibration(source, iterations=80)
    assert fitted == fit_calibration(source, iterations=80)
    assert source == original
    calibration, report = fitted["calibration"], fitted["report"]
    assert calibration["validation_report_id"] == identity(report)
    assert report["data_sha256"] == identity(source)
    assert report["iterations_accepted"] > 0
    assert report["final_objective"] < report["initial_objective"]
    assert report["after"]["timing_nll"] < report["before"]["timing_nll"]
    assert report["fitted_parameters"]["fill"] != [1., 0.]
    assert calibration["timing_temperature"] > 0
    assert report["evaluation_scope"] == "CALIBRATION_SET_NOT_HOLDOUT"
    assert report["approved_for_live"] is False
    assert report["after"] == score_calibration(source, calibration)


def test_calibration_metrics_match_runtime_projection_and_conditional_weights():
    source = rows()
    calibration = identity_calibration()
    actual = score_calibration(source, calibration)
    forecasts = [project_outputs(row, dict(close=100., low=99.), calibration) for row in source]
    assert all(forecast.status == "VALID" for forecast in forecasts)
    assert all(f.p_entry <= f.p_rebound for f in forecasts)
    assert actual["projection_count"] == sum("BOUNDED_ISOTONIC_JOINT_REBOUND" in f.repairs for f in forecasts)
    for key, attr in (("fill", "p_fill"), ("rebound", "p_rebound"), ("joint", "p_entry")):
        keep = list(range(len(source))) if key == "fill" else [i for i, row in enumerate(source) if row["labels"]["fill"]]
        total = sum(source[i]["observation_weight"] for i in keep)
        expected = sum(source[i]["observation_weight"] * (getattr(forecasts[i], attr) - source[i]["labels"][key]) ** 2
                       for i in keep) / total
        assert actual["brier"][key] == pytest.approx(expected, abs=1e-14)
    total = sum(row["observation_weight"] for row in source)
    expected = sum(row["observation_weight"] * -math.log(forecast.timing[row["labels"]["timing"]])
                   for row, forecast in zip(source, forecasts)) / total
    assert actual["timing_nll"] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("crossed", [False, True])
def test_optimizer_gradient_is_post_projection_objective(crossed):
    inputs = _inputs(rows(), require_classes=True)
    parameters = np.array([.8, .1, .7, 1.4 if not crossed else -1.4,
                           .6, -1.4 if not crossed else 1.4, .2])
    value, gradient = _loss_gradient(parameters, inputs, .03)
    assert math.isfinite(value)
    numerical = []
    for i in range(7):
        plus, minus = parameters.copy(), parameters.copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        numerical.append((_loss_gradient(plus, inputs, .03)[0] - _loss_gradient(minus, inputs, .03)[0]) / 2e-6)
    assert gradient == pytest.approx(numerical, abs=1e-7)


@pytest.mark.parametrize("head", ["fill", "rebound", "joint"])
def test_missing_class_refuses_calibration_not_identity_fallback(head):
    source = rows()
    for row in source:
        if head == "fill":
            row["labels"].update(fill=1, rebound=0, joint=0)
        elif row["labels"]["fill"]:
            row["labels"][head] = 0
            if head == "rebound":
                row["labels"]["joint"] = 0
    with pytest.raises(ValueError, match="BOTH_CALIBRATION_CLASSES_REQUIRED:" + head):
        fit_calibration(source)


@pytest.mark.parametrize("mutation,message", [
    (lambda r: r[0]["timing_logits"].pop(), "TWENTY_ONE_TIMING_LOGITS_REQUIRED"),
    (lambda r: r[0]["raw_logits"].update(fill=float("nan")), "FINITE_NUMBER_REQUIRED"),
    (lambda r: r[0].update(observation_weight=0.), "POSITIVE_CALIBRATION_WEIGHT_REQUIRED"),
    (lambda r: r[0].update(observation_weight=-1.), "POSITIVE_CALIBRATION_WEIGHT_REQUIRED"),
    (lambda r: r[0]["labels"].update(timing=21), "INVALID_TIMING_LABEL"),
    (lambda r: r[0]["labels"].update(fill=True), "OBSERVED_BINARY_FILL_REQUIRED"),
    (lambda r: r[0]["labels"].update(rebound=0), "NO_FILL_CONDITIONAL_LABELS_MUST_BE_NULL"),
    (lambda r: r[1]["labels"].update(rebound=0, joint=1), "INVALID_CONDITIONAL_EVENT_LABELS"),
    (lambda r: r[1].update(row_id=r[0]["row_id"]), "UNIQUE_CALIBRATION_ROW_IDS_REQUIRED"),
])
def test_invalid_calibration_inputs_are_rejected(mutation, message):
    source = rows()
    mutation(source)
    with pytest.raises(ValueError, match=message):
        fit_calibration(source)


def test_weight_scale_invariance_and_label_conditional_semantics():
    source = rows()
    scaled = deepcopy(source)
    for row in scaled:
        row["observation_weight"] *= 1e200
    expected = score_calibration(source, identity_calibration())
    actual = score_calibration(scaled, identity_calibration())
    assert actual["brier"] == pytest.approx(expected["brier"])
    assert actual["timing_nll"] == pytest.approx(expected["timing_nll"])
    # Heldout-only diagnostics remain possible without both target classes.
    one_class = [row for row in source if row["labels"]["fill"] and row["labels"]["joint"] == 1]
    assert score_calibration(one_class, identity_calibration())["row_count"] == len(one_class)


def test_unrepresentable_positive_weight_is_not_silently_dropped():
    source = rows()
    source[0]["observation_weight"] = 1e-308
    source[1]["observation_weight"] = 1e308
    with pytest.raises(ValueError, match="CALIBRATION_WEIGHT_DYNAMIC_RANGE_UNREPRESENTABLE"):
        fit_calibration(source)


@pytest.mark.parametrize("value", [0., -1., float("inf"), float("nan")])
def test_invalid_temperature_refuses_metrics(value):
    calibration = identity_calibration()
    calibration["timing_temperature"] = value
    with pytest.raises(ValueError):
        score_calibration(rows(), calibration)

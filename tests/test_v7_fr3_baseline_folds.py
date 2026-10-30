"""SYNTHETIC proof/reconstruction ports only; no model fit or market claim."""
from copy import deepcopy

import pytest

from domain.les_v7 import training
from domain.les_v7.contracts import CONTEXTS, identity, timestamp
from infrastructure.les_v7_fr2_archive import (
    _saved_head_provider, _verify_baseline_fold_reconstruction,
)
from tests.test_v7_fr3_training import inputs


def row(name, cutoff, available):
    return dict(row_id=name, cutoff=cutoff, label_available_at=available,
        labels={"fill": 0}, observation_weight=1., features={}, plan=None, context="REVERSAL")


@pytest.fixture
def archive():
    names, bounds = ["slope5"], [[-10., 10.]]
    config = dict(SYNTHETIC_TEST_ONLY=True, reference_policy={"version": "FR3_SOURCE_BOUND_LIMIT_H20_R1"},
        head_solver="STANDARDIZED_CONVEX_R2", folds=[
            dict(fold_id="SYNTHETIC_JUNE", fit_cutoff="2024-06-05T13:30:00+08:00",
                prediction_start="2024-06-06T13:30:00+08:00", prediction_end="2024-06-07T13:30:00+08:00"),
            dict(fold_id="SYNTHETIC_JULY", fit_cutoff="2024-07-05T13:30:00+08:00",
                prediction_start="2024-07-08T13:30:00+08:00", prediction_end="2024-07-09T13:30:00+08:00")])
    rows = [row("SYNTHETIC_OLD", "2024-06-03T05:30:00+00:00", "2024-06-04T05:30:00+00:00"),
            row("SYNTHETIC_JUNE_TEST", "2024-06-06T05:30:00+00:00", "2024-06-20T05:30:00+00:00"),
            row("SYNTHETIC_JULY_TEST", "2024-07-08T05:30:00+00:00", "2024-07-25T05:30:00+00:00")]
    past = rows[:2]
    heads = {"SYNTHETIC_HEAD_ONLY": [1.]}
    report = dict(schema="v7-standardized-convex-head-fit/v2", fit_cutoff=config["folds"][1]["fit_cutoff"],
        feature_count=1, converged=True, training_row_count=len(past),
        ordered_training_row_ids_sha256=identity([r["row_id"] for r in past]),
        raw_heads=heads, heads_sha256=identity(heads),
        last_label_available_at=past[-1]["label_available_at"])
    folds = []
    for index, declared in enumerate(config["folds"]):
        fit, start, end = (timestamp(declared[k]) for k in ("fit_cutoff", "prediction_start", "prediction_end"))
        selected = [r for r in rows if timestamp(r["cutoff"]) < fit and timestamp(r["label_available_at"]) <= fit]
        testing = [r for r in rows if start <= timestamp(r["cutoff"]) <= end]
        fold = dict(fold_id=declared["fold_id"], fit_cutoff=fit.isoformat(), prediction_at=start.isoformat(),
            prediction_end=end.isoformat(), training_row_ids=[r["row_id"] for r in selected],
            prediction_row_ids=[r["row_id"] for r in testing])
        if index == 0:
            fold.update(status="NOT_READY", reason="INSUFFICIENT_HEAD_TRAINING_ROWS")
        else:
            fold.update(status="FITTED", head_fit_report_id=identity(report), heads_sha256=identity(heads),
                last_training_label_available_at=timestamp(past[-1]["label_available_at"]).isoformat(),
                calibration_reason="INSUFFICIENT_PAST_OOF_CALIBRATION_ROWS")
        folds.append(fold)
    proof = training._baseline_fold_proof(config, names, bounds, folds)
    return dict(config=config, names=names, bounds=bounds, rows=rows, folds=folds,
        reports={identity(report): report}, proof=proof)


def provider(case, **options):
    arguments = dict(baseline_folds=case["proof"], rows=case["rows"])
    arguments.update(options)
    return _saved_head_provider(case["config"], case["names"], case["bounds"], case["reports"], **arguments)


def bind_heads(case, callback, index, target=None, training_rows=None):
    declared = case["config"]["folds"][index]
    fit = timestamp(declared["fit_cutoff"])
    selected = [r for r in case["rows"] if timestamp(r["cutoff"]) < fit and timestamp(r["label_available_at"]) <= fit]
    return callback(selected if training_rows is None else training_rows, case["config"], fit.isoformat(),
        case["names"], case["bounds"], {} if target is None else target)


def reseal(case):
    case["proof"]["sha256"] = identity(case["proof"]["payload"])


def test_native_compact_proof_binds_all_ordered_ids_without_duplicate_arrays(archive):
    proof = archive["proof"]
    assert proof["sha256"] == identity(proof["payload"])
    saved = proof["payload"]["folds"]
    assert saved[0]["training_row_ids"] == dict(schema="les-v7-id-array-reference/v1",
        count=1, sha256=identity(["SYNTHETIC_OLD"]))
    assert saved[1]["prediction_row_ids"]["count"] == 1
    assert all(not isinstance(f[k], list) for f in saved for k in ("training_row_ids", "prediction_row_ids"))
    before = deepcopy(archive)
    callback = provider(archive)
    with pytest.raises(ValueError, match="^INSUFFICIENT_HEAD_TRAINING_ROWS$"):
        bind_heads(archive, callback, 0)
    assert bind_heads(archive, callback, 1) == next(iter(archive["reports"].values()))["raw_heads"]
    assert archive == before


def test_oof_reconstruction_replays_skip_and_complete_fold_metadata_without_any_fit(archive, monkeypatch):
    monkeypatch.setattr(training, "_fit", lambda *_a, **_k: pytest.fail("reconstruction must never refit"))
    monkeypatch.setattr(training, "_vectors", lambda *_a: ([1.], None))
    monkeypatch.setattr(training, "head_outputs", lambda *_a: dict(SYNTHETIC_RAW_OUTPUT=True))
    def no_past(*_a):
        raise ValueError("INSUFFICIENT_PAST_OOF_CALIBRATION_ROWS")
    monkeypatch.setattr(training, "_calibrate", no_past)
    predictions, folds, _ = training._oof(archive["rows"], archive["config"], archive["names"],
        archive["bounds"], head_provider=provider(archive))
    assert len(predictions) == 1 and predictions[0]["forecast"]["status"] == "NOT_READY"
    assert folds == archive["folds"]
    _verify_baseline_fold_reconstruction(folds, archive["proof"])


def test_same_heads_at_different_fit_dates_keep_exact_receipts_and_complete_reproduction(archive, monkeypatch):
    """Mock coefficients only; both native fold metadata/reconstruction paths run."""
    heads = {"SYNTHETIC_SAME_COEFFICIENTS": [1.]}
    def synthetic_fit(rows, config, cutoff, names, bounds, *, reports):
        report = dict(schema="v7-standardized-convex-head-fit/v2", fit_cutoff=cutoff,
            feature_count=len(names), converged=True, training_row_count=len(rows),
            ordered_training_row_ids_sha256=identity([r["row_id"] for r in rows]),
            raw_heads=heads, heads_sha256=identity(heads),
            last_label_available_at=max(timestamp(r["label_available_at"]) for r in rows).isoformat())
        reports[identity(report)] = report
        return heads
    monkeypatch.setattr(training, "_fit", synthetic_fit)
    monkeypatch.setattr(training, "_vectors", lambda *_a: ([1.], None))
    monkeypatch.setattr(training, "head_outputs", lambda *_a: dict(SYNTHETIC_RAW_OUTPUT=True))
    def no_calibrator(*_a): raise ValueError("INSUFFICIENT_PAST_OOF_CALIBRATION_ROWS")
    monkeypatch.setattr(training, "_calibrate", no_calibrator)
    original = training._oof(archive["rows"], archive["config"], archive["names"], archive["bounds"])
    assert len(original[1]) == len(original[2]) == 2
    for fold in original[1]:
        report = original[2][fold["head_fit_report_id"]]
        assert timestamp(report["fit_cutoff"]) == timestamp(fold["fit_cutoff"])
    assert len({fold["head_fit_report_id"] for fold in original[1]}) == 2
    proof = training._baseline_fold_proof(archive["config"], archive["names"], archive["bounds"], original[1])
    monkeypatch.setattr(training, "_fit", lambda *_a, **_k: pytest.fail("reconstruction must not fit"))
    callback = _saved_head_provider(archive["config"], archive["names"], archive["bounds"], original[2],
        baseline_folds=proof, rows=archive["rows"])
    rebuilt = training._oof(archive["rows"], archive["config"], archive["names"], archive["bounds"], head_provider=callback)
    assert rebuilt == original
    _verify_baseline_fold_reconstruction(rebuilt[1], proof)


def test_final_receipt_selector_binds_time_features_and_all_training_ids_but_keeps_legacy_behavior(archive):
    heads = next(iter(archive["reports"].values()))["raw_heads"]
    later = deepcopy(next(iter(archive["reports"].values())))
    earlier = dict(deepcopy(later), fit_cutoff=archive["config"]["folds"][0]["fit_cutoff"])
    reports = {identity(earlier): earlier, identity(later): later}
    selected = training._head_fit_receipt_id(archive["config"], reports, heads, archive["rows"][:2],
        later["fit_cutoff"], archive["names"])
    assert selected == identity(later)
    legacy = dict(archive["config"], reference_policy={"version": "SYNTHETIC_LEGACY"})
    assert training._head_fit_receipt_id(legacy, reports, heads, archive["rows"][:2],
        later["fit_cutoff"], archive["names"]) == identity(earlier)
    for rows, names in ((list(reversed(archive["rows"][:2])), archive["names"]),
                        (archive["rows"][:2], ["slope5", "extra"])):
        with pytest.raises(ValueError, match="EXACT_HEAD_FIT_RECEIPT_REQUIRED"):
            training._head_fit_receipt_id(archive["config"], reports, heads, rows, later["fit_cutoff"], names)


def test_legacy_archive_without_new_proof_keeps_original_strict_missing_head_behavior(archive):
    callback = provider(archive, baseline_folds=None, rows=None)
    with pytest.raises(ValueError, match="FR2_EXACT_ARCHIVED_HEAD_NOT_FOUND"):
        bind_heads(archive, callback, 0)
    assert bind_heads(archive, callback, 1) == next(iter(archive["reports"].values()))["raw_heads"]


def test_empty_prediction_window_still_requires_and_reuses_the_real_fitted_head(archive, monkeypatch):
    archive["rows"].pop()
    fold = archive["folds"][1]
    fold.update(prediction_row_ids=[], status="NOT_READY", reason="OOF_FOLD_HAS_NO_ELIGIBLE_PREDICTIONS")
    for key in ("head_fit_report_id", "heads_sha256", "last_training_label_available_at", "calibration_reason"):
        fold.pop(key, None)
    archive["proof"] = training._baseline_fold_proof(archive["config"], archive["names"], archive["bounds"], archive["folds"])
    monkeypatch.setattr(training, "_fit", lambda *_a, **_k: pytest.fail("no refit"))
    monkeypatch.setattr(training, "_calibrate", lambda *_a, **_k: pytest.fail("no predictions to calibrate"))
    predictions, folds, reports = training._oof(archive["rows"], archive["config"], archive["names"],
        archive["bounds"], head_provider=provider(archive))
    assert predictions == [] and reports == archive["reports"]
    assert folds == archive["folds"]
    _verify_baseline_fold_reconstruction(folds, archive["proof"])


@pytest.mark.parametrize("damage", ["checksum", "config", "feature_order", "bounds", "missing_fold",
    "fold_date", "training_ids", "prediction_ids", "duplicate_row", "wrong_policy", "not_an_object"])
def test_rehashed_or_unbound_proof_cannot_authorize_a_skip(archive, damage):
    body = archive["proof"]["payload"]
    if damage == "checksum": archive["proof"]["sha256"] = "0" * 64
    elif damage == "config": body["config_id"] = "0" * 64
    elif damage == "feature_order": body["feature_names"] = ["other"]
    elif damage == "bounds": body["feature_bounds"][0][1] += 1
    elif damage == "missing_fold": body["folds"].pop()
    elif damage == "fold_date": body["folds"][0]["fit_cutoff"] = "2024-06-04T13:30:00+08:00"
    elif damage == "training_ids": body["folds"][0]["training_row_ids"]["sha256"] = "0" * 64
    elif damage == "prediction_ids": body["folds"][0]["prediction_row_ids"]["count"] += 1
    elif damage == "duplicate_row": archive["rows"].append(archive["rows"][0])
    elif damage == "wrong_policy": archive["config"]["reference_policy"]["version"] = "OLD_UNVERIFIED"
    else: body["folds"][0] = "NOT_A_NATIVE_FOLD"
    if damage != "checksum": reseal(archive)
    with pytest.raises(ValueError, match="FR3_NATIVE_BASELINE_FOLD"):
        provider(archive)


def test_fit_time_request_and_row_order_cannot_change_after_proof_validation(archive):
    callback = provider(archive)
    with pytest.raises(ValueError, match="TRAINING_REQUEST_MISMATCH"):
        bind_heads(archive, callback, 1, training_rows=list(reversed(archive["rows"][:2])))


def test_disjoint_windows_can_reuse_same_failed_fit_without_new_restriction(archive):
    archive["config"]["folds"][1]["fit_cutoff"] = archive["config"]["folds"][0]["fit_cutoff"]
    fold = archive["folds"][1]
    fold.update(fit_cutoff=archive["folds"][0]["fit_cutoff"], training_row_ids=["SYNTHETIC_OLD"],
        status="NOT_READY", reason="INSUFFICIENT_HEAD_TRAINING_ROWS")
    for key in ("head_fit_report_id", "heads_sha256", "last_training_label_available_at", "calibration_reason"):
        fold.pop(key, None)
    archive["reports"].clear()
    archive["proof"] = training._baseline_fold_proof(archive["config"], archive["names"], archive["bounds"], archive["folds"])
    callback = provider(archive)
    for index in (0, 1):
        with pytest.raises(ValueError, match="^INSUFFICIENT_HEAD_TRAINING_ROWS$"):
            bind_heads(archive, callback, index)


def test_fitted_head_absence_corruption_or_wrong_binding_is_not_a_skip(archive):
    archive["reports"].clear()
    callback = provider(archive)
    with pytest.raises(ValueError, match="FR2_EXACT_ARCHIVED_HEAD_NOT_FOUND"):
        bind_heads(archive, callback, 1)


def test_native_nonconvergence_can_be_replayed_only_with_exact_failed_head_receipt(archive):
    old = next(iter(archive["reports"].values()))
    failed = dict(deepcopy(old), fit_cutoff=archive["config"]["folds"][0]["fit_cutoff"], converged=False,
        training_row_count=1, ordered_training_row_ids_sha256=identity(["SYNTHETIC_OLD"]),
        last_label_available_at=archive["rows"][0]["label_available_at"])
    archive["reports"][identity(failed)] = failed
    archive["proof"]["payload"]["folds"][0]["reason"] = "HEAD_SOLVER_NOT_CONVERGED"
    reseal(archive)
    target = {}
    with pytest.raises(ValueError, match="^HEAD_SOLVER_NOT_CONVERGED$"):
        bind_heads(archive, provider(archive), 0, target)
    assert target == {identity(failed): failed}
    failed["converged"] = True  # Hash/semantics both no longer match the retained receipt.
    with pytest.raises(ValueError, match="HEAD_TRAINING_BINDING_MISMATCH"):
        bind_heads(archive, provider(archive), 0)


def test_skip_proof_cannot_hide_a_converged_head_or_unknown_skip_reason(archive):
    fold = archive["proof"]["payload"]["folds"][1]
    fold.update(status="NOT_READY", reason="INSUFFICIENT_HEAD_TRAINING_ROWS")
    reseal(archive)
    with pytest.raises(ValueError, match="SKIPPED_HEAD_CONFLICT"):
        bind_heads(archive, provider(archive), 1)
    fold.pop("heads_sha256")
    fold.pop("head_fit_report_id")
    fold["reason"] = "ARBITRARY_CALLER_SKIP"
    reseal(archive)
    # Not recognized as native readiness: the normal head is required and the
    # complete recomputed fold census must still equal the retained proof.
    assert bind_heads(archive, provider(archive), 1)
    with pytest.raises(ValueError, match="REPRODUCTION_MISMATCH"):
        _verify_baseline_fold_reconstruction(archive["folds"], archive["proof"])


def test_fr3_producer_retains_proof_before_a_later_baseline_readiness_failure(inputs, monkeypatch):
    """Only dispatch/retention is mocked; no actual head or weight fit runs."""
    config = inputs[2]
    fit, label = "2026-03-03T05:30:00+00:00", "2026-05-01T05:30:00+00:00"
    rows = [dict(row("SYNTHETIC_TRAIN_"+str(i), fit, label), context=c, utility=.5,
        plan=None) for i, c in enumerate(CONTEXTS)]
    rows += [dict(row("SYNTHETIC_HOLDOUT_"+str(i), f"2026-08-0{i+3}T05:30:00+00:00",
        "2026-09-01T05:30:00+00:00"), context=c, utility=None, plan=None) for i, c in enumerate(CONTEXTS)]
    forecast = dict(status="VALID", quantiles=[1., 2., 3.], timing=[1.]+[0.]*20,
        p_fill=.5, p_rebound=.5, p_entry=.25, calibration_version="SYNTHETIC_ONLY")
    predictions = [dict(row_id=r["row_id"], label_available_at=label, forecast=forecast,
        head_fit_cutoff="2026-03-02T05:30:00+00:00", head_last_label_available_at="2026-03-01T05:30:00+00:00",
        calibration_fit_cutoff="2026-03-02T05:30:00+00:00", calibration_last_label_available_at="2026-03-01T05:30:00+00:00",
        prediction_id="SYNTHETIC_PREDICTION_"+str(i)) for i, r in enumerate(rows[:3])]
    folds = [dict(fold_id="SYNTHETIC_ONLY", training_row_ids=[r["row_id"] for r in rows[:3]],
        prediction_row_ids=[], status="NOT_READY", reason="INSUFFICIENT_HEAD_TRAINING_ROWS")]
    names, bounds = config["feature_names"], config["feature_bounds"]
    def oof(_rows, _config, current_names, _bounds):
        return (predictions, [], {}) if current_names == names else ([], folds, {})
    def fit_head(_rows, _config, _cutoff, current_names, _bounds, **_kw):
        if current_names != names:
            raise ValueError("INSUFFICIENT_SYNTHETIC_BASELINE_STOP")
        return {}
    monkeypatch.setattr(training, "_oof", oof)
    monkeypatch.setattr(training, "_fit", fit_head)
    monkeypatch.setattr(training, "_calibrate", lambda *_a: ({}, {}))
    monkeypatch.setattr(training, "_holdout", lambda *_a: ({}, []))
    monkeypatch.setattr(training, "factor_vector", lambda *_a, **_k: ((.5,)*6, {}))
    monkeypatch.setattr(training, "fit_factor_weights", lambda *_a, **_k: {"status": "CONVERGED"})
    result = training._train_prepared(dict(sha256="SYNTHETIC_DATASET",
        payload={"dataset_kind": "SYNTHETIC_TEST_ONLY"}), config, rows, [])
    assert result["status"] == "NOT_READY"
    assert result["reason_codes"] == ["BASELINE_ABLATION_NOT_READY:INSUFFICIENT_SYNTHETIC_BASELINE_STOP"]
    expected_names = [name for name in names if not training.is_active_fractal_feature(name)]
    expected_bounds = [bound for name, bound in zip(names, bounds) if name in expected_names]
    assert result["baseline_folds"] == training._baseline_fold_proof(config, expected_names, expected_bounds, folds)

"""SYNTHETIC_TEST_ONLY: engineering contracts, never market skill evidence.

All histories, paths, calendars and outcome costs in this module are invented
and remain in memory or pytest temporary storage. The full pipeline test uses
the native features/labels/OOF/calibration/weight/holdout implementation.
"""
from copy import deepcopy
import json
import random

import numpy as np
import pandas as pd
import pytest

from domain.les_v7.contracts import CONTEXTS, FEATURE_VERSION, identity, timestamp
from domain.les_v7.decision import make_plan, select_context
from domain.les_v7.features import compute_features, select_bars
from domain.les_v7.localization import FrozenBundle, head_outputs, predict, project_outputs
from domain.les_v7.quality_policy import default_quality_policy
from domain.les_v7.research import fit_factor_weights, fit_heads
from domain.les_v7.training import prepare_events, train_offline, validate_training_config


GROUPS = ("2025-03-03", "2025-04-07", "2025-05-13", "2025-06-18")


def synthetic_config():
    compare = lambda field, op, value: dict(field=field, op=op, value=value)
    return dict(
        schema="les-v7-training-config/v1",
        feature_names=["slope5", "dfa512"],
        feature_bounds=[[-10., 10.], [0., 3.]],
        context_rules={
            "REVERSAL": compare("close", "lt", 105.),
            "TREND_PULLBACK": dict(op="all", args=[compare("close", "gt", 110.), compare("close", "lt", 120.)]),
            "LEADERSHIP_CONTINUATION": compare("close", "gt", 125.),
        },
        trigger_rules=dict(near_pullback_support=compare("close", "gt", 0.),
                           tight_base_or_retest=compare("close", "gt", 0.)),
        reference_policy=dict(policy_id="REFERENCE_H20", version="SYNTHETIC_TEST_ONLY",
            entry_rule="CUTOFF_CLOSE_NEXT_SESSION_BAND", support_rule="STRICT_PIVOT_ELSE_PRIOR_LL20",
            reference_shares=1, entry_band_atr=[0., .25], cost_up_per_share=.02,
            cost_stop_per_share=.02, leadership_rr_min=1.8),
        quality_policy=default_quality_policy(),
        fit_cutoff="2025-06-17T13:30:00+08:00", validation_start="2025-06-18T13:30:00+08:00",
        validation_end="2025-06-18T13:30:00+08:00", evaluation_cutoff="2025-07-21T13:30:00+08:00",
        folds=[dict(fold_id="synthetic-past-1", fit_cutoff="2025-04-04T13:30:00+08:00",
                    prediction_start="2025-04-07T13:30:00+08:00", prediction_end="2025-04-07T13:30:00+08:00"),
               dict(fold_id="synthetic-past-2", fit_cutoff="2025-05-12T13:30:00+08:00",
                    prediction_start="2025-05-13T13:30:00+08:00", prediction_end="2025-05-13T13:30:00+08:00")],
        min_head_rows=2, min_calibration_rows=2, min_weight_rows=3,
        min_context_weight_rows=1, min_holdout_rows=3,
        head_iterations=20, calibration_iterations=30, regularization=.01,
        null_seed=19, null_replicates=16,
        weight_solver=dict(max_iterations=1000, tolerance=1e-6),
        # Deliberately permissive synthetic engineering checks, not deployment
        # criteria, market validation or evidence of forecasting accuracy.
        acceptance=dict(max_brier=1., max_timing_nll=100., max_pinball_loss=10.),
    )


def seal(payload):
    return dict(payload=payload, sha256=identity(payload))


def synthetic_published_candidate(candidate):
    """In-memory loader fixture only; no historical deployment assertion."""
    return dict(deepcopy(candidate), trained_until="2026-10-13T10:00:00+08:00")


def native_seed(symbol, market, cutoff, bars, evidence_ids, config):
    selected = select_bars(pd.DataFrame(bars), cutoff)
    features, _ = compute_features(selected)
    context, _ = select_context(features, config["context_rules"])
    rows = selected.astype(object).where(pd.notna(selected), None).to_dict("records")
    for row in rows:
        row["Date"] = pd.Timestamp(row["Date"]).date().isoformat()
    evidence_id = identity(dict(symbol=symbol, market=market, cutoff=timestamp(cutoff).isoformat(),
        bars=rows, data_rules=FEATURE_VERSION, external_evidence=evidence_ids))
    return features, make_plan(context, features, config["reference_policy"], evidence_id)


def synthetic_dataset(config=None):
    config = synthetic_config() if config is None else config
    histories, events = [], []
    all_sessions = set()
    for group_index, day in enumerate(GROUPS):
        for context_index, context in enumerate(CONTEXTS):
            for outcome in ("success", "stop", "no_fill"):
                symbol = f"SYNTHETIC_{group_index}_{context_index}_{outcome}"
                dates = pd.bdate_range(end=day, periods=540)
                future = pd.bdate_range(start=pd.Timestamp(day) + pd.Timedelta(days=1), periods=20)
                all_sessions.update(dates.strftime("%Y-%m-%d"))
                all_sessions.update(future.strftime("%Y-%m-%d"))
                x = np.arange(len(dates))
                close = 80. + .06 * x + .8 * np.sin(x / 4)
                if context == "REVERSAL":
                    close[-40:] = np.r_[np.linspace(close[-41], 99., 30),
                        [99.3, 99.6, 99.9, 100.2, 100.1, 100.4, 100.8, 101.2, 101.5, 101.8]]
                elif context == "TREND_PULLBACK":
                    close[-6:] = close[-7] + np.array([.3, .5, .7, .6, .3, .2])
                else:
                    close[-10:] = np.linspace(close[-11], close[-11] + 1., 10)
                    close += 20.
                bars = [dict(Date=d.date().isoformat(), Open=float(c - .1), High=float(c + .8),
                    Low=float(c - .8), Close=float(c), Volume=100000., Amount=float(c * 100000.),
                    Bid=float(c - .02), Ask=float(c + .02), **{"TAIEX Close": 10000.,
                    "Available At": d.date().isoformat() + "T13:30:00+08:00", "Complete": True})
                    for d, c in zip(dates, close)]
                cutoff = day + "T13:30:00+08:00"
                evidence_ids = ["SYNTHETIC_TEST_ONLY:" + symbol]
                features, plan = native_seed(symbol, "TWSE", cutoff, bars, evidence_ids, config)
                assert plan is not None and plan.context == context
                for index, date in enumerate(future):
                    if outcome == "stop":
                        c = plan.stop - .1 if index >= 1 else plan.entry
                    else:
                        c = plan.entry + index * .6
                    bars.append(dict(Date=date.date().isoformat(), Open=float(c), High=float(c + .8),
                        Low=float(c - .4), Close=float(c), Volume=100000., Amount=float(c * 100000.),
                        Bid=float(c - .02), Ask=float(c + .02), **{"TAIEX Close": 10000.,
                        "Available At": date.date().isoformat() + "T13:30:00+08:00", "Complete": True}))
                final_at = future[-1].date().isoformat() + "T13:30:00+08:00"
                coordinate = "SYNTHETIC_TWD:" + symbol
                execution = dict(policy_id=plan.policy_id, policy_version=plan.policy_version,
                    plan_id=plan.plan_id, coordinate_id=coordinate, path_evidence_ids=evidence_ids,
                    path_available_at=final_at, filled=outcome != "no_fill")
                if outcome != "no_fill":
                    execution.update(entry=plan.entry, initial_stop=plan.stop, cutoff_atr=plan.atr,
                        shares=1, filled_at=future[0].date().isoformat() + "T09:05:00+08:00",
                        path_order_verified=True, net_2r_first_hit_at=(future[16].date().isoformat() + "T10:00:00+08:00"
                            if outcome == "success" else None),
                        initial_stop_first_hit_at=(future[1].date().isoformat() + "T10:00:00+08:00"
                            if outcome == "stop" else None),
                        realized_at=future[16 if outcome == "success" else 1].date().isoformat() + "T10:00:00+08:00",
                        realized_net_pnl=(2. if outcome == "success" else -1.) * (plan.entry - plan.stop),
                        cost_policy_version="SYNTHETIC_COST_AFTER_FEES_R1")
                histories.append(dict(symbol=symbol, market="TWSE", coordinate_id=coordinate, bars=bars))
                events.append(dict(symbol=symbol, market="TWSE", cutoff=cutoff, evidence_ids=evidence_ids,
                    execution=execution, execution_sha256=identity(execution)))
    payload = dict(schema="les-v7-training-dataset/v1", dataset_kind="SYNTHETIC_TEST_ONLY",
        price_unit="TWD", volume_unit="SHARES", amount_unit="TWD",
        source_evidence_ids=["SYNTHETIC_TEST_ONLY_NOT_MARKET_EVIDENCE"],
        universe_policy="SYNTHETIC_ALL_EVENTS_WITHOUT_WINNER_SELECTION", universe=sorted(
            [[item["symbol"], item["market"]] for item in histories]), histories=histories, events=events,
        calendars=dict(TWSE=dict(source="SYNTHETIC_TEST_ONLY", sessions=sorted(all_sessions),
            available_at="2022-01-01T00:00:00+08:00", evidence_ids=["SYNTHETIC_FROZEN_SESSION_FIXTURE"])))
    return seal(payload)


@pytest.fixture(scope="module")
def full_inputs():
    config = synthetic_config()
    return synthetic_dataset(config), config


@pytest.fixture(scope="module")
def full_result(full_inputs):
    dataset, config = full_inputs
    return train_offline(dataset, config)


def test_native_end_to_end_fits_oof_calibration_context_weights_holdout_and_freeze(full_inputs, full_result):
    result = full_result
    assert result["status"] == "READY_TO_FREEZE", result["reason_codes"]
    assert result["research_only"] and not result["approved_for_live"]
    assert result["source_authenticity"] == "NOT_INDEPENDENTLY_CERTIFIED"
    assert result["counts"]["prepared"] == 36 and result["counts"]["holdout"] == 9
    assert set(result["counts"]["weight_contexts"]) == set(CONTEXTS)
    payload = result["candidate"]
    assert payload["trained_until"] is None
    with pytest.raises(ValueError, match="MODEL_BUNDLE_NOT_PUBLISHED"):
        FrozenBundle.load(seal(payload))
    assert payload["training_manifest"]["dataset_kind"] == "SYNTHETIC_TEST_ONLY"
    ids = payload["validation_report_ids"]
    assert {"calibration", "holdout", "fractal_null", "projection", "ablation"} <= set(ids)
    assert all(identity(result["reports"][rid]) == rid for rid in ids.values())
    assert result["reports"][ids["fractal_null"]]["status"] == "PASS"
    assert result["reports"][ids["projection"]]["status"] == "PASS"
    assert result["reports"][ids["ablation"]]["same_row_ids"]
    rows, _ = prepare_events(*full_inputs)
    row = next(r for r in rows if r["cutoff"].startswith(GROUPS[-1]))
    # Retrospective OOF/holdout inference is separate from runtime availability.
    prediction = project_outputs(head_outputs(payload["heads"], row["vector"]),
                                 row["features"], payload["calibration"])
    assert prediction.status == "VALID"
    assert prediction.p_entry <= prediction.p_rebound
    assert sum(prediction.timing) == pytest.approx(1.)


def test_repeat_is_bitwise_deterministic_and_input_unmodified(full_inputs, full_result):
    dataset, config = full_inputs
    before_dataset, before_config = identity(dataset), identity(config)
    assert train_offline(dataset, config) == full_result
    assert identity(dataset) == before_dataset and identity(config) == before_config


def test_research_candidate_cannot_be_loaded_into_daily_execution(tmp_path, full_result):
    from config.settings import AppSettings, SettingsError
    path = tmp_path / "SYNTHETIC_TEST_ONLY_frozen_bundle.json"
    path.write_text(json.dumps(seal(synthetic_published_candidate(full_result["candidate"]))), encoding="utf-8")
    with pytest.raises(SettingsError, match="SEPARATE_DEPLOYMENT_REVIEW"):
        AppSettings.from_env({"TWSE_DAILY_ENTRY_MODEL": "v7", "TWSE_LES_V7_BUNDLE": str(path)},
                             mail_enabled=False, mops_enabled=False, validate_paths=False)


@pytest.mark.parametrize("change", ["report", "approval"])
def test_frozen_new_producer_rejects_tampered_report_and_approval(full_result, change):
    payload = synthetic_published_candidate(full_result["candidate"])
    if change == "report":
        report_id = payload["validation_report_ids"]["projection"]
        payload["validation_reports"][report_id]["status"] = "SYNTHETIC_TAMPER"
    else:
        payload["approved_for_live"] = True
    with pytest.raises(ValueError, match="VALIDATION_REPORT_CONTENT"):
        FrozenBundle.load(seal(payload))


def test_frozen_model_cannot_be_available_before_its_holdout_labels(full_result):
    payload = synthetic_published_candidate(full_result["candidate"])
    payload["trained_until"] = payload["fit_cutoff"]
    with pytest.raises(ValueError, match="VALIDATION_NOT_AVAILABLE_AT_PUBLICATION"):
        FrozenBundle.load(seal(payload))


def test_native_candidate_publication_uses_actual_time_and_verifiable_reports(tmp_path, full_inputs, full_result):
    from application.les_v7_training import train_request
    from infrastructure.storage.les_v7_training import LesV7TrainingRepository

    class SyntheticGuard:
        def retain(self, objects):
            return dict(source=objects.put(b"SYNTHETIC_TEST_ONLY source identity"),
                        environment=objects.put_json(dict(kind="SYNTHETIC_TEST_ONLY environment")),
                        parameters=objects.put_json(dict(kind="SYNTHETIC_TEST_ONLY parameters")))

        def check(self):
            pass

    published_at = "2026-10-13T10:00:00+08:00"
    repository = LesV7TrainingRepository(tmp_path, guard_factory=SyntheticGuard, clock=lambda: published_at)
    dataset, config = full_inputs
    # full_result was produced by the actual native training pipeline above;
    # reuse it for publication rather than duplicate expensive fitting here.
    receipt = train_request(dataset, config, repository, allow_synthetic=True,
                            trainer=lambda *_: deepcopy(full_result))
    frozen = json.loads((tmp_path / receipt["job_id"] / "frozen_bundle.json").read_text())
    assert FrozenBundle.load(frozen).record()["trained_until"] == published_at
    assert receipt["first_published_at"] == published_at
    assert receipt["execution_action"] == "NO_ACTION"
    assert repository.verify(tmp_path / receipt["job_id"])["bundle_sha256"] == frozen["sha256"]
    rows, _ = prepare_events(*full_inputs)
    row = rows[-1]
    assert predict(FrozenBundle.load(frozen), row["features"], row["plan"], cutoff=row["cutoff"]).status == "NOT_READY"


def test_event_and_history_order_do_not_change_fits(full_inputs, full_result):
    dataset, config = full_inputs
    payload = deepcopy(dataset["payload"])
    random.Random(37).shuffle(payload["histories"])
    random.Random(71).shuffle(payload["events"])
    result = train_offline(seal(payload), config)
    assert result["status"] == full_result["status"]
    for key in ("heads", "calibration", "weights"):
        assert result["candidate"][key] == full_result["candidate"][key]
    assert result["oof_predictions"] == full_result["oof_predictions"]


def test_future_market_rows_cannot_change_features_labels_or_fits(full_inputs, full_result):
    dataset, config = full_inputs
    payload = deepcopy(dataset["payload"])
    for history in payload["histories"]:
        history["bars"].append(dict(history["bars"][-1], Date="2025-08-01", Open=1., High=2., Low=.5,
            Close=1., Amount=100000., Bid=.99, Ask=1.01, **{"Available At": "2025-08-01T13:30:00+08:00"}))
    result = train_offline(seal(payload), config)
    for key in ("heads", "calibration", "weights"):
        assert result["candidate"][key] == full_result["candidate"][key]
    assert result["oof_predictions"] == full_result["oof_predictions"]


def test_every_oof_fit_and_calibrator_has_only_past_matured_labels(full_inputs, full_result):
    rows, _ = prepare_events(*full_inputs)
    by_id = {r["row_id"]: r for r in rows}
    for fold in full_result["folds"]:
        assert fold["status"] == "FITTED"
        for row_id in fold["training_row_ids"]:
            row = by_id[row_id]
            assert timestamp(row["cutoff"]) < timestamp(row["label_available_at"]) <= timestamp(fold["fit_cutoff"])
        assert set(fold["training_row_ids"]).isdisjoint(fold["prediction_row_ids"])
    first = [p for p in full_result["oof_predictions"] if p["fold_id"] == "synthetic-past-1"]
    later = [p for p in full_result["oof_predictions"] if p["fold_id"] == "synthetic-past-2"]
    assert first and later and all(p["forecast"]["status"] == "NOT_READY" for p in first)
    for prediction in later:
        assert prediction["forecast"]["status"] == "VALID"
        assert timestamp(prediction["head_last_label_available_at"]) <= timestamp(prediction["head_fit_cutoff"]) < timestamp(prediction["cutoff"])
        assert timestamp(prediction["calibration_last_label_available_at"]) <= timestamp(prediction["calibration_fit_cutoff"]) < timestamp(prediction["cutoff"])
    manifest = full_result["candidate"]["training_manifest"]
    valid_ids = {p["prediction_id"] for p in later if by_id[p["row_id"]]["utility"] is not None}
    assert set(manifest["weight_prediction_ids"]) == valid_ids


def test_no_fill_never_gets_conditional_rebound_or_joint_and_ambiguous_path_is_excluded(full_inputs):
    dataset, config = full_inputs
    rows, _ = prepare_events(dataset, config)
    no_fill = [r for r in rows if r["labels"]["fill"] == 0]
    assert len(no_fill) == 12
    assert all(r["labels"]["rebound"] is None and r["labels"]["joint"] is None and r["utility"] is None for r in no_fill)
    payload = deepcopy(dataset["payload"])
    event = next(e for e in payload["events"] if e["execution"]["filled"])
    event["execution"]["path_order_verified"] = False
    event["execution_sha256"] = identity(event["execution"])
    repaired, diagnostics = prepare_events(seal(payload), config)
    assert len(repaired) == len(rows) - 1
    assert any(d.get("excluded_reason") == "PATH_AMBIGUOUS" for d in diagnostics)


@pytest.mark.parametrize("balanced", [False, True])
def test_final_holdout_path_labels_do_not_change_heads_calibration_or_weights(full_inputs, full_result, balanced):
    dataset, config = full_inputs
    payload = deepcopy(dataset["payload"])
    histories = {history["symbol"]: history for history in payload["histories"]}
    for event in payload["events"]:
        if (event["cutoff"].startswith(GROUPS[-1]) and event["execution"]["filled"]
                and (balanced or event["execution"].get("net_2r_first_hit_at") is not None)):
            fill = event["execution"]
            success = balanced and fill["net_2r_first_hit_at"] is None
            history = histories[event["symbol"]]
            future = history["bars"][-20:]
            for index, bar in enumerate(future):
                c = (fill["entry"] + index * .6 if success else
                     fill["initial_stop"] - .1 if index >= 1 else fill["entry"])
                bar.update(Open=c, High=c + .8, Low=c - .4, Close=c,
                           Amount=c * 100000., Bid=c - .02, Ask=c + .02)
            fill["net_2r_first_hit_at"] = future[16]["Date"] + "T10:00:00+08:00" if success else None
            fill["initial_stop_first_hit_at"] = None if success else future[1]["Date"] + "T10:00:00+08:00"
            fill["realized_at"] = future[16 if success else 1]["Date"] + "T10:00:00+08:00"
            fill["realized_net_pnl"] = (2. if success else -1.) * (fill["entry"] - fill["initial_stop"])
            event["execution_sha256"] = identity(event["execution"])
    result = train_offline(seal(payload), config)
    assert result["status"] == "READY_TO_FREEZE"
    for key in ("heads", "calibration", "weights"):
        assert result["candidate"][key] == full_result["candidate"][key]
    holdout = full_result["candidate"]["validation_report_ids"]["holdout"]
    altered_holdout = result["candidate"]["validation_report_ids"]["holdout"]
    before = full_result["reports"][holdout]
    after = result["reports"][altered_holdout]
    assert after["label_sha256"] != before["label_sha256"]
    assert after["heads_sha256"] == before["heads_sha256"]
    assert after["calibration_sha256"] == before["calibration_sha256"]
    if balanced:
        # Identical aggregate calibration metrics do not identify the labels
        # used by an independently evaluated report. Its label hash still must.
        assert after["metrics"] == before["metrics"]


def test_sparse_folds_and_missing_paths_stay_not_ready_no_fake_probability(full_inputs):
    dataset, config = full_inputs
    sparse = dict(config, min_head_rows=1000)
    result = train_offline(dataset, sparse)
    assert result["status"] == "NOT_READY" and result["candidate"] is None
    assert not result["oof_predictions"] and all(f["status"] == "NOT_READY" for f in result["folds"])
    payload = deepcopy(dataset["payload"])
    for event in payload["events"]:
        event.pop("execution")
        event.pop("execution_sha256")
    result = train_offline(seal(payload), config)
    assert result["status"] == "NOT_READY" and result["candidate"] is None
    assert result["counts"]["prepared"] == 0
    assert result["counts"]["exclusions"] == {"PATH_NOT_PROVIDED": 36}


def test_less_than_513_cutoff_closes_is_a_feature_gap_not_an_imputed_probability(full_inputs):
    dataset, config = full_inputs
    payload = deepcopy(dataset["payload"])
    # Remove only early warmup, not a future-label session. The resulting 512
    # cutoff closes cannot supply 512 return observations to DFA512.
    for history in payload["histories"]:
        history["bars"] = history["bars"][28:]
    rows, diagnostics = prepare_events(seal(payload), config)
    assert rows == []
    assert all(d["excluded_reason"] == "TRAINING_FEATURES_INCOMPLETE" for d in diagnostics)
    assert all(d["feature_audit"]["fractal_history"]["dfa512"]["available_close_count"] == 512 for d in diagnostics)


@pytest.mark.parametrize("change,pattern", [
    ("hash", "HASH_OR_SCHEMA"), ("unit", "UNITS_UNIVERSE"), ("sources", "UNITS_UNIVERSE"),
    ("calendar_source", "FROZEN_CALENDAR"), ("calendar_order", "FROZEN_CALENDAR"),
    ("execution_hash", "EXECUTION_EVIDENCE_HASH"), ("bars_order", "SORTED_UNIQUE"),
    ("incomplete", "INCOMPLETE_DAILY_BAR"), ("naive_availability", "TIMEZONE"),
])
def test_training_evidence_fails_closed(full_inputs, change, pattern):
    dataset, config = full_inputs
    payload = deepcopy(dataset["payload"])
    if change == "unit": payload["price_unit"] = "TWD_X_1000"
    elif change == "sources": payload["source_evidence_ids"] = []
    elif change == "calendar_source": payload["calendars"]["TWSE"]["source"] = "UNVERIFIED_WEEKDAYS"
    elif change == "calendar_order": payload["calendars"]["TWSE"]["sessions"].reverse()
    elif change == "execution_hash": payload["events"][0]["execution"]["filled"] = "PASS"
    elif change == "bars_order": payload["histories"][0]["bars"].reverse()
    elif change == "incomplete": payload["histories"][0]["bars"][0]["Complete"] = False
    elif change == "naive_availability": payload["histories"][0]["bars"][0]["Available At"] = "2023-01-01T13:30:00"
    value = seal(payload)
    if change == "hash": value["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=pattern):
        prepare_events(value, config)


@pytest.mark.parametrize("key", ["fill", "rebound", "joint"])
def test_head_fit_rejects_single_binary_class_and_future_transform(key):
    cutoff, matured = "2025-01-01T13:30:00+08:00", "2025-02-01T13:30:00+08:00"
    rows = [dict(cutoff=cutoff, label_available_at=matured, feature_fit_cutoff=cutoff,
        features=[.1, .5], fill=1, rebound=i, joint=i, log_bottom_ratio=-.02,
        bottom_timing=i, observation_weight=1.) for i in (0, 1)]
    for row in rows:
        row[key] = 0
        if key == "fill":
            row["rebound"] = row["joint"] = None
    with pytest.raises(ValueError, match="BOTH_BINARY|INVALID_CONDITIONAL"):
        fit_heads(rows, fit_cutoff="2025-02-02T13:30:00+08:00", iterations=2)
    rows[0]["feature_fit_cutoff"] = "2025-01-02T13:30:00+08:00"
    with pytest.raises(ValueError, match="FUTURE_SCALER"):
        fit_heads(rows, fit_cutoff="2025-02-02T13:30:00+08:00", iterations=2)


def test_weight_solver_rejects_in_sample_and_calibrator_leakage():
    row = dict(prediction_kind="IN_SAMPLE", factors=[.5] * 6, utility=.5, context="REVERSAL",
        cutoff="2025-03-01T13:30:00+08:00", label_available_at="2025-04-01T13:30:00+08:00",
        head_fit_cutoff="2025-02-15T13:30:00+08:00", head_last_label_available_at="2025-02-14T13:30:00+08:00",
        calibration_fit_cutoff="2025-03-01T13:30:00+08:00", calibration_last_label_available_at="2025-02-28T13:30:00+08:00")
    with pytest.raises(ValueError, match="REQUIRES_OOF"):
        fit_factor_weights([row], fit_cutoff="2025-04-02T13:30:00+08:00")
    row["prediction_kind"] = "STRICTLY_PAST_OOF"
    with pytest.raises(ValueError, match="CALIBRATION_TIME_LEAKAGE"):
        fit_factor_weights([row], fit_cutoff="2025-04-02T13:30:00+08:00")


@pytest.mark.parametrize("change,pattern", [("overlap", "TIME_LEAKAGE"), ("future_holdout", "VALIDATION_TIME"),
    ("no_fractal", "FRACTAL_AND_BASELINE"), ("shadow", "SHADOW"), ("unbounded", "TRAINING_COUNT")])
def test_config_requires_frozen_supported_and_temporal_policy(change, pattern):
    config = synthetic_config()
    if change == "overlap": config["folds"][1]["prediction_start"] = config["folds"][0]["prediction_start"]
    elif change == "future_holdout": config["validation_start"] = config["fit_cutoff"]
    elif change == "no_fractal": config["feature_names"] = ["slope5", "rsi"]
    elif change == "shadow": config["feature_names"] = ["mfdfa_alpha", "dfa512"]
    elif change == "unbounded": config["null_replicates"] = 100000
    with pytest.raises(ValueError, match=pattern): validate_training_config(config)

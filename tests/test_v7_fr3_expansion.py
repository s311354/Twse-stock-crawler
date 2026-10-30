"""SYNTHETIC calendars/configs only; no official source or model skill claim."""
from copy import deepcopy
from datetime import date, timedelta

import pytest

from application.les_v7_fr3_expansion import plan_fr3_expansion, verify_fr3_expansion_plan
from domain.les_v7.contracts import identity
from domain.les_v7.fr3_plan import policy
from domain.les_v7.quality_policy import r6_quality_policy
from tests.strategies.test_v7_offline_training import synthetic_config
from tests.test_v7_dataset_builder import builder_inputs


@pytest.fixture
def templates():
    _, build = builder_inputs()
    build = deepcopy(build)
    path = build["seed"]["reference_policy"]["cost_path_policy"]
    path["cost_policy"].update(effective_start="2024-01-01", effective_end="2027-12-31")
    build["seed"]["reference_policy"] = policy(build["seed"]["reference_policy"],
        available_at="2026-10-23T15:21:04Z")
    build["path_policy"] = deepcopy(build["seed"]["reference_policy"]["cost_path_policy"])
    build["evaluation_cutoff"] = "2026-10-22T13:30:00+08:00"
    build["max_events"] = 4096
    training = synthetic_config()
    training.update(deepcopy(build["seed"]))
    training.update(quality_policy=r6_quality_policy(), evaluation_cutoff=build["evaluation_cutoff"],
        min_holdout_rows=500, min_weight_rows=500, min_context_weight_rows=100)
    return build, training


@pytest.fixture
def sessions():
    # Deliberately invented weekdays plus an exceptional Saturday. They are
    # NOT an official calendar and must never be used for production training.
    result, day = [], date(2021, 1, 4)
    while day <= date(2026, 10, 23):
        if day.weekday() < 5 and day.isoformat() not in {"2026-10-23", "2024-07-04"}:
            result.append(day.isoformat())
        day += timedelta(days=1)
    result.append("2024-07-06")
    return sorted(result)


def run(templates, sessions, **options):
    arguments = dict(sessions=sessions, history_start="2021-01-04", through="2026-10-23",
        universe_id=identity([["SYNTHETIC", "TWSE"]]), calendar_id=identity({"SYNTHETIC": sessions}))
    arguments.update(options)
    return plan_fr3_expansion(*templates, **arguments)


def test_outcome_blind_plan_has_stable_content_identity_and_detached_configs(templates, sessions):
    original = deepcopy((templates, sessions))
    result = run(templates, sessions)
    assert result["plan_id"] == identity(result["payload"])
    assert run(templates, sessions) == result
    assert (templates, sessions) == original
    assert verify_fr3_expansion_plan(result) == result
    result["payload"]["training_config"]["reference_policy"]["leadership_rr_min"] = 999
    assert (templates, sessions) == original


def test_final_180_mature_open_dates_and_20_future_session_horizon(templates, sessions):
    body = run(templates, sessions)["payload"]
    mature = sessions[:-20]
    eligible = [day for day in mature if day >= "2024-01-01"]
    holdout = [day for window in body["comparison_windows"] for day in window["sessions"]]
    assert holdout == eligible[-180:]
    assert [len(window["sessions"]) for window in body["comparison_windows"]] == [60] * 3
    assert body["build_config"]["event_start"] == "2024-01-01"
    assert body["build_config"]["event_end"] == sessions[-21]
    assert body["training_config"]["fit_cutoff"][:10] == sessions[sessions.index(holdout[0]) - 1]
    assert body["training_config"]["validation_start"][:10] == holdout[0]
    assert body["training_config"]["validation_end"][:10] == holdout[-1]
    assert body["training_config"]["evaluation_cutoff"] == "2026-10-23T13:30:00+08:00"
    assert body["history_end"] == "2026-10-23"  # Through may be officially CLOSED.
    assert body["source_sessions"][-1] == "2026-10-22"


def test_six_calendar_month_initial_training_and_monthly_past_only_oof(templates, sessions):
    body = run(templates, sessions)["payload"]
    config = body["training_config"]
    folds = config["folds"]
    assert folds[0]["prediction_start"] == "2024-07-01T13:30:00+08:00"
    previous = None
    for fold in folds:
        start, end = fold["prediction_start"][:10], fold["prediction_end"][:10]
        expected_fit = sessions[sessions.index(start) - 1]
        assert fold["fit_cutoff"][:10] == expected_fit
        assert fold["fit_cutoff"] < fold["prediction_start"] <= fold["prediction_end"] < config["fit_cutoff"]
        assert start[:7] == end[:7]
        assert previous is None or previous < fold["prediction_start"]
        previous = fold["prediction_end"]
    assert "2024-07-06" in body["candidate_sessions"]  # No weekday reclassification.
    assert "2024-07-04" not in body["candidate_sessions"]


def test_no_economic_policy_minimum_or_model_parameter_changes(templates, sessions):
    body = run(templates, sessions)["payload"]
    for source, changed, allowed in ((templates[0], body["build_config"],
            {"event_start", "event_end", "evaluation_cutoff"}),
            (templates[1], body["training_config"],
             {"fit_cutoff", "validation_start", "validation_end", "evaluation_cutoff", "folds"})):
        assert {k: v for k, v in source.items() if k not in allowed} == {
            k: v for k, v in changed.items() if k not in allowed}
    assert body["native_minima"]["min_holdout_rows"] == 500
    assert body["native_minima"]["min_context_weight_rows"] == 100
    assert body["native_minima"]["min_weight_rows"] == 500
    assert body["capacity"]["batch_size_upper_bound"] == 4096 // body["capacity"]["event_calendar_days"]
    assert body["capacity"]["sample_upper_bounds_verified"] is False
    assert body["status"] == "CALENDAR_PLAN_ONLY_NOT_TRAINING_READY"
    assert body["holdout_interpretation"] == "RETROSPECTIVE_DEVELOPMENT_NOT_FRESH_CONFIRMATION"
    assert body["historical_deployment_claim"] is body["native_prepared_reuse_allowed"] is False
    assert body["model_fitted"] is body["approved_for_live"] is False


@pytest.mark.parametrize("minimum,value", [("min_holdout_rows", 499),
    ("min_context_weight_rows", 99), ("min_weight_rows", 499)])
def test_lowered_native_engineering_minima_are_rejected(templates, sessions, minimum, value):
    templates[1][minimum] = value
    with pytest.raises(ValueError, match="MINIMA_MUST_NOT_BE_LOWERED"):
        run(templates, sessions)


@pytest.mark.parametrize("damage", ["reversed", "duplicate", "noncanonical", "outside", "boolean"])
def test_calendar_is_bounded_unique_and_canonical_not_inferred(templates, sessions, damage):
    if damage == "reversed":
        sessions.reverse()
    elif damage == "duplicate":
        sessions.insert(10, sessions[10])
    elif damage == "noncanonical":
        sessions[0] = "20210104"
    elif damage == "outside":
        sessions.append("2026-10-24")
    else:
        sessions[0] = True
    with pytest.raises(ValueError, match="FR3_EXPANSION"):
        run(templates, sessions)


def test_cannot_plan_without_explicit_calendar_content_binding(templates, sessions):
    with pytest.raises(ValueError, match="CONTENT_ID_REQUIRED:calendar_id"):
        run(templates, sessions, calendar_id=None)


def test_not_enough_mature_dates_does_not_shorten_holdout_or_horizon(templates, sessions):
    with pytest.raises(ValueError, match="INSUFFICIENT_MATURE_OPEN_SESSIONS"):
        run(templates, [day for day in sessions if day >= "2026-01-01"][-190:])


def test_cost_validity_is_not_extended_or_backdated(templates, sessions):
    for config in (templates[0]["path_policy"],
                   templates[0]["seed"]["reference_policy"]["cost_path_policy"],
                   templates[1]["reference_policy"]["cost_path_policy"]):
        config["cost_policy"]["effective_end"] = "2026-10-22"
    with pytest.raises(ValueError, match="COST_POLICY_DOES_NOT_COVER_SOURCE_END"):
        run(templates, sessions)


def test_changed_calendar_or_scope_changes_plan_identity_not_business_policy(templates, sessions):
    original = run(templates, sessions)
    changed = run(templates, sessions, calendar_id="a" * 64)
    assert changed["plan_id"] != original["plan_id"]
    assert changed["payload"]["training_config"] == original["payload"]["training_config"]
    assert run(templates, sessions, universe_id="b" * 64)["plan_id"] != original["plan_id"]


def test_recompute_refuses_rehashed_wrong_date_partition(templates, sessions):
    result = run(templates, sessions)
    result["payload"]["comparison_windows"][0]["sessions"].pop()
    result["plan_id"] = identity(result["payload"])
    with pytest.raises(ValueError, match="RECOMPUTATION_MISMATCH"):
        verify_fr3_expansion_plan(result)


def test_one_pure_planning_operation_never_reads_or_writes_artifacts(templates, sessions, monkeypatch):
    import builtins
    monkeypatch.setattr(builtins, "open", lambda *_a, **_k: pytest.fail("planning must remain pure"))
    result = run(templates, sessions)
    assert verify_fr3_expansion_plan(result) == result

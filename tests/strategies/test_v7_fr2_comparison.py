"""SYNTHETIC engineering fixtures, never real efficacy evidence."""
from copy import deepcopy

import pytest

from domain.les_v7.contracts import identity
from domain.les_v7 import fr2_comparison as module


def seal(body):
    return dict(payload=body, sha256=identity(body))


def comparison():
    conditions = {k: "SYNTHETIC_" + k for k in module.CONDITION_KEYS}
    models = {}
    for name in ("M0", "M1"):
        models[name] = seal(dict(feature_names=["return20", "atr_fraction"] + (["dfa512"] if name == "M1" else []),
            heads_id=name + "_heads", calibration_id=name + "_calibration", weights_id=name + "_weights",
            training_config_id="SYNTHETIC_MATCHED_CONFIG", fit_cutoff="2026-01-01T05:30:00Z",
            last_training_label_available_at="2025-12-31T05:30:00Z",
            calibration_last_label_available_at="2025-12-31T05:30:00Z"))
    sessions = ["2026-01-05", "2026-01-06", "2026-01-07"]
    arms = {}
    for name in module.ARMS:
        rows = []
        for index, session in enumerate(sessions):
            cutoff = session + "T05:30:00+00:00"
            row = dict(symbol="SYNTHETIC", market="TWSE", cutoff=cutoff,
                label_available_at="2026-02-20T05:30:00Z", bottom=90, close=100, atr=5,
                labels=dict(timing=1, fill=1, rebound=index % 2, joint=0),
                forecast=dict(quantiles=[85, 91 if name.startswith("M1") else 92, 95],
                    timing=[1 / 21] * 21, p_fill=.8, p_rebound=.6, p_entry=.2))
            row["row_id"] = identity(dict(symbol=row["symbol"], market=row["market"], cutoff=cutoff))
            rows.append(row)
        arms[name] = seal(dict(conditions=deepcopy(conditions), model_id=models[name[:2]]["sha256"],
            entry_policy_id=name[-2:], execution_policy_id="SYNTHETIC_EXEC", events=rows))
    return seal(dict(schema=module.SCHEMA, research_only=True, conditions=conditions,
        models=models, arms=arms, test_sessions=sessions, calendar_evidence_ids=["SYNTHETIC_CALENDAR"],
        entry_policy_ids=dict(P0="P0", P1="P1"), execution_policy_id="SYNTHETIC_EXEC",
        evaluation_cutoff="2026-03-01T05:30:00Z", test_windows=[dict(sessions=sessions)],
        holdout_use="DEVELOPMENT_ALREADY_INSPECTED", predeclared_before_outcomes=False))


def edit_arm(record, name, edit):
    body = deepcopy(record["payload"])
    arm = body["arms"][name]["payload"]
    edit(arm)
    body["arms"][name] = seal(arm)
    return seal(body)


def test_four_arms_keep_same_forecasts_but_small_existing_data_is_not_efficacy():
    record = comparison()
    before = deepcopy(record)
    result = module.compare(record)
    assert record == before
    assert result == module.compare(record)
    assert result["status"] == "NOT_ASSESSABLE"
    assert set(result["arms"]) == set(module.ARMS)
    assert all(a["forecast_status"] == "COMPLETED" and a["account_status"] == "NOT_ASSESSABLE" for a in result["arms"].values())
    assert result["forecast_comparisons"]["P1"]["fractal"]["median_bottom_error_atr"] == .2
    assert "DEVELOPMENT_OR_NON_PREDECLARED_HOLDOUT" in result["reason_codes"]
    assert "THREE_DISJOINT_60_SESSION_WINDOWS_REQUIRED" in result["reason_codes"]
    assert all(p["status"] == "NOT_ASSESSABLE" for p in result["after_cost_comparisons"].values())
    assert result["order_authority"] is False


def test_returned_json_cannot_mutate_inputs_or_acceptance_policy(monkeypatch):
    # Isolate the global even on a failing pre-fix run; the regression itself
    # must not contaminate other engineering tests by changing the policy.
    monkeypatch.setattr(module, "POLICY", deepcopy(module.POLICY))
    record = comparison()
    unchanged_input, unchanged_policy = deepcopy(record), deepcopy(module.POLICY)
    original = module.compare(record)
    edited = module.compare(record)
    edited["conditions"]["dataset_id"] = "CALLER_MUTATION"
    edited["acceptance_policy"]["minimum_paired_events"] = 0
    edited["arms"]["M0_P0"]["metrics"]["pinball_loss"] = 999
    assert record == unchanged_input and module.POLICY == unchanged_policy
    assert module.compare(record) == original


@pytest.mark.parametrize("field", ["dataset_id", "universe_id", "split_id", "cost_policy_id", "initial_account_id"])
def test_changed_conditions_cannot_be_called_matched(field):
    record = edit_arm(comparison(), "M1_P1", lambda a: a["conditions"].update({field: "OTHER"}))
    with pytest.raises(ValueError, match="CONDITION_OR_MODEL_MISMATCH"):
        module.compare(record)


def test_admission_policy_cannot_edit_probabilities():
    record = edit_arm(comparison(), "M1_P1", lambda a: a["events"][0]["forecast"].update(p_fill=.9))
    with pytest.raises(ValueError, match="CANNOT_MODIFY_PREDICTIVE_PROBABILITIES"):
        module.compare(record)


@pytest.mark.parametrize("change,reason", [
    (lambda r: r.update(label_available_at="2026-04-01T00:00:00Z"), "TIME_ISOLATION"),
    (lambda r: r["labels"].update(fill=0), "CONDITIONAL_LABELS_MUST_BE_NULL"),
    (lambda r: r["forecast"].update(p_entry=.9), "PROJECTION_INVALID"),
    (lambda r: r["forecast"].update(quantiles=[95, 90, 85]), "PROJECTION_INVALID"),
])
def test_future_labels_and_invalid_projections_rejected(change, reason):
    record = edit_arm(comparison(), "M1_P1", lambda a: change(a["events"][0]))
    with pytest.raises(ValueError, match=reason):
        module.compare(record)


def test_different_labels_cannot_hide_behind_same_row_ids():
    record = edit_arm(comparison(), "M1_P1", lambda a: a["events"][0]["labels"].update(timing=2))
    with pytest.raises(ValueError, match="EVENT_OR_LABEL_PAIRING_MISMATCH"):
        module.compare(record)


def test_missing_arm_not_filled_with_baseline_results():
    body = comparison()["payload"]
    del body["arms"]["M0_P1"]
    report = module.compare(seal(body))
    assert report["arms"]["M0_P1"]["status"] == "NOT_ASSESSABLE"
    assert "MISSING_ARM:M0_P1" in report["reason_codes"]


def test_account_unknown_is_not_assumed_empty_and_nonzero_costs_require_binding():
    def add(arm):
        arm["account_path"] = seal(dict(initial_account_id=arm["conditions"]["initial_account_id"],
            cost_policy_id="WRONG", external_cashflows=[], net_of_costs=True,
            execution_evidence_ids=["SYNTHETIC"], sessions=["2026-01-05", "2026-01-06", "2026-01-07"],
            net_equity=[1000000, 1000000, 1000000, 1000000], completed_trades=[]))
    with pytest.raises(ValueError, match="ACCOUNT_COST_OR_SESSION_BINDING_INVALID"):
        module.compare(edit_arm(comparison(), "M0_P0", add))


def test_hash_mismatch_is_not_not_assessable():
    record = comparison()
    record["payload"]["arms"].clear()
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        module.compare(record)


def test_no_fractal_counterpart_is_not_daily_rule_baseline():
    record = comparison()
    body = record["payload"]
    model = body["models"]["M0"]["payload"]
    model["feature_names"].append("dfa512")
    body["models"]["M0"] = seal(model)
    with pytest.raises(ValueError, match="TRAINED_NO_FRACTAL_COUNTERPART_REQUIRED"):
        module.compare(seal(body))


def test_bootstrap_clusters_dates_and_has_fixed_seed():
    # Exercise inference mechanics without pretending 42 synthetic sessions
    # meet the predeclared 180-session real-data acceptance requirement.
    groups = [[i, i + 1000] for i in range(42)]
    statistic = lambda items: sum(sum(g) for g in items) / len(items)
    a = module._bootstrap(groups, statistic, tests=16)
    assert a == module._bootstrap(groups, statistic, tests=16)
    assert a["block_sessions"] == 21 and a["family_tests"] == 16
    assert a["lower"] < a["upper"]


def test_sparse_conditional_labels_make_ci_unknown_not_crash_or_drop_resamples():
    left, right, days = [], [], []
    for index in range(42):
        days.append(f"SYNTHETIC_{index:03}")
        left.append(dict(error=2., pinball=2., timing=2., fill=.2,
            rebound=.2 if index == 0 else None, joint=.2 if index == 0 else None,
            covered=1., width=1., weight=1.))
        right.append(dict(error=1., pinball=1., timing=1., fill=.1,
            rebound=.1 if index == 0 else None, joint=.1 if index == 0 else None,
            covered=1., width=1., weight=1.))
    result = module._forecast_comparison(left, right, days)
    for name in ("rebound_brier", "joint_brier"):
        assert result["criteria"][name]["confidence_interval"] is None
        assert result["criteria"][name]["status"] == "NOT_ASSESSABLE"
    assert module._bootstrap(list(range(42)), lambda _: float("nan"), tests=16) is None

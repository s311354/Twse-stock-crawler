"""Additional synthetic temporal/solver boundaries; no market/deployment claims."""
from copy import deepcopy

import pytest

from domain.les_v7.contracts import PRIOR, identity
from domain.les_v7.research import fit_factor_weights
from domain.les_v7.training import prepare_events, train_offline, validate_training_config
from tests.strategies.test_v7_offline_training import GROUPS, seal, synthetic_config, synthetic_dataset


@pytest.fixture(scope="module")
def inputs():
    config = synthetic_config()
    return synthetic_dataset(config), config


def _observed_no_fill(event):
    old = event["execution"]
    event["execution"] = {k: old[k] for k in ("policy_id", "policy_version", "plan_id",
        "coordinate_id", "path_evidence_ids", "path_available_at")}
    event["execution"]["filled"] = False
    event["execution_sha256"] = identity(event["execution"])


def test_v7_weight_solver_uses_v7_anchor_not_v6_default(monkeypatch):
    captured = {}
    def solver(x, y, contexts, **options):
        captured.update(options)
        return dict(status="CONVERGED", weights={})
    monkeypatch.setattr("domain.strategies.low_entry_v6_weights.fit_convex_weights", solver)
    row = dict(factors=[.5] * 6, utility=.5, context="REVERSAL", prediction_kind="STRICTLY_PAST_OOF",
        head_last_label_available_at="2024-01-01T00:00:00Z", head_fit_cutoff="2024-01-02T00:00:00Z",
        cutoff="2024-02-01T00:00:00Z", label_available_at="2024-03-01T00:00:00Z")
    fit_factor_weights([row], fit_cutoff="2024-04-01T00:00:00Z")
    assert captured["prior"] == PRIOR
    with pytest.raises(ValueError, match="V7_PRIOR_IS_FROZEN"):
        fit_factor_weights([row], fit_cutoff="2024-04-01T00:00:00Z", prior=[1., 0., 0., 0., 0., 0.])


def test_preclose_event_cannot_label_unknown_current_session_low(inputs):
    dataset, config = inputs
    payload = deepcopy(dataset["payload"])
    payload["events"][0]["cutoff"] = payload["events"][0]["cutoff"].replace("13:30:00", "13:29:59")
    rows, diagnostics = prepare_events(seal(payload), config)
    assert len(rows) == 35
    assert any(d.get("excluded_reason") == "NO_COMPLETE_BAR_FOR_EVENT_SESSION" for d in diagnostics)


def test_past_calibrator_without_observed_fills_stays_not_ready(inputs):
    dataset, config = inputs
    payload = deepcopy(dataset["payload"])
    for event in payload["events"]:
        if event["cutoff"].startswith(GROUPS[1]):
            _observed_no_fill(event)
    result = train_offline(seal(payload), config)
    assert result["status"] == "NOT_READY" and result["candidate"] is None
    assert result["folds"][1]["calibration_reason"] == "CONDITIONAL_CALIBRATION_ROWS_REQUIRED"
    assert all(p["forecast"]["status"] == "NOT_READY" for p in result["oof_predictions"])


def test_no_fill_only_holdout_saves_gap_not_fake_conditional_metrics(inputs):
    dataset, config = inputs
    payload = deepcopy(dataset["payload"])
    for event in payload["events"]:
        if event["cutoff"].startswith(GROUPS[-1]):
            _observed_no_fill(event)
    result = train_offline(seal(payload), config)
    assert result["status"] == "NOT_READY" and result["candidate"] is None
    assert result["reason_codes"] == ["INDEPENDENT_HOLDOUT_NOT_READY:CONDITIONAL_CALIBRATION_ROWS_REQUIRED"]


@pytest.mark.parametrize("change,reason", [
    ("trigger", "TRIGGER_RULES_INCOMPLETE"),
    ("future_rule", "CONTEXT_CANNOT_READ_FORECAST_OR_SCORE"),
    ("operator", "UNKNOWN_RULE_OPERATOR"),
    ("policy", "UNSUPPORTED_OR_INCOMPLETE_REFERENCE_POLICY"),
])
def test_frozen_rule_and_reference_contract_validated_without_data(change, reason):
    config = synthetic_config()
    if change == "trigger":
        config["trigger_rules"].pop("tight_base_or_retest")
    elif change == "future_rule":
        config["context_rules"]["REVERSAL"]["field"] = "p_rebound"
    elif change == "operator":
        config["context_rules"]["REVERSAL"]["op"] = "eval"
    else:
        config["reference_policy"]["entry_rule"] = "RETROSPECTIVE_LOW"
    with pytest.raises(ValueError, match=reason):
        validate_training_config(config)

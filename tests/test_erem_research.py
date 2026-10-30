"""Synthetic recorded tapes only; never an alternative account backtest."""
from copy import deepcopy

import pytest

from application.erem_research import compare_recorded_erem


def tape():
    result = dict(jobId="SYNTHETIC_ONLY", evidencePolicy="SYNTHETIC_NOT_OOS", daily=[])
    for day in (1, 2, 3, 4, 5, 6):
        snapshot = dict(cutoff=f"2026-09-{day:02}T13:30:00+08:00", holdings=[
            dict(code="SYNTHETIC_A", shares=20, entryDate="2026-09-01", entryLES=70., entryDecision="BUY_SMALL")])
        result["daily"].append(dict(date=f"2026-09-{day:02}", openingSnapshot=snapshot,
            research=dict(accountSnapshot=deepcopy(snapshot)), decisions=[dict(code="SYNTHETIC_A",
                eremDecision="REDUCE 25%", hc=50.,
                eremReasons=[f"HoldingDays={day-1}；GraceProfile=NONE；決策=REDUCE 25%"])],
            fills=[dict(code="SYNTHETIC_A", side="SELL", reason="EREM_REDUCE", shares=5)]))
    return result


def test_recorded_comparison_does_not_invent_new_account_returns_or_change_input():
    run = tape()
    before = deepcopy(run)
    result = compare_recorded_erem(run)
    assert run == before
    assert [row["research_decision"] for row in result["rows"]] == [
        "HOLD_REVIEW", "REDUCE 25%", "HOLD_REVIEW", "HOLD_REVIEW", "HOLD_REVIEW", "REDUCE 25%"]
    assert result["net_return_effect"] is None and not result["sample_out_of_sample_validated"]
    assert result["performance_status"].startswith("NOT_ASSESSABLE")
    assert result["policy"]["performance_validated"] is False
    assert all(row["counterfactual_fill"] == "NOT_COMPUTED" for row in result["observed_sell_diagnostics"])


def test_future_rows_cannot_change_earlier_decisions():
    run = tape()
    before = compare_recorded_erem(run)
    run["daily"] = run["daily"][:3]
    earlier = compare_recorded_erem(run)
    assert earlier["rows"] == before["rows"][:3]


@pytest.mark.parametrize("mutate", [
    lambda run: run["daily"][0].pop("openingSnapshot"),
    lambda run: run["daily"][0]["openingSnapshot"].update(cash=0),
    lambda run: run["daily"].reverse(),
    lambda run: run["daily"].append(deepcopy(run["daily"][-1])),
    lambda run: run["daily"][0]["decisions"].append(deepcopy(run["daily"][0]["decisions"][0])),
])
def test_missing_conflicting_or_duplicate_source_is_rejected(mutate):
    run = tape()
    mutate(run)
    with pytest.raises(ValueError, match="EREM_RESEARCH"):
        compare_recorded_erem(run)


def test_missing_grace_proof_is_unknown_not_assumed_false():
    run = tape()
    run["daily"][0]["decisions"][0]["eremReasons"] = ["HoldingDays=0"]
    result = compare_recorded_erem(run)
    assert result["rows"][0]["research_status"] == "GRACE_INPUT_UNKNOWN"
    assert result["rows"][1]["research_status"] == "SOFT_REDUCTION_PENDING_CONFIRMATION"


def test_external_risk_exit_remains_outside_soft_reduction_filter():
    run = tape()
    run["daily"][1]["fills"][0]["reason"] = "RISK_STOP"
    result = compare_recorded_erem(run)
    assert result["observed_sell_diagnostics"][1]["external_risk_exit_unchanged"] is True

"""SYNTHETIC ONLY: a valid non-entry report is not an executable H20 plan.

Exercise the actual daily/FR2/native-validation path, without source fetching,
model training, SMTP, Persistence or account writes. No plan is deleted from a
generated report to fabricate the primary missing-plan case.
"""
from copy import deepcopy
from datetime import date, timedelta
import json

import numpy as np
import pytest

from application.low_entry_v7_daily import build_daily_v7, daily_report, finalize_daily_report
from domain.les_v7.contracts import canonical, identity
from domain.les_v7.fr2_execution import assess_execution, validate_execution_assessment
from domain.les_v7.fr2_policy import restore_original
from domain.les_v7.report import validate_report
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from infrastructure.report.low_entry_v7_renderer import render_reports
from tests.test_v7_fr2_eps_daily import research_history
from tests.test_v7_fr2_execution import inputs, quote


def planless_history(*, account=True):
    history = research_history()
    # The unchanged synthetic model only permits REVERSAL. An uninterrupted
    # rising path fails its native reversal guard and thus naturally has no
    # selected context/reference plan; no model/policy or report is edited.
    close = np.linspace(90., 100., len(history)) * 1000.
    for column, offset in (("Close", 0.), ("Open", -.1), ("High", 2.),
                           ("Low", -2.), ("Bid", -.05), ("Ask", .05)):
        history[column] = close + offset * 1000.
    history["Amount"] = history.Close / 1000. * history.Volume
    if account:
        history.attrs["les_v6_portfolio"] = dict(value=1_000_000., available_cash=1_000_000.,
            intended_weight=.25, capacity_weight=.25, positions=[], as_of=history.attrs[SIGNAL_DATE_ATTR])
    return history


@pytest.fixture(scope="module")
def missing_plan_report():
    return daily_report(planless_history())


def assert_non_executable(assessment):
    assert assessment["state"] == "UNKNOWN"
    assert assessment["indicative_shares"] == assessment["filled_shares"] == 0
    assert assessment["allocation_authorized"] is False
    assert assessment["fill_eligible"] is False
    assert assessment["live_order_authorized"] is False
    assert assessment["maximum_total_weight"] is None
    assert "PLAN_OR_BOUND_COSTS_INVALID:ORIGINAL_H20_PLAN_MISSING" in assessment["reason_codes"]


def test_real_daily_non_entry_plan_absence_produces_valid_unknown_execution(missing_plan_report):
    report = missing_plan_report
    validate_report(report)
    original = restore_original(report)
    assert report["envelope"]["event"]["context"] == "NO_CLEAR_CONTEXT"
    assert report["diagnostics"]["reference_plan"] is None
    assert report["envelope"]["event"]["plan_id"] is None
    assert original["diagnostics"]["reference_plan"] is None
    assert original["envelope"]["event"]["plan_id"] is None
    assert report["envelope"]["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert_non_executable(report["fr2_execution"])
    assert "ACCOUNT_INPUTS_UNKNOWN" not in report["fr2_execution"]["reason_codes"]
    assert report["fr2_execution"]["quote_assessment"]["state"] == "NOT_PROVIDED"
    assert "pae_authorization" not in report


def next_weekday(day):
    following = date.fromisoformat(day) + timedelta(days=1)
    while following.weekday() > 4:
        following += timedelta(days=1)
    return following.isoformat()


@pytest.mark.parametrize("quote_kind", ["none", "source-bound-next-open", "flag-only"])
def test_public_execution_and_revalidation_cannot_fill_missing_plan(missing_plan_report, quote_kind):
    report = deepcopy(missing_plan_report)
    before = canonical(report)
    admission = report["diagnostics"]["fr2"]["admission"]
    account = report["fr2_execution"]["supplied_account"]
    supplied_quote = (None if quote_kind == "none" else dict(verified=True, price=100.)
        if quote_kind == "flag-only" else quote(report, session=next_weekday(report["signal_date"])))
    checked = assess_execution(report, admission, account, quote=supplied_quote)
    assert_non_executable(checked)
    validate_execution_assessment(report, admission, checked)
    if quote_kind == "source-bound-next-open":
        # A fully bound synthetic official open passes source/calendar checks
        # and specifically reaches the absent-plan branch, not a stale date.
        assert checked["quote_assessment"]["state"] == "UNKNOWN"
        assert checked["quote_assessment"]["reason_codes"] == [
            "QUOTE_EVIDENCE_UNKNOWN:ORIGINAL_H20_PLAN_MISSING"]
    assert canonical(report) == before
    report["fr2_execution"] = checked
    validate_report(report)


def test_missing_account_and_missing_plan_are_both_reported_without_exception():
    report = daily_report(planless_history(account=False))
    validate_report(report)
    assert_non_executable(report["fr2_execution"])
    assert "ACCOUNT_INPUTS_UNKNOWN" in report["fr2_execution"]["reason_codes"]


def test_finalization_retry_preserves_scores_and_rebuilds_unknown_assessment(missing_plan_report):
    report = deepcopy(missing_plan_report)
    before = deepcopy(report)
    account = report["fr2_execution"]["supplied_account"]
    actual = finalize_daily_report(report, account=account)
    assert actual == before
    assert actual["envelope"]["score"] == before["envelope"]["score"]
    assert_non_executable(actual["fr2_execution"])
    validate_report(actual)


@pytest.mark.parametrize("damage", ["invented-event-id", "unbound-plan"])
def test_malformed_plan_binding_is_not_downgraded_to_harmless_absence(missing_plan_report, damage):
    report = deepcopy(missing_plan_report)
    if damage == "invented-event-id":
        report["envelope"]["event"]["plan_id"] = "f" * 64
    else:
        report["diagnostics"]["reference_plan"] = deepcopy(inputs()[0]["diagnostics"]["reference_plan"])
    with pytest.raises(ValueError):
        assess_execution(report, report["diagnostics"]["fr2"]["admission"],
            report["fr2_execution"]["supplied_account"])


def test_rehashed_allocation_cannot_promote_planless_report(missing_plan_report):
    report = deepcopy(missing_plan_report)
    admission = report["diagnostics"]["fr2"]["admission"]
    assessment = report["fr2_execution"]
    assessment.update(state="PASS", indicative_shares=1, allocation_authorized=True)
    assessment["assessment_id"] = identity({key: value for key, value in assessment.items()
        if key != "assessment_id"})
    with pytest.raises(ValueError, match="FR2_EXECUTION_ASSESSMENT_BINDING_MISMATCH"):
        validate_execution_assessment(report, admission, assessment)
    with pytest.raises(ValueError):
        validate_report(report)


def test_planless_report_remains_renderable_without_changing_native_results(missing_plan_report):
    before = canonical(missing_plan_report)
    html = render_reports([missing_plan_report])
    assert html
    assert canonical(missing_plan_report) == before
    validate_report(missing_plan_report)


def test_daily_batch_continues_from_planless_to_plan_ready_history():
    # Two independent histories of one SYNTHETIC security isolate the public
    # list assembly boundary; they do not declare a duplicate real universe.
    missing, ready = planless_history(), research_history()
    assert missing.attrs["les_v7_bundle_json"] == ready.attrs["les_v7_bundle_json"]
    result = build_daily_v7([missing, ready])
    assert len(result) == 2
    reports = [json.loads(value) for value in result["LES v7 Report"]]
    assert reports[0]["diagnostics"]["reference_plan"] is None
    assert reports[1]["diagnostics"]["reference_plan"] is not None
    for report in reports:
        validate_report(report)
        assert report["fr2_execution"]["filled_shares"] == 0
        assert report["fr2_execution"]["live_order_authorized"] is False
    assert_non_executable(reports[0]["fr2_execution"])

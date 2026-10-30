"""SYNTHETIC diagnostic shapes; no market or source certification."""
from domain.les_v7.routing_r17 import _structure_failures, _structure_unknowns


def prepared(audit):
    return {"selection": {"short_base_audit": audit}}


def test_early_missing_source_without_check_rows_is_still_unknown():
    value = prepared(dict(state="UNKNOWN", checks={}, reasons=["SOURCE_CLOSE_OR_ATR_MISSING_OR_CONFLICTING"]))
    assert _structure_unknowns(value) == ["STRUCTURE_INPUTS_UNKNOWN"]
    assert _structure_failures(value) == []


def test_missing_calendar_and_nested_floor_are_not_a_known_unfinished_pattern():
    value = prepared(dict(state="UNKNOWN", checks=dict(FULL_PRIOR_120=None),
        session_coverage=dict(state="UNKNOWN"), support_audit=dict(state="UNKNOWN", checks={})))
    assert _structure_unknowns(value) == [
        "OFFICIAL_SESSION_COVERAGE", "STRUCTURE_FULL_PRIOR_120", "STRUCTURE_INPUTS_UNKNOWN"]


def test_known_missing_open_bar_is_a_source_failure_not_verified_or_unknown():
    value = prepared(dict(state="FAIL", checks=dict(FULL_PRIOR_120=False),
        session_coverage=dict(state="FAIL", missing=["2026-10-01"])))
    assert _structure_unknowns(value) == []
    assert _structure_failures(value) == ["OFFICIAL_SESSION_COVERAGE_FAIL"]


def test_known_weak_strength_is_not_a_source_failure():
    value = prepared(dict(state="PASS", checks=dict(FULL_PRIOR_120=True),
        soft_checks=dict(CLV=False, POSITIVE_RS20=False), session_coverage=dict(state="PASS")))
    assert _structure_unknowns(value) == []
    assert _structure_failures(value) == []

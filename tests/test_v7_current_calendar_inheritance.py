"""Synthetic capture-to-native regression, not a claim of real market buys.

Exercise the deployed selector and actual capture boundary. A prepopulated
calendar in a model fixture would hide the R20+ runtime collection regression.
"""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from application.low_entry_v7_daily import daily_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.contracts import timestamp
from domain.les_v7.policy_r23 import BASELINE_VERSION
from domain.les_v7.report import validate_report
from tests.test_native_source_runtime import capture_fixture, crawler_stub
from tests.test_v7_r23_pipeline import fixture
from tests.test_v7_r19_pipeline import current_market


def deployed_fixture():
    # The predecessor's test fixture is between R23 publication and daily
    # activation. Use a later explicit clock, never mock the deployed selector.
    with patch("tests.test_v7_r19_pipeline.AT", "2026-10-28T07:30:00Z"):
        return current_market(fixture())


@pytest.mark.parametrize("version", [f"V7_RULE_BASED_BASELINE_R{n}" for n in range(17, 24)])
def test_all_current_calendar_consumers_collect_once_and_share_436_selections(tmp_path, monkeypatch, version):
    import domain.les_v7.rule_baseline as rules
    value = deployed_fixture()
    cutoff = value.signal_dates["2330"]
    original = value.native_source_selections["2330"]["calendar_evidence"]
    collector = Mock(return_value=deepcopy(original))
    capture, *_ = capture_fixture(tmp_path)
    capture.clock = lambda: timestamp(cutoff)
    capture.calendar_collector = collector
    # Valid symbol syntax, deliberately absent from the synthetic master.
    symbols = ["2330", *[f"TEST{n:04d}" for n in range(435)]]
    crawler = crawler_stub(dates=value.dates[0], stocknumbers=symbols)
    monkeypatch.setattr(rules, "current_source_baseline_version", lambda _: version)
    capture.collect(crawler, market="TWSE")
    collector.assert_called_once()
    selections = capture.selections(crawler, cutoff=cutoff)
    assert set(selections) == set(symbols)
    calendars = [row["calendar_evidence"] for row in selections.values()]
    assert all(calendar is calendars[0] for calendar in calendars)
    with pytest.raises(TypeError, match="READ_ONLY"):
        calendars[0]["documents"]["2026"]["payload"]["stat"] = "tamper"


@pytest.mark.parametrize("damage", [None, "capture_failed", "missing_original", "future_original", "missing_restriction"])
def test_current_deployed_r23_uses_captured_calendar_and_keeps_missing_evidence_blocked(tmp_path, damage):
    value = deployed_fixture()
    expected_input = value
    cutoff = value.signal_dates["2330"]
    sources = deepcopy(value.native_source_selections)
    original = sources["2330"].pop("calendar_evidence")
    if damage == "missing_original":
        original["documents"] = {}
    elif damage == "future_original":
        original["documents"]["2026"]["publication_proof"]["published_at"] = "2030-01-01T00:00:00Z"
    collector = Mock(return_value=original)
    if damage == "capture_failed":
        collector.side_effect = ValueError("SYNTHETIC_CALENDAR_UNAVAILABLE")
    capture, *_ = capture_fixture(tmp_path)
    capture.clock = lambda: timestamp(cutoff)
    capture.calendar_collector = collector
    crawler = crawler_stub(dates=value.dates[0])
    capture.collect(crawler, market="TWSE")
    captured = capture.selections(crawler, cutoff=cutoff)["2330"]
    if "calendar_evidence" in captured:
        sources["2330"]["calendar_evidence"] = captured["calendar_evidence"]
    value = replace(value, native_source_selections=sources,
        trading_restrictions={} if damage == "missing_restriction" else value.trading_restrictions)
    history = MarketHistoryAssembler().build(capture_comparison_contexts(value), 0)
    report = daily_report(history)
    validate_report(report)
    d, e = report["diagnostics"], report["envelope"]
    assert d["rule_baseline"]["policy"]["policy_version"] == BASELINE_VERSION
    states = {row["gate_id"]: row["state"] for row in d["gate_receipts"]}
    if damage is None:
        assert e["strategy"]["strategy_decision"] == "BUY_SMALL"
        assert states["STRESS_RISK"] == "PASS"
        assert d["structure_plan"]["conditional_entry"]["state"] == "UNFILLED"
        # Restoring the real capture wire must reproduce exactly the report
        # from the identical prepopulated source fixture, not introduce a
        # different score, policy, account, plan, gate or sizing calculation.
        expected = daily_report(MarketHistoryAssembler().build(
            capture_comparison_contexts(expected_input), 0))
        assert report == expected
    else:
        assert e["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
        if damage == "missing_restriction":
            assert states["TRADING_STATUS"] == "UNKNOWN"
        else:
            assert states["STRESS_RISK"] == "UNKNOWN"
        assert not report["pae_authorization"]["authorized"]
    assert not report["pae_authorization"]["live_order_authorized"]
    collector.assert_called_once()

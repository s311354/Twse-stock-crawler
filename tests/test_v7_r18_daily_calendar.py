"""SYNTHETIC ONLY: real deployed selector and capture-to-daily calendar wiring.

No model/gate/plan/score/decision is mocked. Source transports use existing
synthetic fixtures; these tests do not certify real exchange evidence or fills.
"""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest

from application.low_entry_v7_daily import daily_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from domain.les_v7.contracts import timestamp
from domain.les_v7.policy_r18 import BASELINE_VERSION, SUPPORT_RESISTANCE_LIMIT
from domain.les_v7.report import validate_report
from tests.test_native_source_runtime import capture_fixture, crawler_stub
from tests.test_v7_r18_pipeline import neutral_market


@pytest.mark.parametrize("damage", [None, "capture_failed", "missing_original", "future_original"])
def test_deployed_r18_capture_reaches_plan_and_never_converts_unknown_to_pass(tmp_path, damage):
    value = neutral_market()
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
    # Use the actual runtime capture/projection, not the prepopulated fixture
    # which hid the missing R18 collection branch in earlier pipeline tests.
    if "calendar_evidence" in captured:
        sources["2330"]["calendar_evidence"] = captured["calendar_evidence"]
    value = replace(value, native_source_selections=sources)
    history = MarketHistoryAssembler().build(capture_comparison_contexts(value), 0)
    report = daily_report(history)
    validate_report(report)
    d, e = report["diagnostics"], report["envelope"]
    assert d["rule_baseline"]["policy"]["policy_version"] == BASELINE_VERSION
    states = {row["gate_id"]: row["state"] for row in d["gate_receipts"]}
    if damage is None:
        assert e["strategy"]["strategy_decision"] == "BUY_SMALL"
        assert states["STRESS_RISK"] == "PASS"
        assert d["reference_plan"]["setup_variant"] == SUPPORT_RESISTANCE_LIMIT
        assert d["structure_plan"]["conditional_entry"]["state"] == "UNFILLED"
        assert not report["pae_authorization"]["live_order_authorized"]
    else:
        assert e["strategy"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
        assert states["STRESS_RISK"] == "UNKNOWN"
        assert not report["pae_authorization"]["authorized"]
        assert "reference_entry_route" not in d
    collector.assert_called_once()

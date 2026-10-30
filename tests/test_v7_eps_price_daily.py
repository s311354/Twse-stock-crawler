"""SYNTHETIC ONLY: native daily source/method wiring; no actual market repair."""
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from application.low_entry_v7_daily import _native_source_packet, daily_report
from application.market_history import MarketHistoryAssembler
from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.les_v7.contracts import identity
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import BASELINE_POLICY, LEGACY_BASELINE_POLICY, R2_BASELINE_POLICY
from domain.les_v7.valuation_earnings import valuation_earnings
from domain.les_v7.valuation_eps_receipt import INDEPENDENT_PRODUCER
from domain.les_v7.valuation_input_policy import input_policy
from domain.les_v7.valuation_price import policy
from tests.strategies.test_daily_v7_rule_baseline import rule_market, CUTOFF
from tests.strategies.test_v7_rule_baseline import legal_bars
from tests.test_v7_eps_price_receipts import source


def packet_context():
    row = source()
    earnings = row.pop("earnings_source")
    return dict(symbol="SYNTHETIC", market="TWSE", as_of=CUTOFF, rule_baseline=True,
                filings=[row], valuation_earnings={row["content_sha256"]: earnings})


def test_current_rule_packet_keeps_exact_eps_and_comparisons_without_expanding_core_financial_graph():
    context = packet_context(); before = deepcopy(context)
    bars = legal_bars()
    history = bars.copy(); history.attrs["les_v6_analysis_session"] = "2026-10-14"
    packet = _native_source_packet(history, bars, context)
    row = packet["valuation_filings"][0]
    assert row["earnings_source"]["fields"]["diluted_eps"] == 12.
    assert row["earnings_source"]["comparative_ytd_fields"]["diluted_eps"] == 10.
    assert row["source_url"] == context["filings"][0]["source_url"]
    assert row["comparative_ytd_fields"]["revenue"] == 100000.
    assert "total_assets" not in row and "cash" not in row
    assert context == before


@pytest.mark.parametrize("version", [LEGACY_BASELINE_POLICY, R2_BASELINE_POLICY])
def test_old_rule_packet_does_not_receive_new_earnings_or_method_inputs(version):
    context = packet_context(); context["rule_baseline_policy_version"] = version
    bars = legal_bars(); history = bars.copy()
    packet = _native_source_packet(history, bars, context)
    row = packet["valuation_filings"][0]
    assert "earnings_source" not in row and "comparative_ytd_fields" not in row
    assert "source_url" not in row


def test_learned_packet_does_not_receive_unused_new_eps_inputs_or_changed_identity():
    context = packet_context(); context.pop("rule_baseline")
    bars = legal_bars(); history = bars.copy()
    before = identity(_native_source_packet(history, bars, context))
    context["valuation_earnings"] = {"UNUSED_SOURCEMAP": {"SYNTHETIC_FUTURE": True}}
    assert identity(_native_source_packet(history, bars, context)) == before


def rule_market_with_earnings():
    market = rule_market(); snapshot = deepcopy(market.fundamental_snapshots["2330"])
    additions = {}
    for row in snapshot["les_v6_mops_filings"]:
        period = pd.Period(row["period_end"], freq="Q")
        if period.quarter != 4: continue
        # Reuse this fixture's exact reported contexts, not a fabricated NI/EPS
        # denominator. These are synthetic originals, not authenticity proof.
        contexts = row["field_evidence"]["basic_eps"]["contexts"]
        facts = tuple(FiledFact(c["concept"], c["numeric_value"], None, c["start_date"], c["end_date"],
            c.get("instant"), unit=c["unit"], context_id=c.get("context_id"),
            dimensions=tuple(tuple(d) for d in c.get("dimensions", ()))) for c in contexts)
        # The older Quality fixture has quarter EPS but no full-year EPS.
        # Add an explicit SYNTHETIC annual disclosure for this port-only test;
        # real raw-byte/hash verification is tested separately in the repository.
        facts += tuple(FiledFact("ifrs-full:BasicEarningsLossPerShare", value, None,
            f"{year}-01-01", f"{year}-12-31", None, unit="iso4217:TWD/xbrli:shares",
            context_id="SYNTHETIC_ONLY_FULL_ANNUAL_EPS")
            for year, value in ((period.year, 2.4), (period.year - 1, 1.7)))
        filing = FiledFinancials(period.year, 4, row["available_at"], row["revision_number"], facts,
            content_sha256=row["content_sha256"], source_url=row["source_url"],
            period_end=row["period_end"], statement_scope=row["scope"])
        additions[row["content_sha256"]] = valuation_earnings(filing, symbol="2330", original_verified=True, as_of=CUTOFF)
    snapshot["les_v7_context"] = dict(valuation_earnings=additions)
    return replace(market, fundamental_snapshots={"2330": snapshot})


def test_native_daily_with_reported_eps_finishes_valuation_instead_of_demanding_unreported_shares():
    market = rule_market_with_earnings(); before = deepcopy(market.fundamental_snapshots)
    report = daily_report(MarketHistoryAssembler().build(market, 0))
    validate_report(report)
    baseline = report["diagnostics"]["rule_baseline"]
    native = report["diagnostics"]["native_evidence"]
    assert baseline["policy"]["policy_version"] == BASELINE_POLICY
    assert native["valuation_policy"] == policy()
    gate = native["produced_receipts"]["VALUATION"]
    assert gate["gate_version"] == INDEPENDENT_PRODUCER
    assert native["valuation_input_policy"] == input_policy()
    assert gate["state"] == "FAIL"  # genuine price-valuation failure, not permission to buy
    assert gate["valuation_diagnostics"]["scenarios"] is not None
    assert gate["valuation_diagnostics"]["inputs"]["eps_basis"] == "BASIC"
    assert not any("SHARES" in reason for reason in gate["reason_codes"])
    assert market.fundamental_snapshots == before
    assert report["pae_authorization"]["authorized"] is False


@pytest.mark.parametrize("damage", ["policy", "result"])
def test_native_report_cannot_rehash_a_different_method_or_fake_valuation_pass(damage):
    report = daily_report(MarketHistoryAssembler().build(rule_market_with_earnings(), 0))
    native = report["diagnostics"]["native_evidence"]
    if damage == "policy": native["valuation_policy"]["required_return"] = .01
    else:
        native["produced_receipts"]["VALUATION"]["state"] = "PASS"
        native["produced_receipts"]["VALUATION"]["valuation_diagnostics"]["status"] = "PASS"
    with pytest.raises(ValueError): validate_report(report)

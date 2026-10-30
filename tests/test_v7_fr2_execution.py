"""SYNTHETIC contract tests only; no market prediction, actual fill or account."""
from copy import deepcopy
from dataclasses import asdict
import hashlib

import pytest

from domain.les_v7.contracts import ReferencePlan, canonical, identity
from domain.les_v7.fr2_execution import (assess_execution, component, validate_execution_assessment)
from tests.test_research_archive import proof
from tests.test_v7_reference_path import fixture as path_fixture

CUTOFF = "2026-10-12T05:30:00+00:00"


def inputs(signal="BUY_SMALL", *, warning=False):
    cost = path_fixture()[2]
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC_H20_R1", "b" * 64, "REVERSAL",
        100., 98., 106., 2., (99.8, 100.5), 1, .5, .5)
    report = dict(symbol="SYNTHETIC_ONLY", signal_date="2026-10-12", envelope=dict(
        event=dict(security_id="SYNTHETIC_ONLY", cutoff_at=CUTOFF, evidence_id="a" * 64,
            plan_id=plan.plan_id, signal_id="c" * 64, context="REVERSAL"),
        score=dict(final_score=80.), strategy=dict(strategy_decision=signal)),
        diagnostics=dict(gate_receipts=[dict(gate_id=g, state="FAIL" if warning and g == "VALUATION" else "PASS")
            for g in ("SAFETY", "DATA_INTEGRITY", "TRADING_STATUS", "VALUATION", "STRESS_RISK")],
            quality=dict(status="PASS"), reference_plan=asdict(plan),
            gate_bindings=dict(SAFETY=dict(market="TWSE")),
            strategy_policy=dict(payload=dict(reference_policy=dict(cost_path_policy=cost))),
            features=dict(adv=100000000., spread=.001, close=100.)))
    admission = dict(version="FR_ENTRY_R2", decision=signal, buy_allowed=signal == "BUY", small_allowed=True,
        valuation_warning=warning, final_score=80.)
    admission["assessment_id"] = identity(admission)
    report["diagnostics"]["fr2"] = dict(admission=deepcopy(admission))
    account = dict(value=1000000., available_cash=1000000., intended_weight=.25, capacity_weight=.25,
        as_of=CUTOFF, positions=[])
    return report, admission, account


def assess(items, **kwargs):
    return assess_execution(*items, validate_native=False, **kwargs)


def position(symbol="OTHER", shares=100, *, cost=.5, stop=98.):
    risk = dict(schema="les-v7-fr2-position-risk/v1", symbol=symbol, market="TWSE", shares=shares,
        currency="TWD", price_unit="TWD", quantity_unit="shares",
        as_of=CUTOFF, source_available_at=CUTOFF, reviewed_by="SYNTHETIC_TEST_PROVIDER",
        source_evidence_ids=["SYNTHETIC_MARK_STOP_ORIGINAL"], cost_policy_id="d" * 64,
        mark_price=100., stop_price=stop, cost_stop_per_share=cost)
    return dict(symbol=symbol, shares=shares, risk_evidence=risk, risk_evidence_id=identity(risk))


def quote(report, *, price=100., session="2026-10-13"):
    payload = dict(symbol=report["symbol"], market="TWSE", session=session, phase="OFFICIAL_OPEN",
        price_unit="TWD", volume_unit="shares", price=price, bid=price - .05, ask=price + .05,
        observed_at=session + "T09:00:00+08:00")
    body = canonical(payload)
    binding = dict(kind="FR2_OFFICIAL_OPEN_QUOTE", symbol=report["symbol"], market="TWSE",
        session=session, content_sha256=identity(payload))
    publication = proof(binding, publication=session + "T09:00:00+08:00", precision="TIMESTAMP")
    publication.update(quotes=[body], verified_at=session + "T09:00:01+08:00")
    publication["document"].update(body_utf8=body, sha256=hashlib.sha256(body.encode()).hexdigest(),
        retrieved_at=session + "T09:00:01+08:00")
    annual = dict(stat="ok", fields=["日期", "名稱", "說明"], data=[["2026-01-01", "元旦休市", "休市"]])
    calendar = dict(source="TWSE_OFFICIAL", days={}, announcements=[], documents={"2026": dict(payload=annual,
        publication_proof=proof(dict(kind="ANNUAL_CALENDAR", market="TWSE", year=2026,
            content_sha256=identity(annual)), publication="2025-12-01"))})
    return dict(payload=payload, publication_proof=publication,
        checked_at=session + "T09:00:02+08:00", calendar_evidence=calendar)


def test_component_identity_is_stable_detached_and_includes_all_caps():
    value = component()
    value["total_symbol_caps"]["BUY"] = 1.
    assert component()["total_symbol_caps"]["BUY"] == .05
    assert component()["component_id"] == identity({k: v for k, v in component().items() if k != "component_id"})


@pytest.mark.parametrize("signal,warning,cap", [("BUY", False, .05), ("BUY_SMALL", False, .01), ("BUY_SMALL", True, .005)])
def test_total_symbol_caps_are_direct_and_do_not_quarter_small_again(signal, warning, cap):
    items = inputs(signal, warning=warning)
    original = deepcopy(items)
    result = assess(items)
    assert result["state"] == "PASS" and result["allocation_authorized"]
    assert result["maximum_total_weight"] == cap
    assert result["planned_cash_debit"] <= cap * items[2]["value"]
    assert result["indicative_shares"] >= int(cap * 1000000 / 100.5) - 1
    assert result["planned_stop_loss"] <= (2500 if signal == "BUY" else 500)
    assert result["filled_shares"] == 0 and not result["fill_eligible"] and not result["live_order_authorized"]
    validate_execution_assessment(*items[:2], result, validate_native=False)
    assert items == original  # Includes the original model score and plan.


def test_existing_same_symbol_consumes_total_cap_without_changing_score():
    items = inputs()
    items[2]["positions"] = [position("SYNTHETIC_ONLY", 80)]
    items[2]["available_cash"] = 992000.
    result = assess(items)
    assert result["indicative_shares"] == 19
    assert result["current_weight"] == .008
    assert result["incremental_weight_capacity"] == .002
    assert items[0]["envelope"]["score"]["final_score"] == 80.


def test_portfolio_risk_budget_includes_all_positions_not_only_candidate():
    items = inputs("BUY")
    items[2]["positions"] = [position(shares=3600)]  # 9000 planned loss.
    items[2]["available_cash"] = 640000.
    result = assess(items)
    assert result["existing_portfolio_risk_fraction"] == .009
    assert result["planned_stop_loss"] <= 1000.
    assert result["indicative_shares"] < assess(inputs("BUY"))["indicative_shares"]
    items[2]["positions"] = [position(shares=4000)]
    items[2]["available_cash"] = 600000.
    assert assess(items)["indicative_shares"] == 0


@pytest.mark.parametrize("damage", ["account", "cash", "holdings", "risk", "cost", "stop", "source", "future", "cutoff", "hash", "shares"])
def test_missing_or_invalid_account_risk_evidence_is_unknown_zero_allocation(damage):
    report, admission, account = inputs()
    account["positions"] = [position()]
    account["available_cash"] = 990000.
    if damage == "account": account = None
    elif damage == "cash": account.pop("available_cash")
    elif damage == "holdings": account.pop("positions")
    elif damage == "risk": account["positions"][0].pop("risk_evidence")
    else:
        row = account["positions"][0]
        if damage == "cost": row["risk_evidence"].pop("cost_policy_id")
        elif damage == "stop": row["risk_evidence"]["stop_price"] = 100.
        elif damage == "source": row["risk_evidence"]["source_evidence_ids"] = []
        elif damage == "future": row["risk_evidence"]["source_available_at"] = "2026-10-13T00:00:00Z"
        elif damage == "cutoff": row["risk_evidence"]["as_of"] = "2026-10-11T05:30:00Z"
        elif damage == "shares": row["shares"] = 1.5
        if damage != "hash": row["risk_evidence_id"] = identity(row["risk_evidence"])
        else: row["risk_evidence_id"] = "0" * 64
    result = assess((report, admission, account))
    assert result["state"] == "UNKNOWN" and result["indicative_shares"] == 0
    assert admission["decision"] == "BUY_SMALL"
    validate_execution_assessment(report, admission, result, validate_native=False)


@pytest.mark.parametrize("field,value", [("value", True), ("available_cash", float("nan")), ("capacity_weight", float("inf"))])
def test_nonfinite_or_bool_account_never_authorizes(field, value):
    items = inputs()
    items[2][field] = value
    # NaN cannot be retained as canonical JSON; malformed JSON inputs reject.
    if isinstance(value, float):
        with pytest.raises(ValueError): assess(items)
    else:
        assert not assess(items)["allocation_authorized"]


@pytest.mark.parametrize("gate,state", [("SAFETY", "FAIL"), ("TRADING_STATUS", "UNKNOWN"), ("STRESS_RISK", "FAIL"), ("VALUATION", "UNKNOWN")])
def test_hard_risk_unknown_restriction_and_invalid_stop_are_not_compensated(gate, state):
    items = inputs()
    next(g for g in items[0]["diagnostics"]["gate_receipts"] if g["gate_id"] == gate)["state"] = state
    assert assess(items)["indicative_shares"] == 0


def test_minimum_fees_use_actual_quantity_and_cash_does_not_assume_zero_costs():
    items = inputs("BUY")
    policy = items[0]["diagnostics"]["strategy_policy"]["payload"]["reference_policy"]["cost_path_policy"]
    policy["cost_policy"].update(minimum_buy_fee="20", minimum_sell_fee="20")
    items[2].update(available_cash=1010., value=101010., positions=[position(shares=1000, stop=99.99, cost=0.)])
    result = assess(items)
    assert result["indicative_shares"] == 9
    assert result["planned_cash_debit"] == pytest.approx(924.5)


def test_bound_assessment_detects_account_plan_gate_and_output_tampering():
    items = inputs()
    result = assess(items)
    for key in ("indicative_shares", "planned_stop_loss", "supplied_account", "component_id"):
        changed = deepcopy(result)
        changed[key] = {} if key == "supplied_account" else 999
        with pytest.raises(ValueError, match="BINDING_MISMATCH"):
            validate_execution_assessment(*items[:2], changed, validate_native=False)
    for key in ("plan", "gate"):
        changed_report = deepcopy(items[0])
        if key == "plan": changed_report["diagnostics"]["reference_plan"]["stop"] -= .5
        else: changed_report["diagnostics"]["gate_receipts"][0]["state"] = "UNKNOWN"
        with pytest.raises(ValueError, match="BINDING_MISMATCH"):
            validate_execution_assessment(changed_report, items[1], result, validate_native=False)


def test_public_entry_validates_native_before_derivation(monkeypatch):
    calls = []
    monkeypatch.setattr("domain.les_v7.report.validate_report", lambda report: calls.append(report))
    result = assess_execution(*inputs())
    validate_execution_assessment(*inputs()[:2], result)
    assert len(calls) == 2


def test_next_official_open_is_source_bound_and_no_fill_is_claimed():
    items = inputs()
    original = deepcopy(items)
    result = assess(items, quote=quote(items[0]))
    assert result["quote_assessment"]["state"] == "PASS"
    assert result["fill_eligible"] and result["filled_shares"] == 0
    assert result["live_order_authorized"] is False
    validate_execution_assessment(*items[:2], result, validate_native=False)
    assert items == original


@pytest.mark.parametrize("damage", ["out_of_band", "later_session", "intraday", "missing_calendar", "hash", "future_source", "flag_only", "unknown_account", "wrong_market_quote", "wrong_market_calendar"])
def test_later_touch_stale_missing_or_outside_price_cannot_fill(damage):
    items = inputs()
    value = quote(items[0], price=101. if damage == "out_of_band" else 100.,
        session="2026-10-14" if damage == "later_session" else "2026-10-13")
    if damage == "intraday": value["payload"]["phase"] = "INTRADAY_TOUCH"
    elif damage == "missing_calendar": value["calendar_evidence"]["documents"] = {}
    elif damage == "hash": value["publication_proof"]["document"]["sha256"] = "0" * 64
    elif damage == "future_source": value["publication_proof"]["document"]["retrieved_at"] = "2026-10-14T00:00:00Z"
    elif damage == "flag_only": value = dict(verified=True, price=100.)
    elif damage == "unknown_account": items = (*items[:2], None)
    elif damage == "wrong_market_quote": value["publication_proof"]["document"]["url"] = "https://www.tpex.org.tw/test-only"
    elif damage == "wrong_market_calendar": value["calendar_evidence"]["documents"]["2026"]["publication_proof"]["document"]["url"] = "https://www.tpex.org.tw/test-only"
    result = assess(items, quote=value)
    assert not result["fill_eligible"] and result["filled_shares"] == 0
    assert result["quote_assessment"]["state"] in {"UNKNOWN", "FAIL"}


def test_same_symbol_different_account_does_not_share_authorization_identity():
    items = inputs()
    first = assess(items)
    items[2].update(value=1000., available_cash=1000.)
    second = assess(items)
    assert first["assessment_id"] != second["assessment_id"]
    assert first["account_id"] != second["account_id"]
    assert first["indicative_shares"] > second["indicative_shares"]


def test_off_grid_band_budgets_inside_original_band_without_rewriting_plan():
    items = inputs()
    original_plan = items[0]["diagnostics"]["reference_plan"]
    original_plan["band"] = [99.8, 100.2]
    from domain.les_v7.plan_r18 import plan_from_record
    items[0]["envelope"]["event"]["plan_id"] = plan_from_record(original_plan).plan_id
    result = assess(items)
    assert result["sizing_entry_limit"] == 100.
    assert original_plan["band"] == [99.8, 100.2]


def test_source_bound_cost_policy_must_cover_signal_session():
    items = inputs()
    items[0]["diagnostics"]["strategy_policy"]["payload"]["reference_policy"]["cost_path_policy"]["cost_policy"]["effective_end"] = "2026-09-30"
    result = assess(items)
    assert result["state"] == "UNKNOWN" and not result["allocation_authorized"]


def test_actual_open_revalidates_original_rr_at_actual_size():
    items = inputs()
    plan = items[0]["diagnostics"]["reference_plan"]
    plan.update(target=104.)
    from domain.les_v7.plan_r18 import plan_from_record
    items[0]["envelope"]["event"]["plan_id"] = plan_from_record(plan).plan_id
    result = assess(items, quote=quote(items[0]))
    assert result["allocation_authorized"]  # Source/native plan assessed independently.
    assert not result["fill_eligible"]
    assert result["quote_assessment"]["reason_codes"] == ["OFFICIAL_OPEN_RISK_OR_NET_RR_FAILED"]


def test_no_entry_qualification_can_be_invented_by_tampering_admission():
    items = inputs()
    items[1]["decision"] = "BUY"
    with pytest.raises(ValueError, match="ADMISSION_REQUIRED"):
        assess(items)


@pytest.mark.parametrize("damage", ["decision", "cutoff", "evidence", "model", "quality"])
def test_newly_hashed_admission_cannot_authorize_different_report_or_sources(damage):
    items = inputs()
    changed = deepcopy(items[1])
    if damage == "decision": changed.update(decision="BUY", buy_allowed=True)
    elif damage == "cutoff": changed["cutoff"] = "2026-10-11T05:30:00Z"
    elif damage == "evidence": changed["evidence_id"] = "0" * 64
    elif damage == "model": changed["bundle_sha256"] = "0" * 64
    else: changed["quality_result_id"] = "0" * 64
    changed["assessment_id"] = identity({k: v for k, v in changed.items() if k != "assessment_id"})
    if damage != "decision":
        items[0]["diagnostics"]["fr2"]["admission"] = deepcopy(changed)
    with pytest.raises(ValueError, match="ADMISSION_BINDING_MISMATCH"):
        assess((items[0], changed, items[2]))


def test_armed_report_cannot_be_paired_with_even_correctly_hashed_buy_admission():
    items = inputs()
    items[0]["envelope"]["strategy"]["strategy_decision"] = "ARMED"
    with pytest.raises(ValueError, match="ADMISSION_BINDING_MISMATCH"):
        assess(items)


@pytest.mark.parametrize("with_holdings", [False, True])
def test_unexplained_nav_cannot_enlarge_account_risk_budget(with_holdings):
    items = inputs()
    if with_holdings:
        items[2]["positions"] = [position(shares=100)]
        items[2]["available_cash"] = 990000.
    items[2]["value"] += 50000.
    result = assess(items)
    assert result["state"] == "UNKNOWN" and result["indicative_shares"] == 0
    assert any("NAV_CASH_HOLDING_MARKS_MISMATCH" in reason for reason in result["reason_codes"])


def test_four_positions_cannot_allocate_a_fifth_symbol_but_existing_name_can_add():
    items = inputs()
    items[2]["positions"] = [position("OTHER" + str(i), shares=10) for i in range(4)]
    items[2]["available_cash"] = 996000.
    result = assess(items)
    assert result["state"] == "FAIL" and result["indicative_shares"] == 0
    assert "PORTFOLIO_FOUR_POSITIONS_NO_NEW_SYMBOL_SLOT" in result["reason_codes"]
    items[2]["positions"][0] = position("SYNTHETIC_ONLY", shares=10)
    assert assess(items)["indicative_shares"] > 0

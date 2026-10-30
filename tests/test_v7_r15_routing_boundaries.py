"""SYNTHETIC boundary stubs, NOT native reports or official-source evidence.

These isolate routing and already-validated allocation arithmetic. Upstream
source/quality/native recomputation is covered by test_v7_r15_pipeline.
"""
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, ROUND_FLOOR
from types import SimpleNamespace

import pytest
import domain.strategies  # noqa: F401 -- initialize the existing plugin graph

from domain.les_v7.allocation import _authorize_validated
from domain.les_v7.contracts import FACTORS, ReferencePlan, identity
from domain.les_v7.plan_r10 import evaluate_entry
from domain.les_v7.policy_r15 import AVAILABLE_AT, BASELINE_VERSION
from domain.les_v7.reference_path import estimate_reference_costs
from domain.les_v7.routing_r15 import score_routes
from domain.les_v7.rule_baseline import CONFIG, baseline_policy
from domain.les_v7.valuation_recovery_r15 import evaluate_recovery_price_valuation
from tests.strategies.test_v7_entry_policy_r13 import gates, quality
from tests.strategies.test_v7_entry_policy_r15 import admission, valuation
from tests.strategies.test_v7_rule_baseline import unit_features
from tests.test_v7_baseline_allocation import account, boundary_report


def ledger(raw):
    budgets = dict(structure=30., trend=10., relative_strength=20., demand=25., momentum=5., execution=10.)
    return dict(conditions=[dict(id=name, group=name, observed=True, weight=weight,
            fraction=raw / 100., earned_points=weight * raw / 100.) for name, weight in budgets.items()],
        components={name: dict(total_weight=weight, earned_points=weight * raw / 100.)
            for name, weight in budgets.items()})


def plan(*, stop=98., target=109., atr=2.):
    p = baseline_policy(version=BASELINE_VERSION)["reference_policy"]
    costs = estimate_reference_costs(100., stop, target, p["reference_shares"], p["cost_path_policy"])
    return ReferencePlan("REFERENCE_H20", p["version"], "a" * 64, "REVERSAL",
        100., stop, target, atr, (100., 100.), p["reference_shares"],
        costs["cost_up_per_share"], costs["cost_stop_per_share"])


def stub_admission(*, state="ALLOWED", core_points=10., tactical_points=3.):
    """Routing-only contract double, not a source-validated valuation verdict."""
    return dict(state=state, quality=dict(evidence_ready=True), core_points=core_points,
        tactical_points=tactical_points,
        core_buy_allowed=state == "ALLOWED", tactical_small_allowed=state == "ALLOWED",
        core_grade=dict(state="ALLOWED", points=core_points),
        tactical_grade=dict(state="ALLOWED", price_points=tactical_points,
            bear_points=0., points=tactical_points, reason="SYNTHETIC_ROUTING_STUB"))


def route(monkeypatch, *, raw=85., p=None, states=None, risk=(), stub=None, confirmed=True, variant=None):
    policy = baseline_policy(version=BASELINE_VERSION)
    features = unit_features(); features.update(close=100., low=99., support=98.)
    r = valuation(ratio=.7)
    original = r["valuation_diagnostics"]
    diagnostic = evaluate_recovery_price_valuation(legacy_result=dict(inputs=dict(original["inputs"], current_price=100.)),
        comparables=original["comparables"], recovery=original["recovery"])
    r.update(state=diagnostic["status"], valuation_diagnostics=diagnostic)
    prepared = dict(policy=policy, selection=dict(setup_variant=variant))
    if stub is not None:
        monkeypatch.setattr("domain.les_v7.entry_policy_r15.assess", lambda *args, **kwargs: deepcopy(stub))
    return score_routes(prepared, selection=SimpleNamespace(context="REVERSAL"), features=features,
        scored=dict(score_evidence=ledger(raw), factor_budgets={name: 1. for name in FACTORS}),
        trigger=dict(starter=True, confirmed=confirmed), plan=p or plan(), quality=quality(),
        gates=gates(r, **(states or {})), native_features=dict(close=100., support=98., spread=.001, adv=1e8),
        risks=list(risk), starter_gate=True, confirmed_gate=True,
        valuation_receipt=r, entry_binding=dict(symbol="SYNTHETIC", cutoff=AVAILABLE_AT), config=CONFIG)


def test_core_buy_publishes_core_deduction_not_the_higher_tactical_score(monkeypatch):
    result = route(monkeypatch, stub=stub_admission(), raw=85.)
    routes = result["r15_route_evaluation"]
    assert routes["core"]["score"]["final_score"] == 75.
    assert routes["tactical"]["score"]["final_score"] == 82.
    assert result["decision"] == "BUY" and routes["selected_route"] == "CORE"
    assert result["score"]["final_score"] == 75.


def test_tactical_score_above_72_cannot_upgrade_core_below_72(monkeypatch):
    result = route(monkeypatch, stub=stub_admission(), raw=78.)
    routes = result["r15_route_evaluation"]
    assert routes["core"]["score"]["final_score"] == 68.
    assert routes["tactical"]["score"]["final_score"] == 75.
    assert result["decision"] == "BUY_SMALL" and routes["selected_route"] == "TACTICAL"
    assert not routes["core"]["buy_eligible"]


@pytest.mark.parametrize("stop,target", [(96., 107.), (98., 104.)])
def test_tactical_two_atr_or_13_rr_never_becomes_core_buy(monkeypatch, stop, target):
    p = plan(stop=stop, target=target)
    reference = baseline_policy(version=BASELINE_VERSION)["reference_policy"]
    assert evaluate_entry(p, 100., reference)["state"] == "FAIL"
    assert evaluate_entry(p, 100., reference, starter=True)["state"] == "PASS"
    result = route(monkeypatch, p=p, stub=stub_admission(core_points=0., tactical_points=0.), raw=100.)
    assert result["decision"] == "BUY_SMALL"
    assert not result["r15_route_evaluation"]["core"]["buy_eligible"]


@pytest.mark.parametrize("stop,target", [(95.99, 110.), (98., 103.5)])
def test_tactical_risk_failure_cannot_be_bought_even_with_100_points(monkeypatch, stop, target):
    result = route(monkeypatch, p=plan(stop=stop, target=target),
        stub=stub_admission(core_points=0., tactical_points=0.), raw=100.)
    assert result["decision"] not in {"BUY", "BUY_SMALL"}
    assert not result["r15_route_evaluation"]["tactical"]["small_eligible"]


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "STRESS_RISK"])
@pytest.mark.parametrize("state", ["UNKNOWN", "FAIL"])
def test_real_admission_hard_unknown_or_fail_blocks_both_routes(monkeypatch, gate, state):
    result = route(monkeypatch, raw=100., states={gate: state}, risk=[gate + "_FAIL"] if state == "FAIL" else [])
    assert result["decision"] not in {"BUY", "BUY_SMALL"}
    routes = result["r15_route_evaluation"]
    assert not routes["core"]["buy_eligible"] and not routes["tactical"]["small_eligible"]


def allocation_report():
    report = boundary_report("BUY_SMALL")
    report["envelope"]["event"]["security_id"] = "SYNTHETIC"
    report["envelope"]["event"]["cutoff_at"] = AVAILABLE_AT
    report["diagnostics"]["rule_baseline"]["policy"] = baseline_policy(version=BASELINE_VERSION)
    # Deliberate boundary stub: this source support is not certified. Large
    # absolute per-share risk makes held-risk consumption bind before weight.
    p = plan(stop=50., target=180., atr=30.)
    report["diagnostics"]["reference_plan"] = asdict(p)
    report["diagnostics"]["entry_admission"] = admission(valuation(ratio=2.))
    report["diagnostics"]["reference_entry_route"] = dict(route_id="SYNTHETIC_BOUNDARY_ONLY")
    return report


def explicit_account(**changes):
    return dict(account(), as_of=AVAILABLE_AT, **changes)


def test_total_caps_and_half_tenth_percent_risk_deduct_existing_holdings_once():
    report = allocation_report()
    empty = _authorize_validated(report, explicit_account())
    held = _authorize_validated(report, explicit_account(positions=[dict(symbol="SYNTHETIC", shares=5)]))
    exhausted = _authorize_validated(report, explicit_account(positions=[dict(symbol="SYNTHETIC", shares=10)]))
    loss = Decimal(str(empty["bound_plan_loss_per_share"]))
    expected = int((Decimal("500") / loss).to_integral_value(rounding=ROUND_FLOOR))
    assert empty["small_risk_fraction"] == .0005 and empty["indicative_shares"] == expected
    assert held["indicative_shares"] == expected - 5
    assert (Decimal(held["indicative_shares"] + 5) * loss) <= Decimal("500")
    assert exhausted["current_weight"] < exhausted["maximum_total_weight"]
    assert exhausted["indicative_shares"] == 0 and not exhausted["authorized"]
    assert not empty["live_order_authorized"]


def test_pae_capacity_and_cash_limit_size_but_do_not_change_score_or_admission():
    report = allocation_report(); before = deepcopy(report)
    limited = _authorize_validated(report, explicit_account(intended_weight=.0004, capacity_weight=.0002))
    assert limited["maximum_total_weight"] == .0002
    cashless = _authorize_validated(report, explicit_account(available_cash=0.))
    assert not cashless["authorized"] and cashless["indicative_shares"] == 0
    assert report == before
    assert report["diagnostics"]["entry_admission"]["assessment_id"] == identity({
        k: v for k, v in report["diagnostics"]["entry_admission"].items() if k != "assessment_id"})

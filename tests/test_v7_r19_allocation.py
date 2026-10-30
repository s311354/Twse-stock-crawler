"""Synthetic already-validated quota boundary; not an official research report."""
from copy import deepcopy
from decimal import Decimal

import pytest

from domain.les_v7.allocation import _authorize_validated, validate_authorization
from domain.les_v7.contracts import identity
from domain.les_v7.entry_policy_r18 import assess
from domain.les_v7.policy_r18 import SUPPORT_RESISTANCE_LIMIT
from domain.les_v7.policy_r19 import AVAILABLE_AT, BASELINE_VERSION
from domain.les_v7.rule_baseline import baseline_policy
from tests.test_v7_r16_allocation import report as previous_report
from tests.test_v7_r15_routing_boundaries import explicit_account
from tests.strategies.test_v7_entry_policy_r13 import gates, quality
from tests.strategies.test_v7_entry_policy_r15 import valuation


def report():
    result = previous_report()
    result["envelope"]["event"]["cutoff_at"] = AVAILABLE_AT
    d = result["diagnostics"]
    policy = baseline_policy(version=BASELINE_VERSION)
    d["rule_baseline"] = dict(policy=policy, policy_id=identity(policy))
    d["reference_plan"].update(entry=100., stop=98., target=140., band=[99., 100.],
        signal_close=100., entry_basis="STRUCTURAL_CONDITIONAL_BAND")
    value = valuation()
    d["entry_admission"] = assess(quality(), gates(value), value,
        symbol="SYNTHETIC", cutoff=AVAILABLE_AT,
        current_price=value["valuation_diagnostics"]["inputs"]["current_price"],
        setup_variant=SUPPORT_RESISTANCE_LIMIT)
    return result


def account(**values):
    return dict(explicit_account(**values), as_of=AVAILABLE_AT)


def test_higher_band_limit_reprices_costs_and_reduces_shares_under_same_loss_budget():
    original = report()
    old = _authorize_validated(original, account())
    widened = deepcopy(original)
    widened["diagnostics"]["reference_plan"].update(band=[99., 102.], entry=102.)
    before = deepcopy(widened)
    new = _authorize_validated(widened, account())
    assert 0 < new["indicative_shares"] < old["indicative_shares"]
    assert new["sizing_entry_limit"] == 102.
    assert new["bound_plan_loss_per_share"] > old["bound_plan_loss_per_share"]
    assert new["small_risk_fraction"] == old["small_risk_fraction"] == .0005
    assert Decimal(str(new["bound_plan_loss_per_share"])) * new["indicative_shares"] <= Decimal("500")
    assert new["policy_version"] == "LES_V7_PAE_SIGNAL_AUTHORIZATION_R8"
    assert new["risk_basis"] == "R19_REPRICED_BOUND_COSTS_AT_ENTRY_BAND_LIMIT"
    assert new["maximum_total_weight"] <= .01
    assert not new["live_order_authorized"]
    assert widened == before
    validate_authorization(dict(widened, pae_authorization=new))


def test_r19_total_risk_includes_existing_holdings_at_signal_close():
    r = report()
    r["diagnostics"]["reference_plan"]["signal_close"] = 120.
    quota = _authorize_validated(r, account(positions=[dict(symbol="SYNTHETIC", shares=23)]))
    assert quota["current_weight"] < quota["maximum_total_weight"]
    assert Decimal(str(quota["bound_held_loss_per_share"])) * 23 > Decimal("500")
    assert quota["indicative_shares"] == 0 and not quota["authorized"]


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "STRESS_RISK"])
@pytest.mark.parametrize("state", ["FAIL", "UNKNOWN"])
def test_wider_band_does_not_compensate_for_hard_gate_failure_or_unknown(gate, state):
    r = report()
    next(row for row in r["diagnostics"]["gate_receipts"] if row["gate_id"] == gate)["state"] = state
    quota = _authorize_validated(r, account())
    assert quota["indicative_shares"] == 0 and not quota["authorized"]
    assert "V7_HARD_GATES_NOT_PASS" in quota["reason_codes"]

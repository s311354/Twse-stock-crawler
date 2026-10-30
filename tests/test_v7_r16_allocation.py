"""SYNTHETIC arithmetic boundary only; not a source-certified report."""
from copy import deepcopy
from decimal import Decimal

from domain.les_v7.allocation import _authorize_validated
from domain.les_v7.contracts import identity
from domain.les_v7.entry_policy_r16 import assess
from domain.les_v7.policy_r16 import AVAILABLE_AT, BASELINE_VERSION, EARLY_PULLBACK
from domain.les_v7.rule_baseline import baseline_policy
from tests.test_v7_r15_routing_boundaries import allocation_report, explicit_account
from tests.strategies.test_v7_entry_policy_r13 import gates, quality
from tests.strategies.test_v7_entry_policy_r15 import valuation


def report():
    r = allocation_report()
    r["envelope"]["event"]["cutoff_at"] = AVAILABLE_AT
    d = r["diagnostics"]
    d["rule_baseline"]["policy"] = baseline_policy(version=BASELINE_VERSION)
    # Deliberate boundary double: 120 mark vs 100 conditional price. Source
    # validation happens upstream; here exercise already-validated math only.
    d["reference_plan"].update(entry=100., stop=98., target=140., band=[99., 100.],
        signal_close=120., entry_basis="STRUCTURAL_CONDITIONAL_BAND")
    v = valuation()
    d["entry_admission"] = assess(quality(), gates(v), v, symbol="SYNTHETIC",
        cutoff=AVAILABLE_AT, current_price=v["valuation_diagnostics"]["inputs"]["current_price"],
        setup_variant=EARLY_PULLBACK)
    return r


def account(**values):
    return dict(explicit_account(**values), as_of=AVAILABLE_AT)


def test_conditional_discount_never_exceeds_signal_close_total_exposure_cap():
    r = report(); original = deepcopy(r)
    a = _authorize_validated(r, account())
    assert a["indicative_shares"] > 0
    assert a["indicative_shares"] * 120. / 1e6 <= a["maximum_total_weight"]
    assert a["exposure_price"] == 120.
    assert a["bound_held_loss_per_share"] > a["bound_plan_loss_per_share"]
    assert r == original and not a["live_order_authorized"]


def test_existing_mark_to_stop_risk_not_understated_by_cheaper_conditional_price():
    r = report()
    a = _authorize_validated(r, account(positions=[dict(symbol="SYNTHETIC", shares=23)]))
    assert a["current_weight"] < a["maximum_total_weight"]
    # Twenty-three existing shares exhaust the .05% planned NAV loss, even
    # though exposure cap alone still has room for additional shares.
    assert Decimal(str(a["bound_held_loss_per_share"])) * 23 > Decimal("500")
    assert a["indicative_shares"] == 0 and not a["authorized"]
    assert r["diagnostics"]["entry_admission"]["assessment_id"] == identity({
        k: v for k, v in r["diagnostics"]["entry_admission"].items() if k != "assessment_id"})

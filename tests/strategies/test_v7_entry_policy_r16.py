"""SYNTHETIC ONLY: R16 admission caps, not official-source certification."""
import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.entry_policy_r16 import assess, policy
from domain.les_v7.policy_r16 import AVAILABLE_AT, EARLY_PULLBACK, PRICE_DISCOVERY
from tests.strategies.test_v7_entry_policy_r13 import gates, quality
from tests.strategies.test_v7_entry_policy_r15 import valuation, admission as previous


def evaluate(variant, *, ratio=.7, states=None):
    r = valuation(ratio=ratio)
    return assess(quality(), gates(r, **(states or {})), r, symbol="SYNTHETIC",
        cutoff=AVAILABLE_AT, current_price=r["valuation_diagnostics"]["inputs"]["current_price"],
        setup_variant=variant)


@pytest.mark.parametrize("variant,cap,risk", [(EARLY_PULLBACK, .005, .0005),
                                             (PRICE_DISCOVERY, .0025, .00025)])
def test_new_starters_are_small_only_with_absolute_total_caps(variant, cap, risk):
    result = evaluate(variant)
    assert result["small_allowed"] and not result["buy_allowed"]
    assert result["tactical_total_cap"] == result["small_total_cap"] == cap
    assert result["tactical_risk_fraction"] == risk
    assert result["policy_id"] == identity(policy())
    assert result["assessment_id"] == identity({k: v for k, v in result.items() if k != "assessment_id"})
    assert result["position_limits_affect_score"] is False


@pytest.mark.parametrize("variant", [EARLY_PULLBACK, PRICE_DISCOVERY])
def test_valuation_deductions_unchanged_and_stricter_total_caps_win(variant):
    old = previous(valuation(ratio=2.))
    new = evaluate(variant, ratio=2.)
    assert new["tactical_points"] == old["tactical_points"] == 12.
    assert new["small_total_cap"] == min(old["small_total_cap"], new["route_total_cap"])
    assert new["valuation"]["native_state"] == "FAIL"


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "STRESS_RISK"])
@pytest.mark.parametrize("state", ["UNKNOWN", "FAIL"])
@pytest.mark.parametrize("variant", [EARLY_PULLBACK, PRICE_DISCOVERY])
def test_new_structure_never_compensates_unknown_or_failed_safety(variant, gate, state):
    result = evaluate(variant, states={gate: state})
    assert not result["buy_allowed"] and not result["small_allowed"]
    assert result["state"] == ("UNKNOWN" if state == "UNKNOWN" else "BLOCKED")


def test_unrecognized_variant_fails_closed():
    with pytest.raises(ValueError, match="R16_UNKNOWN_SETUP_VARIANT"):
        evaluate("FAKE_HIGH_SCORE")

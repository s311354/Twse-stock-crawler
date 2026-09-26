"""Scenario payoffs, not forecasts or calibrated probabilities; no market I/O."""

from collections.abc import Mapping

from domain.strategies.low_entry_v6_quality import number


def payoff_scenarios(plan: Mapping, *, buy_cost_rate: float, sell_cost_rate: float) -> dict:
    """Use only a frozen plan and explicit cost assumptions.

    Target/stop are conditional outcomes, not promises of fills. A two-outcome
    break-even probability is a required success rate, not an estimated one.
    Missing targets (including price discovery) must not be replaced with 2R or
    a future high. This evidence cannot grant entry or override any gate.
    """
    buy, sell = number(buy_cost_rate), number(sell_cost_rate)
    if buy is None or sell is None or not (0 <= buy < 1 and 0 <= sell < 1):
        raise ValueError("INVALID_PAYOFF_COST_ASSUMPTION")
    entry, stop, target = (number(plan.get(key)) for key in
                           ("entry_price", "stop_price", "target_price"))
    result = {
        "basis": "CONDITIONAL_PRICE_SCENARIOS_NOT_FORECAST",
        "status": "UNAVAILABLE", "entry_price": entry,
        "target_price": target, "stop_price": stop,
        "gross_target_return": None, "gross_stop_return": None,
        "net_target_return": None, "net_stop_return": None,
        "entry_cost_per_share": None, "net_risk_per_share": None,
        "net_reward_per_share": None,
        "net_reward_risk": None, "break_even_win_rate": None,
        "expected_return": None, "estimated_win_rate": None,
        "forecast_status": "NOT_CALIBRATED",
        "cost_assumptions": {"buy_rate": buy, "sell_rate": sell,
                             "slippage_included": False, "dividends_included": False},
        "stop_fill_guaranteed": False,
    }
    if entry is None or entry <= 0:
        return result
    entry_cost = number(entry * (1 + buy))
    if entry_cost is None:
        return result
    result["entry_cost_per_share"] = entry_cost
    if stop is not None and 0 < stop < entry:
        result["net_risk_per_share"] = entry_cost - stop * (1 - sell)
        result["gross_stop_return"] = stop / entry - 1
        result["net_stop_return"] = -result["net_risk_per_share"] / entry_cost
    if target is None or target <= entry:
        result["status"] = "NO_VALID_UPSIDE_TARGET"
        return result
    result["gross_target_return"] = target / entry - 1
    result["net_reward_per_share"] = target * (1 - sell) - entry_cost
    result["net_target_return"] = result["net_reward_per_share"] / entry_cost
    loss, reward = result["net_stop_return"], result["net_target_return"]
    if loss is None:
        result["status"] = "NO_VALID_STOP"
    elif reward <= 0:
        result["status"] = "COST_EXCEEDS_TARGET_UPSIDE"
    else:
        result["status"] = "AVAILABLE"
        result["net_reward_risk"] = reward / -loss
        result["break_even_win_rate"] = -loss / (reward - loss)
    return result

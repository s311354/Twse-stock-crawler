"""Conditional return arithmetic must never masquerade as a forecast."""
import pytest

from domain.strategies.low_entry_v6_payoff import payoff_scenarios
from domain.strategies.low_entry_score_v6 import assess_execution
from tests.strategies.test_low_entry_score_v6 import ACCOUNT, execution_report


def payoff(entry=100, stop=95, target=120, buy=.001425, sell=.004425):
    return payoff_scenarios(dict(entry_price=entry, stop_price=stop, target_price=target),
                            buy_cost_rate=buy, sell_cost_rate=sell)


def test_cost_adjusted_target_stop_and_required_success_rate():
    result = payoff()
    win = 120 * .995575 / 100.1425 - 1
    loss = 95 * .995575 / 100.1425 - 1
    assert result["status"] == "AVAILABLE"
    assert result["gross_target_return"] == pytest.approx(.2)
    assert result["gross_stop_return"] == pytest.approx(-.05)
    assert result["net_target_return"] == pytest.approx(win)
    assert result["net_stop_return"] == pytest.approx(loss)
    assert result["net_reward_risk"] == pytest.approx(win / -loss)
    assert result["break_even_win_rate"] == pytest.approx(-loss / (win - loss))
    assert result["expected_return"] is result["estimated_win_rate"] is None
    assert result["forecast_status"] == "NOT_CALIBRATED"
    assert result["stop_fill_guaranteed"] is False


@pytest.mark.parametrize("target", [None, 0, 100, 90, float("nan"), float("inf"), True])
def test_price_discovery_or_invalid_target_is_not_fabricated_upside(target):
    result = payoff(target=target)
    assert result["status"] == "NO_VALID_UPSIDE_TARGET"
    assert result["gross_target_return"] is result["net_target_return"] is None
    assert result["net_reward_risk"] is result["break_even_win_rate"] is None
    assert result["net_stop_return"] < 0


@pytest.mark.parametrize("stop", [None, 0, 100, 105, float("nan"), True])
def test_invalid_stop_does_not_produce_reward_risk(stop):
    result = payoff(stop=stop)
    assert result["status"] == "NO_VALID_STOP"
    assert result["net_reward_risk"] is result["break_even_win_rate"] is None


@pytest.mark.parametrize("entry", [None, 0, -1, float("inf"), True])
def test_invalid_entry_fails_closed(entry):
    assert payoff(entry=entry)["status"] == "UNAVAILABLE"


def test_tiny_gross_reward_can_be_negative_after_costs():
    result = payoff(target=100.1)
    assert result["status"] == "COST_EXCEEDS_TARGET_UPSIDE"
    assert result["net_target_return"] < 0
    assert result["break_even_win_rate"] is None


def test_zero_cost_assumption_is_supported_and_more_cost_reduces_rr():
    assert payoff(buy=0, sell=0)["net_reward_risk"] == pytest.approx(4)
    assert payoff()["net_reward_risk"] < payoff(buy=0, sell=0)["net_reward_risk"]


@pytest.mark.parametrize("invalid", [None, -1, 1, float("nan"), True])
def test_invalid_costs_are_not_silently_replaced(invalid):
    with pytest.raises(ValueError, match="INVALID_PAYOFF_COST_ASSUMPTION"):
        payoff(buy=invalid)


def test_fresh_execution_payoff_uses_live_ask_and_frozen_stop_target():
    report = execution_report()
    quote = dict(symbol="2330", timestamp="2026-09-14T09:05:00+08:00",
                 price=100.6, bid=100.5, ask=100.7, restricted=False)
    result = assess_execution(report, quote, now=quote["timestamp"],
                              trading_sessions=["2026-09-11", "2026-09-14"], portfolio=ACCOUNT)
    assert result["action"] == "ELIGIBLE"
    assert result["payoff_scenarios"]["entry_price"] == 100.7
    assert result["payoff_scenarios"]["stop_price"] == report["entry"]["stop_price"]
    assert result["payoff_scenarios"]["target_price"] == report["entry"]["target_price"]
    assert result["is_fill"] is False

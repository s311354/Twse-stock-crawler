from twse.analyzer import compute_signal_features
from twse.analyzer import evaluate_low_entry
from twse.analyzer import max_profit
from twse.analyzer import max_profit_k_transactions
from twse.analyzer import max_profit_unlimited
from twse.analyzer import max_profit_with_fee


def test_profit_calculations() -> None:
    prices = [100, 120, 90, 130]

    assert max_profit(prices) == 40
    assert max_profit_unlimited(prices) == 60
    assert max_profit_k_transactions(5, prices) == 60
    assert max_profit_with_fee(prices, 5) == 50


def test_compute_signal_features() -> None:
    features = compute_signal_features(
        closes=[100, 102, 104, 105, 108],
        highs=[101, 103, 105, 106, 110],
        lows=[99, 101, 103, 104, 106],
    )

    assert features.trend_score == 1
    assert features.mom == 0.0588
    assert features.cs == 0.5
    assert features.stop_loss == 104.76
    assert features.take_profit == 113.4


def test_low_entry_buy_decision() -> None:
    decision = evaluate_low_entry(
        current_price=101,
        high_price=110,
        low_price=100,
        stop_loss=98,
        take_profit=107,
        mom=0.01,
        cs=0.2,
    )

    assert decision.position_score == 0.1
    assert decision.rr == 2.0
    assert decision.decision == "BUY"
    assert "Position Score <= 0.3=Yes" in decision.reason


def test_low_entry_invalid_risk_is_no_buy() -> None:
    decision = evaluate_low_entry(
        current_price=100,
        high_price=110,
        low_price=100,
        stop_loss=100,
        take_profit=105,
        mom=0.1,
        cs=0.5,
    )

    assert decision.rr is None
    assert decision.decision == "NO BUY"
    assert "無效計算（風險過高）" in decision.reason

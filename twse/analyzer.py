"""Analysis and trading-signal helpers for TWSE daily data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SignalFeatures:
    trend_score: int | None
    mom: float | None
    cs: float | None
    stop_loss: float | None
    take_profit: float | None


@dataclass(frozen=True)
class LowEntryDecision:
    position_score: float | None
    rr: float | None
    decision: str
    reason: str


def max_profit(prices: Iterable[float]) -> float:
    hold, not_hold = -float("inf"), 0
    for price in prices:
        not_hold = max(not_hold, hold + price)
        hold = max(hold, -price)
    return not_hold


def max_profit_unlimited(prices: Iterable[float]) -> float:
    prices = list(prices)
    return sum(max(prices[index] - prices[index - 1], 0) for index in range(1, len(prices)))


def max_profit_k_transactions(k: int, prices: Iterable[float]) -> float:
    prices = list(prices)
    if not prices:
        return 0

    dp = [[0 for _ in range(len(prices))] for _ in range(k + 1)]
    for trans_k in range(1, k + 1):
        balance_after_buy = -prices[0]
        for day in range(1, len(prices)):
            dp[trans_k][day] = max(dp[trans_k][day - 1], balance_after_buy + prices[day])
            balance_after_buy = max(balance_after_buy, dp[trans_k - 1][day - 1] - prices[day])
    return dp[k][-1]


def max_profit_with_fee(prices: Iterable[float], fee: float) -> float:
    hold, not_hold = -float("inf"), 0
    for price in prices:
        not_hold = max(not_hold, hold + price)
        hold = max(hold, not_hold - price - fee)
    return not_hold


def compute_signal_features(
    closes: Iterable[float],
    highs: Iterable[float],
    lows: Iterable[float],
    ma_window: int = 5,
    mom_lag: int = 3,
) -> SignalFeatures:
    close_series = pd.Series(list(closes), dtype="float64")
    high_series = pd.Series(list(highs), dtype="float64")
    low_series = pd.Series(list(lows), dtype="float64")

    if close_series.empty:
        return SignalFeatures(None, None, None, None, None)

    # Use the last ma_window prices when calculating a moving average.
    ma = close_series.rolling(window=ma_window, min_periods=ma_window).mean().iloc[-1]
    current_close = close_series.iloc[-1]
    trend_score = int(pd.notna(ma) and current_close > ma)

    previous_close = close_series.shift(mom_lag).iloc[-1]
    mom = (current_close - previous_close) / previous_close if pd.notna(previous_close) and previous_close != 0 else None

    current_low = low_series.iloc[-1]
    cs_denominator = high_series.iloc[-1] - current_low
    cs = (current_close - current_low) / cs_denominator if cs_denominator != 0 else None

    stop_loss_by_price = current_close * 0.97
    stop_loss = max(stop_loss_by_price, ma) if pd.notna(ma) else stop_loss_by_price
    take_profit = current_close * 1.05

    return SignalFeatures(
        trend_score=trend_score,
        mom=round(mom, 4) if mom is not None else None,
        cs=round(cs, 4) if cs is not None else None,
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
    )


def evaluate_low_entry(
    current_price: float | None,
    high_price: float | None,
    low_price: float | None,
    stop_loss: float | None,
    take_profit: float | None,
    mom: float | None,
    cs: float | None,
    invalid_risk_threshold: float = 1e-9,
) -> LowEntryDecision:
    price_range = high_price - low_price if high_price is not None and low_price is not None else None
    position_score = (current_price - low_price) / price_range if price_range and price_range != 0 else None

    risk = current_price - stop_loss if current_price is not None and stop_loss is not None else None
    reward = take_profit - current_price if take_profit is not None and current_price is not None else None
    invalid_risk = risk is None or risk <= invalid_risk_threshold
    rr = reward / risk if not invalid_risk and reward is not None else None

    checks = {
        "Position Score <= 0.3": position_score is not None and position_score <= 0.3,
        "MOM > 0": mom is not None and mom > 0,
        "CS >= 0.2": cs is not None and cs >= 0.2,
        "RR >= 1.5": rr is not None and rr >= 1.5,
    }

    decision = "BUY" if not invalid_risk and all(checks.values()) else "NO BUY"
    risk_note = "無效計算（風險過高）" if invalid_risk else "有效"
    reason = "；".join(f"{condition}={'Yes' if passed else 'No'}" for condition, passed in checks.items())
    reason = f"{reason}；風險計算={risk_note}"

    return LowEntryDecision(
        position_score=round(position_score, 2) if position_score is not None else None,
        rr=round(rr, 2) if rr is not None else None,
        decision=decision,
        reason=reason,
    )


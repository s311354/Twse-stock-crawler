"""Causal daily features for v6, built on the shared feature pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.features import FeatureFrame
from domain.strategies.low_entry_v6_score_evidence import DEPENDENCIES


# Fixed, deliberately short causal window. ATR tolerances are frozen at the
# breakout bar so later volatility cannot widen an already established base.
BREAKOUT_RETEST_MAX_AGE = 5
BREAKOUT_RETEST_LOWER_ATR = .30
BREAKOUT_RETEST_UPPER_ATR = .50


def _consolidation_features(f: pd.DataFrame) -> pd.DataFrame:
    """A compact three-session shelf, known only after two held right bars.

    Daily lows may oscillate. The first low must remain the floor, above an
    already confirmed higher swing low. Freeze ATR at that first bar; neither
    a later volatility spike nor a new lower low may manufacture a tighter stop.
    This is short-base evidence, not a replacement for the swing-low series.
    """
    numeric = f.reindex(columns=("open", "high", "low", "close", "volume", "atr14", "support")).apply(pd.to_numeric, errors="coerce")
    o, h, l, c, v, atr, support = (numeric[key] for key in numeric)
    valid = (np.isfinite(numeric).all(axis=1) & (l > 0) & (v > 0) & (atr > 0)
             & (h >= pd.concat([o, c], axis=1).max(axis=1))
             & (l <= pd.concat([o, c], axis=1).min(axis=1)))
    ready = f.reindex(columns=("history_ready", "market_ready", "support_intact")).eq(True).all(axis=1)
    floor = l.shift(2)
    ceiling = h.rolling(3, min_periods=3).max()
    confirmed = ((valid & ready).rolling(3, min_periods=3).sum().eq(3)
                 & f.get("higher_low", pd.Series(False, index=f.index)).shift(2).eq(True)
                 & (floor > support.shift(2)) & (floor > support)
                 & (l.rolling(3, min_periods=3).min() >= floor)
                 & (ceiling - floor <= 1.5 * atr.shift(2))
                 & (c >= (ceiling + floor) / 2) & (c >= c.shift(2)))
    return pd.DataFrame({"consolidation_confirmed": confirmed,
                         "consolidation_support": floor.where(confirmed)}, index=f.index)


def _breakout_retest_features(f: pd.DataFrame) -> pd.DataFrame:
    """Recognize a fresh breakout followed by a separately completed held retest.

    The newest qualified breakout replaces earlier evidence. Confirmation is
    emitted only on a qualifying retest bar, never carried forward or backdated.
    A break or missing aligned bar permanently invalidates that event.
    """
    columns = ("open", "high", "low", "close", "volume", "hh20", "atr14",
               "clv", "volume_ratio", "ema20_slope", "rs20")
    values = f.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ready = f.reindex(columns=("history_ready", "market_ready")).eq(True).all(axis=1).to_numpy()
    confirmed = np.zeros(len(f), dtype=bool)
    supports = np.full(len(f), np.nan)
    levels = np.full(len(f), np.nan)
    ages = np.full(len(f), np.nan)
    active = None
    previous_close = np.nan
    for position, row in enumerate(values):
        opening, high, low, close, volume, level, atr, clv, volume_ratio, slope, rs20 = row
        valid_bar = (ready[position] and np.isfinite(row[:5]).all() and low > 0
                     and high >= max(opening, close) and low <= min(opening, close) and volume > 0)
        if not valid_bar:
            active, previous_close = None, np.nan
            continue
        fresh = (np.isfinite(row[5:]).all() and level > 0 and atr > 0
                 and close > level and previous_close <= level and clv >= .5
                 and volume_ratio >= 1.1 and slope > 0 and rs20 > 0)
        if fresh:
            # Do not use the breakout bar's low as evidence of a later retest.
            active = (position, level, atr, np.inf)
        elif active is not None:
            event_position, event_level, event_atr, lowest = active
            age = position - event_position
            lower = event_level - BREAKOUT_RETEST_LOWER_ATR * event_atr
            upper = event_level + BREAKOUT_RETEST_UPPER_ATR * event_atr
            if age > BREAKOUT_RETEST_MAX_AGE or low < lower or close < event_level:
                active = None
            else:
                lowest = min(lowest, low)
                active = (event_position, event_level, event_atr, lowest)
                if low <= upper and close >= opening and np.isfinite(clv) and clv >= .4:
                    confirmed[position] = True
                    supports[position] = min(event_level, lowest)
                    levels[position] = event_level
                    ages[position] = age
        previous_close = close
    return pd.DataFrame({
        "breakout_retest_confirmed": confirmed,
        "breakout_retest_support": supports,
        "breakout_retest_level": levels,
        "breakout_retest_age": ages,
    }, index=f.index)


def _wilder(values: pd.Series, window: int = 14) -> pd.Series:
    """SMA seed followed by Wilder smoothing; missing evidence resets warm-up."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    seed: list[float] = []
    prior = None
    for i, value in enumerate(values):
        if not np.isfinite(value):
            seed, prior = [], None
        elif prior is None:
            seed.append(value)
            if len(seed) == window:
                prior = sum(seed) / window
                result.iloc[i] = prior
        else:
            prior = (prior * (window - 1) + value) / window
            result.iloc[i] = prior
    return result


def build_features(data: pd.DataFrame, shared: FeatureFrame) -> pd.DataFrame:
    shared.assert_compatible(data)
    if shared.config.vwap_short_window != 20 or shared.config.bollinger_window != 20:
        raise ValueError("LES v6 requires 20-session VWAP and Bollinger shared features")
    f = shared.to_pandas()
    c, h, l, o, v = (f[k] for k in ("close", "high", "low", "open", "volume"))
    valid_bar = (np.isfinite(pd.concat([o, h, l, c, v], axis=1)).all(axis=1)
                 & (l > 0) & (h >= pd.concat([o, c], axis=1).max(axis=1))
                 & (l <= pd.concat([o, c], axis=1).min(axis=1)) & (v > 0))
    f["history_ready"] = valid_bar.rolling(60, min_periods=60).sum().eq(60)
    valid_market = np.isfinite(f["taiex"]) & f["taiex"].gt(0)
    f["market_ready"] = valid_market.rolling(21, min_periods=21).sum().eq(21)
    atr = _wilder(f["true_range"].where(valid_bar))
    f["atr14"] = atr
    f["atr_percent"] = atr / c
    f["atr_expansion"] = atr / atr.rolling(20).mean()
    f["atr_stable"] = (f["atr_expansion"] <= 1.2) & (atr <= atr.shift(1))
    for window, lag in ((5, 3), (10, 5), (20, 5)):
        f[f"ema{window}_slope"] = f[f"ema{window}"].diff(lag) / (lag * atr)
        f[f"reclaim{window}"] = (c > f[f"ema{window}"]) & (c.shift(1) <= f[f"ema{window}"].shift(1))
    gain, loss = _wilder(c.diff().clip(lower=0)), _wilder(-c.diff().clip(upper=0))
    f["rsi14"] = (100 - 100 / (1 + gain / loss)).where(loss != 0, np.where(gain > 0, 100, 50))
    f["rsi_cross40"] = (f["rsi14"] > 40) & (f["rsi14"].shift(1).rolling(5).min() < 40)
    f["rsi_pullback"] = f["rsi14"].shift(1).between(40, 55) & (f["rsi14"] > f["rsi14"].shift(1))
    f["macd_improving"] = f["macd_histogram_strict"].diff().gt(0).rolling(2).sum().eq(2)
    f["clv"] = ((2 * c - h - l) / (h - l)).where(h != l, 0.)
    f["cmf20"] = (f["clv"] * v).rolling(20).sum() / v.rolling(20).sum()
    f["cmf_improving"] = f["cmf20"] > f["cmf20"].shift(3)
    f["obv_slope5"] = f["obv"].rolling(5).apply(lambda values: np.dot(values, np.arange(5) - 2) / 10, raw=True)
    f["volume_ratio"] = v / v.rolling(20).mean()
    up = v.where(c > c.shift(1))
    f["up_volume_ratio"] = up / up.rolling(20, min_periods=1).mean()
    down = v.where(c < c.shift(1))
    down_average = down.rolling(20, min_periods=1).mean()
    f["down_volume_contracts"] = down.rolling(3, min_periods=1).mean() < down_average
    f["down_volume_expands"] = (c < c.shift(1)) & (v > down_average.shift(1))
    f["amount_ma20"] = f["amount"].rolling(20).mean()
    f["amount_ratio"] = f["amount"] / f["amount_ma20"]
    f["spread"] = ((f["ask"] - f["bid"]) / ((f["ask"] + f["bid"]) / 2)).where((f["bid"] > 0) & (f["ask"] >= f["bid"]))
    f["average_trade_size"] = v / f["trades"]
    f["vwap20"] = f["vwap20_typical"]
    f["position60"] = (c - l.rolling(60).min()) / (h.rolling(60).max() - l.rolling(60).min()).replace(0, np.nan)
    f["drawdown60"] = c / c.rolling(60).max() - 1
    f["drawdown20"] = c / c.rolling(20).max() - 1
    f["bollinger_position"] = (c - f["bollinger_lower"]) / (f["bollinger_upper"] - f["bollinger_lower"]).replace(0, np.nan)
    f["gap"] = o / c.shift(1) - 1
    f["hh20"] = h.shift(1).rolling(20).max()
    f["hh60"] = h.shift(1).rolling(60).max()
    f["previous_high"] = h.shift(1)
    prior_low20 = l.where(valid_bar).shift(1).rolling(20, min_periods=20).min()
    f["new_low"] = valid_bar & (l < prior_low20)
    f["new_lows3"] = f["new_low"].rolling(3).sum()
    # Stabilization means two completed sessions without a new 20-session low,
    # not two successively rising daily lows. Oscillating/equal-low bases remain
    # eligible for all three setups; trigger, support and risk still decide entry.
    # Compare each bar to its own *prior* floor. Including today's low in that
    # floor, or negating an unavailable new_low flag, would falsely grant safety.
    held_floor = valid_bar & prior_low20.notna() & (l >= prior_low20)
    f["no_new_low2"] = held_floor.rolling(2, min_periods=2).sum().eq(2)
    # A pivot at t-2 becomes visible at t, never at its original position.
    for label, prices, low_pivot in (("swing_low", l, True), ("swing_high", h, False)):
        pivot = prices.shift(2)
        comparisons = [pivot <= prices.shift(k) if low_pivot else pivot > prices.shift(k) for k in (0, 1, 3, 4)]
        confirmed = pd.concat(comparisons, axis=1).all(axis=1)
        if low_pivot:
            # Equal-low shelves are support, but a flat five-bar plateau is not
            # a pivot: there must be a strictly higher low on each side.
            confirmed &= ((pivot < prices.shift(3)) | (pivot < prices.shift(4)))
            confirmed &= ((pivot < prices) | (pivot < prices.shift(1)))
        events = pivot.where(confirmed)
        if low_pivot:
            # A repeat test of the same floor must not erase the prior distinct
            # floor and turn an established higher low into a false negative.
            distinct = events.dropna()
            events = distinct[distinct.ne(distinct.shift(1))].reindex(events.index)
        f[label] = events.ffill()
        previous_events = events.dropna().shift(1).reindex(events.index).ffill()
        f[f"previous_{label}"] = previous_events
    f["higher_low"] = f["swing_low"] > f["previous_swing_low"]
    f["higher_high"] = f["swing_high"] > f["previous_swing_high"]
    f["support"] = f["swing_low"]
    f["support_intact"] = (l >= f["support"]) & (c > f["support"])
    f["support_break"] = c < f["support"]
    f["undercut_event"] = (l < f["support"].shift(1)) & (c > f["support"].shift(1)) & (f["clv"] > .5) & (f["volume_ratio"] >= 1.2)
    f["undercut_reclaim"] = f["undercut_event"].shift(1, fill_value=False) & (l >= l.shift(1)) & (c > f["support"].shift(2))
    f["stop_support"] = f["support"].where(~f["undercut_reclaim"], l.shift(1))
    # Only consecutive, date-aligned observations are accepted. No ffill or compressed return window.
    aligned = np.isfinite(c) & c.gt(0) & valid_market
    f["rs20"] = (f["r20"] - f["taiex_r20"]).where(aligned.rolling(21).sum().eq(21))
    f["r60"] = c.pct_change(60, fill_method=None)
    f["rs60"] = (f["r60"] - f["taiex"].pct_change(60, fill_method=None)).where(aligned.rolling(61).sum().eq(61))
    f["rs_line"] = c.where(aligned) / f["taiex"].where(aligned)
    f["rs_slope5"] = (np.log(f["rs_line"] / f["rs_line"].shift(5)) / 5).where(aligned.rolling(6).sum().eq(6))
    f["rs_improving"] = f["rs20"] > f["rs20"].shift(3)
    f["rs_near_high"] = f["rs_line"] >= .98 * f["rs_line"].rolling(20).max()
    f["resilience"] = f["drawdown20"] - (f["taiex"] / f["taiex"].rolling(20).max() - 1)
    f["resilience_improving"] = (f["resilience"] > f["resilience"].shift(3)) | (f["resilience"] > 0)
    benchmark_slope = f["taiex_ema20_strict"].diff(5)
    f["market_regime"] = np.select([
        (f["taiex"] > f["taiex_ema20_strict"]) & (benchmark_slope >= 0),
        (f["taiex"] < f["taiex_ema20_strict"]) & (benchmark_slope < 0),
    ], ["RISK_ON", "RISK_OFF"], default="NEUTRAL")
    f["ema20_extension"] = (c - f["ema20"]) / atr
    f["breakout_extension"] = (c - f["hh20"]) / atr
    f["break_previous_high"] = c > f["previous_high"]
    f["restricted"] = data["Restricted"] if "Restricted" in data else None
    f = f.join(_breakout_retest_features(f)).join(_consolidation_features(f))
    # Comparisons with NaN above intentionally fail closed for admission. Keep
    # their observability separately: False during warm-up is not evidence of
    # an actually failed condition in a partial-data score estimate.
    known = {key: np.isfinite(f[key]) for key in f.select_dtypes(include="number")}
    window = lambda mask, size: mask.rolling(size, min_periods=size).sum().eq(size)
    for period in (5, 10, 20):
        known[f"ema{period}"] &= window(valid_bar, period)
        known[f"ema{period}_slope"] &= window(valid_bar, period + (3 if period == 5 else 5))
        known[f"reclaim{period}"] = known[f"ema{period}"] & known[f"ema{period}"].shift(1, fill_value=False)
    known.update({
        "atr_stable": known["atr_expansion"] & known["atr14"].shift(1, fill_value=False),
        "rsi_cross40": window(known["rsi14"], 6),
        "rsi_pullback": window(known["rsi14"], 2),
        "macd_improving": window(known["macd_histogram_strict"], 3),
        "cmf_improving": known["cmf20"] & known["cmf20"].shift(3, fill_value=False),
        "down_volume_contracts": down.rolling(3, min_periods=1).mean().notna() & down_average.notna(),
        "new_lows3": window(valid_bar & prior_low20.notna(), 3),
        "no_new_low2": window(valid_bar & prior_low20.notna(), 2),
        "higher_low": known["swing_low"] & known["previous_swing_low"],
        "higher_high": known["swing_high"] & known["previous_swing_high"],
        "support_intact": valid_bar & known["support"],
        "break_previous_high": valid_bar & valid_bar.shift(1, fill_value=False),
        "rs_improving": known["rs20"] & known["rs20"].shift(3, fill_value=False),
        "rs_near_high": window(aligned, 20),
        "resilience_improving": known["resilience"] & (f["resilience"].gt(0) | known["resilience"].shift(3, fill_value=False)),
        "restricted": f["restricted"].map(lambda value: str(value) in {"True", "False"}),
        # A down/flat day is a known failed up-volume test, not missing data.
        "up_volume_ratio": window(valid_bar, 2) & (f["up_volume_ratio"].notna() | c.le(c.shift(1))),
    })
    # Some shared indicators tolerate holes/fill volume internally. Partial
    # estimates must not interpret those synthetic observations as evidence.
    for key, periods in {"clv": 1, "obv_slope5": 6, "volume_ratio": 20,
                         "cmf20": 20, "cmf_improving": 23, "vwap20": 20,
                         "position60": 60, "drawdown60": 60, "drawdown20": 20,
                         "ema20_extension": 20, "breakout_extension": 21,
                         "down_volume_contracts": 20}.items():
        known[key] &= window(valid_bar, periods)
    scored_fields = {key for groups in DEPENDENCIES.values() for fields in groups.values()
                     for condition in fields for key in condition if not key.startswith("@")}
    scored_masks = {key: value for key, value in known.items() if key in scored_fields}
    f["score_availability"] = pd.Series(pd.DataFrame(scored_masks, index=f.index).to_dict("records"), index=f.index)
    f["stock_history_observations"] = valid_bar.rolling(60, min_periods=1).sum()
    f["market_history_observations"] = aligned.rolling(21, min_periods=1).sum()
    return f

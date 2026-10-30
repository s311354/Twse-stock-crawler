"""Cutoff-only feature extension; strict pivots become known at s+2 close."""
from __future__ import annotations

from dataclasses import asdict
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from domain.features import FeatureFrame, SharedFeaturePipeline
from domain.les_v7.contracts import FEATURE_VERSION, timestamp
from domain.les_v7.fractals import (DFA_WINDOWS, HIGUCHI_WINDOWS, DFA_CHANGE_LAG,
                                   dfa, higuchi, mfdfa, lppl_shadow, fractal_history_requirements)

CLOSE_SUPPORT_FEATURE_VERSION = "V7_BASE_FRACTAL_CLOSE_SUPPORT_R2"
SUPPORTED_FEATURE_VERSIONS = frozenset({FEATURE_VERSION, CLOSE_SUPPORT_FEATURE_VERSION})


def close_support_policy():
    return dict(version="STRICT_PIVOT_CONFIRMED_AT_SECOND_RIGHT_CLOSE_R2",
        confirmation_lag_sessions=2, additional_delay_sessions=0,
        cutoff_mode="COMPLETED_DAILY_CLOSE")


def _profile_audit(feature_version, **values):
    result = dict(profile=feature_version, **values)
    if feature_version == CLOSE_SUPPORT_FEATURE_VERSION:
        result["support_policy"] = close_support_policy()
    return result


def select_bars(data: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    if data.empty:
        return data.copy(deep=True)
    if "Date" not in data or not data.index.is_unique:
        raise ValueError("UNIQUE_DATED_BARS_REQUIRED")
    dates = pd.to_datetime(data["Date"], errors="raise")
    if (dates.isna().any() or dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all()
            or dates.duplicated().any() or not dates.is_monotonic_increasing):
        raise ValueError("SORTED_UNIQUE_LOCAL_SESSION_DATES_REQUIRED")
    close_at = dates.dt.tz_localize("Asia/Taipei") + pd.Timedelta(hours=13, minutes=30)
    mask = close_at <= timestamp(cutoff)
    if "Available At" in data:
        # Availability is evidence time, never silently localized from a naive timestamp.
        available = pd.Series([timestamp(v) if pd.notna(v) else pd.NaT for v in data["Available At"]], index=data.index)
        mask &= available <= timestamp(cutoff)
    result = data.loc[mask].copy(deep=True)
    if "Available At" in result:
        result["Available At"] = available.loc[mask].map(lambda v: v.isoformat())
    if "Complete" in result and not result["Complete"].map(lambda v: isinstance(v, (bool, np.bool_)) and bool(v)).all():
        raise ValueError("INCOMPLETE_DAILY_BAR")
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(result[k].map(lambda v: isinstance(v, (bool, np.bool_))).any() for k in required if k in result):
        raise ValueError("BOOLEAN_IS_NOT_OHLCV")
    values = result.reindex(columns=required).apply(pd.to_numeric, errors="coerce")
    if not (np.isfinite(values).all(axis=1) & (values[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
            & (values["Volume"] >= 0) & (values["High"] >= values[["Open", "Close"]].max(axis=1))
            & (values["Low"] <= values[["Open", "Close"]].min(axis=1))).all():
        raise ValueError("INVALID_OHLCV")
    for key in ("Amount", "Trades", "Bid", "Ask", "TAIEX Close"):
        if key in result:
            if result[key].map(lambda v: isinstance(v, (bool, np.bool_))).any():
                raise ValueError("BOOLEAN_IS_NOT_MARKET_EVIDENCE")
            result[key] = pd.to_numeric(result[key], errors="raise")
            if np.isinf(result[key].to_numpy(dtype=float)).any():
                raise ValueError("INFINITE_MARKET_EVIDENCE")
    for key in ("Amount", "Trades"):
        if key in result and (result[key] < 0).any():
            raise ValueError("NEGATIVE_MARKET_AMOUNT_OR_COUNT")
    result[required] = values
    return result


def wilder(values: pd.Series, window=14) -> pd.Series:
    output = np.full(len(values), np.nan)
    seed, prior = [], None
    for i, value in enumerate(values):
        if not np.isfinite(value):
            seed, prior = [], None
        elif prior is None:
            seed.append(value)
            if len(seed) == window:
                prior = float(np.mean(seed))
                output[i] = prior
        else:
            prior = (prior * (window - 1) + value) / window
            output[i] = prior
    return pd.Series(output, index=values.index)


def _latest_resilience(market: pd.Series, close: pd.Series, rs20: pd.Series) -> float:
    """Final-row value, using the first market peak in the same 21 sessions."""
    if len(market) < 21:
        return np.nan
    m, stock = market.iloc[-21:], close.iloc[-21:]
    if not m.notna().all():
        return np.nan
    peak = int(np.argmax(m.to_numpy()))
    return (np.log(stock.iloc[-1] / stock.iloc[peak]) - np.log(m.iloc[-1] / m.iloc[peak])
            if peak < 20 and m.iloc[-1] < m.iloc[peak] else rs20.iloc[-1])


def compute_features(data: pd.DataFrame, *, shared: FeatureFrame | None = None,
                     shadow: bool = False, feature_version: str = FEATURE_VERSION) -> tuple[dict, dict]:
    if not isinstance(feature_version, str) or feature_version not in SUPPORTED_FEATURE_VERSIONS:
        raise ValueError("V7_UNSUPPORTED_FEATURE_VERSION")
    if data.empty:
        return {}, _profile_audit(feature_version, status="NO_BARS",
                        fractal_history=fractal_history_requirements(0))
    shared = shared if shared is not None else SharedFeaturePipeline().build(data)
    shared.assert_compatible(data)
    f = shared.to_pandas()
    o, h, l, c, v = (f[k] for k in ("open", "high", "low", "close", "volume"))
    previous = c.shift(1)
    # The first TR has no previous close; no fabricated complete 14-TR seed.
    tr = pd.concat((h - l, (h - previous).abs(), (l - previous).abs()), axis=1).max(axis=1)
    tr = tr.where(previous.notna())
    atr = wilder(tr)
    f["atr"] = atr
    # Causal volatility in the same relative-price coordinate as the low-price
    # heads. Absolute ATR alone cannot compare differently priced securities.
    f["atr_fraction"] = atr / c
    f["atr_expansion"] = atr / atr.rolling(20).mean()
    for n, lag in ((5, 3), (10, 5), (20, 5)):
        f[f"ema{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
        f[f"slope{n}"] = f[f"ema{n}"].diff(lag) / (lag * atr)
    f["prior_slope20"] = f["slope20"].shift(5)
    gain, loss = wilder(c.diff().clip(lower=0)), wilder(-c.diff().clip(upper=0))
    f["rsi"] = (100 - 100 / (1 + gain / loss)).where(loss != 0, np.where(gain > 0, 100, 50))
    f["rsi"] = f["rsi"].where(gain.notna() & loss.notna())
    macd = c.ewm(span=12, adjust=False, min_periods=12).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    histogram = macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["hist_delta3_atr"] = histogram.diff(3) / atr
    f["rsi_delta3"] = f["rsi"].diff(3)
    f["clv"] = ((2 * c - h - l) / (h - l)).where(h != l, 0.)
    f["cmf"] = (f["clv"] * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_change5"] = obv.diff(5) / v.rolling(5).sum().replace(0, np.nan)
    f["vr"] = v / v.shift(1).rolling(20).mean().replace(0, np.nan)
    f["adv"] = f["amount"].shift(1).rolling(20).mean().replace(0, np.nan)
    f["amount_ratio"] = f["amount"] / f["adv"]
    f["spread"] = ((f["ask"] - f["bid"]) / ((f["ask"] + f["bid"]) / 2)).where(
        (f["bid"] > 0) & (f["ask"] >= f["bid"]))
    # Amount VWAP is never silently interchanged with the typical-price proxy.
    f["vwap20"] = f["amount"].rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    for n in (5, 20, 60):
        f[f"return{n}"] = c / c.shift(n) - 1
    market = f["taiex"].where(np.isfinite(f["taiex"]) & (f["taiex"] > 0))
    aligned = c.where(market.notna())
    for n in (20, 60):
        ready = market.notna().rolling(n + 1).sum().eq(n + 1)
        f[f"rs{n}"] = (aligned / aligned.shift(n) - market / market.shift(n)).where(ready)
    f["rs20_change5"] = f["rs20"].diff(5)
    f["rs_slope"] = np.log((aligned / market) / (aligned / market).shift(5)) / 5
    f["resilience"] = np.nan
    # Only the final feature row is published, and no subsequent rolling
    # calculation uses resilience. Preserve its full 21-session dependency
    # window without slicing/assigning every unused historical row.
    f.iloc[-1, f.columns.get_loc("resilience")] = _latest_resilience(market, c, f["rs20"])
    f["hh20"] = h.shift(1).rolling(20).max()
    f["ll20"] = l.shift(1).rolling(20).min()
    f["previous_high"] = h.shift(1)
    f["previous_close"] = previous
    pivot = l.shift(2)
    confirmed = pd.concat([pivot < l.shift(k) for k in (0, 1, 3, 4)], axis=1).all(axis=1)
    events = pivot.where(confirmed)
    confirmed_support = events.ffill()
    # R1 deliberately retains its extra-session conservative anchor. The
    # explicit close profile may use the pivot only after BOTH right-hand
    # completed bars exist, matching the newer close-based structural plan.
    support = (confirmed_support if feature_version == CLOSE_SUPPORT_FEATURE_VERSION
               else confirmed_support.shift(1)).combine_first(f["ll20"])
    last = events.ffill()
    prior = events.dropna().shift(1).reindex(f.index).ffill()
    f["higher_low"] = (last > prior).where(last.notna() & prior.notna())
    f["support"] = support
    f["support_intact"] = (c >= support).where(support.notna())
    f["reclaim"] = ((l < support) & (c > support)).where(support.notna())
    f["no_low2"] = (l >= support).where(support.notna()).rolling(2).sum().eq(2).where(support.shift(1).notna())
    f["new_lows3"] = (l < f["ll20"]).where(f["ll20"].notna()).rolling(3).sum()
    f["breakout"] = (c > f["hh20"]).where(f["hh20"].notna())
    f["extension"] = (c - f["ema20"]) / atr
    f["breakout_extension"] = (c - f["hh20"]) / atr
    f["drawdown20"] = c / c.rolling(20).max() - 1
    down = v.where(c < previous)
    recent = down.rolling(10, min_periods=3).mean()
    prior_down = down.shift(10).rolling(20, min_periods=3).mean()
    f["down_volume_ratio"] = recent / prior_down.replace(0, np.nan)
    f["cmf_improved"] = (f["cmf"] > f["cmf"].shift(3)).where(f["cmf"].shift(3).notna())
    f["obv_improved"] = (f["obv_change5"] > 0).where(f["obv_change5"].notna())
    f["rsi_cross40"] = ((f["rsi"] > 40) & (f["rsi"].shift(1) <= 40)).where(f["rsi"].shift(1).notna())
    f["hist_improved"] = (histogram > histogram.shift(1)).where(histogram.shift(1).notna())
    row = {}
    for key, value in f.iloc[-1].items():
        if pd.isna(value) or not np.isfinite(value):
            row[key] = None
        elif isinstance(value, (bool, np.bool_)):
            row[key] = bool(value)
        else:
            row[key] = float(value)
    logs = np.log(c.to_numpy())
    returns = np.diff(logs)
    estimates = {}
    for n in DFA_WINDOWS:
        current = dfa(returns[-n:]) if len(returns) >= n else None
        old = dfa(returns[-n - DFA_CHANGE_LAG:-DFA_CHANGE_LAG]) if len(returns) >= n + DFA_CHANGE_LAG else None
        estimates[f"dfa{n}"] = asdict(current) if current else dict(value=None, status="INSUFFICIENT_HISTORY", window=n)
        row[f"dfa{n}"] = current.value if current else None
        row[f"delta_dfa{n}_{DFA_CHANGE_LAG}"] = current.value - old.value if current and old and current.value is not None and old.value is not None else None
    for n in HIGUCHI_WINDOWS:
        estimate = higuchi(logs[-n:]) if len(logs) >= n else None
        estimates[f"higuchi{n}"] = asdict(estimate) if estimate else dict(value=None, status="INSUFFICIENT_HISTORY", window=n)
        row[f"higuchi{n}"] = estimate.value if estimate else None
    estimates["mfdfa"] = mfdfa(returns) if shadow else dict(status="SHADOW_DISABLED")
    estimates["lppl"] = lppl_shadow(logs) if shadow else dict(status="SHADOW_DISABLED")
    return row, _profile_audit(feature_version, status="COMPUTED", fractal=estimates,
                     fractal_history=fractal_history_requirements(len(c)),
                     zero_range=bool(h.iloc[-1] == l.iloc[-1]), vwap_method="AMOUNT_OVER_SHARES",
                     cutoff_session=pd.Timestamp(data["Date"].iloc[-1]).date().isoformat())

"""Causal resilient-base and breakout-quality hypotheses, not entry permission."""
from __future__ import annotations

import numpy as np
import pandas as pd

POLICY = "pit-industry-resilient-breakout/v1"


def leadership_features(f: pd.DataFrame) -> pd.DataFrame:
    """Prior-base windows exclude the breakout bar; never create stop anchors."""
    c, h, l, v, atr = (f[key] for key in ("close", "high", "low", "volume", "atr14"))
    valid = (np.isfinite(f[["open", "high", "low", "close", "volume", "atr14"]]).all(axis=1)
             & (l > 0) & (v > 0) & (atr > 0) & (l <= c) & (c <= h)
             & (l <= f["open"]) & (f["open"] <= h))
    ready = valid.shift(1, fill_value=False).rolling(20, min_periods=20).sum().eq(20)
    prior_high = h.shift(1).rolling(10).max()
    prior_low = l.shift(1).rolling(10).min()
    width = prior_high - prior_low
    range_atr = width / atr.shift(1)
    recent_width = h.shift(1).rolling(5).max() - l.shift(1).rolling(5).min()
    range_ratio = recent_width / width.replace(0, np.nan)
    volume_ratio = v.shift(1).rolling(5).mean() / v.shift(1).rolling(20).mean().replace(0, np.nan)
    # Freeze an already confirmed support and ATR at the start of the base.
    support, support_atr = f["support"].shift(10), atr.shift(10)
    held = prior_low >= support
    hits = pd.DataFrame({lag: l.shift(lag).between(support, support + .5 * support_atr)
                         for lag in range(1, 11)}, index=f.index)
    tests = hits.sum(axis=1)
    repeated = hits[list(range(1, 6))].any(axis=1) & hits[list(range(6, 11))].any(axis=1)
    observed = ready & support.gt(0) & support_atr.gt(0) & range_ratio.notna()
    compression = ((range_atr <= 3.) & (range_ratio <= .8) & (volume_ratio <= .8)
                   & held & repeated).where(observed)
    result = pd.DataFrame(dict(base_range_atr10=range_atr.where(ready),
        base_range_ratio5_10=range_ratio.where(ready), base_volume_ratio5_20=volume_ratio.where(ready),
        base_support=support.where(observed), base_support_tests=tests.where(observed),
        base_compression=compression), index=f.index)
    amount_ratio = f["amount"] / f["amount"].shift(1).rolling(20).mean().replace(0, np.nan)
    amount_valid = np.isfinite(f["amount"]) & f["amount"].gt(0)
    amount_known = amount_valid & amount_valid.shift(1, fill_value=False).rolling(20).sum().eq(20)
    breakout_volume = v / v.shift(1).rolling(20).mean().replace(0, np.nan)
    known = ready & valid & f["hh20"].gt(0) & f["clv"].notna()
    crossed = c > f["hh20"]
    result["clv_breakout_quality"] = ((f["clv"] - .3) / .4).clip(0, 1).where(crossed, 0.).where(known)
    result["breakout_volume_ratio"] = breakout_volume.where(known)
    result["breakout_amount_ratio"] = amount_ratio.where(known & amount_known)
    result["breakout_quality"] = (crossed & f["clv"].ge(.5) & breakout_volume.ge(1.1)
        & amount_ratio.ge(1.2) & f["breakout_extension"].between(0, .75)).where(
            known & amount_known & np.isfinite(f["breakout_extension"]))
    # A market drawdown must actually occur. Compare drawdowns over the SAME
    # path, and keep support frozen before its first material drawdown session.
    downside = pd.DataFrame(index=f.index, columns=("market_correction20", "stock_correction20",
        "downside_capture20", "correction_support", "downside_resilience"), dtype=float)
    if len(f) >= 21:
        windows = np.lib.stride_tricks.sliding_window_view
        market = windows(f["taiex"].to_numpy(dtype=float), 21)
        stock = windows(c.to_numpy(dtype=float), 21)
        ready_windows = (windows(valid.to_numpy(), 21).all(axis=1)
            & f["market_ready"].iloc[20:].eq(True).to_numpy()
            & np.isfinite(market).all(axis=1) & (market > 0).all(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            md = market / np.maximum.accumulate(market, axis=1) - 1
            sd = stock / np.maximum.accumulate(stock, axis=1) - 1
        events = md <= -.03
        tested = ready_windows & events.any(axis=1)
        market_dd = md.min(axis=1)
        stock_dd = np.where(events, sd, np.inf).min(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            capture = stock_dd / market_dd
        first = events.argmax(axis=1)
        anchors = f["support"].to_numpy(dtype=float)[np.arange(len(first)) + np.maximum(first - 1, 0)]
        support_known = tested & np.isfinite(anchors) & (anchors > 0)
        after_start = np.arange(21)[None, :] >= first[:, None]
        held = ((~after_start) | (windows(l.to_numpy(dtype=float), 21) >= anchors[:, None])).all(axis=1)
        downside.iloc[20:, :] = np.column_stack((
            np.where(ready_windows, market_dd, np.nan), np.where(tested, stock_dd, np.nan),
            np.where(tested, capture, np.nan), np.where(support_known, anchors, np.nan),
            np.where(support_known, (capture <= .75) & held, np.nan)))
    result = result.join(downside)
    for key in ("base_compression", "breakout_quality", "downside_resilience"):
        result[key] = result[key].map(lambda value: None if pd.isna(value) else bool(value))
    return result


def recognition(f: dict) -> dict:
    """Only measured True qualifies. This diagnostic grants no shares or gates."""
    from domain.strategies.low_entry_v6_quality import number
    sector = number(f.get("industry_rs20"))
    leader = None if sector is None else sector > 0
    known = all(value is not None for value in (leader, f.get("base_compression"), f.get("downside_resilience")))
    base = leader and f.get("base_compression") is True and f.get("downside_resilience") is True if known else None
    breakout = (base and f.get("breakout_quality") is True
                if known and f.get("breakout_quality") is not None else None)
    return dict(policy_version=POLICY, industry_status=f.get("industry_status", "UNKNOWN"),
        resilient_base=base, leading_breakout=breakout,
        calibration_status="RESEARCH_HYPOTHESIS_NOT_PERFORMANCE_VALIDATED",
        metrics={key: f.get(key) for key in ("industry_rs20", "industry_rs60", "market_correction20",
            "stock_correction20", "downside_capture20", "correction_support", "base_range_atr10",
            "base_range_ratio5_10", "base_volume_ratio5_20", "base_support", "base_support_tests",
            "base_compression", "clv_breakout_quality", "breakout_amount_ratio", "breakout_volume_ratio",
            "breakout_extension", "breakout_quality", "downside_resilience")})

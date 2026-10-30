"""Causal R14 short-base variant; no score, account or future price selection.

Only an otherwise unclassified completed-bar structure can enter this branch.
The fixed five-bar floor is a distinct source-bound support method, not a
strict pivot and never the nearest low chosen to obtain a favorable stop.
"""
from __future__ import annotations

from math import isfinite

import pandas as pd

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.features import select_bars, wilder
from domain.strategies.low_entry_v6_hybrid import ContextSelection, NO_CLEAR_CONTEXT, known_features


VARIANT = "EARLY_SHORT_BASE_BREAKOUT_R14"
SCHEMA = "les-v7-early-short-base-context/v1"
SUPPORT_METHOD = "FROZEN_PRIOR5_FLOOR_TWO_SEPARATE_RETEST_EPISODES"


def policy():
    return dict(version=VARIANT, context="LEADERSHIP_CONTINUATION",
        only_when="ORIGINAL_SELECTED_NO_CLEAR_CONTEXT",
        base_windows=[5, 10], base_width_atr=[3., 4.], base_atr="PRIOR_WILDER_14",
        range_excludes_signal=True, breakout_window=10, maximum_extension_atr=.75,
        minimum_volume_ratio=1.2, minimum_amount_ratio=1.2, participation_window=20,
        minimum_clv=.6, clv_scale="[-1,1]", trend="EMA10_GT_EMA20_AND_EMA20_SLOPE_GT_ZERO",
        relative_strength="RS20_GT_ZERO_AND_RS_SLOPE5_GT_ZERO",
        support_method=SUPPORT_METHOD, touch_zone_atr=.5,
        minimum_touch_episodes=2, episode_separator="LOW_AND_CLOSE_STRICTLY_ABOVE_TOUCH_ZONE",
        signal_low_must_hold_floor=True, volume_contraction_required=False,
        decision="BUY_SMALL_ONLY", minimum_final_score=62.,
        maximum_total_weight=.025, maximum_nav_risk_fraction=.00125,
        score_policy="UNCHANGED_LEADERSHIP_FACTORS_NO_FREE_POINTS",
        performance_validated=False)


context_policy = policy


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def _support(rows, prior_atr):
    prior = rows[-6:-1]
    floor = min(row["Low"] for row in prior)
    upper = floor + .5 * prior_atr
    episodes, active, separated = [], None, True
    for row in prior:
        touch = floor <= row["Low"] <= upper
        if touch:
            if active is None or separated:
                active = dict(first_session=row["Date"], last_session=row["Date"], sessions=[])
                episodes.append(active)
            active["sessions"].append(row["Date"])
            active["last_session"] = row["Date"]
            separated = False
        elif row["Low"] > upper and row["Close"] > upper:
            separated = True
    first_floor = next(row for row in prior if row["Low"] == floor)
    return dict(value=floor, session=first_floor["Date"],
        confirmed_at_session=prior[-1]["Date"], method=SUPPORT_METHOD,
        window_start=prior[0]["Date"], window_end=prior[-1]["Date"],
        prior_atr=prior_atr, touch_zone=[floor, upper],
        touch_episodes=episodes, touch_episode_count=len(episodes),
        signal_low=rows[-1]["Low"], source_rows_sha256=identity(prior))


def assess_short_base(bars, features, *, cutoff):
    """Evaluate only required inputs; an absent optional factor is not invented."""
    audit = dict(schema=SCHEMA, policy_id=identity(policy()), cutoff=timestamp(cutoff).isoformat(),
        state="UNKNOWN", setup_variant=None, checks={}, metrics={}, support=None,
        bars_id=None, reasons=[], source_verification="INHERITED_NATIVE_EVIDENCE_BINDING_NOT_CERTIFIED_HERE")
    if bars is None or bars.empty:
        audit["reasons"] = ["R14_COMPLETED_BARS_MISSING"]
        return audit
    selected = select_bars(bars, cutoff)
    if len(selected) < 21:
        audit["reasons"] = ["R14_PRIOR_20_SESSION_HISTORY_MISSING"]
        return audit
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if "Amount" in selected:
        columns.append("Amount")
    rows = selected[columns].tail(21).to_dict("records")
    for row in rows:
        row["Date"] = str(row["Date"])[:10]
        for key in columns[1:]:
            row[key] = _number(row.get(key))
    audit["bars_id"] = identity(rows)
    c = selected["Close"].astype(float)
    previous = c.shift(1)
    tr = pd.concat((selected["High"] - selected["Low"],
        (selected["High"] - previous).abs(), (selected["Low"] - previous).abs()), axis=1).max(axis=1)
    atr = wilder(tr.where(previous.notna()))
    prior_atr, signal_atr = _number(atr.iloc[-2]), _number(atr.iloc[-1])
    f = known_features(features)
    needed = {key: _number(f.get(key)) for key in ("ema10", "ema20", "ema20_slope", "rs20", "rs_slope5")}
    checks = dict(HISTORY_READY=f.get("history_ready") if isinstance(f.get("history_ready"), bool) else None,
                  MARKET_READY=f.get("market_ready") if isinstance(f.get("market_ready"), bool) else None)
    if prior_atr is None or signal_atr is None or prior_atr <= 0 or signal_atr <= 0:
        checks["ATR_READY"] = None
        audit.update(checks=checks, reasons=["R14_PRIOR_ATR_UNAVAILABLE"])
        return audit
    last = rows[-1]
    close, high, low = last["Close"], last["High"], last["Low"]
    prior5, prior10, prior20 = rows[-6:-1], rows[-11:-1], rows[:-1]
    width5 = max(row["High"] for row in prior5) - min(row["Low"] for row in prior5)
    width10 = max(row["High"] for row in prior10) - min(row["Low"] for row in prior10)
    breakout = max(row["High"] for row in prior10)
    extension = (close - breakout) / prior_atr
    volumes = [row["Volume"] for row in prior20]
    volume_mean = sum(volumes) / 20
    volume_ratio = last["Volume"] / volume_mean if volume_mean > 0 else None
    amounts = [row.get("Amount") for row in prior20]
    amount_mean = sum(amounts) / 20 if all(value is not None and value > 0 for value in amounts) else None
    amount_ratio = (last.get("Amount") / amount_mean
                    if amount_mean and last.get("Amount") is not None and last["Amount"] > 0 else None)
    clv = (2 * close - high - low) / (high - low) if high > low else 0.
    support = _support(rows, prior_atr)
    compare = lambda key, predicate: predicate(needed[key]) if needed[key] is not None else None
    trend = (needed["ema10"] > needed["ema20"] if needed["ema10"] is not None and needed["ema20"] is not None else None)
    checks.update(ATR_READY=True, PRIOR_5_WIDTH=width5 <= 3 * prior_atr,
        PRIOR_10_WIDTH=width10 <= 4 * prior_atr, CLOSE_BREAKOUT=close > breakout,
        BREAKOUT_EXTENSION=0 < extension <= .75,
        BREAKOUT_VOLUME=volume_ratio >= 1.2 if volume_ratio is not None else None,
        BREAKOUT_AMOUNT=amount_ratio >= 1.2 if amount_ratio is not None else None,
        CLOSE_LOCATION=clv >= .6, EMA_UPTREND=trend,
        POSITIVE_EMA20_SLOPE=compare("ema20_slope", lambda value: value > 0),
        POSITIVE_RS20=compare("rs20", lambda value: value > 0),
        POSITIVE_RS_SLOPE5=compare("rs_slope5", lambda value: value > 0),
        TWO_SEPARATED_SUPPORT_TESTS=support["touch_episode_count"] >= 2,
        SIGNAL_HOLDS_FIXED_FLOOR=low >= support["value"])
    reasons = [key for key, value in checks.items() if value is not True]
    state = "UNKNOWN" if any(value is None for value in checks.values()) else "PASS" if not reasons else "FAIL"
    audit.update(state=state, checks=checks, reasons=reasons, support=support,
        setup_variant=VARIANT if state == "PASS" else None,
        metrics=dict(prior_atr=prior_atr, signal_atr=signal_atr,
            prior5_width=width5, prior5_width_atr=width5 / prior_atr,
            prior10_width=width10, prior10_width_atr=width10 / prior_atr,
            breakout_level=breakout, breakout_extension_atr=extension,
            breakout_volume_ratio=volume_ratio, breakout_amount_ratio=amount_ratio, clv=clv, **needed))
    return audit


def select_context(selection, features, *, bars, cutoff):
    """Never rescue an already recognized setup after its risk gate fails."""
    if not isinstance(selection, ContextSelection):
        raise ValueError("R14_ORIGINAL_CONTEXT_SELECTION_REQUIRED")
    if selection.context != NO_CLEAR_CONTEXT:
        return selection, dict(schema=SCHEMA, policy_id=identity(policy()), state="NOT_APPLICABLE",
            setup_variant=None, reasons=["ORIGINAL_SETUP_PRESERVED"], original_selection=selection.record())
    audit = assess_short_base(bars, features, cutoff=cutoff)
    audit["original_selection"] = selection.record()
    if audit["state"] != "PASS":
        return selection, audit
    # Keep the three-context contract with the new variant's actual criteria.
    # The complete old failed checks remain in original_selection; in
    # particular old PRIOR_BASE=false is never relabeled true.
    evidence = tuple((name, tuple(audit["checks"].items()) if name == "LEADERSHIP_CONTINUATION" else checks)
                     for name, checks in selection.evidence)
    return ContextSelection("LEADERSHIP_CONTINUATION", evidence), audit

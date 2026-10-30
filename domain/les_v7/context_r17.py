"""R17 structural interpretation; scored trend/flow strength is not a veto.

Fixed priority and causal price structure still decide interpretation before
the one score. No quality, valuation, risk, account or score is inspected here.
"""
from copy import deepcopy
from math import isfinite

import pandas as pd

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.features import select_bars
from domain.strategies.low_entry_v6_features import (
    BREAKOUT_RETEST_MAX_AGE, BREAKOUT_RETEST_LOWER_ATR, BREAKOUT_RETEST_UPPER_ATR,
    _wilder,
)
from domain.strategies.low_entry_v6_hybrid import (
    ContextSelection, NO_CLEAR_CONTEXT, _Facts, _all, _any, select_context as previous,
)

VERSION = "CAUSAL_STRUCTURE_SCORED_SOFT_CONFIRMATION_R17"


def policy():
    return dict(version=VERSION, priority=["LEADERSHIP_CONTINUATION", "TREND_PULLBACK", "REVERSAL"],
        hard="KNOWN_HISTORY_MARKET_HELD_SUPPORT_AND_CAUSAL_PRICE_CLASSIFICATION",
        soft="EMA_NEAR_TREND_BASE_COMPRESSION_RESILIENCE_ALREADY_IN_SINGLE_FACTOR_LEDGER",
        breakout="ACTUAL_CLOSE_ABOVE_PRIOR20_HIGH_OR_CAUSAL_PRICE_ONLY_RETEST_R17",
        price_retest=dict(maximum_age=BREAKOUT_RETEST_MAX_AGE,
            lower_atr=BREAKOUT_RETEST_LOWER_ATR, upper_atr=BREAKOUT_RETEST_UPPER_ATR,
            source="COMPLETED_OHLCV_PRIOR_H20_AND_SAME_V6_WILDER_ATR_ONLY",
            soft_strengths="NO_EMA_CLV_RS_OR_VOLUME_RATIO_TESTS"),
        score_used_for_selection=False, risk_used_for_selection=False,
        missing_policy="NO_MISSING_HARD_OPERAND_AS_TRUE", performance_validated=False)


def audit(features):
    return dict(policy=policy(), original_evidence=previous(features).record(),
        interpretation=select_context(features).record(),
        price_retest=deepcopy(features.get("r17_price_retest_trace")),
        soft_checks_effect="AUDIT_AND_EXISTING_SCORE_ONLY_NO_EXTRA_BONUS_OR_BOOLEAN_VETO")


def price_structure_features(bars, features, *, cutoff):
    """Return detached R17 price interpretation without changing score inputs.

    The old aggregate ``breakout_retest_*`` fields are intentionally untouched.
    An observed invalid bar cannot carry a prior event through a gap. This
    helper does not authenticate source originals or certify that no official
    session was omitted. A/C and frozen-floor routes separately verify their
    official windows; ordinary anchor plans retain their existing source
    policy without an additional calendar gate. Weekdays do not prove sessions.
    """
    output = deepcopy(dict(features))
    trace = dict(schema="les-v7-price-only-retest/v1", cutoff=timestamp(cutoff).isoformat(),
        policy=policy()["price_retest"], state="UNKNOWN", source_rows_sha256=None,
        available_rows=0, current=None, resets=[], reasons=[])
    output["r17_price_retest_confirmed"] = None
    output["r17_price_retest_trace"] = trace
    if bars is None or bars.empty:
        trace["reasons"] = ["COMPLETED_PRICE_BARS_MISSING"]
        return output
    try:
        selected = select_bars(bars, cutoff)
    except (ValueError, TypeError, KeyError) as error:
        trace["reasons"] = ["PRICE_BAR_SOURCE_INVALID_OR_INCOMPLETE"]
        trace["validation_error"] = str(error)
        return output
    if selected.empty:
        trace["reasons"] = ["COMPLETED_PRICE_BARS_MISSING"]
        return output
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    rows = selected[columns].to_dict("records")
    for row in rows:
        row["Date"] = str(row["Date"])[:10]
        for key in columns[1:]: row[key] = float(row[key])
    trace.update(source_rows_sha256=identity(rows), available_rows=len(rows))
    o, h, l, c, v = (selected[name].astype(float) for name in columns[1:])
    valid = (pd.concat((o, h, l, c, v), axis=1).notna().all(axis=1)
        & (l > 0) & (h >= pd.concat((o, c), axis=1).max(axis=1))
        & (l <= pd.concat((o, c), axis=1).min(axis=1)) & (v > 0))
    previous = c.shift(1)
    true_range = pd.concat((h - l, (h - previous).abs(), (l - previous).abs()), axis=1).max(axis=1)
    # Preserve v6's first-TR seed, missing-value reset and full warm-up; do not
    # substitute the distinct native-v7 ATR initialization for this old event.
    atr = _wilder(true_range.where(valid))
    prior_high = h.shift(1).rolling(20, min_periods=20).max()
    ready = valid.rolling(60, min_periods=60).sum().eq(60)
    active, previous_close, last_confirmed = None, None, False
    for index, row in enumerate(rows):
        day, opening, high, low, close, volume = (row[name] for name in columns)
        level, current_atr = float(prior_high.iloc[index]), float(atr.iloc[index])
        if not bool(valid.iloc[index]) or not bool(ready.iloc[index]):
            if active is not None:
                trace["resets"].append(dict(session=day, reason="INVALID_BAR_OR_HISTORY_RESET"))
            active, previous_close, last_confirmed = None, None, False
            continue
        fresh = (isfinite(level) and isfinite(current_atr) and level > 0 and current_atr > 0
            and close > level and previous_close is not None and previous_close <= level)
        last_confirmed = False
        if fresh:
            active = dict(index=index, session=day, level=level, atr=current_atr, lowest=None)
        elif active is not None:
            age = index - active["index"]
            lower = active["level"] - BREAKOUT_RETEST_LOWER_ATR * active["atr"]
            upper = active["level"] + BREAKOUT_RETEST_UPPER_ATR * active["atr"]
            if age > BREAKOUT_RETEST_MAX_AGE or low < lower or close < active["level"]:
                trace["resets"].append(dict(session=day, reason="PRICE_RETEST_EXPIRED_OR_BROKEN",
                    breakout_session=active["session"]))
                active = None
            else:
                active["lowest"] = low if active["lowest"] is None else min(active["lowest"], low)
                if low <= upper and close >= opening:
                    last_confirmed = True
                    trace["current"] = dict(session=day, breakout_session=active["session"],
                        age=age, level=active["level"], frozen_atr=active["atr"],
                        support=min(active["level"], active["lowest"]), band=[lower, upper])
        previous_close = close
    last_ready = bool(ready.iloc[-1]) and isfinite(float(atr.iloc[-1])) and isfinite(float(prior_high.iloc[-1]))
    if last_ready:
        output["r17_price_retest_confirmed"] = bool(last_confirmed)
        trace["state"] = "VERIFIED_RETEST" if last_confirmed else "VERIFIED_NOT_RETEST"
        if not last_confirmed: trace["current"] = None
    else:
        trace["current"] = None
        trace["reasons"] = ["PRICE_RETEST_HISTORY_OR_CURRENT_BAR_UNKNOWN"]
    return output


def select_context(features):
    f = _Facts(features)
    ready = _all(f.b("history_ready"), f.b("market_ready"))
    held = _all(f.b("support_intact"),
        f.compare(("low", "support", "close"), lambda low, support, close: 0 < support <= low < close))
    # The old breakout_quality includes volume/CLV/trend. Actual price must
    # establish its own breakout regardless of those already-scored strengths.
    actual_break = f.compare(("close", "hh20"), lambda close, high: close > high > 0)
    # Never trust the old strength-aggregated flag when the new source-bound
    # feature is absent. The ordinary fixed factor ledger still uses it.
    retest = f.b("r17_price_retest_confirmed")
    recovery_location = _any(f.compare(("drawdown60",), lambda value: value <= -.06),
        f.compare(("position60",), lambda value: value <= .4), f.b("undercut_reclaim"))
    checks = (
        ("LEADERSHIP_CONTINUATION", (("HISTORY_AND_MARKET_READY", ready), ("HELD_SUPPORT", held),
            ("ACTUAL_BREAKOUT_OR_VERIFIED_RETEST", _any(actual_break, retest)),
            ("HIGH_ZONE", f.compare(("position60",), lambda value: value >= .6)))),
        ("TREND_PULLBACK", (("HISTORY_AND_MARKET_READY", ready), ("HELD_SUPPORT", held),
            ("HELD_LOW", f.b("no_new_low2")),
            ("PULLBACK", f.compare(("drawdown20",), lambda value: -.15 <= value <= -.01)))),
        ("REVERSAL", (("HISTORY_AND_MARKET_READY", ready), ("HELD_SUPPORT", held),
            ("LOW_ZONE", recovery_location),
            ("HELD_LOW", _any(f.b("no_new_low2"), f.b("higher_low"), f.b("consolidation_confirmed"))))),
    )
    selected = next((name for name, items in checks if all(value is True for _, value in items)), NO_CLEAR_CONTEXT)
    return ContextSelection(selected, checks)

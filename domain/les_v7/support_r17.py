"""Source-bound frozen-floor confirmation, independent of scored patterns.

No EMA/RS/volume, score, valuation or account operand participates. The floor
is selected before risk calculations and never moved after an RR failure.
"""
from __future__ import annotations

from copy import deepcopy

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.plan_r16 import _atr, _rows
from domain.les_v7.session_coverage_r17 import verify_window

SCHEMA = "les-v7-causal-frozen-floor/v1"
METHOD = "FROZEN_PRIOR5_FLOOR_SOURCE_CONFIRMATION_R17"


def assess_floor(bars, *, cutoff, market, session_evidence=None,
                 consecutive=False, impulse_high=None):
    """Use the first minimum and two subsequent *prior* completed closes.

    C retains the R15 non-consecutive confirmation definition. A requests
    consecutive confirmations and the source-bound impulse chronology.
    Calendar completeness is rederived from the bound source, not a PASS flag.
    """
    result = dict(schema=SCHEMA, method=METHOD, cutoff=timestamp(cutoff).isoformat(),
        state="UNKNOWN", support=None, checks={}, reasons=[], metrics={},
        source_authenticity="INHERITED_NATIVE_SOURCE_BINDING_NOT_CERTIFIED_BY_HASH")
    if bars is None or bars.empty:
        result["reasons"] = ["COMPLETED_SOURCE_BARS_MISSING"]
        return result
    selected, rows = _rows(bars, cutoff)
    if len(rows) < 21:
        result["reasons"] = ["PRIOR_20_COMPLETED_SESSIONS_MISSING"]
        return result
    prior_atr, signal_atr = _atr(selected)
    if prior_atr is None or signal_atr is None or min(prior_atr, signal_atr) <= 0:
        result["reasons"] = ["CAUSAL_WILDER_ATR_MISSING"]
        return result
    coverage = verify_window(session_evidence, market=market, cutoff=cutoff,
        observed_sessions=[row["Date"] for row in rows[-21:]], prior_sessions=20)
    prior = rows[-6:-1]
    floor = min(row["Low"] for row in prior)
    first = next(index for index, row in enumerate(prior) if row["Low"] == floor)
    formation = prior[first]["Date"]
    level = floor + .5 * prior_atr
    confirmations, run = [], []
    for row in prior[first + 1:]:
        passed = row["Close"] >= level if consecutive else row["Close"] > level
        if passed:
            run.append(row["Date"])
            confirmations.append(row["Date"])
            if consecutive and len(run) >= 2:
                confirmations = run[:2]
                break
        elif consecutive:
            run = []
    confirmed = len(run) >= 2 if consecutive else len(confirmations) >= 2
    last = rows[-1]
    checks = dict(FULL_PRIOR_20=(True if coverage["state"] == "PASS" else
        False if coverage["state"] == "FAIL" else None),
        TWO_SUBSEQUENT_PRIOR_CLOSES=confirmed, SIGNAL_HOLDS_FLOOR=last["Low"] >= floor,
        SIGNAL_CLOSE_ABOVE_FLOOR=last["Close"] > floor)
    if consecutive:
        checks["FLOOR_AFTER_IMPULSE"] = formation > impulse_high["session"] if impulse_high else None
    support = dict(value=floor, session=formation, formation_session=formation,
        confirmed_at_session=confirmations[1] if confirmed else None,
        confirmation_sessions=confirmations if confirmed else [],
        confirmation_level=level, confirmation_comparison="GTE" if consecutive else "GT",
        consecutive=consecutive, method=METHOD,
        window_start=prior[0]["Date"], window_end=prior[-1]["Date"],
        source_rows_sha256=identity(prior), impulse_high=deepcopy(impulse_high))
    reasons = [name for name, passed in checks.items() if passed is not True]
    state = "UNKNOWN" if any(passed is None for passed in checks.values()) else "FAIL" if reasons else "PASS"
    result.update(state=state, reasons=reasons, checks=checks, support=support,
        metrics=dict(prior_atr=prior_atr, signal_atr=signal_atr,
            prior5_width=max(row["High"] for row in prior) - floor,
            prior10_width=max(row["High"] for row in rows[-11:-1]) - min(row["Low"] for row in rows[-11:-1])),
        session_coverage=coverage, bars_id=identity(rows[-21:]))
    result["assessment_id"] = identity(result)
    return result

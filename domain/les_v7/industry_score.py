"""Source-bound industry RS operands for the fixed-budget rule baseline.

This only reconnects two existing four/two-point factor slots. It neither
changes their curves nor redistributes points when evidence is unavailable.
The existing industry selector owns source, publication and membership policy.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from domain.industry_benchmark import aligned_index, sector_code
from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.features import select_bars

VERSION = "SOURCE_BOUND_INDUSTRY_RS_R1"
SCHEMA = "les-v7-industry-score-evidence/v1"
HORIZON_VERSION = "SOURCE_BOUND_INDUSTRY_RS_R2_PER_HORIZON"


def industry_score_policy(*, version=VERSION) -> dict:
    """A detached declaration, not a fitted model or extra score budget."""
    if version == HORIZON_VERSION:
        return dict(industry_score_policy(), version=HORIZON_VERSION,
            alignment="INDEPENDENT_21_61_EXACT_DATE_WINDOWS_HISTORICAL_MEMBERSHIP_REQUIRED",
            partial="ONLY_VERIFIED_HORIZON_SCORED_NO_WEIGHT_REDISTRIBUTION")
    if version != VERSION:
        raise ValueError("UNKNOWN_INDUSTRY_SCORE_POLICY")
    return dict(version=VERSION,
        formula="STOCK_SIMPLE_RETURN_MINUS_OFFICIAL_INDUSTRY_SIMPLE_RETURN",
        horizons={"20": dict(weight=4), "60": dict(weight=2)},
        missing="ZERO_POINTS_NO_RENORMALIZATION",
        alignment="EXACT_DATES_NO_FORWARD_FILL_HISTORICAL_MEMBERSHIP_REQUIRED")


def apply_industry_features(features, bars, packet, *, symbol, market, sector, cutoff, version=VERSION):
    """Return detached features and cutoff-selected source diagnostics.

    Naked factor operands or stale passed flags cannot fill a missing packet.
    No quote, financial, trigger, admission, risk or position policy is changed.
    """
    if version == HORIZON_VERSION:
        return _per_horizon(features, bars, packet, symbol=symbol, market=market, sector=sector, cutoff=cutoff)
    if version != VERSION:
        raise ValueError("UNKNOWN_INDUSTRY_SCORE_POLICY")
    result = json.loads(canonical(features))
    result.update(industry_rs20=None, industry_rs60=None, industry_status="UNKNOWN")
    result["score_availability"] = dict(result.get("score_availability") or {},
        industry_rs20=False, industry_rs60=False)
    audit = dict(schema=SCHEMA, policy_id=identity(industry_score_policy()), symbol=symbol,
        market=market, sector=sector_code(sector), cutoff=timestamp(cutoff).isoformat(),
        status="UNKNOWN", reason="INDUSTRY_BENCHMARK_MISSING", window=None,
        periods={"20": dict(available=False, value=None), "60": dict(available=False, value=None)},
        evidence=None)
    try:
        selected = select_bars(bars, cutoff).tail(61)
        if selected.empty:
            audit["reason"] = "INDUSTRY_STOCK_WINDOW_MISSING"
            return result, audit
        close = pd.to_numeric(selected["Close"], errors="coerce")
        if result.get("close") is None or result["close"] != float(close.iloc[-1]):
            audit["reason"] = "INDUSTRY_STOCK_CLOSE_BINDING_MISMATCH"
            return result, audit
        dates = pd.to_datetime(selected["Date"])
        audit["window"] = dict(start=dates.iloc[0].date().isoformat(),
            end=dates.iloc[-1].date().isoformat(), observations=len(dates))
        # ReadOnlyEvidence uses immutable tuple/list facades. Normalize this
        # bounded official packet to the existing selector's list contract.
        normalized = json.loads(canonical(packet)) if packet is not None else None
        index, evidence = aligned_index(normalized, dates, symbol=symbol, sector=sector,
                                        market=market, as_of=cutoff)
        audit.update(status=evidence["status"], reason=evidence["reason"], evidence=evidence)
        result["industry_status"] = evidence["status"]
        for horizon in (20, 60):
            paired = pd.concat((close.tail(horizon + 1), index.tail(horizon + 1)), axis=1)
            available = (evidence["status"] == "PASS" and len(paired) == horizon + 1
                and bool(np.isfinite(paired.to_numpy(dtype=float)).all()) and bool(paired.gt(0).all().all()))
            value = float(paired.iloc[-1, 0] / paired.iloc[0, 0]
                - paired.iloc[-1, 1] / paired.iloc[0, 1]) if available else None
            name = f"industry_rs{horizon}"
            result[name], result["score_availability"][name] = value, available
            audit["periods"][str(horizon)] = dict(available=available, value=value)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        # Invalid evidence is unscored, never an implicit verified index.
        result.update(industry_rs20=None, industry_rs60=None, industry_status="UNKNOWN")
        result["score_availability"].update(industry_rs20=False, industry_rs60=False)
        audit.update(status="UNKNOWN", reason=str(error) or "INDUSTRY_SCORE_INPUT_UNVERIFIED")
        audit["periods"] = {str(n): dict(available=False, value=None) for n in (20, 60)}
    return result, audit


def _per_horizon(features, bars, packet, *, symbol, market, sector, cutoff):
    """A gap outside RS20's dependencies cannot erase verified RS20 evidence.

    Each selector still verifies exact dates, versions, market, membership and
    availability at this cutoff. No current classification substitutes for a
    missing historical constituent record, and no dates are forward-filled.
    """
    result = json.loads(canonical(features))
    result.update(industry_rs20=None, industry_rs60=None, industry_status="UNKNOWN",
        industry_period_status={"20": "UNKNOWN", "60": "UNKNOWN"})
    result["score_availability"] = dict(result.get("score_availability") or {},
        industry_rs20=False, industry_rs60=False)
    audit = dict(schema="les-v7-industry-score-evidence/v2", policy_id=identity(
        industry_score_policy(version=HORIZON_VERSION)), symbol=symbol, market=market,
        sector=sector_code(sector), cutoff=timestamp(cutoff).isoformat(), status="UNKNOWN",
        reason="INDUSTRY_BENCHMARK_MISSING", periods={})
    try:
        selected = select_bars(bars, cutoff).tail(61)
        if selected.empty:
            raise ValueError("INDUSTRY_STOCK_WINDOW_MISSING")
        close = pd.to_numeric(selected["Close"], errors="coerce")
        if result.get("close") is None or result["close"] != float(close.iloc[-1]):
            raise ValueError("INDUSTRY_STOCK_CLOSE_BINDING_MISMATCH")
        normalized = json.loads(canonical(packet)) if packet is not None else None
        dates = pd.to_datetime(selected["Date"])
        for horizon in (20, 60):
            window, prices = dates.tail(horizon + 1), close.tail(horizon + 1)
            index, evidence = aligned_index(normalized, window, symbol=symbol,
                sector=sector, market=market, as_of=cutoff)
            paired = pd.concat((prices, index), axis=1)
            available = (evidence["status"] == "PASS" and len(paired) == horizon + 1
                and bool(np.isfinite(paired.to_numpy(dtype=float)).all()) and bool(paired.gt(0).all().all()))
            value = float(paired.iloc[-1, 0] / paired.iloc[0, 0]
                - paired.iloc[-1, 1] / paired.iloc[0, 1]) if available else None
            name = f"industry_rs{horizon}"
            result[name], result["score_availability"][name] = value, available
            result["industry_period_status"][str(horizon)] = "PASS" if available else "UNKNOWN"
            audit["periods"][str(horizon)] = dict(available=available, value=value,
                window=dict(start=window.iloc[0].date().isoformat(), end=window.iloc[-1].date().isoformat(),
                            observations=len(window)), evidence=evidence,
                reason=evidence["reason"] if len(window) == horizon + 1 else "INDUSTRY_HORIZON_HISTORY_SHORT")
        available = sum(p["available"] for p in audit["periods"].values())
        audit["status"] = result["industry_status"] = "PASS" if available == 2 else "PARTIAL" if available else "UNKNOWN"
        audit["reason"] = ("VERIFIED_INDEPENDENT_HORIZONS" if available == 2 else
            "PARTIAL_INDEPENDENT_HORIZON_EVIDENCE" if available else
            audit["periods"]["20"]["reason"])
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        result.update(industry_rs20=None, industry_rs60=None, industry_status="UNKNOWN",
            industry_period_status={"20": "UNKNOWN", "60": "UNKNOWN"})
        result["score_availability"].update(industry_rs20=False, industry_rs60=False)
        audit.update(status="UNKNOWN", reason=str(error) or "INDUSTRY_SCORE_INPUT_UNVERIFIED")
        audit["periods"] = {str(h): dict(available=False, value=None) for h in (20, 60)}
    return result, audit

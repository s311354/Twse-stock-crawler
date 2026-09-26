"""Validate restriction evidence at decision time, independently of daily bars."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from domain.restriction_announcements import HISTORICAL_SCOPE, check_history


RESTRICTION_ENDPOINTS = ("exchangeReport/TWTAWU", "exchangeReport/TWT85U", "announcement/punish")
TPEX_RESTRICTION_ENDPOINTS = (
    "tpex_spendi_today", "tpex_spendi_history", "tpex_cmode", "tpex_disposal_information",
)
RESTRICTION_CONTRACTS = {
    "TWSE": ("TWSE_OPENAPI", RESTRICTION_ENDPOINTS),
    "TPEX": ("TPEX_OPENAPI", TPEX_RESTRICTION_ENDPOINTS),
}
CURRENT_RESTRICTION_SCOPE = "CURRENT_AS_OF"
MAX_RESTRICTION_AGE = pd.Timedelta(hours=4)


def restriction_check(evidence: Mapping, *, market_data_date: str, decision_time: object,
                      market: str = "TWSE", allow_historical: bool = False,
                      symbol: str | None = None) -> dict:
    """A current list proves current eligibility, never a past session's status.

    Legacy unscoped records are accepted only for same-session decisions. Every
    path requires all official lists for the instrument's market, an aware
    timestamp, and a fresh observation. Evidence cannot choose its own market.
    Market-data freshness is checked separately against the batch session.
    """
    result = {"status": "UNKNOWN", "restricted": None, "reason": "MISSING_RESTRICTION_EVIDENCE",
              "market_data_date": market_data_date, "observation_date": None,
              "decision_date": None, "scope": None}

    def fail(reason):
        result["reason"] = reason
        return result

    if not isinstance(evidence, Mapping) or not evidence:
        return result
    if evidence.get("scope") == HISTORICAL_SCOPE:
        if not allow_historical:
            return fail("HISTORICAL_RESTRICTIONS_REPLAY_ONLY")
        if symbol is None or evidence.get("symbol") != symbol:
            return fail("HISTORICAL_RESTRICTION_SYMBOL_MISMATCH")
        return check_history(evidence, market=market, market_data_date=market_data_date,
                             decision_time=decision_time)
    if not isinstance(market, str) or market not in RESTRICTION_CONTRACTS:
        return fail("UNSUPPORTED_RESTRICTION_MARKET")
    source, required_endpoints = RESTRICTION_CONTRACTS[market]
    # Legacy TWSE producers did not include market. TPEx has no legacy contract.
    if evidence.get("market", "TWSE") != market:
        return fail("RESTRICTION_MARKET_MISMATCH")
    try:
        observed = pd.Timestamp(evidence.get("available_at"))
        cutoff = pd.Timestamp(decision_time)
        bar = pd.Timestamp(market_data_date)
        if (pd.isna(observed) or pd.isna(cutoff) or pd.isna(bar)
                or observed.tzinfo is None or cutoff.tzinfo is None
                or bar.tzinfo is not None or bar != bar.normalize()):
            return fail("INVALID_RESTRICTION_TIMESTAMP")
    except (TypeError, ValueError):
        return fail("INVALID_RESTRICTION_TIMESTAMP")
    observed_date = observed.tz_convert("Asia/Taipei").date().isoformat()
    decision_date = cutoff.tz_convert("Asia/Taipei").date().isoformat()
    result.update(observation_date=observed_date, decision_date=decision_date,
                  scope=evidence.get("scope", "SESSION"))
    if observed > cutoff:
        return fail("FUTURE_RESTRICTION_EVIDENCE")
    if observed_date != decision_date or cutoff - observed > MAX_RESTRICTION_AGE:
        return fail("STALE_RESTRICTION_EVIDENCE")
    if (bar.tz_localize("Asia/Taipei") + pd.Timedelta(hours=13, minutes=30)) > cutoff:
        return fail("INCOMPLETE_OR_FUTURE_MARKET_DATA")
    if evidence.get("session_date") != observed_date:
        return fail("RESTRICTION_DATE_MISMATCH")
    scope = evidence.get("scope")
    if market == "TPEX" and scope != CURRENT_RESTRICTION_SCOPE:
        return fail("RESTRICTION_SCOPE_REQUIRED")
    if scope not in (None, CURRENT_RESTRICTION_SCOPE):
        return fail("UNSUPPORTED_RESTRICTION_SCOPE")
    if scope == CURRENT_RESTRICTION_SCOPE:
        if evidence.get("observation_date") != observed_date:
            return fail("RESTRICTION_DATE_MISMATCH")
    elif market_data_date != observed_date:
        return fail("RESTRICTION_SCOPE_REQUIRED")
    endpoints = evidence.get("endpoints")
    if (evidence.get("source") != source or not isinstance(endpoints, (list, tuple))
            or any(endpoint not in endpoints for endpoint in required_endpoints)
            or type(evidence.get("restricted")) is not bool):
        return fail("INCOMPLETE_RESTRICTION_EVIDENCE")
    restricted = evidence["restricted"]
    result.update(status="RESTRICTED" if restricted else "CLEAR", restricted=restricted,
                  reason="CURRENT_RESTRICTION_CONFIRMED" if scope else "SESSION_RESTRICTION_CONFIRMED")
    return result

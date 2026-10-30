"""Deterministic current-source join; old quote rows remain independently kept."""
from collections.abc import Mapping
from copy import deepcopy
from math import isclose, isfinite

from domain.les_v7.official_pe import validate_pe_selection
from domain.security_master import verify_compact_security_selection


def security_state(selection, *, symbol, market, cutoff):
    if selection is None:
        return dict(status="UNKNOWN", reason="SECURITY_MASTER_CAPTURE_MISSING", issuer_symbol=None)
    checked = verify_compact_security_selection(dict(selection), cutoff=cutoff)
    if checked["market_symbol"] != symbol or checked["market"] != market:
        raise ValueError("V7_SECURITY_SOURCE_BINDING_MISMATCH")
    return checked


def _number(value):
    return (float(value) if not isinstance(value, bool) and isinstance(value, (int, float))
            and isfinite(value) else None)


def join_official_pe(rows, selections, *, symbol, market, cutoff, source_policy=None):
    scoped = False
    if source_policy is not None:
        from domain.les_v7.source_policy import require_source_policy, SCOPED_VERSION, AUDIT_SOURCE_VERSION
        checked = require_source_policy(source_policy, cutoff=cutoff)
        scoped = checked["policy_version"] in {SCOPED_VERSION, AUDIT_SOURCE_VERSION}
    if not isinstance(selections, (list, tuple)) or len(selections) > 60:
        raise ValueError("V7_BOUNDED_OFFICIAL_PE_SELECTIONS_REQUIRED")
    mapped = {}
    for raw in selections:
        if not isinstance(raw, Mapping) or raw.get("session_date") in mapped:
            raise ValueError("V7_OFFICIAL_PE_SELECTION_DATE_CONFLICT")
        day = raw.get("session_date")
        missing = dict(market=market, symbol=symbol, session_date=day, pe=None, status="MISSING",
            reason="OFFICIAL_PE_OBSERVATION_MISSING", source_record_ids=[],
            source_available_at=None, selection_proofs=[])
        # This is an explicit absence of a local observation, not an official
        # clearance or a positive P/E. A purported observed row still needs its
        # inclusion/absence proof and cannot be downgraded to this shape.
        if dict(raw) == missing:
            from domain.les_v7.official_pe import session_date
            session_date(day)
            mapped[day] = missing
        else:
            mapped[day] = validate_pe_selection(dict(raw), market=market, symbol=symbol,
                trade_date=day, cutoff=cutoff)
    joined, diagnostics = [], []
    for value in rows:
        row = deepcopy(dict(value)); day = str(row["Date"])[:10]
        selected = mapped.get(day)
        if selected and selected.get("selection_proofs"):
            quote_pe, quote_close = _number(row.get("PE")), _number(row.get("Close"))
            close = _number(selected.get("reported_close"))
            close_conflict = (close is not None and (quote_close is None or
                not isclose(close, quote_close, rel_tol=0., abs_tol=1e-9)))
            pe_conflict = (quote_pe is not None and
                (selected["status"] != "REPORTED" or not isclose(quote_pe, selected["pe"], rel_tol=0., abs_tol=1e-9)))
            if close_conflict or pe_conflict:
                row["PE"] = None
                if scoped:
                    diagnostics.append(dict(session_date=day,
                        reason="OFFICIAL_CLOSE_QUOTE_CONFLICT" if close_conflict else "OFFICIAL_PE_RATIO_CONFLICT",
                        conflict_fields=(["Close"] if close_conflict else []) + (["PE"] if pe_conflict else []),
                        blocks_data_integrity=close_conflict,
                        affected_component="PRICE_IDENTITY" if close_conflict else "PE", state="UNKNOWN",
                        quote_values=dict(Close=quote_close, PE=quote_pe),
                        official_values=dict(Close=close, PE=selected.get("pe"), status=selected["status"]),
                        source_record_ids=list(selected.get("source_record_ids", ())),
                        source_available_at=selected.get("source_available_at"),
                        quarantine="PE_NONE_NO_SELECTION_OR_IMPUTATION"))
                else:
                    diagnostics.append(dict(session_date=day, reason="OFFICIAL_PE_QUOTE_CONFLICT"))
            else:
                row["PE"] = selected["pe"] if selected["status"] == "REPORTED" else None
        elif market == "TPEX":
            # This quote endpoint has no P/E field. A cached number without a
            # separately bound official ratio table cannot manufacture one.
            row["PE"] = None
        joined.append(row)
    if set(mapped) - {str(row["Date"])[:10] for row in rows}:
        raise ValueError("V7_OFFICIAL_PE_OUTSIDE_SELECTED_WINDOW")
    return joined, diagnostics


def scoped_pe_inputs(rows):
    """Assess the retained declared P/E window without imputing any ratio.

    This is a required valuation component, not OHLC/security identity. A
    quarantined current ratio cannot be supplied by an EPS valuation fallback.
    """
    window=list(rows)[-60:]
    values=[_number(row.get("PE")) for row in window]
    current=values[-1] if values else None
    positive=sum(value is not None and value>0 for value in values)
    reasons=[]
    if current is None or current<=0:reasons.append("CURRENT_PE_UNAVAILABLE_OR_NON_POSITIVE")
    if positive<20:reasons.append("PE_HISTORY_MINIMUM_20_POSITIVE_REQUIRED")
    return dict(state="UNKNOWN" if reasons else "PASS",window_limit=60,
        minimum_positive_observations=20,observed_rows=len(window),positive_observations=positive,
        current_pe=current,reasons=reasons,imputed_values=False)


def validate_source_projection(packet, *, symbol, market, cutoff, source_policy=None):
    quote = packet.get("quote_market_rows")
    if not isinstance(quote, (list, tuple)):
        raise ValueError("V7_ORIGINAL_QUOTE_ROWS_REQUIRED")
    joined, diagnostics = join_official_pe(quote, packet.get("official_pe", ()),
        symbol=symbol, market=market, cutoff=cutoff, source_policy=source_policy)
    from domain.les_v7.contracts import identity
    if identity(joined) != identity(packet.get("market_rows")) or diagnostics != packet.get("source_join_diagnostics", []):
        raise ValueError("V7_SOURCE_PROJECTION_RECOMPUTATION_MISMATCH")
    return security_state(packet.get("security_evidence"), symbol=symbol, market=market, cutoff=cutoff), diagnostics

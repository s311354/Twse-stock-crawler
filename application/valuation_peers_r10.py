"""Capture cutoff-bound annual-EPS comparisons from the existing fixed universe.

No requests, forecast defaults or financial selection shortcuts. Source packets
are projected once before stock-worker isolation. Monthly earnings must still
have been available at that month's close in the domain validator.
"""
from dataclasses import replace
import json

import pandas as pd

from application.evidence_context import ReadOnlyEvidence
from application.market_history import MarketHistoryAssembler
from application.snapshot_retention import materialize_history_snapshot
from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.features import select_bars
from domain.numeric_history import metadata_free_frame


def _independent_prices(selection, *, rows, symbol, cutoff, source, verify_selection=None):
    """R12-only compact projection of a once-verified retained cache capture.

    This is a current observation of final historical prices, not historical
    EPS availability. The domain still verifies annual originals at each month.
    Missing history preserves the original conservative fallback.
    """
    if selection is None:
        return rows, source
    checked = json.loads(canonical(selection))
    if (checked.get("schema") != "les-v7-monthly-valuation-selection/v1"
            or checked.get("symbol") != symbol or checked.get("market") != "TWSE"
            or checked.get("price_unit") != "TWD" or checked.get("status") not in {"COMPLETE", "PARTIAL", "UNKNOWN"}
            or checked.get("selection_id") != identity({k: v for k, v in checked.items() if k != "selection_id"})
            or timestamp(checked["observed_at"]) > timestamp(cutoff)):
        raise ValueError("R12_VALUATION_HISTORY_SELECTION_INVALID")
    bound_source = dict(source)
    if source.get("source_available_at") is not None:
        bound_source["source_available_at"] = max(source["source_available_at"], checked["observed_at"], key=timestamp)
    if verify_selection is None:
        # A self-hash is not proof that the named retained original contains
        # these prices. Non-I/O callers must supply the verification port.
        return rows, dict(bound_source, source_final=False, valuation_history_selection=checked,
                          valuation_history_verification_error="ORIGINAL_SOURCE_VERIFIER_NOT_PROVIDED")
    verified = verify_selection(checked)
    if canonical(verified) != canonical(checked):
        raise ValueError("R12_VALUATION_HISTORY_VERIFIER_CHANGED_INPUT")
    if not rows:
        return rows, source
    if checked["status"] == "UNKNOWN":
        # A conflicting/unverifiable price history cannot be selected around
        # just because other months would meet the minimum sample count.
        return rows, dict(bound_source, source_final=False, valuation_history_selection=checked)
    dates = [r["Date"] for r in checked["rows"]]
    if dates != sorted(set(dates)):
        raise ValueError("R12_VALUATION_HISTORY_DATE_CONFLICT")
    current = rows[-1]
    original = {r["Date"]: r["Close"] for r in rows}
    conflicts = [dict(date=row["Date"], strategy_close=original[row["Date"]], monthly_close=row["Close"])
                 for row in checked["rows"] if row["Date"] in original and row["Close"] != original[row["Date"]]]
    if conflicts:
        return rows, dict(bound_source, source_final=False, valuation_history_selection=checked,
            valuation_history_conflicts=conflicts)
    selected = [row for row in checked["rows"] if row["Date"] < current["Date"]] + [current]
    projected = dict(bound_source, bars_sha256=identity(selected),
        source_available_at=max(source["source_available_at"], checked["observed_at"], key=timestamp),
        source_record_ids=[*source["source_record_ids"], checked["capture_evidence_id"],
                           checked["selection_id"], identity(selected)],
        valuation_history_selection=checked)
    return selected, projected


def annual_projection(snapshot, cutoff):
    selected = []
    companions = snapshot.get("les_v7_context", {}).get("valuation_earnings", {})
    for filing in snapshot.get("les_v6_mops_filings", snapshot.get("les_v6_filings", ())):
        try:
            if pd.Period(filing["period_end"], freq="Q").quarter != 4:
                continue
            if timestamp(filing["available_at"]) > timestamp(cutoff):
                continue
        except (KeyError, TypeError, ValueError):
            # Retain invalid provenance for the validator, never select around it.
            pass
        fields = ("symbol", "market", "source", "source_url", "scope", "currency", "unit",
                  "period_end", "flow_basis", "available_at", "revision_id", "revision_number",
                  "content_sha256", "filing_schema", "revenue", "operating_income")
        row = {key: filing.get(key) for key in fields}
        row["field_evidence"] = {key: filing.get("field_evidence", {}).get(key, {})
                                 for key in ("revenue", "operating_income")}
        for group in ("comparative_ytd_fields", "comparative_ytd_field_evidence"):
            row[group] = {key: filing.get(group, {}).get(key) for key in ("revenue", "operating_income")}
        row["earnings_source"] = companions.get(filing.get("content_sha256"))
        selected.append(row)
    return tuple(selected)


def recovery_projection(records, cutoff):
    """Retain only exact YTD dependencies, not another full financial graph."""
    result = []
    metadata = ("symbol", "market", "source", "source_url", "scope", "currency", "unit", "period_end",
                "flow_basis", "available_at", "revision_id", "revision_number", "content_sha256", "filing_schema")
    for filing in records:
        try:
            if timestamp(filing["available_at"]) > timestamp(cutoff):
                continue
            if pd.Period(filing["period_end"], freq="Q").quarter == 4:
                continue
        except (KeyError, TypeError, ValueError):
            pass  # Invalid provenance must remain visible to the validator.
        row = {key: filing.get(key) for key in (*metadata, "revenue", "operating_income", "cfo")}
        row["field_evidence"] = {key: filing.get("field_evidence", {}).get(key, {})
                                 for key in ("revenue", "operating_income", "cfo")}
        for group in ("comparative_ytd_fields", "comparative_ytd_field_evidence"):
            row[group] = {key: filing.get(group, {}).get(key) for key in ("revenue", "operating_income")}
        if "quarter_flow_source" in filing:
            row["quarter_flow_source"] = {key: filing["quarter_flow_source"].get(key) for key in metadata}
        result.append(row)
    return tuple(result)


def capture_comparison_contexts(market, *, verify_valuation_history=None):
    """Return a detached market view; legacy/model/replay paths are unchanged."""
    from domain.les_v7.rule_baseline import current_source_baseline_version, RECOVERY_POLICIES, CLOSE_POLICIES
    if (market.entry_model != "v7" or not market.les_v7_rule_baseline or not market.les_v7_native_evidence
            or not market.les_v7_source_repair or market.replay_as_of):
        return market
    active = {symbol: timestamp(cutoff).isoformat() for symbol, cutoff in market.signal_dates.items()
              if current_source_baseline_version(cutoff) in RECOVERY_POLICIES}
    if not active:
        return market
    assembler, blocks, current_quote_sources = MarketHistoryAssembler(), {}, {}
    universe = sorted(str(s) for s in market.stock_numbers)
    for item, symbol in enumerate(market.stock_numbers):
        cutoff = market.signal_dates.get(symbol)
        if cutoff is None:
            continue
        history = assembler.build(market, item, include_evidence=False)
        raw = metadata_free_frame(history)
        for field in ("Open", "High", "Low", "Close", "Bid", "Ask"):
            raw[field] = pd.to_numeric(raw[field], errors="coerce") / 1000.
        selected = select_bars(raw, cutoff)
        rows = selected.astype(object).where(pd.notna(selected), None).to_dict("records")
        for row in rows:
            row["Date"] = pd.Timestamp(row["Date"]).date().isoformat()
        source = dict(history.attrs.get("les_v7_market_evidence", {}))
        source.update(bars_sha256=identity(rows), source_record_ids=[identity(rows)])
        current_quote_sources[symbol] = source
        if current_source_baseline_version(cutoff) in CLOSE_POLICIES:
            rows, source = _independent_prices(market.native_source_selections.get(symbol, {}).get("valuation_price_history"),
                rows=rows, symbol=symbol, cutoff=cutoff, source=source,
                verify_selection=verify_valuation_history)
        snapshot = materialize_history_snapshot(market, symbol, cutoff=cutoff)
        blocks[symbol] = ReadOnlyEvidence(dict(symbol=symbol,
            security_evidence=market.native_source_selections.get(symbol, {}).get("security"),
            market_rows=rows, market_evidence=source, annual_filings=annual_projection(snapshot, cutoff)))
        del history, raw, selected, snapshot
    sources = dict(market.native_source_selections)
    for symbol, cutoff in active.items():
        issuer = blocks.get(symbol)
        peers, exclusions = [], []
        instrument = (issuer or {}).get("security_evidence") or {}
        industry = (instrument.get("instrument") or {}).get("sector")
        for other, block in blocks.items():
            if other == symbol:
                continue
            selection = block.get("security_evidence") or {}
            sector = (selection.get("instrument") or {}).get("sector")
            if not industry or sector != industry:
                exclusions.append(dict(symbol=other, reason="DIFFERENT_OR_UNVERIFIED_OFFICIAL_INDUSTRY"))
                continue
            if selection.get("status") == "OUT_OF_SCOPE":
                from domain.security_master import verify_compact_security_selection
                verified = verify_compact_security_selection(json.loads(canonical(selection)), cutoff=cutoff)
                # Only authenticated scope exclusion is allowed. Unknown or
                # contradictory ordinary-share identity is not relabelled.
                if verified["status"] == "OUT_OF_SCOPE":
                    exclusions.append(dict(symbol=other, reason="VERIFIED_NON_COMMON_SECURITY_OUT_OF_SCOPE",
                                           source_id=verified["selection_id"]))
                    continue
            # Peers need only the current quote, not the issuer's whole history.
            rows = block["market_rows"][-1:]
            # The peer comparison uses only its current quote. An independent
            # monthly-history gap/conflict belongs to that stock as an issuer,
            # not to this already-bound current market observation. Keep the
            # full compact monthly selection only on its issuer projection.
            peer_source = (current_quote_sources[other]
                if current_source_baseline_version(cutoff) in CLOSE_POLICIES else block["market_evidence"])
            source = dict(peer_source, bars_sha256=identity(rows),
                          source_record_ids=[*peer_source["source_record_ids"], identity(rows)])
            annuals = block["annual_filings"]
            try:
                latest = max(pd.Period(row["period_end"], freq="Q") for row in annuals)
                # Retain every version of the latest period so conflicts cannot
                # be selected away. Older years are needed only by the issuer.
                annuals = tuple(row for row in annuals if pd.Period(row["period_end"], freq="Q") == latest)
            except (KeyError, TypeError, ValueError):
                pass
            peers.append(ReadOnlyEvidence(dict(block, market_rows=rows, market_evidence=source, annual_filings=annuals)))
        exclusions.extend(dict(symbol=s, reason="MARKET_CUTOFF_NOT_PROVIDED") for s in universe if s not in blocks)
        context = ReadOnlyEvidence(dict(schema="les-v7-recovery-context/v1", symbol=symbol,
            market=market.market_source, cutoff=cutoff, issuer=issuer, peers=tuple(peers),
            universe_symbols=universe, exclusions=exclusions))
        sources[symbol] = dict(sources.get(symbol, {}), r10_valuation_context=context)
    return replace(market, native_source_selections=sources)

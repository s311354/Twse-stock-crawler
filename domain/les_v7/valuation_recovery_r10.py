"""R10 price-only recovery scenarios and cutoff-bound comparable evidence.

This is a versioned assumption policy, not a calibrated price forecast. The
capture owner verifies original bytes; this port revalidates the retained
security inclusion, annual EPS proofs, quote hash and temporal selection.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, localcontext
import json
from math import isfinite
from statistics import median
from zoneinfo import ZoneInfo

import pandas as pd

from domain.les_v7.contracts import canonical, identity, timestamp
from domain.les_v7.valuation_price import _number


POLICY_VERSION = "V7_RECOVERY_INDUSTRY_PRICE_RETURN_R10"
AVAILABLE_AT = "2026-10-14T16:24:29Z"
CONTEXT_SCHEMA = "les-v7-recovery-context/v1"
INDUSTRIES = {
    "半導體業": (.20, 30.), "電子零組件業": (.20, 30.),
    "電腦及週邊設備業": (.15, 25.), "光電業": (.15, 25.),
    "通信網路業": (.15, 25.), "通訊網路業": (.15, 25.),
    "電子通路業": (.10, 20.), "其他電子業": (.10, 20.),
}


def policy():
    return dict(schema="les-v7-price-return-valuation/v2", policy_version=POLICY_VERSION,
        available_at=AVAILABLE_AT, return_basis="PRICE_RETURN_ONLY", performance_validated=False,
        maximum_annual_age_days=550, required_return=.10, maximum_bear_permanent_loss=.35,
        basic_eps_haircut=.90, dividends="EXCLUDED_NOT_IMPUTED",
        dilution="NOT_SEPARATELY_MODELED_NO_OBSERVED_ZERO_CLAIM",
        eps_anchor="LATEST_VERIFIED_ANNUAL_EPS_NOT_PRIOR_OR_OFFICIAL_PE_IMPLIED_EPS",
        recovery=dict(annual_weight=.75, recent_weight=.25,
            annual_growth="MIN_REPORTED_ANNUAL_EPS_AND_REVENUE_GROWTH",
            recent_growth="MIN_SAME_ORIGINAL_YTD_REVENUE_AND_OPERATING_INCOME_GROWTH",
            conditions=["CURRENT_AND_PRIOR_REVENUE_POSITIVE", "CURRENT_AND_PRIOR_OPERATING_INCOME_POSITIVE",
                        "REVENUE_AND_OPERATING_INCOME_GROW", "LATEST_VERIFIED_CFO_POSITIVE"],
            missing_or_failed_recent_conditions="ANNUAL_GROWTH_ONLY_NO_RECOVERY_CREDIT",
            annualization="NO_INTERIM_EPS_MULTIPLICATION_OR_SYNTHETIC_TTM"),
        industry_profiles={name: dict(maximum_growth=growth, maximum_exit_pe=pe)
                           for name, (growth, pe) in INDUSTRIES.items()},
        comparables=dict(minimum_peer_count=10, minimum_issuer_month_count=24,
            historical_month_window=36, issuer_weight=.50, peer_weight=.50,
            minimum_exit_pe=5., required_same=["market", "industry", "eps_basis", "scope", "currency"],
            peer_fiscal_year="SAME_LATEST_VERIFIED_ANNUAL_YEAR",
            historical_earnings="LATEST_ANNUAL_ORIGINAL_AVAILABLE_AT_EACH_MONTH_CLOSE",
            insufficient="EXPLICIT_ORIGINAL_R1_FALLBACK", conflict="UNKNOWN"),
        base=dict(years=5, minimum_growth=-.10, long_run_growth=.03, annual_difference_decay=.50, eps_factor=1.),
        bear=dict(years=5, growth_increment=-.05, minimum_growth=-.10,
                  eps_factor=.90, exit_pe_multiplier=.80, minimum_exit_pe=5.),
        bull=dict(years=5, growth_increment=.05, maximum_growth="INDUSTRY_BASE_CAP_PLUS_5PP",
                  eps_factor=1.10, exit_pe_multiplier=1.20, maximum_exit_pe="INDUSTRY_BASE_CAP_TIMES_1.2"),
        formula=dict(terminal_eps="HAIRCUT_ANNUAL_EPS * PRODUCT(1 + ANNUAL_SCENARIO_GROWTH) * EPS_FACTOR",
            future_price="TERMINAL_EPS * EXIT_PE", fair_price="FUTURE_PRICE / 1.1 ** 5",
            price_cagr="(FUTURE_PRICE / CURRENT_PRICE) ** (1 / 5) - 1",
            bear_loss="MAX(0, 1 - BEAR_FAIR_PRICE / CURRENT_PRICE)"),
        bear_metric="DISCOUNTED_SCENARIO_VALUE_SHORTFALL_NOT_DRAWDOWN_OR_LOSS_PROBABILITY",
        pass_conditions=["BASE_PRICE_CAGR_GTE_REQUIRED_RETURN", "BASE_FAIR_PRICE_GT_CURRENT_PRICE", "BEAR_LOSS_LTE_MAXIMUM"])


def _annual(row, *, symbol, market, cutoff, basis=None, year=None, scope=None):
    from domain.les_v7.valuation_eps_receipt import _eps, _same_original, _unreported_diluted_eps
    try:
        period = pd.Period(row["period_end"], freq="Q")
        age = (timestamp(cutoff).astimezone(ZoneInfo("Asia/Taipei")).date() - period.end_time.date()).days
        if (period.quarter != 4 or timestamp(row["available_at"]) > timestamp(cutoff)
                or period.end_time.date() > timestamp(cutoff).astimezone(ZoneInfo("Asia/Taipei")).date()
                or age < 0 or age > 550
                or row.get("symbol") != symbol or row.get("market") != market
                or row.get("source") != "MOPS_IXBRL" or row.get("scope") not in {"consolidated", "individual"}
                or (scope is not None and row["scope"] != scope)
                or row.get("currency") != "TWD" or row.get("unit") != "thousand"
                or row.get("flow_basis") not in {"ANNUAL", "YTD"}
                or (year is not None and period.year != year)):
            return None
        source = row.get("earnings_source")
        if not _same_original(source, row):
            if isinstance(source, Mapping) and source.get("status") == "VERIFIED_ORIGINAL":
                raise ValueError("R10_ANNUAL_EPS_COMPANION_SOURCE_CONFLICT")
            return None
        actual_basis, field = "DILUTED", "diluted_eps"
        value, proof = _eps(source, field, period.year)
        if value is None and (proof.get("status") == "RESOLVED" or "CONFLICT" in str(proof.get("reason", ""))):
            raise ValueError("R10_ANNUAL_EPS_REPORTED_PROOF_CONFLICT")
        if value is None and _unreported_diluted_eps(source, period.year):
            actual_basis, field = "BASIC", "basic_eps"
            value, proof = _eps(source, field, period.year)
            if value is None and (proof.get("status") == "RESOLVED" or "CONFLICT" in str(proof.get("reason", ""))):
                raise ValueError("R10_ANNUAL_EPS_REPORTED_PROOF_CONFLICT")
        if value is None or value <= 0 or (basis is not None and actual_basis != basis):
            return None
        return dict(eps=value, basis=actual_basis, year=period.year, scope=row["scope"],
                    source_id=identity(row), available_at=row["available_at"])
    except ValueError as error:
        if str(error).startswith("R10_ANNUAL_EPS_"):
            raise
        return None
    except (KeyError, TypeError, AttributeError):
        return None


def _original_at(rows, cutoff):
    eligible = {}
    for row in rows:
        period = pd.Period(row["period_end"], freq="Q")
        at = timestamp(row["available_at"])
        if period.quarter != 4 or at > timestamp(cutoff):
            continue
        prior = eligible.get(period.year)
        if prior is not None and timestamp(prior["available_at"]) == at and identity(prior) != identity(row):
            raise ValueError("CONFLICTING_R10_ANNUAL_VERSIONS")
        if prior is None or timestamp(prior["available_at"]) < at:
            eligible[period.year] = row
    return eligible[max(eligible)] if eligible else None


def _block(block, *, market, cutoff):
    from domain.security_master import verify_compact_security_selection
    from domain.les_v7.native_evidence import _market_source
    symbol = block["symbol"]
    # The compact security contract intentionally compares list-shaped fields.
    # Normalize only this small inclusion proof, not the financial/quote graph.
    selection = verify_compact_security_selection(json.loads(canonical(block["security_evidence"])), cutoff=cutoff)
    if (selection.get("market_symbol") != symbol or selection.get("market") != market
            or selection.get("status") != "VERIFIED"):
        raise ValueError("R10_SECURITY_SOURCE_UNVERIFIED")
    industry = selection["instrument"].get("sector")
    rows = block["market_rows"]
    packet = dict(market_evidence=block["market_evidence"], expected_session=str(rows[-1]["Date"])[:10],
                  historical_replay=block.get("historical_replay") is True)
    if _market_source(packet, symbol=symbol, market=market, cutoff=cutoff, rows=rows) is None:
        raise ValueError("R10_PRICE_SOURCE_HASH_OR_TIME_UNVERIFIED")
    dates = [str(row["Date"])[:10] for row in rows]
    if dates != sorted(set(dates)) or any(pd.Timestamp(day).date() > timestamp(cutoff).astimezone(ZoneInfo("Asia/Taipei")).date()
                                       for day in dates):
        raise ValueError("R10_QUOTE_DATES_CONFLICT_OR_FUTURE")
    closes = [timestamp(day + "T13:30:00+08:00") for day in dates]
    if any(close > timestamp(cutoff) or close > timestamp(block["market_evidence"]["source_available_at"])
           for close in closes):
        raise ValueError("R10_QUOTE_CLOSE_NOT_AVAILABLE_AT_CUTOFF_OR_OBSERVATION")
    if any(_number(row.get("Close")) is None or row["Close"] <= 0 for row in rows):
        raise ValueError("R10_QUOTE_PRICE_UNVERIFIED")
    return dict(symbol=symbol, industry=industry, rows=rows, annuals=block.get("annual_filings", ()),
        source_ids=[selection["selection_id"], identity(rows)],
        available_at=max(selection["master"]["available_at"],
                         block["market_evidence"]["source_available_at"], key=timestamp))


def validate_comparables(context, *, symbol, market, cutoff, eps_basis, fiscal_year, scope, annual_source=None,
                        expected_session=None, current_price=None):
    """Historical R10/R13 selection contract; its output remains unchanged."""
    return _select_comparables(context, symbol=symbol, market=market, cutoff=cutoff, eps_basis=eps_basis,
        fiscal_year=fiscal_year, scope=scope, annual_source=annual_source,
        expected_session=expected_session, current_price=current_price)


def validate_comparable_components(context, *, symbol, market, cutoff, eps_basis, fiscal_year, scope,
                                   annual_source=None, expected_session=None, current_price=None):
    """R14 retains independently verified peer and issuer-history components.

    Both use the same original/cutoff/universe validator as R10. A short issuer
    history is a known sample gap, not a conflicting peer proof. Nothing here
    declares raw official rolling P/E comparable to annual-EPS multiples.
    """
    result = _select_comparables(context, symbol=symbol, market=market, cutoff=cutoff, eps_basis=eps_basis,
        fiscal_year=fiscal_year, scope=scope, annual_source=annual_source,
        expected_session=expected_session, current_price=current_price, components=True)
    if result.get("schema") == "les-v7-comparable-components/v1":
        return result
    # No naked median or fabricated month is added for an unavailable context.
    return dict(result, schema="les-v7-comparable-components/v1", issuer_history=dict(
        status="UNKNOWN" if result["status"] == "UNKNOWN" else "INSUFFICIENT",
        count=0, minimum=24, median=None, sources=[]), peers=dict(
        status="UNKNOWN" if result["status"] == "UNKNOWN" else "INSUFFICIENT",
        count=0, minimum=10, median=None, sources=[]))


def validate_comparable_components_r15(context, *, symbol, market, cutoff, eps_basis, fiscal_year, scope,
                                      annual_source=None, expected_session=None, current_price=None):
    """R15 only: quarantine failed third-party peers, never own-source conflicts.

    The full fixed universe is checked before any peer is inspected. Every
    retained peer still passes the exact original source/period/unit/cutoff
    checks; no conflict value enters the median or the ten-peer minimum.
    """
    result = _select_comparables(context, symbol=symbol, market=market, cutoff=cutoff,
        eps_basis=eps_basis, fiscal_year=fiscal_year, scope=scope, annual_source=annual_source,
        expected_session=expected_session, current_price=current_price,
        components=True, quarantine_peers=True)
    if "issuer_history" not in result:
        state = "UNKNOWN" if result["status"] == "UNKNOWN" else "INSUFFICIENT"
        result.update(issuer_history=dict(status=state, count=0, minimum=24, median=None, sources=[]),
                      peers=dict(status=state, count=0, minimum=10, median=None, sources=[]))
    return dict(result, schema="les-v7-comparable-components/v2",
        peer_validation_policy="R15_FIXED_UNIVERSE_ROLE_BOUND_PEER_QUARANTINE")


def _peer_quarantine_record(raw, error):
    """Small immutable source references; never copy full financial originals."""
    annuals = raw.get("annual_filings", ())
    return dict(symbol=raw["symbol"], role="THIRD_PARTY_PEER", reason=str(error) or "R15_PEER_SOURCE_UNVERIFIED",
        block_id=identity(raw), security_evidence_id=identity(raw.get("security_evidence")),
        market_evidence_id=identity(raw.get("market_evidence")), quote_rows_id=identity(raw.get("market_rows")),
        annual_sources=[dict(source_id=identity(row), content_sha256=row.get("content_sha256"),
            revision_id=row.get("revision_id"), available_at=row.get("available_at"))
            for row in annuals if isinstance(row, Mapping)] if isinstance(annuals, (list, tuple)) else [])


def _select_comparables(context, *, symbol, market, cutoff, eps_basis, fiscal_year, scope, annual_source=None,
                        expected_session=None, current_price=None, components=False, quarantine_peers=False):
    """Derive PE operands from full retained proofs, never accept naked medians.

    Missing/inadequate input uses R1; a supplied contradictory or malformed
    source is UNKNOWN. Future annual originals are omitted at each month, not
    repaired by assigning their latest EPS to an earlier quote.
    """
    insufficient = lambda reason: dict(status="INSUFFICIENT", reasons=[reason])
    if context is None:
        return insufficient("R10_COMPARABLE_CONTEXT_MISSING")
    try:
        if (not isinstance(context, Mapping) or context.get("schema") != CONTEXT_SCHEMA
                or context.get("symbol") != symbol or context.get("market") != market
                or timestamp(context["cutoff"]) != timestamp(cutoff)):
            raise ValueError("R10_COMPARABLE_CONTEXT_BINDING_INVALID")
        universe = context["universe_symbols"]
        blocks = [context["issuer"], *context.get("peers", ())]
        symbols = [block["symbol"] for block in blocks]
        excluded = [row["symbol"] for row in context.get("exclusions", ())]
        if (not isinstance(universe, (list, tuple)) or len(universe) != len(set(universe))
                or len(symbols + excluded) != len(set(symbols + excluded))
                or set(symbols + excluded) != set(universe) or symbols[0] != symbol
                or any(not row.get("reason") for row in context.get("exclusions", ()))):
            raise ValueError("R10_COMPARABLE_UNIVERSE_CONFLICT")
        issuer = _block(blocks[0], market=market, cutoff=cutoff)
        if (expected_session is not None and str(issuer["rows"][-1]["Date"])[:10] != expected_session
                or current_price is not None and (_number(current_price) is None
                    or issuer["rows"][-1]["Close"] != current_price)):
            raise ValueError("R10_ISSUER_QUOTE_STRATEGY_BINDING_CONFLICT")
        industry = issuer["industry"]
        if industry not in INDUSTRIES:
            if not components:
                return insufficient("R10_INDUSTRY_PROFILE_NOT_DEFINED")
            if not isinstance(industry, str) or not industry.strip():
                raise ValueError("R14_OFFICIAL_INDUSTRY_IDENTITY_UNVERIFIED")
            # A verified official classification absent from a fixed policy
            # table is not a source contradiction. Continue through every
            # supplied annual/peer proof before permitting explicit fallback.
        latest = _original_at(issuer["annuals"], cutoff)
        actual = _annual(latest, symbol=symbol, market=market, cutoff=cutoff, basis=eps_basis,
                         year=fiscal_year, scope=scope) if latest else None
        if actual is None:
            if components:
                raise ValueError("R14_ISSUER_ANNUAL_PROOF_UNAVAILABLE")
            return insufficient("R10_ISSUER_ANNUAL_PROOF_UNAVAILABLE")
        if annual_source is not None and any(latest.get(key) != annual_source.get(key)
                                            for key in ("content_sha256", "revision_id", "available_at")):
            raise ValueError("R10_ISSUER_VALUATION_ANNUAL_SOURCE_CONFLICT")
        daily = {row["Date"][:7]: row for row in issuer["rows"]}
        current_month = timestamp(cutoff).astimezone(ZoneInfo("Asia/Taipei")).strftime("%Y-%m")
        months = sorted(month for month in daily if month < current_month)[-36:]
        issuer_values, monthly_sources = [], []
        for month in months:
            row = daily[month]
            at = datetime.fromisoformat(str(row["Date"])[:10] + "T13:30:00").replace(tzinfo=ZoneInfo("Asia/Taipei")).isoformat()
            original = _original_at(issuer["annuals"], at)
            eps = _annual(original, symbol=symbol, market=market, cutoff=at, basis=eps_basis, scope=scope) if original else None
            if eps is None:
                continue
            value = _number(row["Close"] / eps["eps"])
            if value is None or value <= 0:
                raise ValueError("R10_ISSUER_PE_NONFINITE")
            issuer_values.append(value)
            monthly_sources.append(dict(month=month, quote_date=str(row["Date"])[:10], annual=eps))
        peer_values, peer_sources, peer_exclusions, peer_quarantine = [], [], [], []
        source_times = [issuer["available_at"], actual["available_at"]]
        session = str(issuer["rows"][-1]["Date"])[:10]
        for raw in blocks[1:]:
            try:
                peer = _block(raw, market=market, cutoff=cutoff)
                if peer["industry"] != industry:
                    peer_exclusions.append(_peer_quarantine_record(raw, "DIFFERENT_OFFICIAL_INDUSTRY") if quarantine_peers
                        else dict(symbol=peer["symbol"], reason="DIFFERENT_OFFICIAL_INDUSTRY")); continue
                original = _original_at(peer["annuals"], cutoff)
                eps = _annual(original, symbol=peer["symbol"], market=market, cutoff=cutoff, basis=eps_basis,
                              year=fiscal_year, scope=scope) if original else None
                if eps is None or str(peer["rows"][-1]["Date"])[:10] != session:
                    peer_exclusions.append(_peer_quarantine_record(raw, "INCOMPARABLE_ANNUAL_OR_QUOTE_SESSION") if quarantine_peers
                        else dict(symbol=peer["symbol"], reason="INCOMPARABLE_ANNUAL_OR_QUOTE_SESSION")); continue
                value = _number(peer["rows"][-1]["Close"] / eps["eps"])
                if value is None or value <= 0:
                    raise ValueError("R10_PEER_PE_NONFINITE")
                peer_values.append(value); peer_sources.append(dict(symbol=peer["symbol"], annual=eps, source_ids=peer["source_ids"]))
                source_times.extend((peer["available_at"], eps["available_at"]))
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
                if not quarantine_peers:
                    raise
                peer_quarantine.append(_peer_quarantine_record(raw, error))
        counts = dict(peer_count=len(peer_values), issuer_month_count=len(issuer_values))
        if components:
            issuer_ready, peer_ready = counts["issuer_month_count"] >= 24, counts["peer_count"] >= 10
            state = "VERIFIED" if issuer_ready and peer_ready else "PARTIAL" if peer_ready else "INSUFFICIENT"
            result = dict(schema="les-v7-comparable-components/v1", status=state,
                reasons=[] if state == "VERIFIED" else ["R14_COMPARABLE_COMPONENT_COVERAGE_INSUFFICIENT"],
                industry=industry, **counts, context_id=identity(context),
                industry_source_status="VERIFIED_OFFICIAL_CLASSIFICATION",
                industry_profile_status="DEFINED" if industry in INDUSTRIES else "NOT_DEFINED",
                issuer_source_ids=issuer["source_ids"], peer_exclusions=peer_exclusions,
                available_at=max(source_times, key=timestamp),
                issuer_history=dict(status="VERIFIED" if issuer_ready else "INSUFFICIENT",
                    count=counts["issuer_month_count"], minimum=24,
                    median=median(issuer_values) if issuer_values else None, sources=monthly_sources),
                peers=dict(status="VERIFIED" if peer_ready else "INSUFFICIENT",
                    count=counts["peer_count"], minimum=10,
                    median=median(peer_values) if peer_values else None, sources=peer_sources))
            if quarantine_peers:
                result.update(peer_quarantine=peer_quarantine, fixed_universe_id=identity(universe),
                    fixed_universe_symbols=list(universe), fixed_exclusions=json.loads(canonical(context.get("exclusions", ()))))
            return result
        if counts["peer_count"] < 10 or counts["issuer_month_count"] < 24:
            return dict(status="INSUFFICIENT", reasons=["R10_COMPARABLE_COVERAGE_INSUFFICIENT"], **counts,
                        peer_exclusions=peer_exclusions)
        return dict(status="VERIFIED", reasons=[], industry=industry,
            issuer_pe_median=median(issuer_values), peer_pe_median=median(peer_values), **counts,
            context_id=identity(context), issuer_source_ids=issuer["source_ids"], monthly_sources=monthly_sources,
            peer_sources=peer_sources, peer_exclusions=peer_exclusions,
            available_at=max(source_times, key=timestamp))
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        return dict(status="UNKNOWN", reasons=[str(error) or "R10_COMPARABLE_SOURCE_UNVERIFIED"])


def select_recovery(rows, *, symbol, market, cutoff, scope):
    """Validate latest same-original YTD amounts; never annualize interim EPS."""
    from domain.les_v7.valuation_eps_receipt import _amount
    from domain.strategies.low_entry_v6_quality_evidence import validated_proof_value, validate_field
    eligible = {}
    try:
        for row in rows:
            period = pd.Period(row["period_end"], freq="Q")
            if timestamp(row["available_at"]) > timestamp(cutoff):
                continue
            if period.quarter == 4:
                continue
            prior = eligible.get(str(period))
            if prior and timestamp(prior["available_at"]) == timestamp(row["available_at"]) and identity(prior) != identity(row):
                raise ValueError("CONFLICTING_R10_RECOVERY_VERSIONS")
            if prior is None or timestamp(prior["available_at"]) < timestamp(row["available_at"]):
                eligible[str(period)] = row
        if not eligible:
            return dict(status="NOT_AVAILABLE", reasons=["R10_RECENT_YTD_NOT_AVAILABLE"])
        key = max(eligible, key=lambda value: pd.Period(value, freq="Q"))
        row, period = eligible[key], pd.Period(key, freq="Q")
        if period.end_time.date() > timestamp(cutoff).astimezone(ZoneInfo("Asia/Taipei")).date():
            raise ValueError("R10_RECOVERY_PERIOD_NOT_COMPLETE_AT_CUTOFF")
        if (row.get("symbol") != symbol or row.get("market") != market or row.get("scope") != scope
                or row.get("source") != "MOPS_IXBRL" or row.get("currency") != "TWD"
                or row.get("unit") != "thousand" or row.get("flow_basis") != "YTD"):
            raise ValueError("R10_RECOVERY_SOURCE_OR_SCOPE_UNVERIFIED")
        values, proofs = {}, {}
        source_times = [row["available_at"]]
        for field in ("revenue", "operating_income", "cfo"):
            if field == "cfo":
                checked = validate_field(row, field, period=period, cutoff=pd.Timestamp(timestamp(cutoff)))
                proof = row.get("field_evidence", {}).get(field, {})
                origin = row.get("quarter_flow_source", row)
                if (origin.get("scope") != scope or origin.get("currency") != "TWD"
                        or origin.get("unit") != "thousand" or origin.get("period_end") != row["period_end"]):
                    raise ValueError("R10_RECENT_CFO_SOURCE_SCOPE_OR_PERIOD_CONFLICT")
                source_times.append(origin["available_at"])
                values["current_" + field] = (validated_proof_value(proof, origin, field,
                    start=f"{period.year}-01-01", end=period.end_time.date().isoformat())
                    if checked["status"] == "VERIFIED_REPORTED" and all(checked["checks"].values()) else None)
                proofs["current_" + field] = proof
            else:
                values["current_" + field], proofs["current_" + field] = _amount(row, field, period, cutoff)
            if values["current_" + field] is None:
                original_proof = row.get("field_evidence", {}).get(field, {})
                if original_proof.get("status") == "RESOLVED" or "CONFLICT" in str(original_proof.get("reason", "")):
                    raise ValueError("R10_RECENT_REPORTED_PROOF_CONFLICT:" + field)
                return dict(status="NOT_AVAILABLE", reasons=["R10_RECENT_FIELD_UNVERIFIED:" + field])
            if field == "cfo":
                continue
            proof = row.get("comparative_ytd_field_evidence", {}).get(field, {})
            previous = period - 4
            parsed = validated_proof_value(proof, row, field, start=f"{previous.year}-01-01",
                                           end=previous.end_time.date().isoformat())
            retained = _number(row.get("comparative_ytd_fields", {}).get(field))
            if parsed is None or retained is None or abs(parsed - retained) > .5:
                if proof.get("status") == "RESOLVED" or "CONFLICT" in str(proof.get("reason", "")):
                    raise ValueError("R10_RECENT_COMPARATIVE_PROOF_CONFLICT:" + field)
                return dict(status="NOT_AVAILABLE", reasons=["R10_RECENT_COMPARATIVE_UNVERIFIED:" + field])
            values["prior_" + field], proofs["prior_" + field] = parsed, proof
        good = all(values[key] > 0 for key in values) and all(values["current_" + field] > values["prior_" + field]
                                                                          for field in ("revenue", "operating_income"))
        return dict(status="VERIFIED" if good else "NO_RECOVERY", reasons=[] if good else ["R10_RECOVERY_CONDITIONS_NOT_MET"],
            values=values, proof_ids={key: identity(proof) for key, proof in proofs.items()},
            source_id=identity(row), available_at=max(source_times, key=timestamp), period=str(period))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        return dict(status="UNKNOWN", reasons=[str(error) or "R10_RECOVERY_SOURCE_UNVERIFIED"])


def evaluate_recovery_price_valuation(*, legacy_result, comparables, recovery=None):
    """Run new arithmetic only after original annual prerequisites and proofs.

    Fallback retains the complete R1 result and exact economics. A missing
    optional recovery operand earns no recovery credit; conflicts stay UNKNOWN.
    """
    fixed, recovery = policy(), recovery or dict(status="NOT_AVAILABLE", reasons=["R10_RECENT_YTD_NOT_AVAILABLE"])
    def safe_json(value):
        if isinstance(value, Mapping):
            return {key: safe_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe_json(item) for item in value]
        return None if isinstance(value, float) and not isfinite(value) else value

    result = deepcopy(legacy_result)
    result.update(schema=fixed["schema"], policy_version=POLICY_VERSION, policy_id=identity(fixed), policy=fixed,
        selected_method="RECOVERY_INDUSTRY_R10", comparables=safe_json(comparables), recovery=safe_json(recovery),
        fallback_policy_id=legacy_result.get("policy_id"), fallback_result=deepcopy(legacy_result),
        bear_metric=fixed["bear_metric"], discounted_bear_value_shortfall=None)
    if comparables.get("status") == "UNKNOWN" or recovery.get("status") == "UNKNOWN":
        result.update(status="UNKNOWN", reasons=comparables.get("reasons", []) + recovery.get("reasons", []),
                      scenarios=None, anchor=None, gate_checks=None, bear_permanent_loss=None)
        return result
    if comparables.get("status") != "VERIFIED":
        result.update(selected_method="ORIGINAL_R1_FALLBACK", fallback_reasons=comparables.get("reasons", []),
                      discounted_bear_value_shortfall=legacy_result.get("bear_permanent_loss"))
        return result
    inputs = legacy_result.get("inputs", {})
    required = ("current_eps", "prior_eps", "current_revenue", "prior_revenue", "current_margin", "prior_margin", "current_price")
    if any(_number(inputs.get(key)) is None or inputs[key] <= 0 for key in required):
        result.update(selected_method="ORIGINAL_R1_PREREQUISITE_FAILURE")
        return result
    industry = comparables.get("industry")
    if industry not in INDUSTRIES or inputs.get("eps_basis") not in {"BASIC", "DILUTED"}:
        result.update(status="UNKNOWN", reasons=["R10_ARITHMETIC_BINDING_INVALID"], scenarios=None)
        return result
    maximum_growth, maximum_pe = INDUSTRIES[industry]
    try:
        issuer_pe, peer_pe = (_number(comparables.get(key)) for key in ("issuer_pe_median", "peer_pe_median"))
        if issuer_pe is None or peer_pe is None or min(issuer_pe, peer_pe) <= 0:
            raise ValueError("R10_COMPARABLE_PE_INVALID")
        with localcontext() as ctx:
            ctx.prec = 34
            values = {key: Decimal(str(inputs[key])) for key in required}
            eps_growth = float(values["current_eps"] / values["prior_eps"] - 1)
            revenue_growth = float(values["current_revenue"] / values["prior_revenue"] - 1)
            annual_growth = min(eps_growth, revenue_growth)
            blended, recent_growth = annual_growth, None
            if recovery.get("status") == "VERIFIED":
                recent = recovery["values"]
                if any(_number(recent.get(key)) is None or recent[key] <= 0 for key in
                       ("current_revenue", "prior_revenue", "current_operating_income", "prior_operating_income", "current_cfo")):
                    raise ValueError("R10_RECOVERY_OPERAND_INVALID")
                recent_growth = min(float(Decimal(str(recent["current_" + field])) / Decimal(str(recent["prior_" + field])) - 1)
                                    for field in ("revenue", "operating_income"))
                if recent_growth <= 0:
                    raise ValueError("R10_RECOVERY_CONDITION_CONFLICT")
                blended = .75 * annual_growth + .25 * recent_growth
            if not all(isfinite(value) for value in (eps_growth, revenue_growth, annual_growth, blended)):
                raise ValueError("R10_NUMERICAL_RESULT_NOT_FINITE")
            growth = min(maximum_growth, max(-.10, blended))
            pe = min(maximum_pe, max(5., .50 * issuer_pe + .50 * peer_pe))
            anchor = values["current_eps"] * Decimal(".9" if inputs["eps_basis"] == "BASIC" else "1")
            base_growth = [.03 + (growth - .03) * .5 ** year for year in range(5)]
            scenarios, exact = {}, {}
            for name, factor, exit_pe, rates in (
                ("BASE", 1., pe, base_growth),
                ("BEAR", .90, max(5., pe * .80), [max(-.10, rate - .05) for rate in base_growth]),
                ("BULL", 1.10, min(maximum_pe * 1.20, pe * 1.20), [min(maximum_growth + .05, rate + .05) for rate in base_growth])):
                terminal = anchor * Decimal(str(factor))
                for rate in rates:
                    terminal *= 1 + Decimal(str(rate))
                future = terminal * Decimal(str(exit_pe)); fair = future / Decimal("1.1") ** 5
                ratio = float(future / values["current_price"])
                if not all(isfinite(float(value)) and value > 0 for value in (terminal, future, fair)) or not isfinite(ratio) or ratio <= 0:
                    raise ValueError("R10_NUMERICAL_RESULT_NOT_FINITE")
                scenarios[name] = dict(years=5, annual_growth=rates, growth=rates[0], eps_factor=factor,
                    exit_pe=exit_pe, terminal_eps=float(terminal), future_price=float(future), fair_price=float(fair),
                    price_cagr=ratio ** .2 - 1)
                exact[name] = fair
            loss = max(Decimal(0), 1 - exact["BEAR"] / values["current_price"])
            checks = dict(BASE_PRICE_CAGR_GTE_REQUIRED_RETURN=exact["BASE"] >= values["current_price"],
                BASE_FAIR_PRICE_GT_CURRENT_PRICE=exact["BASE"] > values["current_price"],
                BEAR_LOSS_LTE_MAXIMUM=loss <= Decimal(".35"))
        result.update(status="PASS" if all(checks.values()) else "FAIL",
            reasons=[key for key, passed in checks.items() if not passed], scenarios=scenarios, gate_checks=checks,
            anchor=dict(current_reported_eps=inputs["current_eps"], prior_reported_eps=inputs["prior_eps"],
                before_basic_haircut=inputs["current_eps"], scenario_anchor=float(anchor), basis=inputs["eps_basis"]),
            observed_growth=dict(eps=eps_growth, revenue=revenue_growth, annual=annual_growth,
                recent_recovery=recent_growth, blended=blended, bounded_first_year=growth),
            bear_permanent_loss=float(loss), discounted_bear_value_shortfall=float(loss))
    except (KeyError, ValueError, TypeError, OverflowError, ArithmeticError) as error:
        result.update(status="UNKNOWN", reasons=[str(error) or "R10_NUMERICAL_RESULT_NOT_FINITE"],
                      scenarios=None, gate_checks=None, anchor=None, bear_permanent_loss=None)
    return result

"""Source-bound daily v7 gates, independent of a model or legacy PASS flags.

These receipts describe completed-day research, never live order authority.
An absent source time/finality/scope is unknown; numeric OHLC alone is not a
point-in-time publication proof. Native policy callers opt in explicitly.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from functools import lru_cache
import json
from math import isfinite

import pandas as pd

from domain.les_v7.contracts import FEATURE_VERSION, GateReceipt, ReferencePlan, canonical, identity, timestamp
from domain.les_v7.decision import make_plan, vetoes
from domain.les_v7.features import compute_features, select_bars
from domain.les_v7.policy_r13 import PLAN_VERSION as R13_PLAN_VERSION, risk_limits
from domain.les_v7.policy_r14 import PLAN_VERSION as R14_PLAN_VERSION
from domain.les_v7.policy_r15 import PLAN_VERSION as R15_PLAN_VERSION
from domain.les_v7.policy_r16 import PLAN_VERSION as R16_PLAN_VERSION
from domain.les_v7.policy_r17 import PLAN_VERSION as R17_PLAN_VERSION
from domain.les_v7.policy_r18 import PLAN_VERSION as R18_PLAN_VERSION
from domain.les_v7.policy_r19 import PLAN_VERSION as R19_PLAN_VERSION
from domain.strategies.low_entry_v6_quality_evidence import validate_field
from domain.trading_restrictions import restriction_check
from domain.valuation_v5 import LowEntryV5ValuationProducer, ValuationProductionStatus


SOURCE_SCHEMA = "les-v7-native-source/v1"
PRODUCER_VERSION = "LES_V7_NATIVE_GATES_R1"
SOURCE_PRODUCER_VERSION = "LES_V7_NATIVE_SOURCE_GATES_R2"
SCOPED_SOURCE_PRODUCER_VERSION = "LES_V7_NATIVE_SOURCE_GATES_R3_SCOPED_PE"
CLOSE_SAFETY_PRODUCER_VERSION = "LES_V7_NATIVE_CLOSE_SAFETY_R3"
R13_STRESS_PRODUCER_VERSION = "LES_V7_NATIVE_STRESS_RISK_R13"
R14_STRESS_PRODUCER_VERSION = "LES_V7_NATIVE_STRESS_RISK_R14"
R15_STRESS_PRODUCER_VERSION = "LES_V7_NATIVE_STRESS_RISK_R15"
REQUIRED_GATES = ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK")
SAFETY_FIELDS = ("support", "close", "open", "clv", "vr", "previous_close", "slope5", "slope20",
                 "atr_expansion", "return5", "atr", "reclaim", "new_lows3")
VALUATION_FIELDS = ("revenue", "operating_income", "net_income", "diluted_shares", "dividends")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        return None
    return float(value)


def _receipt(gate, state, *, binding, source, available, cutoff, reasons, result,
             producer_version=PRODUCER_VERSION):
    """Content IDs identify actual operands and the reproducible check, not signatures."""
    at = timestamp(available)
    if at > timestamp(cutoff):
        return None
    source_id = identity(source)
    proof = dict(producer=producer_version, gate_id=gate, source_id=source_id,
                 input_hash=identity(binding), result=result)
    return dict(gate_id=gate, gate_version=producer_version, state=state,
        input_hash=identity(binding), source_available_at=at.isoformat(),
        checked_at=timestamp(cutoff).isoformat(), reason_codes=sorted(set(reasons)),
        proof_record_ids=[identity(proof)], input_ids=[source_id])


def _market_source(packet, *, symbol, market, cutoff, rows):
    """Only explicit captured availability/finality may certify the source port."""
    source = packet.get("market_evidence", {})
    if not isinstance(source, Mapping) or not rows:
        return None
    try:
        available = timestamp(source["source_available_at"])
        expected = packet.get("expected_session")
        session = str(rows[-1]["Date"])[:10]
        if (source.get("symbol") != symbol or source.get("market") != market
                or source.get("session_date") != session or expected != session
                or source.get("source") not in {"TWSE", "TWSE_DAILY", "TWSE_OPENAPI", "TPEX", "TPEX_DAILY", "TPEX_OPENAPI"}
                or not source["source"].startswith(market)
                or source.get("price_unit") != "TWD" or source.get("volume_unit") != "shares"
                or source.get("amount_unit") != "TWD" or source.get("source_final") is not True
                or source.get("bars_sha256") != identity(rows) or available > timestamp(cutoff)
                or not isinstance(source.get("source_record_ids"), (list, tuple))
                or identity(rows) not in source["source_record_ids"]):
            return None
        if any("Available At" in row and timestamp(row["Available At"]) > timestamp(cutoff) for row in rows):
            return None
        if packet.get("historical_replay") is True and source.get("availability_basis") not in {
                "ORIGINAL_OBSERVATION", "VERIFIED_HISTORICAL_PUBLICATION"}:
            return None
        return source
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _restriction(packet, *, symbol, market, cutoff, session, binding):
    raw = packet.get("restriction_evidence", {})
    checked = restriction_check(raw, market_data_date=session, decision_time=cutoff,
        market=market, symbol=symbol, allow_historical=packet.get("historical_replay") is True)
    if checked["status"] == "UNKNOWN":
        # A captured incomplete current response may support an UNKNOWN
        # diagnostic, never clearance. Absent/future timestamps have no receipt.
        try:
            available = raw["available_at"]
            timestamp(available)
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        return _receipt("TRADING_STATUS", "UNKNOWN", binding=binding, source=raw,
            available=available, cutoff=cutoff, reasons=[checked["reason"]], result=checked)
    if raw.get("scope") == "HISTORICAL_ANNOUNCEMENTS":
        # Historical availability comes from verified publication/coverage
        # evidence, not today's download clock or a guessed past timestamp.
        times = [row["verified_at"] for row in raw.get("coverage", ())
                 if row.get("status") == "VERIFIED"]
        for row in raw.get("announcements", ()):
            from domain.restriction_announcements import publication_bounds
            _, upper = publication_bounds(row)
            if upper <= pd.Timestamp(timestamp(cutoff)):
                times.append(upper.isoformat())
        if not times:
            return None
        available = max(times, key=timestamp)
    else:
        available = raw["available_at"]
    return _receipt("TRADING_STATUS", "FAIL" if checked["restricted"] else "PASS",
        binding=binding, source=raw, available=available, cutoff=cutoff,
        reasons=[checked["reason"]], result=checked)


def _verified_shares(row, period, cutoff):
    """Weighted-average annual share count needs its own share-unit context."""
    proof = row.get("field_evidence", {}).get("diluted_shares", {})
    value = _number(row.get("diluted_shares"))
    if (value is None or value <= 0 or proof.get("status") != "RESOLVED"
            or proof.get("unit") != "shares" or proof.get("basis") not in {"YTD", "ANNUAL"}
            or proof.get("start_date") != f"{period.year}-01-01"
            or proof.get("end_date") != period.end_time.date().isoformat()
            or any(proof.get(k) != row.get(k) for k in
                   ("symbol", "source", "content_sha256", "revision_id", "scope", "currency", "available_at"))):
        return None
    try:
        if timestamp(proof["available_at"]) > timestamp(cutoff):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    contexts = [node for node in proof.get("contexts", ())
        if not node.get("dimensions") and node.get("start_date") == proof["start_date"]
        and node.get("end_date") == proof["end_date"]]
    if not contexts or any(node.get("unit") != "xbrli:shares" or _number(node.get("numeric_value")) != value for node in contexts):
        return None
    return value


def _valuation(packet, *, symbol, market, cutoff, features, quality, bars, binding, market_source,
               assessment_sink=None):
    """Run the existing scenario producer only on independently verified originals.

    The ordinary v5 valuation sidecar remains unchanged. This v7 port does not
    import its PASS string, unverified TTM supplements or missing dividends as
    zero. Incomplete annual share/dividend evidence therefore remains UNKNOWN.
    """
    originals = {}

    def unknown(reason):
        if assessment_sink is not None:
            assessment_sink.update(state="UNKNOWN", reason=reason)
        if market_source is None:
            return None
        known = [row["available_at"] for row in originals.values()]
        available = max([market_source["source_available_at"], *known], key=timestamp)
        return _receipt("VALUATION", "UNKNOWN", binding=binding,
            source=dict(market_source=market_source, selected_annual_originals=originals),
            available=available, cutoff=cutoff, reasons=[reason], result=dict(reason=reason))

    for row in packet.get("valuation_filings", ()):
        try:
            period = pd.Period(row["period_end"], freq="Q")
            if period.quarter != 4 or timestamp(row["available_at"]) > timestamp(cutoff):
                continue
            prior = originals.get(period.year)
            if prior and timestamp(row["available_at"]) == timestamp(prior["available_at"]) and identity(row) != identity(prior):
                return unknown("CONFLICTING_ANNUAL_VALUATION_VERSIONS")
            if prior is None or timestamp(row["available_at"]) > timestamp(prior["available_at"]):
                originals[period.year] = row
        except (KeyError, TypeError, ValueError, AttributeError):
            return unknown("ANNUAL_VALUATION_AVAILABILITY_UNVERIFIED")
    years = sorted(originals)[-5:]
    if len(years) < 2 or years != list(range(years[0], years[-1] + 1)):
        return unknown("COMPARABLE_ANNUAL_VALUATION_HISTORY_MISSING")
    annual_rows = []
    for year in years:
        row, period = originals[year], pd.Period(f"{year}Q4", freq="Q")
        if (row.get("symbol") != symbol or row.get("market") != market
                or row.get("source") not in {"MOPS_IXBRL", "MOPS_XML"}
                or row.get("scope") not in {"consolidated", "individual"}
                or row.get("flow_basis") not in {"YTD", "ANNUAL"}):
            return unknown("ANNUAL_VALUATION_SOURCE_OR_SCOPE_UNVERIFIED")
        values = {}
        for field in ("revenue", "operating_income", "net_income", "dividends"):
            check = validate_field(row, field, period=period, cutoff=pd.Timestamp(timestamp(cutoff)))
            if (check["evidence_level"] != "CANONICAL_ORIGINAL" or check["value"] is None
                    or not all(check["checks"].values())):
                return unknown("ANNUAL_VALUATION_FIELD_UNVERIFIED:" + field)
            values[field] = check["value"]
        shares = _verified_shares(row, period, cutoff)
        if shares is None or values["revenue"] <= 0:
            return unknown("ANNUAL_VALUATION_SHARES_OR_REVENUE_UNVERIFIED")
        annual_rows.append(dict(fiscal_year=year, available_at=row["available_at"],
            net_income=values["net_income"], revenue=values["revenue"],
            operating_margin=values["operating_income"] / values["revenue"],
            dividends_paid=values["dividends"], diluted_shares=shares / 1000.,
            source_id=identity(row), scope=row["scope"]))
    if (not annual_rows or len({row["scope"] for row in annual_rows}) != 1
            or quality.get("core_inputs_pass") is not True or market_source is None):
        return unknown("ANNUAL_VALUATION_COMPARABILITY_OR_QUALITY_UNVERIFIED")
    pe = pd.to_numeric(bars.get("PE", pd.Series(dtype=float)), errors="coerce")
    median = pe.where(pe > 0).rolling(60, min_periods=20).median().iloc[-1] if len(pe) else None
    if median is None or pd.isna(median):
        median = pe.iloc[-1] if len(pe) else None
    result = LowEntryV5ValuationProducer().produce(pd.DataFrame(annual_rows), {},
        signal_date=cutoff, current_price=features.get("close"), recent_pe_median=median)
    if assessment_sink is not None:
        assessment_sink.update(state="COMPLETE" if result.status is ValuationProductionStatus.COMPLETE else "UNKNOWN",
            snapshot=result.as_snapshot_fields(), selected_annual_ids=[row["source_id"] for row in annual_rows])
    if result.status is ValuationProductionStatus.UNKNOWN:
        return unknown("VALUATION_SCENARIO_INPUTS_UNKNOWN:" + ";".join(result.reasons))
    available = max([market_source["source_available_at"], *[row["available_at"] for row in annual_rows]], key=timestamp)
    return _receipt("VALUATION", "PASS" if result.gate_passed else "FAIL", binding=binding,
        source=dict(market_source=market_source, annual_rows=annual_rows,
                    quality_result_id=identity(quality), recent_pe_median=float(median)),
        available=available, cutoff=cutoff, reasons=list(result.reasons), result=result.as_snapshot_fields())


def produce_native_receipts(*, symbol, market, cutoff, bars, features, quality, plan, risks, bindings,
                            source_packet, reference_policy, valuation_policy=None, valuation_input_policy=None,
                            source_policy=None, feature_version=FEATURE_VERSION):
    """Five pure producers; sources are mandatory and legacy receipts are not read."""
    if source_packet.get("schema") != SOURCE_SCHEMA:
        raise ValueError("V7_NATIVE_EVIDENCE_SOURCE_SCHEMA_REQUIRED")
    from domain.les_v7.features import CLOSE_SUPPORT_FEATURE_VERSION
    if feature_version not in {FEATURE_VERSION, CLOSE_SUPPORT_FEATURE_VERSION}:
        raise ValueError("V7_NATIVE_FEATURE_VERSION_MISMATCH")
    if valuation_input_policy is not None and valuation_policy is None:
        raise ValueError("V7_VALUATION_INPUT_POLICY_REQUIRES_PRICE_METHOD")
    if any(source_packet.get(key) != expected for key, expected in
           (("symbol", symbol), ("market", market), ("cutoff", timestamp(cutoff).isoformat()))):
        raise ValueError("V7_NATIVE_EVIDENCE_SOURCE_BINDING_MISMATCH")
    if source_packet.get("financial_capture_cutoff") is not None and (
            source_packet.get("historical_replay") is True
            or timestamp(source_packet["financial_capture_cutoff"]) > timestamp(cutoff)):
        raise ValueError("V7_NATIVE_FINANCIAL_CAPTURE_CUTOFF_MISMATCH")
    rows = source_packet.get("market_rows", ())
    actual = bars.astype(object).where(pd.notna(bars), None).to_dict("records")
    for row in actual:
        row["Date"] = pd.Timestamp(row["Date"]).date().isoformat()
    if identity(rows) != identity(actual) or bindings["DATA_INTEGRITY"]["bars_hash"] != identity(actual):
        raise ValueError("V7_NATIVE_MARKET_ROWS_BINDING_MISMATCH")
    receipts = dict.fromkeys(REQUIRED_GATES)
    security = None
    join_diagnostics = []
    scoped_source = False
    if source_policy is not None:
        from domain.les_v7.source_policy import require_source_policy, SCOPED_VERSION, AUDIT_SOURCE_VERSION
        from domain.les_v7.source_projection import validate_source_projection
        require_source_policy(source_policy, cutoff=cutoff)
        scoped_source = source_policy["policy_version"] in {SCOPED_VERSION, AUDIT_SOURCE_VERSION}
        security, join_diagnostics = validate_source_projection(source_packet,
            symbol=symbol, market=market, cutoff=cutoff, source_policy=source_policy)
    source = _market_source(source_packet, symbol=symbol, market=market, cutoff=cutoff, rows=actual)
    session = str(actual[-1]["Date"])[:10] if actual else ""
    receipts["TRADING_STATUS"] = _restriction(source_packet, symbol=symbol, market=market,
        cutoff=cutoff, session=session, binding=bindings["TRADING_STATUS"])
    if source is not None:
        # Reuse the native validator; no weekday-derived calendar or silent
        # stale-stock fallback is introduced by the receipt layer.
        validated = select_bars(bars, cutoff)
        ready = len(validated) >= 61 and len(validated) == len(bars)
        receipts["DATA_INTEGRITY"] = _receipt("DATA_INTEGRITY", "PASS" if ready else "UNKNOWN",
            binding=bindings["DATA_INTEGRITY"], source=source, available=source["source_available_at"],
            cutoff=cutoff, reasons=[] if ready else ["INCOMPLETE_TECHNICAL_HISTORY"],
            result=dict(validated_bars=len(validated), cutoff_session=session, minimum_base_history=61))
        if source_policy is not None:
            identity_ready = security["status"] == "VERIFIED" and security["issuer_symbol"] == symbol
            blocking = ([node for node in join_diagnostics if node["blocks_data_integrity"]]
                        if scoped_source else join_diagnostics)
            state = ("FAIL" if security["status"] == "OUT_OF_SCOPE" else
                     "PASS" if ready and identity_ready and not blocking else "UNKNOWN")
            available = max(source["source_available_at"], source_policy["available_at"],
                security.get("master", {}).get("available_at", source["source_available_at"]), key=timestamp)
            if scoped_source:
                available=max([available,*[node["source_available_at"] for node in blocking
                    if node.get("source_available_at") is not None]],key=timestamp)
            reasons = ([] if state == "PASS" else
                ["SECURITY_OUT_OF_SCOPE"] if state == "FAIL" else
                ([security["reason"]] if not identity_ready else [])
                + [node["reason"] for node in blocking]
                + ([] if ready else ["INCOMPLETE_TECHNICAL_HISTORY"]))
            receipts["DATA_INTEGRITY"] = _receipt("DATA_INTEGRITY", state,
                binding=bindings["DATA_INTEGRITY"], source=dict(market_source=source,
                    security_selection=security, source_policy=source_policy, join_diagnostics=join_diagnostics),
                available=available, cutoff=cutoff, reasons=reasons,
                result=dict(validated_bars=len(validated), security_status=security["status"],
                    issuer_symbol=security.get("issuer_symbol"), source_projection_conflicts=join_diagnostics,
                    **(dict(blocking_source_conflicts=blocking,
                        quarantined_component_conflicts=[node for node in join_diagnostics if not node["blocks_data_integrity"]],
                        pe_source_state="UNKNOWN_QUARANTINED" if join_diagnostics else "NO_CONFLICT_OBSERVED")
                       if scoped_source else {})),
                producer_version=SCOPED_SOURCE_PRODUCER_VERSION if scoped_source else SOURCE_PRODUCER_VERSION)
        known = all(features.get(key) is not None for key in SAFETY_FIELDS)
        computed_risks = tuple(vetoes(features))
        if tuple(risks) != computed_risks:
            raise ValueError("V7_NATIVE_SAFETY_RISK_MISMATCH")
        state = "FAIL" if computed_risks else "PASS" if known else "UNKNOWN"
        receipts["SAFETY"] = _receipt("SAFETY", state, binding=bindings["SAFETY"],
            source=dict(market_source=source, features={k: features.get(k) for k in SAFETY_FIELDS},
                **(dict(feature_version=feature_version) if feature_version != FEATURE_VERSION else {})),
            available=source["source_available_at"], cutoff=cutoff,
            reasons=list(computed_risks) if computed_risks else [] if known else ["VETO_FEATURES_UNKNOWN"],
            result=dict(vetoes=computed_risks, complete=known),
            producer_version=CLOSE_SAFETY_PRODUCER_VERSION if feature_version != FEATURE_VERSION else PRODUCER_VERSION)
        if plan is not None and (reference_policy or {}).get("version") not in {R18_PLAN_VERSION, R19_PLAN_VERSION}:
            from domain.les_v7.plan_r17 import plan_from_record
            reference = plan_from_record(plan)
            r14 = (reference_policy or {}).get("version") == R14_PLAN_VERSION
            r16 = (reference_policy or {}).get("version") == R16_PLAN_VERSION
            r17 = (reference_policy or {}).get("version") == R17_PLAN_VERSION
            r15 = (reference_policy or {}).get("version") in {R15_PLAN_VERSION, R16_PLAN_VERSION, R17_PLAN_VERSION}
            r13 = (reference_policy or {}).get("version") in {R13_PLAN_VERSION, R14_PLAN_VERSION, R15_PLAN_VERSION, R16_PLAN_VERSION, R17_PLAN_VERSION}
            r10 = (reference_policy or {}).get("version") in {"V7_SETUP_STRUCTURE_H20_R10", R13_PLAN_VERSION, R14_PLAN_VERSION, R15_PLAN_VERSION, R16_PLAN_VERSION, R17_PLAN_VERSION}
            calendar_coverage = None
            if r16 or r17:
                from domain.les_v7.plan_r16 import make_plan as conditional_plan
                if r17:
                    from domain.les_v7.plan_r17 import make_plan as conditional_plan
                causal, causal_evidence = conditional_plan(reference.context, features, reference_policy,
                    reference.evidence_id, bars=bars, cutoff=cutoff,
                    session_evidence=source_packet.get("calendar_evidence"))
                calendar_coverage = causal_evidence.get("short_base_audit", {}).get("session_coverage")
            else:
                causal = make_plan(reference.context, features, reference_policy, reference.evidence_id,
                    **(dict(bars=bars, cutoff=cutoff) if r10 else {}))
            if causal is None or identity(asdict(causal)) != identity(plan):
                raise ValueError("V7_NATIVE_REFERENCE_PLAN_MISMATCH")
            spread, adv = _number(features.get("spread")), _number(features.get("adv"))
            participation = reference.entry * reference.reference_shares / adv if adv is not None and adv > 0 else None
            rr = reference.net_rr
            rr_min = (risk_limits()["minimum_net_rr"].get(reference.context) if r13 else
                      2. if reference.context == "REVERSAL" else 1.8 if reference.context == "TREND_PULLBACK"
                      else _number((reference_policy or {}).get("leadership_rr_min")))
            known = spread is not None and participation is not None and rr is not None and rr_min is not None
            if r13:
                from domain.les_v7.plan_r10 import evaluate_entry
                close_check = evaluate_entry(reference, reference.entry, reference_policy, **(dict(starter=True) if r15 else {}))
                failed = close_check["state"] == "FAIL" or (spread is not None and spread > .005) or (
                    participation is not None and participation > .01)
                known = known and close_check["state"] == "PASS"
            else:
                failed = reference.stop_atr > 1.5 or (spread is not None and spread > .005) or (
                    participation is not None and participation > .01) or (rr is not None and rr_min is not None and rr < rr_min)
            if r10:
                from domain.les_v7.plan_r10 import validate_band
                band_check = validate_band(reference, reference_policy, **(dict(starter=True) if r15 else {}))
                failed = failed or band_check["state"] == "FAIL"
                known = known and band_check["state"] == "PASS"
            state = "FAIL" if failed else "PASS" if known else "UNKNOWN"
            risk_source = dict(market_source=source, plan=plan, spread=spread, adv=adv)
            if r13:
                risk_source.update(reference_policy_id=identity(reference_policy),
                    structural_rules=reference_policy["structural_rules"])
            calendar_times = []
            if calendar_coverage is not None:
                risk_source["r17_session_coverage" if r17 else "r16_session_coverage"] = calendar_coverage
                calendar_times = [row["available_at"] for row in calendar_coverage.get("selected_publications", [])]
            route_checks = {}
            if r15:
                route_checks = dict(core=dict(close=evaluate_entry(reference, reference.entry, reference_policy),
                    band=validate_band(reference, reference_policy)),
                    tactical=dict(close=close_check, band=band_check),
                    pass_scope="TACTICAL_SMALL_MINIMUM_NOT_CORE_BUY_AUTHORIZATION")
                risk_source["r15_route_checks"] = route_checks
                rr_min = reference_policy["structural_rules"]["starter_minimum_net_rr"][reference.context]
            receipts["STRESS_RISK"] = _receipt("STRESS_RISK", state, binding=bindings["STRESS_RISK"],
                source=risk_source,
                available=(max([source["source_available_at"], reference_policy["available_at"], *calendar_times], key=timestamp)
                           if r13 else source["source_available_at"]), cutoff=cutoff,
                reasons=["REFERENCE_RISK_FAIL"] if failed else [] if known else ["REFERENCE_RISK_UNKNOWN"],
                result=dict(stop_atr=reference.stop_atr, net_rr=rr, rr_min=rr_min,
                            spread=spread, participation=participation,
                            **(dict(r15_route_checks=route_checks) if r15 else {})),
                **(dict(producer_version="LES_V7_NATIVE_STRESS_RISK_R17" if r17 else "LES_V7_NATIVE_STRESS_RISK_R16" if r16 else R15_STRESS_PRODUCER_VERSION if r15 else R14_STRESS_PRODUCER_VERSION if r14 else R13_STRESS_PRODUCER_VERSION) if r13 else {}))
        if plan is not None and (reference_policy or {}).get("version") in {R18_PLAN_VERSION, R19_PLAN_VERSION}:
            from domain.les_v7.plan_r18 import plan_from_record, make_plan as limit_plan
            from domain.les_v7.risk_r18 import assess_execution
            r19 = reference_policy["version"] == R19_PLAN_VERSION
            if r19:
                from domain.les_v7.plan_r19 import plan_from_record, make_plan as limit_plan
                from domain.les_v7.risk_r19 import assess_execution
            reference = plan_from_record(plan)
            causal, causal_evidence = limit_plan(reference.context, features, reference_policy,
                reference.evidence_id, bars=bars, cutoff=cutoff,
                session_evidence=source_packet.get("calendar_evidence"))
            if causal is None or identity(asdict(causal)) != identity(plan):
                raise ValueError("V7_NATIVE_REFERENCE_PLAN_MISMATCH")
            execution = assess_execution(reference, reference_policy, features)
            coverage = causal_evidence.get("short_base_audit", {}).get("session_coverage")
            times = [row["available_at"] for row in (coverage or {}).get("selected_publications", [])]
            state = execution["state"]
            risk_source = dict(market_source=source, plan=plan,
                reference_policy_id=identity(reference_policy), execution_assessment=execution,
                r18_session_coverage=coverage)
            receipt = _receipt("STRESS_RISK", state, binding=bindings["STRESS_RISK"], source=risk_source,
                available=max([source["source_available_at"], reference_policy["available_at"], *times], key=timestamp),
                cutoff=cutoff, reasons=["REFERENCE_RISK_FAIL"] if state == "FAIL" else
                    [] if state == "PASS" else ["REFERENCE_RISK_UNKNOWN"],
                result=execution, producer_version="LES_V7_NATIVE_STRESS_RISK_R19" if r19 else "LES_V7_NATIVE_STRESS_RISK_R18")
            if receipt is not None:
                receipts["STRESS_RISK"] = dict(receipt, execution_assessment=execution)
    if source_policy is not None and security["status"] != "VERIFIED":
        if source is not None:
            receipts["VALUATION"] = _receipt("VALUATION", "UNKNOWN", binding=bindings["VALUATION"],
                source=dict(security=security, market_source=source, source_policy=source_policy),
                available=max(source["source_available_at"], source_policy["available_at"], key=timestamp),
                cutoff=cutoff, reasons=["ANNUAL_VALUATION_SECURITY_" + security["status"]],
                result=dict(status="UNKNOWN", reason="COMMON_EQUITY_ISSUER_BINDING_REQUIRED"),
                producer_version=SOURCE_PRODUCER_VERSION)
    elif valuation_policy is not None:
        from domain.les_v7.valuation_eps_receipt import produce_price_valuation_receipt
        receipts["VALUATION"] = produce_price_valuation_receipt(source_packet,
            symbol=symbol, market=market, cutoff=cutoff, features=features, quality=quality,
            bars=bars, binding=bindings["VALUATION"], market_source=source, policy=valuation_policy,
            **({"input_policy": valuation_input_policy} if valuation_input_policy is not None else {}))
    else:
        receipts["VALUATION"] = _valuation(source_packet, symbol=symbol, market=market, cutoff=cutoff,
            features=features, quality=quality, bars=bars, binding=bindings["VALUATION"], market_source=source)
    if scoped_source and source is not None:
        from domain.les_v7.source_projection import scoped_pe_inputs
        pe_inputs=scoped_pe_inputs(actual)
        # The legacy cash/share method consumes this P/E window. A verified
        # EPS-price method validates its own multiples and annual sources;
        # an unrelated P/E gap must not become a new global entry veto.
        if pe_inputs["state"] != "PASS" and valuation_policy is None:
            raw_valuation=receipts["VALUATION"]
            clocks=[source["source_available_at"],source_policy["available_at"],
                *([raw_valuation["source_available_at"]] if raw_valuation else []),
                *[row["source_available_at"] for row in source_packet.get("official_pe", ())
                  if row.get("source_available_at") is not None]]
            receipt=_receipt("VALUATION","UNKNOWN",binding=bindings["VALUATION"],
                source=dict(market_source=source,source_policy=source_policy,
                    pe_input_evidence=pe_inputs,source_projection_conflicts=join_diagnostics,
                    raw_valuation_receipt_id=identity(raw_valuation)),
                available=max(clocks,key=timestamp),cutoff=cutoff,
                reasons=pe_inputs["reasons"],result=dict(state="UNKNOWN",pe_input_evidence=pe_inputs,
                    raw_valuation_state=raw_valuation.get("state") if raw_valuation else None),
                producer_version=SCOPED_SOURCE_PRODUCER_VERSION)
            if receipt is not None:
                receipt["valuation_diagnostics"]=dict(status="UNKNOWN",reasons=pe_inputs["reasons"],
                    pe_input_evidence=pe_inputs,
                    raw_valuation_diagnostics_id=identity(raw_valuation.get("valuation_diagnostics") if raw_valuation else None),
                    raw_valuation_receipt=raw_valuation)
            receipts["VALUATION"]=receipt
        elif valuation_policy is not None and receipts["VALUATION"] is not None:
            receipts["VALUATION"].setdefault("valuation_diagnostics", {})["quote_pe_component"] = dict(
                pe_inputs, affects_independent_eps_gate=False,
                dependency_policy="INDEPENDENT_EPS_METHOD_OWNS_ITS_VERIFIED_INPUTS")
    return receipts


@lru_cache(maxsize=16)
def _validated_features(rows_json, cutoff, feature_version):
    """Bounded pure preprocessing cache; immutable JSON values never escape."""
    from domain.les_v7.features import CLOSE_SUPPORT_FEATURE_VERSION
    if feature_version not in {FEATURE_VERSION, CLOSE_SUPPORT_FEATURE_VERSION}:
        raise ValueError("V7_NATIVE_FEATURE_VERSION_MISMATCH")
    bars = select_bars(pd.DataFrame(json.loads(rows_json)), cutoff)
    features, _ = compute_features(bars,
        **(dict(feature_version=feature_version) if feature_version != FEATURE_VERSION else {}))
    return canonical(features)


def validate_native_receipts(report):
    """Recalculate all native receipt checks; a rehashed PASS is not accepted."""
    diagnostic = report["diagnostics"]
    native = diagnostic.get("native_evidence")
    if native is None:
        return
    packet = native.get("source_packet", {})
    if native.get("schema") != "les-v7-native-evidence/v1" or native.get("source_id") != identity(packet):
        raise ValueError("V7_NATIVE_SOURCE_ID_MISMATCH")
    event = report["envelope"]["event"]
    baseline = diagnostic.get("rule_baseline")
    feature_version = baseline["policy"].get("native_feature_version", FEATURE_VERSION) if baseline else FEATURE_VERSION
    expected_features = _validated_features(canonical(packet.get("market_rows", ())),
        event["cutoff_at"], feature_version)
    if expected_features != canonical(diagnostic["features"]):
        raise ValueError("V7_NATIVE_FEATURE_RECOMPUTATION_MISMATCH")
    from domain.les_v7.execution_profile import validate_execution_profile, execution_feature_provider
    partition = validate_execution_profile(diagnostic.get("model_input_profile", {}))
    if partition is not None:
        from domain.les_v7.contracts import ReferencePlan
        seed = diagnostic.get("reference_plan")
        plan = ReferencePlan(**dict(seed, band=tuple(seed["band"]))) if seed else None
        provider = execution_feature_provider(diagnostic["model_input_profile"])
        extension, audit = provider.compute_execution_features(pd.DataFrame(packet.get("market_rows", ())),
            cutoff=event["cutoff_at"], base_features=diagnostic["features"], plan=plan)
        expected_extension = {name: extension[name] for name in partition[1]}
        if (canonical(expected_extension) != canonical(diagnostic.get("execution_features"))
                or canonical(audit) != canonical(diagnostic.get("execution_feature_audit"))):
            raise ValueError("V7_NATIVE_EXECUTION_FEATURE_RECOMPUTATION_MISMATCH")
    elif any(key in diagnostic for key in ("execution_features", "execution_feature_audit")):
        raise ValueError("V7_NATIVE_EXECUTION_FEATURE_PROFILE_REQUIRED")
    if feature_version != FEATURE_VERSION and diagnostic["feature_audit"].get("profile") != feature_version:
        raise ValueError("V7_NATIVE_FEATURE_PROFILE_BINDING_MISMATCH")
    if feature_version != FEATURE_VERSION:
        from domain.les_v7.features import close_support_policy
        if identity(diagnostic["feature_audit"].get("support_policy")) != identity(close_support_policy()):
            raise ValueError("V7_NATIVE_SUPPORT_POLICY_BINDING_MISMATCH")
    expected = produce_native_receipts(symbol=report["symbol"], market=packet.get("market"),
        cutoff=event["cutoff_at"], bars=pd.DataFrame(packet.get("market_rows", ())),
        features=diagnostic["features"], quality=diagnostic["quality"],
        plan=diagnostic["reference_plan"], risks=vetoes(diagnostic["features"]),
        bindings=diagnostic["gate_bindings"], source_packet=packet,
        reference_policy=native.get("reference_policy"),
        **({"valuation_policy": native["valuation_policy"]} if "valuation_policy" in native else {}),
        **({"valuation_input_policy": native["valuation_input_policy"]} if "valuation_input_policy" in native else {}),
        **({"source_policy": native["source_policy"]} if "source_policy" in native else {}),
        **(dict(feature_version=feature_version) if feature_version != FEATURE_VERSION else {}))
    if identity(expected) != identity(native.get("produced_receipts")):
        raise ValueError("V7_NATIVE_RECEIPT_RECOMPUTATION_MISMATCH")
    checked = [asdict(GateReceipt.bind(expected.get(gate), gate_id=gate,
        inputs=diagnostic["gate_bindings"][gate], cutoff=event["cutoff_at"])) for gate in REQUIRED_GATES]
    if identity(checked) != identity(diagnostic["gate_receipts"]):
        raise ValueError("V7_NATIVE_GATE_PROJECTION_MISMATCH")

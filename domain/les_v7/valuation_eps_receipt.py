"""Source-bound price-only EPS method; old native valuation remains unchanged."""
from __future__ import annotations

from collections.abc import Mapping
from math import isclose

import pandas as pd

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.valuation_earnings import SCHEMA, MAPPING_VERSION, EPS_CONCEPTS
from domain.les_v7.valuation_price import policy as fixed_policy, evaluate_price_valuation
from domain.strategies.low_entry_v6_quality_evidence import validate_field, validated_proof_value
from domain.strategies.low_entry_v6_quality import number


PRODUCER = "LES_V7_NATIVE_EPS_PRICE_GATES_R1"
INDEPENDENT_PRODUCER = "LES_V7_NATIVE_EPS_PRICE_GATES_R2"
RECOVERY_PRODUCER = "LES_V7_NATIVE_RECOVERY_INDUSTRY_PRICE_GATES_R10"
RELAXED_PRODUCER = "LES_V7_NATIVE_ANNUAL_EPS_GRADED_PRICE_GATES_R13"
COMPONENT_PRODUCER = "LES_V7_NATIVE_RECOVERY_COMPONENT_PRICE_GATES_R14"
TACTICAL_PRODUCER = "LES_V7_NATIVE_ROLE_BOUND_RECOVERY_PRICE_GATES_R15"
SOURCE_KEYS = ("symbol", "source", "source_url", "period_end", "available_at", "content_sha256",
               "revision_id", "scope", "currency", "unit")


def _same_original(source, row):
    return (isinstance(source, Mapping) and source.get("schema") == SCHEMA
        and source.get("mapping_version") == MAPPING_VERSION and source.get("status") == "VERIFIED_ORIGINAL"
        and source.get("market") == row.get("market")
        and all(source.get(key) == row.get(key) for key in SOURCE_KEYS))


def _eps(source, field, year, *, comparative=False):
    values, proofs = (("comparative_ytd_fields", "comparative_ytd_field_evidence") if comparative
                      else ("fields", "field_evidence"))
    value = number(source.get(values, {}).get(field))
    proof = source.get(proofs, {}).get(field, {})
    start, end = f"{year}-01-01", f"{year}-12-31"
    if (value is None or proof.get("basis") != "ANNUAL"
            or proof.get("mapping_version") != MAPPING_VERSION):
        return None, proof
    contexts = [c for c in proof.get("contexts", ()) if isinstance(c, Mapping) and not c.get("dimensions")
        and c.get("start_date") == start and c.get("end_date") == end]
    if not contexts or any(c.get("concept") != EPS_CONCEPTS[field] or c.get("instant") is not None for c in contexts):
        return None, proof
    # The shared validator's basic_eps argument selects EPS unit semantics,
    # not a different QName. Exact TOTAL basic/diluted QName is checked above.
    parsed = validated_proof_value(proof, source, "basic_eps", start=start, end=end)
    if parsed is None or not isclose(parsed, value, rel_tol=0., abs_tol=1e-9):
        return None, proof
    # Tolerance establishes consistency, not a substitute for the original.
    # Returning the retained approximation could turn reported zero into
    # positive earnings (or a tiny reported positive amount into zero).
    return parsed, proof


def _unreported_diluted_eps(source, year, *, comparative=False):
    """Require the mapper's complete non-reporting contract, not a reason flag.

    Retained diluted contexts or a resolved value contradict non-reporting.
    A missing/invalid proof cannot authorize the BASIC method either: both
    annual operands must explicitly describe absence in this same original.
    """
    values_key, proofs_key = (("comparative_ytd_fields", "comparative_ytd_field_evidence")
                             if comparative else ("fields", "field_evidence"))
    values, proofs = source.get(values_key), source.get(proofs_key)
    if (not isinstance(values, Mapping) or not isinstance(proofs, Mapping)
            or "diluted_eps" not in values or values["diluted_eps"] is not None):
        return False
    proof = proofs.get("diluted_eps")
    if not isinstance(proof, Mapping):
        return False
    contexts = proof.get("contexts")
    return (proof.get("status") == "UNKNOWN" and proof.get("category") == "MISSING_DATA"
        and proof.get("reason") == "TOTAL_EPS_CONCEPT_NOT_REPORTED"
        and proof.get("mapping_version") == MAPPING_VERSION
        and proof.get("basis") == "ANNUAL" and proof.get("unit") == "TWD/share"
        and proof.get("start_date") == f"{year}-01-01"
        and proof.get("end_date") == f"{year}-12-31"
        and isinstance(contexts, (list, tuple)) and not contexts
        and all(key in proof and key in source and proof[key] == source[key]
                for key in (*[key for key in SOURCE_KEYS if key != "unit"], "revision_number")))


def _amount(row, field, period, cutoff, *, comparative=False):
    if not comparative:
        checked = validate_field(row, field, period=period, cutoff=pd.Timestamp(timestamp(cutoff)))
        if (checked["evidence_level"] != "CANONICAL_ORIGINAL" or not all(checked["checks"].values())
                or checked["status"] != "VERIFIED_REPORTED"):
            return None, checked
        proof = row.get("field_evidence", {}).get(field, {})
        parsed = validated_proof_value(proof, row, field, start=f"{period.year}-01-01",
                                       end=period.end_time.date().isoformat())
        return parsed, checked
    value = number(row.get("comparative_ytd_fields", {}).get(field))
    proof = row.get("comparative_ytd_field_evidence", {}).get(field, {})
    year = period.year - 1
    if value is None or not isinstance(proof, Mapping) or proof.get("basis") not in {"ANNUAL", "YTD"}:
        return None, proof
    parsed = validated_proof_value(proof, row, field, start=f"{year}-01-01", end=f"{year}-12-31")
    if (parsed is None or not isclose(parsed, value, rel_tol=0., abs_tol=.5)
            or any(proof.get(k) != row.get(k) for k in ("symbol", "source", "source_url"))):
        return None, proof
    return parsed, proof


def produce_price_valuation_receipt(packet, *, symbol, market, cutoff, features, quality,
                                    bars, binding, market_source, policy, input_policy=None):
    """Revalidate the latest original and same-original annual comparisons.

    Never substitute a prior original, total income/EPS inferred shares,
    unverified basic EPS after a conflicting diluted fact, or current data at
    an older cutoff. Known financial weakness is FAIL, missing proof UNKNOWN.
    """
    from domain.les_v7.native_evidence import _receipt
    from domain.les_v7.valuation_recovery_r10 import policy as recovery_policy
    from domain.les_v7.valuation_relaxed_r13 import policy as relaxed_policy
    from domain.les_v7.valuation_recovery_r14 import policy as component_policy
    from domain.les_v7.valuation_recovery_r15 import policy as tactical_policy
    from domain.les_v7.fr3_policy import eps_policy as fr3_policy
    fr3_method = identity(policy) == identity(fr3_policy())
    recovery_method = identity(policy) == identity(recovery_policy())
    relaxed_method = identity(policy) == identity(relaxed_policy())
    component_method = identity(policy) == identity(component_policy())
    tactical_method = identity(policy) == identity(tactical_policy())
    structured_method = recovery_method or relaxed_method or component_method or tactical_method
    if identity(policy) != identity(fixed_policy()) and not structured_method and not fr3_method:
        raise ValueError("V7_EPS_PRICE_VALUATION_POLICY_MISMATCH")
    if timestamp(policy["available_at"]) > timestamp(cutoff):
        raise ValueError("V7_EPS_PRICE_VALUATION_NOT_AVAILABLE_AT_CUTOFF")
    source_policy = None
    if input_policy is not None:
        from domain.les_v7.valuation_input_policy import require_input_policy
        source_policy = require_input_policy(input_policy, cutoff=cutoff)
        if (structured_method or fr3_method) and source_policy["valuation_method_policy_id"] != identity(policy):
            raise ValueError("V7_RECOVERY_PRICE_INPUT_POLICY_METHOD_MISMATCH")
    originals = {}

    def finish(state, reasons, *, result=None, source=None):
        if market_source is None: return None
        additional_times = []
        if structured_method and result is not None:
            for key in ("comparables", "recovery"):
                evidence = result.get(key, {})
                statuses = ({"VERIFIED", "PARTIAL", "INSUFFICIENT", "NO_RECOVERY"} if component_method or tactical_method else
                            {"VERIFIED", "NO_RECOVERY"} if relaxed_method else {"VERIFIED"})
                if evidence.get("status") in statuses and evidence.get("available_at"):
                    additional_times.append(evidence["available_at"])
        available = max([market_source["source_available_at"], policy["available_at"],
            *([source_policy["available_at"]] if source_policy else []),
            *[row["available_at"] for row in originals.values()], *additional_times], key=timestamp)
        diagnostic = result or dict(status=state, reasons=list(reasons), policy=policy,
            policy_id=identity(policy), return_basis="PRICE_RETURN_ONLY")
        if source_policy is not None:
            diagnostic = dict(diagnostic, input_policy=source_policy, input_policy_id=identity(source_policy),
                quality_dependency=source_policy["quality_dependency"],
                entry_quality_required=True)
        receipt = _receipt("VALUATION", state, binding=binding,
            source=dict(market_source=market_source, selected_originals=originals, policy=policy,
                        quality_result_id=identity(quality), operands=source,
                        **({"valuation_input_policy": source_policy} if source_policy else {})),
            available=available, cutoff=cutoff, reasons=reasons, result=diagnostic,
            producer_version="FR3_NATIVE_ANNUAL_EPS_PRICE_R2" if fr3_method else TACTICAL_PRODUCER if tactical_method else COMPONENT_PRODUCER if component_method else RELAXED_PRODUCER if relaxed_method else RECOVERY_PRODUCER if recovery_method else INDEPENDENT_PRODUCER if source_policy is not None else PRODUCER)
        if receipt is not None:
            receipt["valuation_diagnostics"] = diagnostic
        return receipt

    for row in packet.get("valuation_filings", ()):
        try:
            period = pd.Period(row["period_end"], freq="Q")
            if timestamp(row["available_at"]) > timestamp(cutoff): continue
            if period.quarter != 4: continue
            prior = originals.get(period.year)
            if prior and timestamp(row["available_at"]) == timestamp(prior["available_at"]) and identity(row) != identity(prior):
                return finish("UNKNOWN", ["CONFLICTING_ANNUAL_EPS_VERSIONS"])
            if prior is None or timestamp(row["available_at"]) > timestamp(prior["available_at"]):
                originals[period.year] = row
        except (TypeError, KeyError, ValueError, AttributeError):
            return finish("UNKNOWN", ["ANNUAL_EPS_AVAILABILITY_UNVERIFIED"])
    if not originals:
        return finish("UNKNOWN", ["ANNUAL_EPS_ORIGINAL_MISSING"])
    year = max(originals)
    row, period = originals[year], pd.Period(f"{year}Q4", freq="Q")
    decision_date = pd.Timestamp(timestamp(cutoff)).tz_convert("Asia/Taipei").date()
    annual_age = (decision_date - period.end_time.date()).days
    if annual_age < 0 or annual_age > policy["maximum_annual_age_days"]:
        return finish("UNKNOWN", ["ANNUAL_EPS_PERIOD_INCOMPLETE_OR_STALE"])
    if (row.get("symbol") != symbol or row.get("market") != market or row.get("source") != "MOPS_IXBRL"
            or row.get("scope") not in {"consolidated", "individual"}
            or row.get("flow_basis") not in {"ANNUAL", "YTD"}
            or row.get("currency") != "TWD" or row.get("unit") != "thousand"
            or not row.get("revision_id") or not row.get("content_sha256") or not row.get("source_url")):
        return finish("UNKNOWN", ["ANNUAL_EPS_SOURCE_OR_SCOPE_UNVERIFIED"])
    earnings = row.get("earnings_source")
    if not _same_original(earnings, row):
        return finish("UNKNOWN", ["ANNUAL_EPS_SOURCE_COMPANION_UNVERIFIED"])
    # Both EPS operands must come from this year's original. This uses the
    # issuer's reported comparative basis, not unrelated old-original values.
    field, basis = "diluted_eps", "DILUTED"
    current, current_proof = _eps(earnings, field, year)
    prior, prior_proof = _eps(earnings, field, year - 1, comparative=True)
    if current is None or prior is None:
        # BASIC is a declared haircut method, not fabricated diluted EPS. A
        # bad unit/conflict/missing comparative is not proof of non-reporting.
        if (not _unreported_diluted_eps(earnings, year)
                or not _unreported_diluted_eps(earnings, year - 1, comparative=True)):
            return finish("UNKNOWN", ["DILUTED_EPS_CURRENT_OR_COMPARATIVE_UNVERIFIED"])
        field, basis = "basic_eps", "BASIC"
        current, current_proof = _eps(earnings, field, year)
        prior, prior_proof = _eps(earnings, field, year - 1, comparative=True)
    if current is None or prior is None:
        return finish("UNKNOWN", ["BASIC_EPS_CURRENT_OR_COMPARATIVE_UNVERIFIED"])
    if fr3_method:
        revision = row.get("revision_number")
        if (type(revision) is not int or revision < 0
                or type(earnings.get("revision_number")) is not int
                or earnings["revision_number"] != revision
                or any(proof.get("source_url") != row["source_url"]
                       or type(proof.get("revision_number")) is not int
                       or proof["revision_number"] != revision
                       for proof in (current_proof, prior_proof))):
            return finish("UNKNOWN", ["FR3_ANNUAL_EPS_SOURCE_VERSION_UNVERIFIED"])
    negative_eps = ["NON_POSITIVE_" + key for key, value in (("CURRENT_EPS", current), ("PRIOR_EPS", prior)) if value <= 0]
    if negative_eps:
        # A proven failed earnings premise remains FAIL even if an unrelated
        # quote or later field is absent. It is not a missing-data diagnosis.
        return finish("FAIL", negative_eps, result=dict(status="FAIL", reasons=negative_eps,
            policy=policy, policy_id=identity(policy), return_basis="PRICE_RETURN_ONLY",
            observed_inputs=dict(current_eps=current, prior_eps=prior, eps_basis=basis)),
            source=dict(current_eps=current_proof, prior_eps=prior_proof))
    values, proofs = {}, dict(current_eps=current_proof, prior_eps=prior_proof)
    for field in ("revenue", "operating_income"):
        for comparative in (False, True):
            key = ("prior_" if comparative else "current_") + field
            values[key], proofs[key] = _amount(row, field, period, cutoff, comparative=comparative)
            if values[key] is None:
                return finish("UNKNOWN", ["ANNUAL_PRICE_VALUATION_FIELD_UNVERIFIED:" + key])
            if values[key] <= 0:
                return finish("FAIL", ["NON_POSITIVE_REPORTED_" + key.upper()],
                    source=dict(field=key, value=values[key], proof=proofs[key]))
    if source_policy is None and quality.get("core_inputs_pass") is not True:
        return finish("UNKNOWN", ["PRICE_VALUATION_QUALITY_INPUTS_UNVERIFIED"])
    raw_pe = bars.get("PE", pd.Series(dtype=float))
    pe = pd.to_numeric(raw_pe.where(~raw_pe.map(lambda value: str(value) in {"True", "False"})), errors="coerce")
    current_pe = number(pe.iloc[-1]) if len(pe) else None
    median = pe.where(pe > 0).rolling(60, min_periods=20).median().iloc[-1] if len(pe) else None
    # No old positive-PE fallback when the actual current quote PE is absent.
    if current_pe is None or current_pe <= 0 or median is None or pd.isna(median):
        return finish("UNKNOWN", ["OFFICIAL_CURRENT_OR_HISTORY_PE_UNVERIFIED"])
    current_revenue, prior_revenue = values["current_revenue"], values["prior_revenue"]
    if current_revenue <= 0 or prior_revenue <= 0:
        return finish("FAIL", ["NON_POSITIVE_REPORTED_REVENUE"])
    evaluator = evaluate_price_valuation
    if fr3_method:
        from domain.les_v7.fr3_policy import evaluate_eps_r2
        evaluator = evaluate_eps_r2
    if relaxed_method:
        from domain.les_v7.valuation_relaxed_r13 import evaluate_annual_price_valuation
        evaluator = evaluate_annual_price_valuation
    if component_method:
        from domain.les_v7.valuation_recovery_r14 import evaluate_annual_price_valuation
        evaluator = evaluate_annual_price_valuation
    if tactical_method:
        from domain.les_v7.valuation_recovery_r15 import evaluate_annual_price_valuation
        evaluator = evaluate_annual_price_valuation
    result = evaluator(current_eps=current, prior_eps=prior,
        current_revenue=current_revenue, prior_revenue=prior_revenue,
        current_margin=values["current_operating_income"] / current_revenue,
        prior_margin=values["prior_operating_income"] / prior_revenue,
        current_price=number(features.get("close")), current_pe=current_pe,
        recent_pe_median=float(median), eps_basis=basis)
    if structured_method:
        from domain.les_v7.valuation_recovery_r10 import (
            evaluate_recovery_price_valuation, select_recovery, validate_comparables)
        comparison_context = packet.get("r10_valuation_context")
        session = str(bars["Date"].iloc[-1])[:10] if "Date" in bars and len(bars) else (market_source or {}).get("session_date")
        if comparison_context is not None and (session is None or number(features.get("close")) is None):
            return finish("UNKNOWN", ["R10_STRATEGY_QUOTE_BINDING_MISSING"])
        selector = validate_comparables
        if component_method:
            from domain.les_v7.valuation_recovery_r10 import validate_comparable_components
            selector = validate_comparable_components
        if tactical_method:
            from domain.les_v7.valuation_recovery_r10 import validate_comparable_components_r15
            selector = validate_comparable_components_r15
        comparables = selector(comparison_context, symbol=symbol,
            market=market, cutoff=cutoff, eps_basis=basis, fiscal_year=year, scope=row["scope"],
            annual_source=row, expected_session=session, current_price=number(features.get("close")))
        recovery = select_recovery(packet.get("filings", ()), symbol=symbol, market=market,
                                   cutoff=cutoff, scope=row["scope"])
        if tactical_method:
            from domain.les_v7.valuation_recovery_r15 import evaluate_recovery_price_valuation as tactical_evaluator
            result = tactical_evaluator(legacy_result=result, comparables=comparables, recovery=recovery)
        elif component_method:
            from domain.les_v7.valuation_recovery_r14 import evaluate_recovery_price_valuation as component_evaluator
            result = component_evaluator(legacy_result=result, comparables=comparables, recovery=recovery)
        elif relaxed_method:
            from domain.les_v7.valuation_relaxed_r13 import evaluate_relaxed_price_valuation
            result = evaluate_relaxed_price_valuation(legacy_result=result, comparables=comparables, recovery=recovery)
        else:
            result = evaluate_recovery_price_valuation(legacy_result=result, comparables=comparables, recovery=recovery)
    return finish(result["status"], result["reasons"], result=result,
        source=dict(eps_basis=basis, fiscal_year=year, source_id=identity(row), proofs=proofs,
                    current_pe=current_pe, recent_pe_median=float(median)))

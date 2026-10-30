"""Explicit R2 adapter: source-bound financial inputs, unchanged v7 budgets.

Original/quarter/bridge values share the existing native proof validators. A
successful path cannot conceal a contradictory verified comparative source;
incomplete paths remain diagnostics. No current-data or global revision-id
fallback, external safety switch, annualized interim amount or fabricated zero.
"""
from __future__ import annotations

from math import fsum, isclose

import pandas as pd

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.quality import CORE, map_metrics, quality_gate, _ratio
from domain.strategies.low_entry_v6_quality import (
    select_quality_inputs, normalize_quarters_with_diagnostics, reported_annual_flow,
)
from domain.strategies.low_entry_v6_quality_evidence import validate_field, validated_proof_value
from domain.strategies.low_entry_v6_ttm import (
    comparative_ttm, _field_dependencies, _impact_attestation,
)

PROFILE = "QG_V7_EVIDENCE_R2"
TTM_POLICY = "V7_TTM_EXACT_FIELD_DEPENDENCIES_R2"
FLOW_FIELDS = (*CORE, "pretax_income", "income_tax", "interest_expense", "capex",
               "intangible_capex", "parent_net_income", "dividends", "share_repurchases")


def _origin(row):
    return row.get("quarter_eps_evidence", row) if row.get("quarter_flow_source") else row


class FinancialInputs:
    """One selected cutoff, bounded per-field memo; no cross-account result cache."""

    def __init__(self, periods, cutoff):
        self.periods, self.cutoff = periods, pd.Timestamp(timestamp(cutoff))
        self.quarters, self.quarter_traces = normalize_quarters_with_diagnostics(periods)
        self.checks, self.ttm_traces, self._values = {}, {}, {}
        self.quarter_available, self.quarter_diagnostics = {}, {}
        self.comparison_checks = {}

    def check(self, field, period, *, instant=False):
        key = f"{period}:{field}"
        if key not in self.checks:
            row = self.periods.get(period)
            self.checks[key] = (validate_field(row, field, period=period, cutoff=self.cutoff, instant=instant)
                if row is not None else dict(field=field, period=str(period), value=None,
                    status="MISSING", category="MISSING_DATA", reason="NO_ELIGIBLE_FILING_FOR_PERIOD",
                    evidence_level="NONE", checks={}))
            check = self.checks[key]
            if row is not None:
                check = dict(check, checks=dict(check["checks"], identity=check["checks"]["identity"] and all(
                    check.get(k) == row.get(k) for k in ("symbol", "market", "scope", "currency", "unit"))))
                self.checks[key] = check
        return self.checks[key]

    def balance(self, field, period):
        check = self.check(field, period, instant=True)
        return check["value"] if check["evidence_level"] == "CANONICAL_ORIGINAL" and all(check["checks"].values()) else None

    def quarter(self, field, period):
        key = f"{period}:{field}"
        row = self.periods.get(period)
        check = self.check(field, period)
        if row is None or check["evidence_level"] != "CANONICAL_ORIGINAL" or check["value"] is None or not all(check["checks"].values()):
            self.quarter_diagnostics[key] = dict(status="UNKNOWN", reason=check["reason"])
            return None
        original = _origin(row)
        proof = original.get("field_evidence", {}).get(field, {})
        available = max(row["_available"], pd.Timestamp(original["available_at"]))
        # Existing explicit source-impact review, not a local revision counter.
        issues = []
        if original.get("correction_description"): issues.append("FILING_CORRECTION")
        basis = original.get("accounting_basis_evidence", {})
        if (basis and basis.get("status") != "RESOLVED"
                and basis.get("reason") != "CURRENT_ACCOUNTING_POLICY_ATTESTATION_MISSING"):
            issues.append("ACCOUNTING_POLICY")
        if issues:
            parsed, dependencies = _field_dependencies(proof, original, proof.get("start_date"), proof.get("end_date"))
            attestation = _impact_attestation(original, field, issues, dependencies, self.cutoff)
            if parsed is None or attestation is None:
                self.quarter_diagnostics[key] = dict(status="UNKNOWN", reason="FIELD_ACCOUNTING_IMPACT_UNVERIFIED", issues=issues)
                return None
            available = max(available, pd.Timestamp(attestation["evidence_available_at"]), pd.Timestamp(attestation["verified_at"]))
        self.quarter_available[key] = available
        if row.get("flow_basis") == "QUARTER" or period.quarter == 1:
            self.quarter_diagnostics[key] = dict(status="VERIFIED_REPORTED", reason="REPORTED_SINGLE_QUARTER")
            return check["value"]
        # Main native flows already have source-bound single-quarter/YTD checks.
        if field in self.quarters.get(period, {}):
            self.quarter_diagnostics[key] = self.quarter_traces[period][field]
            return self.quarters[period][field]
        start, end = period.start_time.date().isoformat(), period.end_time.date().isoformat()
        direct = original.get("quarter_field_evidence", {}).get(field, {})
        value = validated_proof_value(direct, original, field, start=start, end=end)
        if value is not None:
            self.quarter_diagnostics[key] = dict(status="VERIFIED_REPORTED", reason="SOURCE_BOUND_QUARTER_CONTEXT")
            return value
        # Other additive cash/tax/shareholder flows use the SAME original only.
        prior = original.get("prior_ytd_field_evidence", {}).get(field, {})
        previous = validated_proof_value(prior, original, field, start=f"{period.year}-01-01",
                                        end=(period - 1).end_time.date().isoformat())
        self.quarter_diagnostics[key] = dict(status="VERIFIED_DERIVED" if previous is not None else "UNKNOWN",
            reason="SAME_DOCUMENT_YTD_DIFFERENCE" if previous is not None else "SAME_DOCUMENT_QUARTER_CONTEXT_UNAVAILABLE")
        return check["value"] - previous if previous is not None else None

    def ttm(self, field, period):
        if field not in FLOW_FIELDS:
            raise ValueError("TTM_ADDITIVE_FLOW_FIELD_REQUIRED")
        key = f"{period}:{field}"
        if key in self._values: return self._values[key]
        row = self.periods.get(period)
        trace = dict(policy=TTM_POLICY, field=field, period=str(period), status="UNKNOWN",
                     method=None, paths=[], available_at=None, quarter_values_filled=False)
        paths = []
        if row is not None:
            check = self.check(field, period)
            annual = (reported_annual_flow(row, period, field) if check["evidence_level"] == "CANONICAL_ORIGINAL"
                      and check["value"] is not None and all(check["checks"].values()) else None)
            if annual is not None:
                paths.append(dict(method="REPORTED_FULL_YEAR", value=annual,
                    available_at=max(row["_available"], pd.Timestamp(check["available_at"])).isoformat(),
                    source_ids=[check.get("content_sha256")]))
        needed = tuple(period - i for i in range(4))
        values = [self.quarter(field, p) for p in needed]
        if all(v is not None for v in values):
            paths.append(dict(method="FOUR_VERIFIED_QUARTERS", value=fsum(values),
                available_at=max(self.quarter_available[f"{p}:{field}"] for p in needed).isoformat(),
                source_ids=[self.check(field, p).get("content_sha256") for p in needed], operands=values))
        bridge, bridge_trace = comparative_ttm(self.periods, period, field, self.cutoff)
        if bridge is not None:
            paths.append(dict(method="COMPARATIVE_ORIGINAL_TTM", value=bridge,
                available_at=bridge_trace["available_at"],
                source_ids=[r.get("content_sha256") for r in bridge_trace["sources"]], trace=bridge_trace))
        trace["paths"], trace["bridge_diagnostic"] = paths, bridge_trace
        trace["quarter_diagnostics"] = {str(p): self.quarter_diagnostics.get(f"{p}:{field}", {}) for p in needed}
        conflicts = any(g.get("category") == "INCOMPARABLE" for g in bridge_trace.get("gaps", ()))
        if conflicts or paths and any(not isclose(p["value"], paths[0]["value"], rel_tol=0., abs_tol=.5) for p in paths):
            trace.update(status="CONFLICT", reason="TTM_PATH_OR_COMPARATIVE_DEPENDENCY_CONFLICT")
            value = None
        elif paths:
            chosen = paths[0]  # deterministic precedence; successful paths agree
            value = chosen["value"]
            trace.update(status="VERIFIED_REPORTED" if chosen["method"] == "REPORTED_FULL_YEAR" else "VERIFIED_DERIVED",
                method=chosen["method"], available_at=max(pd.Timestamp(p["available_at"]) for p in paths).isoformat(),
                source_ids=sorted(set(s for p in paths for s in p["source_ids"] if s)), value=value)
        else:
            value = None
            trace.update(reason="NO_VERIFIED_TTM_PATH", missing_quarters=[str(p) for p, v in zip(needed, values) if v is None])
        self._values[key], self.ttm_traces[key] = value, trace
        return value

    def comparison_allowed(self, field, period):
        """Validate only this flow's同比 dependencies; keep standalone amounts."""
        key = f"{period}:{field}"
        if key in self.comparison_checks: return self.comparison_checks[key]["state"] == "PASS"
        receipt = dict(field=field, period=str(period), state="PASS", reason="VERIFIED_TTM_SAME_SCOPE_COMPARISON")
        if self.ttm(field, period) is None or self.ttm(field, period - 4) is None:
            receipt.update(state="UNKNOWN", reason="COMPARISON_TTM_SOURCE_UNVERIFIED")
            self.comparison_checks[key] = receipt
            return False
        for source_period in (period, period - 4):
            row = self.periods.get(source_period)
            if row is None:
                receipt.update(state="UNKNOWN", reason="COMPARISON_SOURCE_PERIOD_MISSING")
                break
            original = _origin(row)
            proof = original.get("field_evidence", {}).get(field, {})
            basis = original.get("accounting_basis_evidence", {})
            issues = (["FILING_CORRECTION"] if original.get("correction_description") else [])
            # Missing unrelated note text is not positive evidence of an impact.
            if basis and basis.get("status") != "RESOLVED" and basis.get("reason") != "CURRENT_ACCOUNTING_POLICY_ATTESTATION_MISSING":
                issues.append("ACCOUNTING_POLICY")
            if issues:
                parsed, dependencies = _field_dependencies(proof, original, proof.get("start_date"), proof.get("end_date"))
                if parsed is None or _impact_attestation(original, field, issues, dependencies, self.cutoff) is None:
                    receipt.update(state="UNKNOWN", reason="COMPARISON_FIELD_IMPACT_UNVERIFIED", issues=issues)
                    break
        current = _origin(self.periods.get(period, {}))
        previous = _origin(self.periods.get(period - 4, {}))
        comparative = current.get("comparative_ytd_field_evidence", {}).get(field)
        if receipt["state"] == "PASS" and comparative:
            start, end = f"{period.year - 1}-01-01", (period - 4).end_time.date().isoformat()
            left, _ = _field_dependencies(comparative, current, start, end)
            right, _ = _field_dependencies(previous.get("field_evidence", {}).get(field, {}), previous, start, end)
            if left is None or right is None:
                receipt.update(state="UNKNOWN", reason="COMPARATIVE_FIELD_DEPENDENCY_UNVERIFIED")
            elif not isclose(left, right, rel_tol=0., abs_tol=.5):
                receipt.update(state="CONFLICT", reason="COMPARATIVE_FIELD_DEPENDENCY_VALUES_CHANGED")
        self.comparison_checks[key] = receipt
        return receipt["state"] == "PASS"


def derive_verified_ttm(periods, period, field, cutoff):
    """Public deterministic source/trace API for diagnostics and regressions."""
    inputs = FinancialInputs(periods, cutoff)
    value = inputs.ttm(field, period)
    return value, inputs.ttm_traces[f"{period}:{field}"]


def _empty(reason, policy, *, status="DATA_INSUFFICIENT"):
    result = quality_gate([], core_valid={f: False for f in CORE}, safety_complete=False)
    result.update(profile_id=PROFILE, mapping_version="QG_MAP_R2", quality_policy_id=identity(policy),
                  resolved_policy=policy,
                  status=status, reason_codes=[reason], financial_safety=dict(state="UNKNOWN", reasons=[reason]),
                  ttm_traces={}, field_checks={}, readiness=dict(financial="NOT_READY"))
    return result


def _metrics(inputs, latest):
    """Reuse the existing nineteen curves; only financial source resolution differs."""
    from domain.les_v7.financial_sources import payment_outflow
    ttm = lambda field, offset=0: inputs.ttm(field, latest - offset)
    balance = lambda field, offset=0: inputs.balance(field, latest - offset)
    average = lambda field: ((balance(field) + balance(field, 4)) / 2
        if balance(field) is not None and balance(field, 4) is not None else None)
    t = {field: ttm(field) for field in CORE}
    core_valid = {field: value is not None for field, value in t.items()}
    assets, liabilities, equity, cash, debt, ca, cl = (balance(k) for k in
        ("total_assets", "total_liabilities", "total_equity", "cash", "interest_bearing_debt", "current_assets", "current_liabilities"))
    pretax, interest, tax = (ttm(k) for k in ("pretax_income", "interest_expense", "income_tax"))
    ebit = pretax + interest if pretax is not None and interest is not None else None
    capex_quarters = [(inputs.quarter("capex", latest - i), inputs.quarter("intangible_capex", latest - i)) for i in range(4)]
    capex_missing = sum(any(x is None for x in pair) for pair in capex_quarters)
    payment_evidence, payment_errors = {}, []
    def payment(field):
        raw = ttm(field)
        value, evidence = payment_outflow(raw, field=field, periods=inputs.periods,
            trace=inputs.ttm_traces[f"{latest}:{field}"], cutoff=inputs.cutoff.isoformat())
        payment_evidence[field] = evidence
        if raw is not None and value is None:
            payment_errors.append(field)
        return value
    ppe, intangible = payment("capex"), payment("intangible_capex")
    capex_invalid = any(v is not None and v < 0 for v in (ppe, intangible)) or any(field in payment_errors for field in ("capex", "intangible_capex"))
    capex = ppe + intangible if ppe is not None and intangible is not None and not capex_invalid else None
    fcf = t["cfo"] - capex if t["cfo"] is not None and capex is not None else None
    cfo_quarters = [inputs.quarter("cfo", latest - i) for i in range(4)]
    v = dict(debt_ratio=_ratio(liabilities, assets), net_debt_equity=_ratio(debt - cash if debt is not None and cash is not None else None, equity),
        interest_coverage=_ratio(ebit, interest), current_ratio=_ratio(ca, cl), cash_coverage=_ratio(cash, cl),
        no_current_liabilities=cl == 0,
        positive_cfo_fraction=sum(x > 0 for x in cfo_quarters) / 4 if all(x is not None for x in cfo_quarters) and t["cfo"] is not None and t["cfo"] > 0 else 0. if t["cfo"] is not None and t["cfo"] <= 0 else None,
        cash_conversion=_ratio(t["cfo"], t["net_income"]), net_income_nonpositive=t["net_income"] is not None and t["net_income"] <= 0,
        fcf_to_cfo=_ratio(fcf, t["cfo"]), cfo_nonpositive=t["cfo"] is not None and t["cfo"] <= 0,
        accrual_ratio=_ratio(t["net_income"] - t["cfo"] if all(t[k] is not None for k in ("net_income", "cfo")) else None, average("total_assets")),
        capex_gap=capex is None and capex_missing > 0)
    def capital(offset=0):
        values = [balance(k, offset) for k in ("total_equity", "interest_bearing_debt", "cash")]
        return values[0] + values[1] - values[2] if all(x is not None for x in values) else None
    ic, old_ic = capital(), capital(4)
    rate = _ratio(tax, pretax)
    tax_valid = rate is not None and 0 <= rate <= 1
    nopat = t["operating_income"] * (1 - rate) if tax_valid and t["operating_income"] is not None else None
    old_tax, old_pretax, old_op = ttm("income_tax", 4), ttm("pretax_income", 4), ttm("operating_income", 4)
    if not all(inputs.comparison_allowed(field, latest) for field in ("income_tax", "pretax_income", "operating_income")):
        old_op = None
    old_rate = _ratio(old_tax, old_pretax)
    nopat_comparable = all(inputs.comparison_allowed(field, latest) for field in
                           ("operating_income", "income_tax", "pretax_income"))
    old_nopat = (old_op * (1 - old_rate) if nopat_comparable and old_op is not None
                 and old_rate is not None and 0 <= old_rate <= 1 else None)
    delta_ic = ic - old_ic if ic is not None and old_ic is not None else None
    delta_nopat = nopat - old_nopat if nopat is not None and old_nopat is not None else None
    tax_gap = tax is None or old_tax is None
    roic_shape = (tax_gap and ic is not None and old_ic is not None) or (not tax_gap and tax_valid and (ic is None) != (old_ic is None))
    roic_gaps = [mid for mid in ("R1", "A3") if tax is None or ic is None or old_ic is None or mid == "A3" and tax_gap]
    v.update(roic=_ratio(nopat, (ic + old_ic) / 2 if ic is not None and old_ic is not None else None),
        roic_gap_metrics=roic_gaps, invested_capital_nonpositive=ic is not None and ic <= 0 or old_ic is not None and old_ic <= 0,
        incremental_roic=_ratio(delta_nopat, delta_ic), nopat_history_missing=old_op is None,
        capital_not_increased=delta_ic is not None and delta_nopat is not None and delta_ic <= .01 * max(abs(ic), abs(old_ic)),
        nopat_not_decreased=delta_nopat is not None and delta_nopat >= 0,
        roe=_ratio(ttm("parent_net_income"), average("parent_equity")))
    for name, numerator in (("gross_margin", "cost_of_revenue"), ("operating_margin", "operating_income")):
        now, old = _ratio(t[numerator], t["revenue"]), _ratio(ttm(numerator, 4), ttm("revenue", 4))
        if name == "gross_margin": now, old = (1 - now if now is not None else None), (1 - old if old is not None else None)
        comparable = inputs.comparison_allowed(numerator, latest) and inputs.comparison_allowed("revenue", latest)
        v[name], v[name + "_change"] = now, now - old if comparable and now is not None and old is not None else None
    previous_revenue = ttm("revenue", 4)
    growth = _ratio(t["revenue"], previous_revenue) if inputs.comparison_allowed("revenue", latest) else None
    v.update(revenue_growth=growth - 1 if growth is not None else None,
             revenue_base_nonpositive=previous_revenue is not None and previous_revenue <= 0)
    for name, field, flow in (("dso_change", "accounts_receivable", "revenue"), ("dio_change", "inventory", "cost_of_revenue")):
        now, old = _ratio(balance(field), ttm(flow)), _ratio(balance(field, 4), ttm(flow, 4))
        v[name] = 365 * (now - old) if inputs.comparison_allowed(flow, latest) and now is not None and old is not None else None
    row = inputs.periods[latest]
    share = row.get("adjusted_share_growth_evidence", {})
    v["adjusted_share_growth"] = share.get("value") if (share.get("source_hash") == row.get("content_sha256")
        and share.get("status") == "VERIFIED" and share.get("policy_id") and share.get("available_at")
        and timestamp(share["available_at"]) <= inputs.cutoff.to_pydatetime()) else None
    dividend, buybacks = payment("dividends"), payment("share_repurchases")
    distribution = dividend + buybacks if dividend is not None and buybacks is not None else None
    v.update(distribution_fcf=_ratio(distribution, fcf), distribution_missing=distribution is None,
        no_distribution=distribution == 0 and capex is not None,
        fcf_nonpositive_distribution=fcf is not None and fcf <= 0 and distribution is not None and distribution > 0)
    v["dependency_ids"] = tuple(identity({k: r.get(k) for k in ("symbol", "period_end", "revision_id", "content_sha256")}) for r in inputs.periods.values())
    return v, t, core_valid, dict(capex_ttm=capex, fcf_ttm=fcf, capex_missing_quarters=capex_missing,
        capex_invalid=capex_invalid, roic_shape=roic_shape, interest=interest, debt=debt,
        payment_evidence=payment_evidence, payment_errors=payment_errors)


def evaluate_financials_r2(records, *, symbol, market, cutoff, safety_receipt=None, policy=None):
    from domain.les_v7.quality_policy import validate_quality_policy, native_financial_safety, quality_freshness
    try:
        policy = validate_quality_policy(policy, cutoff, market=market, records=records, symbol=symbol)
    except (ValueError, TypeError) as error:
        return _empty(str(error), policy, status="CONFIG_INCOMPLETE")
    try:
        periods, ledger, excluded = select_quality_inputs(records, symbol=symbol, market=market, cutoff=pd.Timestamp(timestamp(cutoff)))
        if not periods: return _empty("NO_POINT_IN_TIME_FINANCIALS", policy)
        latest = max(periods)
        if str(latest) != policy["latest_required_period"]:
            return _empty("LATEST_REQUIRED_REPORTING_PERIOD_UNAVAILABLE", policy)
        inputs = FinancialInputs(periods, cutoff)
        v, core_ttm, core_valid, detail = _metrics(inputs, latest)
        safety = native_financial_safety(periods[latest], inputs.checks, symbol=symbol, market=market, cutoff=cutoff, policy=policy)
    except (ValueError, TypeError, KeyError) as error:
        return _empty(str(error), policy)
    row = periods[latest]
    debt_free = detail["debt"] == 0 and detail["interest"] == 0 and "VERIFIED_NO_MATERIAL_CAPITALIZED_INTEREST" in safety.get("reasons", ())
    v["debt_free"] = debt_free
    safety_complete = safety["state"] == "PASS" and detail["interest"] is not None and detail["interest"] >= 0 and (detail["interest"] > 0 or debt_free)
    age, freshness = quality_freshness(latest, cutoff, policy)
    checks = list(inputs.checks.values())
    verified = [check for check in checks if check.get("evidence_level") == "CANONICAL_ORIGINAL" and check.get("value") is not None and all(check.get("checks", {}).values())]
    ttm_checks = list(inputs.ttm_traces.values())
    verified_ttm = sum(trace["status"] in {"VERIFIED_REPORTED", "VERIFIED_DERIVED"} for trace in ttm_checks)
    payment_checks = list(detail["payment_evidence"].values())
    verified_payments = sum(note.get("status") in {"VERIFIED_REPORTED", "VERIFIED_DERIVED"} for note in payment_checks)
    comparison_checks = list(inputs.comparison_checks.values())
    verified_comparisons = sum(note["state"] == "PASS" for note in comparison_checks)
    validation_count = len(checks) + len(ttm_checks) + len(payment_checks) + len(comparison_checks)
    validation = 100. * (len(verified) + verified_ttm + verified_payments + verified_comparisons) / validation_count if validation_count else 0.
    source_weights = {"MOPS_IXBRL": 100., "TWSE_OPENAPI": 95., "TPEX_OPENAPI": 95.}
    reliability = fsum(source_weights.get(check.get("source"), 0.) for check in verified) / len(checks) if checks else 0.
    metrics = map_metrics(v)
    # Explicit known economically undefined turnover ratios are observed zero,
    # not missing financial data; no denominator is fabricated or renormalized.
    from dataclasses import replace
    nonmeaningful = {}
    for mid, key in (("C2", "net_income_nonpositive"), ("C3", "cfo_nonpositive"), ("R1", "invested_capital_nonpositive")):
        if v.get(key) is True: nonmeaningful[mid] = "VERIFIED_NONPOSITIVE_RATIO_DENOMINATOR"
    if core_ttm["revenue"] is not None and core_ttm["revenue"] <= 0:
        nonmeaningful.update(R3="VERIFIED_NONPOSITIVE_MARGIN_DENOMINATOR", R4="VERIFIED_NONPOSITIVE_MARGIN_DENOMINATOR")
    if v.get("revenue_base_nonpositive") is True:
        nonmeaningful["G1"] = "VERIFIED_NONPOSITIVE_GROWTH_BASE"
    for mid, field in (("G2", "revenue"), ("G3", "cost_of_revenue")):
        values = [inputs.ttm(field, latest - offset) for offset in (0, 4)]
        if all(x is not None for x in values) and any(x <= 0 for x in values):
            nonmeaningful[mid] = "VERIFIED_NONPOSITIVE_TURNOVER_DENOMINATOR"
    metrics = [replace(m, mapped_score=0., root_gap=None) if m.metric_id in nonmeaningful else m for m in metrics]
    hard = safety["state"] == "FAIL"
    result = quality_gate(metrics, core_valid=core_valid, safety_complete=safety_complete, hard_reject=hard,
        freshness=freshness, validation=validation,
        source_reliability=reliability, gap_shapes=dict(ROIC=detail["roic_shape"], CAPEX=detail["capex_missing_quarters"] == 1))
    if (detail["capex_invalid"] or detail["payment_errors"]) and not hard:
        result.update(status="DATA_INSUFFICIENT", quality_score=None,
            reason_codes=sorted(set(result["reason_codes"] + ["SOURCE_MAPPING_ERROR:PAYMENT_OUTFLOW_UNVERIFIED:" + field for field in detail["payment_errors"]])))
    if age > policy.get("freshness_max_age_days", 275) and result["status"] != "REJECT":
        result.update(status="DATA_INSUFFICIENT", quality_score=None, reason_codes=sorted(set(result["reason_codes"] + ["STALE_FINANCIAL_REPORT"])))
    if not core_valid or not all(core_valid.values()):
        result["reason_codes"] = sorted(set(result["reason_codes"] + ["REQUIRED_TTM_UNAVAILABLE:" + field for field, valid in core_valid.items() if not valid]))
    if not safety_complete:
        result["reason_codes"] = sorted(set(result["reason_codes"] + list(safety.get("reasons", ()))
            + (["FINANCIAL_INTEREST_OR_DEBT_FREE_EVIDENCE_UNVERIFIED"] if detail["interest"] is None or detail["interest"] < 0 or detail["interest"] == 0 and not debt_free else [])))
    for item in result["metric_ledger"]:
        if item["metric_id"] in nonmeaningful: item.update(value_status="NOT_MEANINGFUL", reason=nonmeaningful[item["metric_id"]])
    result.update(profile_id=PROFILE, mapping_version="QG_MAP_R2", quality_policy_id=identity(policy), core_ttm=core_ttm,
        resolved_policy=policy,
        core_ttm_quarters=[str(latest - i) for i in range(4)], latest_period=str(latest),
        missing_quarters=[str(latest - i) for i in range(4) if latest - i not in periods], financial_inputs=v,
        field_checks=inputs.checks, excluded_history=excluded, ttm_traces=inputs.ttm_traces, comparison_checks=inputs.comparison_checks,
        capex_ttm=detail["capex_ttm"], fcf_ttm=detail["fcf_ttm"], capex_missing_quarters=detail["capex_missing_quarters"],
        financial_safety=safety, verification=dict(freshness=freshness, validation=validation,
            source_reliability=reliability, source_age_days=age, verified_checks=len(verified), checked_fields=len(checks),
            verified_ttm=verified_ttm, ttm_checks=len(ttm_checks), verified_payments=verified_payments, payment_checks=len(payment_checks),
            verified_comparisons=verified_comparisons, comparison_checks=len(comparison_checks)),
        payment_normalization=detail["payment_evidence"],
        readiness=dict(financial="READY" if result["status"] == "PASS" else "NOT_READY"))
    return result

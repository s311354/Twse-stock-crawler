"""QG_V7_BOUNDED_GAPS_R1. Reuse native PIT/proof and quarter normalization."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import fsum
import pandas as pd

from domain.les_v7.contracts import finite, identity, ramp, timestamp
from domain.strategies.low_entry_v6_quality import select_quality_inputs, normalize_quarters_with_diagnostics
from domain.strategies.low_entry_v6_quality_evidence import validate_field, validated_proof_value

PROFILE = "QG_V7_BOUNDED_GAPS_R1"
METRICS = {"B1": ("balance_sheet", 6), "B2": ("balance_sheet", 6), "B3": ("balance_sheet", 6),
           "B4": ("balance_sheet", 4), "B5": ("balance_sheet", 3),
           "C1": ("cash_earnings", 5), "C2": ("cash_earnings", 8), "C3": ("cash_earnings", 7), "C4": ("cash_earnings", 5),
           "R1": ("profitability", 10), "R2": ("profitability", 5), "R3": ("profitability", 5), "R4": ("profitability", 5),
           "G1": ("growth_efficiency", 5), "G2": ("growth_efficiency", 5), "G3": ("growth_efficiency", 5),
           "A1": ("capital_allocation", 4), "A2": ("capital_allocation", 3), "A3": ("capital_allocation", 3)}
DOMAINS = dict(balance_sheet=25, cash_earnings=25, profitability=25, growth_efficiency=15, capital_allocation=10)
MINIMA = dict(balance_sheet=1., cash_earnings=.7, profitability=.6, growth_efficiency=1., capital_allocation=.4)
CORE = ("revenue", "cost_of_revenue", "operating_income", "net_income", "cfo")
GAPS = {"ROIC": {"R1", "A3"}, "CAPEX": {"C3", "A2"}}


@dataclass(frozen=True)
class Metric:
    metric_id: str
    mapped_score: float | None
    root_gap: str | None = None
    raw_value: float | None = None
    dependency_fact_ids: tuple[str, ...] = ()


def quality_gate(metrics, *, core_valid, safety_complete, gap_shapes=None, hard_reject=False,
                 config_complete=True, freshness=100., validation=100., source_reliability=100.) -> dict:
    """Pure scoring contract; evidence adapters, not UI flags, establish validity."""
    result = dict(profile_id=PROFILE, mapping_version="QG_MAP_R1", status="UNKNOWN", quality_score=None,
                  earned_points=None, missing_penalty_points=None, data_confidence=None,
                  coverage_total=None, coverage_by_domain={}, gap_qualifier="NOT_EVALUATED",
                  core_ttm_pass=all(core_valid.get(f) is True for f in CORE), mandatory_safety_pass=safety_complete is True,
                  metric_ledger=[dict(asdict(m), domain=METRICS[m.metric_id][0], budget=METRICS[m.metric_id][1],
                     value_status="BOUNDED_GAP" if m.root_gap else "MISSING" if m.mapped_score is None else "OBSERVED",
                     earned_points=METRICS[m.metric_id][1] * m.mapped_score if m.mapped_score is not None else None)
                     for m in metrics], gap_groups=[], reason_codes=[], quality_les_cap=None)
    if config_complete is not True:
        result.update(status="CONFIG_INCOMPLETE", reason_codes=["QUALITY_CONFIG_INCOMPLETE"])
        return result
    if hard_reject is True:
        result.update(status="REJECT", reason_codes=["QUALITY_HARD_REJECT"])
        return result
    if not result["core_ttm_pass"] or safety_complete is not True:
        result.update(status="DATA_INSUFFICIENT", reason_codes=["MANDATORY_TTM_OR_SAFETY_INVALID"])
        return result
    ledger = {m.metric_id: m for m in metrics}
    if len(ledger) != len(metrics) or set(ledger) != set(METRICS):
        raise ValueError("QUALITY_METRIC_LEDGER_INCOMPLETE_OR_DUPLICATE")
    available = {d: 0 for d in DOMAINS}
    terms, groups, reasons = [], set(), []
    for key, (domain, budget) in METRICS.items():
        item = ledger[key]
        if item.mapped_score is not None:
            if item.root_gap is not None:
                raise ValueError("OBSERVED_VALUE_IS_NOT_A_GAP")
            terms.append(budget * finite(item.mapped_score, key, 0, 1))
            available[domain] += budget
        elif item.root_gap in GAPS and key in GAPS[item.root_gap]:
            groups.add(item.root_gap)
        else:
            reasons.append("UNALLOWLISTED_GAP:" + key)
    if any((gap_shapes or {}).get(g) is not True for g in groups):
        reasons.append("GAP_SCOPE_EXCEEDED")
    coverage = sum(available.values()) / 100
    by_domain = {d: available[d] / DOMAINS[d] for d in DOMAINS}
    if coverage < .75 or any(by_domain[d] < MINIMA[d] for d in DOMAINS):
        reasons.append("COVERAGE_INSUFFICIENT")
    if 100 - sum(available.values()) > 23:
        reasons.append("GAP_BUDGET_EXCEEDED")
    confidence = fsum((.4 * coverage * 100, .25 * finite(freshness, "freshness", 0, 100),
                      .25 * finite(validation, "validation", 0, 100), .1 * finite(source_reliability, "source", 0, 100)))
    if confidence < 70:
        reasons.append("CONFIDENCE_LOW")
    result.update(earned_points=fsum(terms), coverage_total=coverage, coverage_by_domain=by_domain,
                  data_confidence=confidence, gap_groups=sorted(groups), reason_codes=sorted(set(reasons)))
    if reasons:
        result["status"] = "PROVISIONAL"
        return result
    penalty = min(6, 3 * len(groups))
    score = max(0., fsum((*terms, -penalty)))
    result.update(quality_score=score, missing_penalty_points=penalty,
                  status="PASS" if score >= 70 else "WATCH_QUALITY" if score >= 60 else "REJECT",
                  gap_qualifier="BOUNDED_ROIC_CAPEX_GAPS" if groups else "COMPLETE",
                  quality_les_cap=100)
    return result


def _ratio(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def map_metrics(v: dict) -> list[Metric]:
    """QG_MAP_R1 fixed curves; unknown inputs are not adverse observed zero."""
    def up(key, a, b): return ramp(v.get(key), a, b)
    def down(key, a, b):
        value = up(key, a, b)
        return 1 - value if value is not None else None
    scores = dict(B1=down("debt_ratio", .3, .7), B2=down("net_debt_equity", 0., 1.),
                  B3=up("interest_coverage", 1.5, 6), B4=up("current_ratio", .8, 1.5),
                  B5=up("cash_coverage", .1, .6), C1=v.get("positive_cfo_fraction"),
                  C2=up("cash_conversion", .5, 1.2), C3=up("fcf_to_cfo", 0., .5),
                  C4=down("accrual_ratio", .02, .15), R1=up("roic", .04, .15),
                  R2=up("roe", .05, .2), G1=up("revenue_growth", -.1, .2),
                  G2=down("dso_change", 0., 15.), G3=down("dio_change", 0., 30.),
                  A1=down("adjusted_share_growth", .02, .1), A2=down("distribution_fcf", 1., 1.5),
                  A3=up("incremental_roic", .04, .15))
    for mid, margin, change, ceiling, lower in (("R3", "gross_margin", "gross_margin_change", .4, -.05),
                                               ("R4", "operating_margin", "operating_margin_change", .2, -.04)):
        level, delta = v.get(margin), v.get(change)
        scores[mid] = (0. if level is not None and level <= 0 else
                       .5 * ramp(level, 0., ceiling) + .5 * ramp(delta, lower, .02)
                       if level is not None and delta is not None else None)
    # Explicit known economic branches, never replacements for missing dependencies.
    for mid, state in (("B3", "debt_free"), ("B4", "no_current_liabilities"), ("B5", "no_current_liabilities")):
        if v.get(state) is True: scores[mid] = 1.
    for mid, key in (("C2", "net_income_nonpositive"), ("C3", "cfo_nonpositive"),
                     ("R1", "invested_capital_nonpositive"), ("G1", "revenue_base_nonpositive")):
        if v.get(key) is True: scores[mid] = 0.
    if v.get("no_distribution") is True: scores["A2"] = 1.
    if v.get("fcf_nonpositive_distribution") is True: scores["A2"] = 0.
    if v.get("capital_not_increased") is True:
        scores["A3"] = .5 if v.get("nopat_not_decreased") is True else 0.
    raw_keys = dict(zip(METRICS, ("debt_ratio", "net_debt_equity", "interest_coverage", "current_ratio", "cash_coverage",
        "positive_cfo_fraction", "cash_conversion", "fcf_to_cfo", "accrual_ratio", "roic", "roe", "gross_margin",
        "operating_margin", "revenue_growth", "dso_change", "dio_change", "adjusted_share_growth", "distribution_fcf", "incremental_roic")))
    result = []
    for key in METRICS:
        root = ("ROIC" if key in v.get("roic_gap_metrics", ()) else
                "CAPEX" if key in GAPS["CAPEX"] and v.get("capex_gap") else None)
        # A3 unrelated history gaps and A2 missing distributions cannot masquerade as roots.
        if key == "A2" and v.get("distribution_missing"): root = None
        if key == "A3" and v.get("nopat_history_missing"): root = None
        score = None if root else scores[key]
        result.append(Metric(key, score, root, v.get(raw_keys[key]), tuple(v.get("dependency_ids", ()))))
    return result


def evaluate_financials(records, *, symbol, market, cutoff, safety_receipt=None, policy=None) -> dict:
    """Latest canonical quarters only. No annual/3-quarter/old-window fallback."""
    from domain.les_v7.contracts import QUALITY_REPORTED_PROFILES
    if policy is not None and policy.get("profile_id") in QUALITY_REPORTED_PROFILES:
        from domain.les_v7.quality_reported import evaluate_financials_reported
        return evaluate_financials_reported(records, symbol=symbol, market=market, cutoff=cutoff,
                                            safety_receipt=safety_receipt, policy=policy)
    if policy is not None and policy.get("profile_id") == "QG_V7_EVIDENCE_R2":
        from domain.les_v7.quality_r2 import evaluate_financials_r2
        return evaluate_financials_r2(records, symbol=symbol, market=market, cutoff=cutoff,
                                      safety_receipt=safety_receipt, policy=policy)
    core_valid = {f: False for f in CORE}
    empty = lambda reason: dict(quality_gate([], core_valid=core_valid, safety_complete=False), reason_codes=[reason])
    if (not policy or not all(policy.get(k) for k in ("filing_schedule_version", "source_mapping_id", "latest_required_period"))
            or policy.get("tax_rule") != "VERIFIED_TTM_EXPENSE_OVER_PRETAX"
            or policy.get("capex_rule") != "PPE_PLUS_INTANGIBLES_POSITIVE_OUTFLOW"):
        return dict(empty("QUALITY_POLICY_DEPENDENCIES_NOT_PROVIDED"), status="CONFIG_INCOMPLETE")
    if not records:
        return empty("NO_POINT_IN_TIME_FINANCIALS")
    try:
        periods, checks, excluded = select_quality_inputs(records, symbol=symbol, market=market, cutoff=pd.Timestamp(timestamp(cutoff)))
        if not periods: return empty("NO_POINT_IN_TIME_FINANCIALS")
        latest = max(periods)
        required_period = pd.Period(policy["latest_required_period"], freq="Q")
        if latest != required_period:
            return dict(empty("LATEST_REQUIRED_REPORTING_PERIOD_UNAVAILABLE"), latest_period=str(latest),
                        missing_quarters=[str(required_period)])
        row = periods[latest]
        quarters, traces = normalize_quarters_with_diagnostics(periods)
    except (ValueError, TypeError) as error:
        return empty(str(error))
    needed = tuple(latest - i for i in range(4))
    basis = row.get("revision_basis_id")
    comparable = bool(basis) and all(periods.get(p, {}).get("revision_basis_id") == basis for p in needed)
    def amount(field, p):
        origin = periods.get(p)
        if origin is None: return None
        check = validate_field(origin, field, period=p, cutoff=pd.Timestamp(timestamp(cutoff)))
        if check["evidence_level"] != "CANONICAL_ORIGINAL" or check["value"] is None: return None
        if field in quarters.get(p, {}): return quarters[p][field]
        if origin.get("flow_basis") == "QUARTER": return check["value"]
        proof = origin.get("quarter_field_evidence", {}).get(field, {})
        return validated_proof_value(proof, origin, field, start=p.start_time.date().isoformat(), end=p.end_time.date().isoformat())
    def ttm(field, offset=0):
        ps = tuple(latest - offset - i for i in range(4))
        values = [amount(field, p) for p in ps]
        compatible = comparable and all(periods.get(p, {}).get("revision_basis_id") == basis for p in ps)
        return fsum(values) if compatible and all(v is not None for v in values) else None
    def balance(field, offset=0):
        p = latest - offset
        origin = periods.get(p)
        if origin is None: return None
        check = validate_field(origin, field, period=p, cutoff=pd.Timestamp(timestamp(cutoff)), instant=True)
        return check["value"] if check["evidence_level"] == "CANONICAL_ORIGINAL" else None
    def average(field):
        a, b = balance(field), balance(field, 4)
        return (a + b) / 2 if a is not None and b is not None else None
    t = {key: ttm(key) for key in CORE}
    core_valid = {key: t[key] is not None for key in CORE}
    assets, liabilities, equity = (balance(k) for k in ("total_assets", "total_liabilities", "total_equity"))
    hard = (equity is not None and equity <= 0 or assets is not None and assets <= 0
            or all(x is not None for x in (assets, liabilities, equity)) and abs(assets - liabilities - equity) / assets >= .01
            or row.get("auditor_opinion") in {"ADVERSE", "DISCLAIMER", "QUALIFIED_MATERIAL_MISSTATEMENT"}
            or row.get("going_concern") is True or row.get("material_accounting_issue") is True)
    safety_fields = ("total_assets", "total_liabilities", "total_equity", "parent_equity", "current_assets",
                     "current_liabilities", "cash", "interest_bearing_debt")
    safety = all(balance(k) is not None for k in safety_fields) and safety_receipt is not None and safety_receipt.state == "PASS"
    v = dict(dependency_ids=tuple(identity({k: r.get(k) for k in ("symbol", "period_end", "revision_id", "content_sha256")})
                                 for r in periods.values()))
    cash, debt, current_assets, current_liabilities = (balance(k) for k in ("cash", "interest_bearing_debt", "current_assets", "current_liabilities"))
    net_debt = debt - cash if debt is not None and cash is not None else None
    pretax, interest, tax = ttm("pretax_income"), ttm("interest_expense"), ttm("income_tax")
    ebit = pretax + interest if pretax is not None and interest is not None else None
    capex, capex_sign_invalid = [], False
    for p in needed:
        ppe, intangible = amount("capex", p), amount("intangible_capex", p)
        capex_sign_invalid |= any(v is not None and v < 0 for v in (ppe, intangible))
        capex.append(ppe + intangible if ppe is not None and intangible is not None and ppe >= 0 and intangible >= 0 else None)
    capex_missing = sum(x is None for x in capex)
    capex_ttm = fsum(capex) if comparable and not capex_missing else None
    fcf = t["cfo"] - capex_ttm if t["cfo"] is not None and capex_ttm is not None else None
    debt_free = bool(debt == 0 and interest == 0 and safety_receipt and
                     "VERIFIED_NO_MATERIAL_CAPITALIZED_INTEREST" in safety_receipt.reasons)
    safety = safety and interest is not None and interest >= 0 and (interest > 0 or debt_free)
    v.update(debt_ratio=_ratio(liabilities, assets), net_debt_equity=_ratio(net_debt, equity),
             interest_coverage=_ratio(ebit, interest), current_ratio=_ratio(current_assets, current_liabilities),
             cash_coverage=_ratio(cash, current_liabilities),
             no_current_liabilities=current_liabilities == 0 and safety, debt_free=debt_free,
             positive_cfo_fraction=sum(amount("cfo", p) > 0 for p in needed) / 4 if core_valid["cfo"] and t["cfo"] > 0 else 0. if core_valid["cfo"] else None,
             cash_conversion=_ratio(t["cfo"], t["net_income"]),
             net_income_nonpositive=t["net_income"] is not None and t["net_income"] <= 0,
             fcf_to_cfo=_ratio(fcf, t["cfo"]), cfo_nonpositive=t["cfo"] is not None and t["cfo"] <= 0,
             accrual_ratio=_ratio(t["net_income"] - t["cfo"] if all(t[k] is not None for k in ("net_income", "cfo")) else None, average("total_assets")),
             capex_gap=capex_missing > 0)
    def capital(offset=0):
        values = [balance(k, offset) for k in ("total_equity", "interest_bearing_debt", "cash")]
        return values[0] + values[1] - values[2] if all(x is not None for x in values) else None
    ic, old_ic = capital(), capital(4)
    rate = _ratio(tax, pretax)
    tax_valid = rate is not None and 0 <= rate <= 1
    nopat = t["operating_income"] * (1 - rate) if tax_valid and t["operating_income"] is not None else None
    old_tax, old_pretax, old_op = ttm("income_tax", 4), ttm("pretax_income", 4), ttm("operating_income", 4)
    old_rate = _ratio(old_tax, old_pretax)
    old_nopat = old_op * (1 - old_rate) if old_op is not None and old_rate is not None and 0 <= old_rate <= 1 else None
    delta_ic = ic - old_ic if ic is not None and old_ic is not None else None
    delta_nopat = nopat - old_nopat if nopat is not None and old_nopat is not None else None
    tax_gap = tax is None or old_tax is None
    roic_shape = (tax_gap and ic is not None and old_ic is not None) or (not tax_gap and tax_valid and (ic is None) != (old_ic is None))
    gap_metrics = []
    if tax is None or ic is None or old_ic is None: gap_metrics.append("R1")
    if tax_gap or ic is None or old_ic is None: gap_metrics.append("A3")
    v.update(roic=_ratio(nopat, (ic + old_ic) / 2 if ic is not None and old_ic is not None else None),
             roic_gap_metrics=gap_metrics, invested_capital_nonpositive=ic is not None and ic <= 0 or old_ic is not None and old_ic <= 0,
             incremental_roic=_ratio(delta_nopat, delta_ic), nopat_history_missing=old_op is None,
             capital_not_increased=delta_ic is not None and delta_ic <= .01 * max(abs(ic), abs(old_ic)),
             nopat_not_decreased=delta_nopat is not None and delta_nopat >= 0,
             roe=_ratio(ttm("parent_net_income"), average("parent_equity")))
    for name, numerator in (("gross_margin", "cost_of_revenue"), ("operating_margin", "operating_income")):
        current = _ratio(t[numerator], t["revenue"])
        old = _ratio(ttm(numerator, 4), ttm("revenue", 4))
        if name == "gross_margin": current, old = (1 - current if current is not None else None), (1 - old if old is not None else None)
        v[name] = current
        v[name + "_change"] = current - old if current is not None and old is not None else None
    prior_revenue = ttm("revenue", 4)
    growth = _ratio(t["revenue"], prior_revenue)
    v["revenue_growth"] = growth - 1 if growth is not None else None
    v["revenue_base_nonpositive"] = prior_revenue is not None and prior_revenue <= 0
    for name, field, flow in (("dso_change", "accounts_receivable", "revenue"), ("dio_change", "inventory", "cost_of_revenue")):
        now = _ratio(balance(field), ttm(flow))
        old = _ratio(balance(field, 4), ttm(flow, 4))
        v[name] = 365 * (now - old) if now is not None and old is not None else None
    # Shares need a shares-unit producer. No TWD-normalized balance masquerades as dilution.
    share = row.get("adjusted_share_growth_evidence", {})
    v["adjusted_share_growth"] = share.get("value") if (share.get("source_hash") == row.get("content_sha256")
        and share.get("status") == "VERIFIED" and share.get("policy_id") and share.get("available_at")
        and timestamp(share["available_at"]) <= timestamp(cutoff)) else None
    dividend, buybacks = ttm("dividends"), ttm("share_repurchases")
    distribution = dividend + buybacks if dividend is not None and buybacks is not None else None
    v.update(distribution_fcf=_ratio(distribution, fcf), distribution_missing=distribution is None,
             no_distribution=distribution == 0 and capex_ttm is not None,
             fcf_nonpositive_distribution=fcf is not None and fcf <= 0 and distribution is not None and distribution > 0)
    metrics = map_metrics(v)
    undefined_day_denominator = any(ttm(field, offset) is not None and ttm(field, offset) <= 0
        for field in ("revenue", "cost_of_revenue") for offset in (0, 4))
    result = quality_gate(metrics, core_valid=core_valid, safety_complete=safety, hard_reject=bool(hard),
        config_complete=not undefined_day_denominator and not capex_sign_invalid,
        gap_shapes=dict(ROIC=roic_shape, CAPEX=comparable and capex_missing == 1),
        source_reliability=100 if all(r.get("source") == "MOPS_IXBRL" for r in periods.values()) else 95)
    result.update(core_ttm=t, core_ttm_quarters=[str(p) for p in needed], latest_period=str(latest),
                  missing_quarters=[str(p) for p in needed if p not in periods],
                  financial_inputs=v, field_checks=checks, excluded_history=excluded,
                  capex_ttm=capex_ttm, fcf_ttm=fcf, capex_missing_quarters=capex_missing)
    if not comparable: result["reason_codes"].append("REVISION_BASIS_UNVERIFIED")
    if undefined_day_denominator: result["reason_codes"].append("DSO_DIO_ECONOMIC_BRANCH_NOT_CONFIGURED")
    if capex_sign_invalid: result["reason_codes"].append("CAPEX_SIGN_MAPPING_INVALID_NOT_BOUNDED_GAP")
    return result

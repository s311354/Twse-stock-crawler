"""Deterministic point-in-time derivation of LES v4 MOPS fundamentals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import re
from typing import Any

import pandas as pd

from domain.text_normalization import normalize_cjk_text


FUNDAMENTAL_DERIVATION_VERSION = (
    "2026-10-05-qs-dc-v20-source-bound-cost-hierarchy"
)


@dataclass(frozen=True)
class FiledFact:
    concept: str
    numeric_value: float | None
    text_value: str | None
    start_date: str | None
    end_date: str | None
    instant: str | None
    dimensions: tuple[tuple[str, str], ...] = ()
    unit: str | None = None
    context_id: str | None = None


@dataclass(frozen=True)
class FiledFinancials:
    fiscal_year: int
    quarter: int
    available_at: str
    revision_number: int
    facts: tuple[FiledFact, ...]
    correction_description: str | None = None
    availability_source: str = "LEGACY_UNVERIFIED"
    filing_id: int | None = None
    content_sha256: str | None = None
    source_url: str | None = None
    correction_announced_on: str | None = None
    period_end: str | None = None
    statement_scope: str = "consolidated"
    audit_status: str | None = None
    # Read from the hash-verified original, never inferred from fiscal dates.
    taxonomy_schema_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class FundamentalBundle:
    annual_fundamentals: pd.DataFrame
    snapshot: dict[str, Any]
    as_of: str | None = None


@dataclass(frozen=True)
class CachedFundamentalHeader:
    """Lightweight active snapshot identity; never a Quality certification."""

    symbol: str
    sector: str
    filing_set_sha256: str
    derivation_version: str
    available_at_epoch: int
    latest_period: tuple[int, int]
    annual_history_years: int
    full_catalog_checked_at_epoch: int
    snapshot_cache_key: str
    filing_schema_version: str | None
    has_mops_filings: bool | None


@dataclass(frozen=True)
class CachedFundamentalBundle:
    """Derived snapshot plus the metadata required for safe fast-path reuse."""

    symbol: str
    sector: str
    filing_set_sha256: str
    derivation_version: str
    available_at_epoch: int
    latest_period: tuple[int, int]
    annual_history_years: int
    full_catalog_checked_at_epoch: int
    bundle: FundamentalBundle


REVENUE = ("ifrs-full:Revenue", "ifrs-full:RevenueFromContractsWithCustomers")
NET_INCOME = (
    "ifrs-full:ProfitLossAttributableToOwnersOfParent",
    "ifrs-full:ProfitLoss",
)
OPERATING_CASH_FLOW = ("ifrs-full:CashFlowsFromUsedInOperatingActivities",)
PPE_CAPEX = (
    "ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
    "ifrs-full:PurchaseOfPropertyPlantAndEquipment",
)
INTANGIBLE_CAPEX = (
    "ifrs-full:PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities",
    "ifrs-full:PurchaseOfIntangibleAssets",
)
CAPEX = PPE_CAPEX
EBIT = ("ifrs-full:ProfitLossFromOperatingActivities",)
GROSS_PROFIT = (
    "ifrs-full:GrossProfit",
    "tifrs-bsci-ci:GrossProfitLossFromOperations",
)
PRETAX_INCOME = ("ifrs-full:ProfitLossBeforeTax",)
INCOME_TAX = ("ifrs-full:IncomeTaxExpenseContinuingOperations", "ifrs-full:IncomeTaxExpense")
DILUTED_EPS = ("ifrs-full:DilutedEarningsLossPerShare", "ifrs-full:BasicEarningsLossPerShare")
CASH = ("ifrs-full:CashAndCashEquivalents",)
PARENT_EQUITY = ("ifrs-full:EquityAttributableToOwnersOfParent",)
TOTAL_EQUITY = ("ifrs-full:Equity",)
EQUITY = PARENT_EQUITY + TOTAL_EQUITY
TOTAL_ASSETS = ("ifrs-full:Assets",)
TOTAL_LIABILITIES = ("ifrs-full:Liabilities",)
CURRENT_ASSETS = ("ifrs-full:CurrentAssets",)
CURRENT_LIABILITIES = ("ifrs-full:CurrentLiabilities",)
FINANCE_COSTS = ("ifrs-full:FinanceCosts",)
DEPRECIATION = ("ifrs-full:AdjustmentsForDepreciationExpense",)
AMORTISATION = ("ifrs-full:AdjustmentsForAmortisationExpense",)
DIVIDENDS_PAID = ("ifrs-full:DividendsPaidClassifiedAsFinancingActivities",)
DEBT_COMPONENTS = (
    "ifrs-full:ShorttermBorrowings",
    "ifrs-full:LongtermBorrowings",
    "tifrs-bsci-ci:LongtermLiabilitiesCurrentPortion",
    "ifrs-full:CurrentBorrowings",
    "ifrs-full:NoncurrentBorrowings",
    "ifrs-full:BondsPayableCurrent",
    "ifrs-full:BondsPayableNoncurrent",
    "ifrs-full:CurrentLeaseLiabilities",
    "ifrs-full:NoncurrentLeaseLiabilities",
    "ifrs-full:CurrentFinanceLeaseLiabilities",
    "ifrs-full:NoncurrentFinanceLeaseLiabilities",
)


def derive_fundamentals(
    filings: list[FiledFinancials],
    *,
    signal_date: str | pd.Timestamp,
    sector: str,
) -> FundamentalBundle:
    """Derive annual metrics and the latest TTM snapshot without look-ahead."""

    parsed_signal_date = pd.to_datetime(signal_date, utc=True, errors="coerce")
    if pd.isna(parsed_signal_date):
        return FundamentalBundle(pd.DataFrame(), {})

    eligible = [
        filing
        for filing in filings
        if (available := pd.to_datetime(filing.available_at, utc=True, errors="coerce")) is not pd.NaT
        and not pd.isna(available)
        and available <= parsed_signal_date
    ]
    latest_by_period: dict[tuple[int, int], FiledFinancials] = {}
    for filing in sorted(eligible, key=lambda item: (item.fiscal_year, item.quarter, item.available_at, item.revision_number)):
        latest_by_period[(filing.fiscal_year, filing.quarter)] = filing
    selected = list(latest_by_period.values())
    # Keep one report scope through annual/TTM arithmetic. Scope fallback must
    # not silently splice an individual annual report into consolidated YTD.
    if selected:
        scope = max(selected, key=lambda f: (f.fiscal_year, f.quarter)).statement_scope
        selected = [f for f in selected if f.statement_scope == scope]
        latest_by_period = {(f.fiscal_year, f.quarter): f for f in selected}
    annual_rows = _derive_annual_rows(latest_by_period)
    annual = pd.DataFrame(annual_rows)
    snapshot = _derive_snapshot(selected, latest_by_period, sector)
    if selected:
        # Prior-year revisions may arrive AFTER the latest fiscal period's
        # release. Derived metrics cannot predate any source they consume.
        available = max(selected, key=lambda f: pd.Timestamp(f.available_at)).available_at
        snapshot["available_at"] = available
        snapshot["source_available_at"] = [f.available_at for f in selected]
        for index, row in annual.iterrows():
            sources = [f for f in selected if f.quarter == 4 and f.fiscal_year <= row["fiscal_year"]]
            annual.at[index, "available_at"] = max(sources, key=lambda f: pd.Timestamp(f.available_at)).available_at
    return FundamentalBundle(annual_fundamentals=annual, snapshot=snapshot)


def _derive_annual_rows(
    filings: dict[tuple[int, int], FiledFinancials],
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    invested_capital_by_year: dict[int, float | None] = {}
    for year in sorted(year for year, quarter in filings if quarter == 4):
        filing = filings[(year, 4)]
        revenue = _flow_value(filing, REVENUE, year)
        net_income = _flow_value(filing, NET_INCOME, year)
        cfo = _flow_value(filing, OPERATING_CASH_FLOW, year)
        capex = _capex_value(filing, year)
        ebit = _flow_value(filing, EBIT, year)
        gross_profit = _flow_value(filing, GROSS_PROFIT, year)
        pretax = _flow_value(filing, PRETAX_INCOME, year)
        tax = _flow_value(filing, INCOME_TAX, year)
        diluted_eps = _flow_value(filing, DILUTED_EPS, year)
        cash = _instant_value(filing, CASH, year)
        equity = _instant_value(filing, EQUITY, year)
        total_equity = _instant_value(filing, TOTAL_EQUITY, year)
        debt = _debt_value(filing, year)
        total_assets = _instant_value(filing, TOTAL_ASSETS, year)
        total_liabilities = _instant_value(filing, TOTAL_LIABILITIES, year)
        current_assets = _instant_value(filing, CURRENT_ASSETS, year)
        current_liabilities = _instant_value(filing, CURRENT_LIABILITIES, year)
        diluted_shares = _safe_div(net_income, diluted_eps)
        gross_margin = _safe_div(gross_profit, revenue)
        operating_margin = _safe_div(ebit, revenue)
        tax_rate = _safe_div(tax, pretax)
        if tax_rate is not None:
            tax_rate = min(max(tax_rate, 0.0), 1.0)
        invested_capital = None
        if equity is not None and debt is not None and cash is not None:
            invested_capital = equity + debt - cash
        invested_capital_by_year[year] = invested_capital
        previous_capital = invested_capital_by_year.get(year - 1)
        average_capital = invested_capital
        if invested_capital is not None and previous_capital is not None:
            average_capital = (invested_capital + previous_capital) / 2
        nopat = ebit * (1 - tax_rate) if ebit is not None and tax_rate is not None else None
        roic = _safe_div(nopat, average_capital)
        rows.append(
            {
                "fiscal_year": year,
                "available_at": filing.available_at,
                "availability_source": filing.availability_source,
                "filing_id": filing.filing_id,
                "revision_number": filing.revision_number,
                "content_sha256": filing.content_sha256,
                "source_url": filing.source_url,
                "correction_announced_on": filing.correction_announced_on,
                "revenue": revenue,
                "net_income": net_income,
                "operating_cash_flow": cfo,
                "capex": capex,
                "free_cash_flow": cfo - capex if cfo is not None and capex is not None else None,
                "roic": roic,
                "gross_margin": gross_margin,
                "operating_margin": operating_margin,
                "diluted_shares": diluted_shares,
                "gross_profit": gross_profit,
                "ebit": ebit,
                "cash": cash,
                "total_debt": debt,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "equity": equity,
                "total_equity": total_equity,
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "dividends_paid": _flow_value(filing, DIVIDENDS_PAID, year),
            }
        )
    return rows


def _derive_snapshot(
    filings: list[FiledFinancials],
    filings_by_period: dict[tuple[int, int], FiledFinancials],
    sector: str,
) -> dict[str, Any]:
    if not filings:
        return {}
    latest = max(filings, key=lambda item: (item.fiscal_year, item.quarter, item.available_at, item.revision_number))
    year = latest.fiscal_year
    revenue_ttm = _ttm_value(latest, filings_by_period, REVENUE)
    net_income_ttm = _ttm_value(latest, filings_by_period, NET_INCOME)
    cfo_ttm = _ttm_value(latest, filings_by_period, OPERATING_CASH_FLOW)
    capex_ttm = _ttm_capex(latest, filings_by_period)
    gross_profit_ttm = _ttm_value(latest, filings_by_period, GROSS_PROFIT)
    ttm_eps = _ttm_eps(latest, filings_by_period)
    ebit_ttm = _ttm_value(latest, filings_by_period, EBIT)
    depreciation_ttm = _ttm_value(latest, filings_by_period, DEPRECIATION)
    amortisation_ttm = _ttm_value(latest, filings_by_period, AMORTISATION)
    interest_expense_ttm = _ttm_value(latest, filings_by_period, FINANCE_COSTS)
    ebitda_ttm = None
    if ebit_ttm is not None and depreciation_ttm is not None and amortisation_ttm is not None:
        ebitda_ttm = ebit_ttm + abs(depreciation_ttm) + abs(amortisation_ttm)
    audit = _derive_audit_evidence(latest)
    going_concern = audit["going_concern"]
    accounting_issue = audit["material_accounting_issue"]
    accounting_concern_classification = audit["accounting_concern_classification"]
    if _is_material_correction(latest.correction_description):
        accounting_issue = True
        accounting_concern_classification = "MATERIAL_RESTATEMENT"

    cash = _instant_value(latest, CASH, year)
    debt = _debt_value(latest, year)
    equity = _instant_value(latest, EQUITY, year)
    total_equity = _instant_value(latest, TOTAL_EQUITY, year)
    total_assets = _instant_value(latest, TOTAL_ASSETS, year)
    total_liabilities = _instant_value(latest, TOTAL_LIABILITIES, year)
    current_assets = _instant_value(latest, CURRENT_ASSETS, year)
    current_liabilities = _instant_value(latest, CURRENT_LIABILITIES, year)
    previous_equity = _instant_value(latest, EQUITY, year - 1)
    previous_assets = _instant_value(latest, TOTAL_ASSETS, year - 1)
    previous_cash = _instant_value(latest, CASH, year - 1)
    previous_debt = _debt_value(latest, year - 1)
    average_equity = _average(equity, previous_equity)
    average_assets = _average(total_assets, previous_assets)
    average_invested_capital = _average(
        _invested_capital(equity, debt, cash),
        _invested_capital(previous_equity, previous_debt, previous_cash),
    )
    pretax_ttm = _ttm_value(latest, filings_by_period, PRETAX_INCOME)
    tax_ttm = _ttm_value(latest, filings_by_period, INCOME_TAX)
    effective_tax_rate = _safe_div(tax_ttm, pretax_ttm)
    if effective_tax_rate is not None:
        effective_tax_rate = min(max(effective_tax_rate, 0.0), 1.0)
    nopat_ttm = (
        ebit_ttm * (1 - effective_tax_rate)
        if ebit_ttm is not None and effective_tax_rate is not None
        else None
    )
    free_cash_flow_ttm = (
        cfo_ttm - capex_ttm
        if cfo_ttm is not None and capex_ttm is not None
        else None
    )
    interest_expense = abs(interest_expense_ttm) if interest_expense_ttm is not None else None
    interest_coverage = None
    if ebit_ttm is not None and interest_expense is not None:
        interest_coverage = (
            None
            if interest_expense == 0
            else _safe_div(ebit_ttm, interest_expense)
        )
    previous_revenue_ttm = _previous_ttm_value(latest, filings_by_period, REVENUE)
    previous_ebit_ttm = _previous_ttm_value(latest, filings_by_period, EBIT)
    previous_gross_profit_ttm = _previous_ttm_value(latest, filings_by_period, GROSS_PROFIT)
    current_gross_margin = _safe_div(gross_profit_ttm, revenue_ttm)
    previous_gross_margin = _safe_div(previous_gross_profit_ttm, previous_revenue_ttm)
    latest_annual = _latest_annual_values(filings_by_period)
    share_dilution_yoy = latest_annual.get("share_dilution_yoy")
    bvps_trend = latest_annual.get("bvps_trend")

    return {
        "derivation_version": FUNDAMENTAL_DERIVATION_VERSION,
        "available_at": latest.available_at,
        "availability_source": latest.availability_source,
        "filing_id": latest.filing_id,
        "revision_number": latest.revision_number,
        "content_sha256": latest.content_sha256,
        "source_url": latest.source_url,
        "correction_announced_on": latest.correction_announced_on,
        "period_end": latest.period_end,
        "statement_scope": latest.statement_scope,
        "audit_status": latest.audit_status,
        "audit_report_text_raw": audit["audit_report_text_raw"],
        "audit_report_text_normalized": audit[
            "audit_report_text_normalized"
        ],
        "ttm_eps": ttm_eps,
        "revenue_ttm": revenue_ttm,
        "net_income_ttm": net_income_ttm,
        "operating_cash_flow_ttm": cfo_ttm,
        "capex_ttm": capex_ttm,
        "gross_profit_ttm": gross_profit_ttm,
        "gross_margin_ttm": current_gross_margin,
        "cash": cash,
        "total_debt": debt,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "equity": equity,
        "total_equity": total_equity,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "ebitda_ttm": ebitda_ttm,
        "ebit_ttm": ebit_ttm,
        "interest_expense_ttm": interest_expense,
        "interest_expense_reported_zero": interest_expense == 0,
        "going_concern": going_concern,
        "material_accounting_issue": accounting_issue,
        "audit_opinion": audit["audit_opinion"],
        "current_audit_opinion": audit["current_audit_opinion"],
        "comparative_audit_opinion": audit["comparative_audit_opinion"],
        "audit_opinion_source": audit["audit_opinion_source"],
        "comparative_audit_opinion_source": audit[
            "comparative_audit_opinion_source"
        ],
        "accounting_concern_classification": accounting_concern_classification,
        "audit_evidence_available": audit["audit_evidence_available"],
        "audit_review_required": audit["audit_review_required"],
        "correction_description": latest.correction_description,
        "sector": sector,
        "free_cash_flow_ttm": free_cash_flow_ttm,
        "roe_ttm": _safe_div(net_income_ttm, average_equity),
        "roa_ttm": _safe_div(net_income_ttm, average_assets),
        "roic_ttm": _safe_div(nopat_ttm, average_invested_capital),
        "cash_conversion": (
            _safe_div(cfo_ttm, net_income_ttm)
            if net_income_ttm is not None and net_income_ttm > 0
            else None
        ),
        "accrual_ratio": (
            _safe_div(net_income_ttm - cfo_ttm, average_assets)
            if net_income_ttm is not None and cfo_ttm is not None
            else None
        ),
        "debt_ratio": _safe_div(total_liabilities, total_assets),
        "net_debt_to_equity": (
            _safe_div(debt - cash, equity)
            if debt is not None and cash is not None
            else None
        ),
        "interest_coverage": interest_coverage,
        "current_ratio": _safe_div(current_assets, current_liabilities),
        "cash_to_current_liabilities": _safe_div(cash, current_liabilities),
        "revenue_growth_ttm_yoy": _growth(revenue_ttm, previous_revenue_ttm),
        "operating_income_growth_ttm_yoy": _growth(ebit_ttm, previous_ebit_ttm),
        "gross_margin_change": (
            current_gross_margin - previous_gross_margin
            if current_gross_margin is not None and previous_gross_margin is not None
            else None
        ),
        "share_dilution_yoy": share_dilution_yoy,
        "bvps_trend": bvps_trend,
    }


def _capex_value(filing: FiledFinancials, year: int) -> float | None:
    values = [
        value
        for aliases in (PPE_CAPEX, INTANGIBLE_CAPEX)
        if (value := _flow_value(filing, aliases, year)) is not None
    ]
    return sum(abs(value) for value in values) if values else None


def _ttm_capex(
    latest: FiledFinancials,
    filings: dict[tuple[int, int], FiledFinancials],
) -> float | None:
    values = [
        value
        for aliases in (PPE_CAPEX, INTANGIBLE_CAPEX)
        if (value := _ttm_value(latest, filings, aliases)) is not None
    ]
    return sum(abs(value) for value in values) if values else None


def _previous_ttm_value(
    latest: FiledFinancials,
    filings: dict[tuple[int, int], FiledFinancials],
    aliases: tuple[str, ...],
) -> float | None:
    previous = filings.get((latest.fiscal_year - 1, latest.quarter))
    if previous is None:
        return None
    return _ttm_value(previous, filings, aliases)


def _average(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return (current + previous) / 2


def _invested_capital(
    equity: float | None,
    debt: float | None,
    cash: float | None,
) -> float | None:
    if equity is None or debt is None or cash is None:
        return None
    return equity + debt - cash


def _growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous <= 0:
        return None
    return current / previous - 1


def _latest_annual_values(
    filings: dict[tuple[int, int], FiledFinancials],
) -> dict[str, float | None]:
    annual = sorted(
        (filing for (year, quarter), filing in filings.items() if quarter == 4),
        key=lambda filing: filing.fiscal_year,
    )
    if len(annual) < 2:
        return {"share_dilution_yoy": None, "bvps_trend": None}
    previous, current = annual[-2:]
    previous_income = _flow_value(previous, NET_INCOME, previous.fiscal_year)
    current_income = _flow_value(current, NET_INCOME, current.fiscal_year)
    previous_shares = _safe_div(
        previous_income,
        _flow_value(previous, DILUTED_EPS, previous.fiscal_year),
    )
    current_shares = _safe_div(
        current_income,
        _flow_value(current, DILUTED_EPS, current.fiscal_year),
    )
    previous_bvps = _safe_div(
        _instant_value(previous, EQUITY, previous.fiscal_year),
        previous_shares,
    )
    current_bvps = _safe_div(
        _instant_value(current, EQUITY, current.fiscal_year),
        current_shares,
    )
    return {
        "share_dilution_yoy": _growth(current_shares, previous_shares),
        "bvps_trend": _growth(current_bvps, previous_bvps),
    }


_OPINION_CONCEPTS = {
    "tifrs-ar:UnqualifiedOpinionAbstract",
    "tifrs-ar:QualifiedOpinionAbstract",
    "tifrs-ar:AdverseOpinionAbstract",
    "tifrs-ar:DisclaimerOfOpinionAbstract",
    "tifrs-ar:UnqualifiedOpinionWithEmphasisOfMatterParagraphsOrOtherMatterParagraphs",
}
_REPORT_BODY_CONCEPT = "tifrs-ar:AccountantsReportBody"
_AUDIT_CONCEPTS = _OPINION_CONCEPTS | {_REPORT_BODY_CONCEPT}
_NON_AFFIRMATIVE_FLAGS = {"", "n", "no", "false", "0", "unknown", "n/a", "none", "未知"}


def _audit_facts_by_period(
    latest: FiledFinancials,
) -> tuple[tuple[FiledFact, ...], tuple[FiledFact, ...], date | None]:
    """Select filing-level facts by context, never mix comparative endpoints.

    Undated legacy facts are a fallback per category (abstract/body), only when
    that category has no dated current facts. Dimensional and invalid/future
    contexts are not evidence for the reporting entity's current opinion.
    """
    try:
        period_end = date.fromisoformat(latest.period_end) if latest.period_end else date(
            latest.fiscal_year, latest.quarter * 3, (31, 30, 30, 31)[latest.quarter - 1]
        )
    except (ValueError, IndexError):
        return (), (), None
    by_date: dict[date, list[FiledFact]] = {}
    undated: list[FiledFact] = []
    for fact in latest.facts:
        if fact.dimensions or fact.concept not in _AUDIT_CONCEPTS:
            continue
        endpoint = fact.end_date or fact.instant
        if endpoint is None:
            if fact.start_date is None:
                undated.append(fact)
            continue
        try:
            fact_date = date.fromisoformat(endpoint)
        except ValueError:
            continue
        if fact_date <= period_end:
            by_date.setdefault(fact_date, []).append(fact)
    current = by_date.get(period_end, [])
    categories = {fact.concept == _REPORT_BODY_CONCEPT for fact in current}
    current = current + [
        fact for fact in undated
        if (fact.concept == _REPORT_BODY_CONCEPT) not in categories
    ]
    prior_end = max((end for end in by_date if end < period_end), default=None)
    prior = by_date[prior_end] if prior_end is not None else []
    return tuple(current), tuple(prior), prior_end


def _derive_audit_evidence(latest: FiledFinancials) -> dict[str, object]:
    current, prior, prior_end = _audit_facts_by_period(latest)
    evidence = _derive_period_audit_evidence(replace(latest, facts=current))
    if prior and prior_end is not None:
        comparative = _derive_period_audit_evidence(replace(
            latest, facts=prior, fiscal_year=prior_end.year,
        ))
        source = str(comparative["audit_opinion_source"])
        opinion = comparative["audit_opinion"]
        # Structured prior facts outrank comparative mentions in the current
        # body, including conflicts/unknown flags; never promote them to PASS.
        if source != "NONE" and (
            source.startswith("ABSTRACT")
            or opinion != "UNKNOWN"
            or evidence["comparative_audit_opinion"] is None
        ):
            if opinion == "UNKNOWN" and _opinion_abstract_is_true(
                prior, "tifrs-ar:QualifiedOpinionAbstract"
            ):
                opinion = "QUALIFIED_UNKNOWN" if source == "ABSTRACT" else "UNKNOWN"
            evidence["comparative_audit_opinion"] = opinion
            evidence["comparative_audit_opinion_source"] = (
                f"{source}_COMPARATIVE"
                if source.startswith("ABSTRACT")
                else "REPORT_BODY_COMPARATIVE"
            )
    # Keep all report bodies for traceability; only period-selected evidence
    # above can affect the current opinion and hard-risk flags.
    raw = " ".join(
        fact.text_value for fact in latest.facts
        if fact.concept == _REPORT_BODY_CONCEPT and fact.text_value
    )
    evidence["audit_report_text_raw"] = raw
    evidence["audit_report_text_normalized"] = normalize_cjk_text(raw)
    return evidence


def _derive_period_audit_evidence(latest: FiledFinancials) -> dict[str, object]:
    facts = latest.facts
    report_bodies = [
        fact.text_value or ""
        for fact in facts
        if fact.concept == "tifrs-ar:AccountantsReportBody" and fact.text_value
    ]
    summary_texts = [
        fact.text_value or ""
        for fact in facts
        if fact.concept in _OPINION_CONCEPTS
        and fact.text_value
        and normalize_cjk_text(fact.text_value).lower()
        not in _NON_AFFIRMATIVE_FLAGS | {"y", "yes", "true", "1"}
    ]
    # Each selected fact starts in its own reporting context. The analysis-only
    # label prevents a prior section in one fact from leaking into the next;
    # raw/normalized evidence returned by the wrapper is never annotated.
    normalized = normalize_cjk_text(
        " ".join("本期:" + text for text in report_bodies + summary_texts)
    ).lower()
    text_evidence_available = bool(normalized.strip())
    adverse_abstract = _opinion_abstract_is_true(
        facts,
        "tifrs-ar:AdverseOpinionAbstract",
    )
    disclaimer_abstract = _opinion_abstract_is_true(
        facts,
        "tifrs-ar:DisclaimerOfOpinionAbstract",
    )
    qualified_abstract = _opinion_abstract_is_true(
        facts,
        "tifrs-ar:QualifiedOpinionAbstract",
    )
    unqualified_abstract = _opinion_abstract_is_true(
        facts,
        (
            "tifrs-ar:UnqualifiedOpinionAbstract",
            (
                "tifrs-ar:"
                "UnqualifiedOpinionWithEmphasisOfMatterParagraphsOrOtherMatterParagraphs"
            ),
        ),
    )
    abstract_values = {
        "ADVERSE": adverse_abstract,
        "DISCLAIMER": disclaimer_abstract,
        "QUALIFIED": qualified_abstract,
        "UNMODIFIED": unqualified_abstract,
    }
    true_abstracts = [
        name for name, is_true in abstract_values.items() if is_true
    ]
    abstracts_present = any(fact.concept in _OPINION_CONCEPTS for fact in facts)
    evidence_available = text_evidence_available or abstracts_present
    abstract_conflict = len(true_abstracts) > 1
    modified_abstract = next(
        (
            name
            for name in ("ADVERSE", "DISCLAIMER", "QUALIFIED")
            if abstract_values[name]
        ),
        None,
    )
    opinion_source = (
        "ABSTRACT_CONFLICT"
        if abstract_conflict
        else "ABSTRACT"
        if true_abstracts
        else "ABSTRACT_UNKNOWN"
        if abstracts_present
        else "REPORT_BODY_FALLBACK"
        if text_evidence_available
        else "NONE"
    )
    going_concern_markers = (
        "繼續經營重大不確定性",
        "持續經營重大不確定性",
        "material uncertainty related to going concern",
    )
    adverse_markers = ("否定意見之基礎", "basis for adverse opinion")
    disclaimer_markers = (
        "無法表示意見之基礎",
        "無法表示結論之基礎",
        "basis for disclaimer of opinion",
    )
    qualified_markers = (
        "保留意見之基礎",
        "保留結論之基礎",
        "basis for qualified opinion",
        "basis for qualified conclusion",
        "basis of qualified opinion",
        "basis of qualified conclusion",
    )
    going_concern = (
        _contains_current_period_marker(
            normalized,
            going_concern_markers,
            latest.fiscal_year,
        )
        if text_evidence_available or unqualified_abstract
        else None
    )
    # Body fallback is absence-based, not truth-based. Present but empty,
    # false, or conflicting structured flags must stay UNKNOWN. A qualified
    # abstract still needs its current basis text to classify the reason.
    infer_from_body = not abstracts_present
    has_adverse_opinion = modified_abstract == "ADVERSE" or (
        infer_from_body
        and _contains_current_period_marker(
            normalized,
            adverse_markers,
            latest.fiscal_year,
        )
    )
    has_disclaimer = modified_abstract == "DISCLAIMER" or (
        infer_from_body
        and _contains_current_period_marker(
            normalized,
            disclaimer_markers,
            latest.fiscal_year,
        )
    )
    has_qualified_opinion = modified_abstract == "QUALIFIED" or (
        infer_from_body
        and _contains_current_period_marker(
            normalized,
            qualified_markers,
            latest.fiscal_year,
        )
    )
    has_unmodified_opinion = unqualified_abstract or (
        infer_from_body
        and _contains_current_period_marker(
            normalized,
            (
                "無保留意見", "無保留結論", "unmodified opinion",
                "unqualified opinion", "unmodified conclusion",
                "並未發現財務報告未允當表達",
            ),
            latest.fiscal_year,
        )
    )

    if abstract_conflict:
        opinion = "UNKNOWN"
        material_issue = None
        concern_classification = "ABSTRACT_CONFLICT"
    elif has_adverse_opinion:
        opinion = "ADVERSE"
        material_issue = False
        concern_classification = "ADVERSE"
    elif has_disclaimer:
        opinion = "DISCLAIMER"
        material_issue = False
        concern_classification = "DISCLAIMER"
    elif has_qualified_opinion:
        qualified_basis = _qualified_basis_text(
            normalized,
            current_fiscal_year=latest.fiscal_year,
        )
        if _is_material_misstatement_basis(qualified_basis):
            opinion = "QUALIFIED_MATERIAL_MISSTATEMENT"
            material_issue = True
            concern_classification = "QUALIFIED_MATERIAL_MISSTATEMENT"
        elif _is_non_significant_component_scope_limitation(qualified_basis):
            opinion = "QUALIFIED_SCOPE_LIMITATION"
            material_issue = False
            concern_classification = "QUALIFIED_SCOPE_LIMITATION"
        else:
            # A qualified flag without a classifiable basis is not proof of a
            # material misstatement. Quality Gate handles it as unknown data.
            opinion = "UNKNOWN"
            material_issue = None
            concern_classification = "UNKNOWN"
    elif has_unmodified_opinion and latest.audit_status == "reviewed":
        opinion = "REVIEWED_UNMODIFIED"
        material_issue = False
        concern_classification = "NONE"
    elif has_unmodified_opinion:
        opinion = "UNMODIFIED"
        material_issue = False
        concern_classification = "NONE"
    else:
        opinion = "UNKNOWN"
        material_issue = None
        concern_classification = "UNKNOWN"
    comparative_opinion = _classify_comparative_opinion(
        normalized,
        current_fiscal_year=latest.fiscal_year,
        audit_status=latest.audit_status,
        adverse_markers=adverse_markers,
        disclaimer_markers=disclaimer_markers,
        qualified_markers=qualified_markers,
    )
    return {
        "going_concern": going_concern,
        "material_accounting_issue": material_issue,
        "audit_opinion": opinion,
        "current_audit_opinion": opinion,
        "comparative_audit_opinion": comparative_opinion,
        "accounting_concern_classification": concern_classification,
        "audit_evidence_available": evidence_available,
        "audit_review_required": evidence_available and opinion == "UNKNOWN",
        "audit_opinion_source": opinion_source,
        "comparative_audit_opinion_source": (
            "REPORT_BODY_COMPARATIVE"
            if comparative_opinion is not None
            else "NONE"
        ),
    }


def _opinion_abstract_is_true(
    facts: tuple[FiledFact, ...],
    concepts: str | tuple[str, ...],
) -> bool:
    accepted = {concepts} if isinstance(concepts, str) else set(concepts)
    for fact in facts:
        if fact.concept not in accepted:
            continue
        text = normalize_cjk_text(fact.text_value).lower()
        if text in _NON_AFFIRMATIVE_FLAGS - {""}:
            continue
        if text or (fact.numeric_value is not None and fact.numeric_value != 0):
            return True
    return False


def _qualified_basis_text(
    report_text: str,
    *,
    current_fiscal_year: int,
    comparative: bool = False,
) -> str:
    """Read basis sections, not references to them or generic responsibilities.

    iXBRL flattens headings into the body. A fixed character window can include
    management's duty to prevent material misstatement, which is not an audit
    finding. Unknown/unrecognized sections remain unclassified, never cleared.
    """
    report_text = _normalize_audit_text(report_text)
    start_markers = (
        "保留意見之基礎",
        "保留結論之基礎",
        "basis for qualified opinion",
        "basis for qualified conclusion",
        "basis of qualified opinion",
        "basis of qualified conclusion",
    )
    candidates: list[str] = []
    section_end = re.compile(
        r"本期:|關鍵查核事項|關鍵核閱事項|其他事項|強調事項|"
        r"管理階層(?:與治理單位)?對(?:合併|個體|個別)?財務(?:報告|報表)之責任|"
        r"會計師(?:查核|核閱)(?:合併|個體|個別)?財務(?:報告|報表)之責任|"
        r"key audit matters|other matter|emphasis of matter|"
        r"responsibilities of (?:management|the auditor)|"
        r"management[’']s responsibilities|auditor[’']s responsibilities|"
        r"保留(?:意見|結論)(?!之基礎)|qualified (?:opinion|conclusion)"
    )
    for marker in start_markers:
        search_from = 0
        while (marker_position := report_text.find(marker, search_from)) >= 0:
            basis_start = marker_position + len(marker)
            prefix = report_text[max(0, marker_position - 48) : marker_position]
            suffix = report_text[basis_start:]
            # Examples: 除保留意見之基礎段所述..., 表示保留意見之基礎。
            # English reports likewise refer to the 'Basis ... section'.
            if (re.match(r"\s*[」\"'’]?\s*(?:段|所述|[。；;]|section\b|paragraph\b)", suffix)
                    or re.search(r"(?:表示|上述|前述|除|described in(?: the)?|refer to(?: the)?)\s*[「\"'‘]?\s*$", prefix)):
                search_from = basis_start
                continue
            is_prior = _is_prior_period_reference(
                report_text[:marker_position],
                current_fiscal_year,
            )
            if is_prior != comparative:
                search_from = basis_start
                continue
            end = section_end.search(report_text, basis_start)
            next_bases = [pos for next_marker in start_markers
                          if (pos := report_text.find(next_marker, basis_start)) >= 0]
            basis_end = min([len(report_text), *(next_bases), *([end.start()] if end else [])])
            candidates.append(report_text[basis_start:basis_end])
            search_from = basis_start
    return " ".join(dict.fromkeys(candidates)).strip()


def _is_non_significant_component_scope_limitation(basis: str) -> bool:
    if not basis:
        return False
    compact_basis = re.sub(r"\s+", "", _normalize_audit_text(basis))
    normalized_basis = _normalize_audit_text(basis)
    # Inability to access evidence/management is not the routine, bounded
    # qualification for unreviewed non-significant components.
    if any(marker in normalized_basis for marker in (
        "未能接觸", "無法接觸", "無法取得", "未能取得", "無法判斷",
        "unable to access", "unable to obtain", "could not obtain", "unable to determine",
    )):
        return False
    chinese_component = any(
        marker in compact_basis
        for marker in (
            "非重要子公司",
            "部分子公司",
            "被投資公司",
            "採用權益法之投資",
            "關聯企業",
        )
    )
    chinese_unreviewed = any(
        marker in compact_basis
        for marker in (
            "未經會計師核閱",
            "未經會計師查核",
            "未經核閱",
            "未經查核",
        )
    )
    english_component = any(
        marker in normalized_basis
        for marker in (
            "non-significant subsidiar",
            "nonsignificant subsidiar",
            "certain subsidiar",
            "investee",
            "associate",
            "equity-method invest",
        )
    )
    english_unreviewed = any(
        marker in normalized_basis
        for marker in ("not reviewed", "not audited", "unaudited")
    )
    return (chinese_component and chinese_unreviewed) or (
        english_component and english_unreviewed
    )


def _normalize_audit_text(value: str) -> str:
    return normalize_cjk_text(value).lower()


def _is_prior_period_reference(
    prefix: str,
    current_fiscal_year: int,
) -> bool:
    """Nearest explicit period label wins, without a 48-character cutoff.

    A prior section remains prior until a current label/year resumes it. This
    also handles a short '前期...本期...' transition without prior dominance.
    Unlabelled text retains the reporting context's period for compatibility.
    """
    compact = re.sub(r"\s+", "", _normalize_audit_text(prefix))
    labels = {
        "前一年度": True, "前期": True, "上期": True, "上年度": True,
        "prioryear": True, "previousyear": True, "priorperiod": True,
        "previousperiod": True, "comparativeperiod": True,
        "本期": False, "本年度": False, "當期": False,
        "currentperiod": False, "currentyear": False,
    }
    candidates = [
        (compact.rfind(label), prior)
        for label, prior in labels.items() if label in compact
    ]
    for match in re.finditer(r"(?<!\d)(?:民國)?(\d{2,4})(?:年度?|q[1-4])", compact):
        year = int(match[1])
        year = year + 1911 if year < 1911 else year
        candidates.append((match.start(), year < current_fiscal_year))
    return max(candidates, default=(-1, False))[1]


def _contains_current_period_marker(
    report_text: str,
    markers: tuple[str, ...],
    current_fiscal_year: int,
) -> bool:
    for marker in markers:
        search_from = 0
        while (position := report_text.find(marker, search_from)) >= 0:
            prefix = report_text[:position]
            if not _is_prior_period_reference(
                prefix,
                current_fiscal_year,
            ):
                return True
            search_from = position + len(marker)
    return False


def _contains_comparative_period_marker(
    report_text: str,
    markers: tuple[str, ...],
    current_fiscal_year: int,
) -> bool:
    for marker in markers:
        search_from = 0
        while (position := report_text.find(marker, search_from)) >= 0:
            prefix = report_text[:position]
            if _is_prior_period_reference(prefix, current_fiscal_year):
                return True
            search_from = position + len(marker)
    return False


def _classify_comparative_opinion(
    report_text: str,
    *,
    current_fiscal_year: int,
    audit_status: str | None,
    adverse_markers: tuple[str, ...],
    disclaimer_markers: tuple[str, ...],
    qualified_markers: tuple[str, ...],
) -> str | None:
    if _contains_comparative_period_marker(
        report_text,
        adverse_markers,
        current_fiscal_year,
    ):
        return "ADVERSE"
    if _contains_comparative_period_marker(
        report_text,
        disclaimer_markers,
        current_fiscal_year,
    ):
        return "DISCLAIMER"
    if _contains_comparative_period_marker(
        report_text,
        qualified_markers,
        current_fiscal_year,
    ):
        basis = _qualified_basis_text(
            report_text,
            current_fiscal_year=current_fiscal_year,
            comparative=True,
        )
        if _is_material_misstatement_basis(basis):
            return "QUALIFIED_MATERIAL_MISSTATEMENT"
        if _is_non_significant_component_scope_limitation(basis):
            return "QUALIFIED_SCOPE_LIMITATION"
        return "QUALIFIED_UNKNOWN"

    unmodified_markers = (
        "無保留意見",
        "無保留結論",
        "unmodified opinion",
        "unqualified opinion",
        "unmodified conclusion",
    )
    if _contains_comparative_period_marker(
        report_text,
        unmodified_markers,
        current_fiscal_year,
    ):
        return (
            "REVIEWED_UNMODIFIED"
            if audit_status == "reviewed"
            else "UNMODIFIED"
        )
    return None


def _is_material_misstatement_basis(basis: str) -> bool:
    if not basis:
        return False
    material_markers = (
        "重大錯誤",
        "重大不實",
        "重大錯報",
        "會計處理不當",
        "會計處理不適當",
        "會計政策不當",
        "未依規定認列",
        "未適當認列",
        "應認列而未認列",
        "未提列",
        "重大短列",
        "重大高估",
        "重大低估",
        "違反國際財務報導準則",
        "material misstatement",
        "materially misstated",
        "inappropriate accounting polic",
        "departure from international financial reporting standards",
        "failure to recognize",
    )
    return any(marker in basis for marker in material_markers)


def _ttm_value(
    latest: FiledFinancials,
    filings: dict[tuple[int, int], FiledFinancials],
    aliases: tuple[str, ...],
) -> float | None:
    year = latest.fiscal_year
    if latest.quarter == 4:
        return _flow_value(latest, aliases, year)
    current_ytd = _flow_value(latest, aliases, year)
    prior_ytd = _flow_value(latest, aliases, year - 1)
    prior_annual_filing = filings.get((year - 1, 4))
    prior_annual = (
        _flow_value(prior_annual_filing, aliases, year - 1)
        if prior_annual_filing is not None
        else None
    )
    if current_ytd is None or prior_ytd is None or prior_annual is None:
        return None
    return current_ytd + prior_annual - prior_ytd


def _ttm_eps(
    latest: FiledFinancials,
    filings: dict[tuple[int, int], FiledFinancials],
) -> float | None:
    """Use TTM parent income divided by a time-weighted share estimate."""

    year = latest.fiscal_year
    if latest.quarter == 4:
        return _flow_value(latest, DILUTED_EPS, year)
    prior_annual_filing = filings.get((year - 1, 4))
    if prior_annual_filing is None:
        return None

    current_income = _flow_value(latest, NET_INCOME, year)
    prior_ytd_income = _flow_value(latest, NET_INCOME, year - 1)
    prior_annual_income = _flow_value(prior_annual_filing, NET_INCOME, year - 1)
    current_shares = _safe_div(current_income, _flow_value(latest, DILUTED_EPS, year))
    prior_ytd_shares = _safe_div(prior_ytd_income, _flow_value(latest, DILUTED_EPS, year - 1))
    prior_annual_shares = _safe_div(
        prior_annual_income,
        _flow_value(prior_annual_filing, DILUTED_EPS, year - 1),
    )
    required = (
        current_income,
        prior_ytd_income,
        prior_annual_income,
        current_shares,
        prior_ytd_shares,
        prior_annual_shares,
    )
    if any(value is None for value in required):
        return None

    elapsed_months = latest.quarter * 3
    remaining_months = 12 - elapsed_months
    prior_remaining_shares = (
        12 * float(prior_annual_shares) - elapsed_months * float(prior_ytd_shares)
    ) / remaining_months
    ttm_weighted_shares = (
        elapsed_months * float(current_shares) + remaining_months * prior_remaining_shares
    ) / 12
    ttm_income = float(current_income) + float(prior_annual_income) - float(prior_ytd_income)
    return _safe_div(ttm_income, ttm_weighted_shares)


def _flow_value(
    filing: FiledFinancials,
    aliases: tuple[str, ...],
    year: int,
) -> float | None:
    for concept in aliases:
        candidates = [
            fact
            for fact in filing.facts
            if fact.concept == concept
            and fact.numeric_value is not None
            and not fact.dimensions
            and fact.start_date == "{}-01-01".format(year)
            and fact.end_date is not None
            and fact.end_date.startswith(str(year))
        ]
        if candidates:
            return float(max(candidates, key=lambda fact: fact.end_date or "").numeric_value)
    return None


def _instant_value(
    filing: FiledFinancials,
    aliases: tuple[str, ...],
    year: int,
) -> float | None:
    for concept in aliases:
        candidates = [
            fact
            for fact in filing.facts
            if fact.concept == concept
            and fact.numeric_value is not None
            and not fact.dimensions
            and fact.instant is not None
            and fact.instant.startswith(str(year))
        ]
        if candidates:
            return float(max(candidates, key=lambda fact: fact.instant or "").numeric_value)
    return None


def _debt_value(filing: FiledFinancials, year: int) -> float | None:
    values = []
    for concept in DEBT_COMPONENTS:
        value = _instant_value(filing, (concept,), year)
        if value is not None:
            values.append(value)
    return sum(values) if values else None


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _is_material_correction(description: str | None) -> bool:
    if not description:
        return False
    normalized = description.lower()
    explicit_restatement_markers = (
        "重大重編",
        "重編財務",
        "重編損益",
        "重編資產負債",
        "重編現金流量",
        "material restatement",
        "materially restated",
    )
    return any(marker in normalized for marker in explicit_restatement_markers)


__all__ = [
    "CachedFundamentalBundle",
    "CachedFundamentalHeader",
    "FUNDAMENTAL_DERIVATION_VERSION",
    "FiledFact",
    "FiledFinancials",
    "FundamentalBundle",
    "derive_fundamentals",
]

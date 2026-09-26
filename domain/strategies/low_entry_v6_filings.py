"""Adapt parsed, dated iXBRL facts to v6's canonical quarterly source contract."""
from __future__ import annotations

import pandas as pd

from domain.fundamental_derivation import FiledFinancials


V6_FILING_SCHEMA_VERSION = "2026-09-25-quarter-eps-context"


def ixbrl_record(symbol: str, filing: FiledFinancials) -> dict:
    period = pd.Period(year=filing.fiscal_year, quarter=filing.quarter, freq="Q")
    end = period.end_time.date().isoformat()
    start = f"{filing.fiscal_year}-01-01"

    def value(*concepts, instant=False, eps=False, quarter=False):
        flow_start = period.start_time.date().isoformat() if quarter else start
        for concept in concepts:
            facts = [fact for fact in filing.facts if fact.concept == concept
                     and not fact.dimensions and fact.numeric_value is not None
                     and ((fact.instant == end) if instant else (fact.start_date == flow_start and fact.end_date == end))
                     and fact.unit == ("iso4217:TWD/xbrli:shares" if eps else "iso4217:TWD")]
            values = {float(fact.numeric_value) for fact in facts}
            if len(values) == 1:
                # iXBRL parser already applies the scale attribute. TWSE snapshots
                # use TWD thousands; EPS remains TWD/share in both sources.
                return values.pop() / (1 if eps else 1000)
        return None

    assets = value("ifrs-full:Assets", instant=True)
    current = value("ifrs-full:CurrentAssets", instant=True)
    revenue = value("ifrs-full:Revenue", "ifrs-full:RevenueFromContractsWithCustomers")
    gross = value("ifrs-full:GrossProfit", "tifrs-bsci-ci:GrossProfitLossFromOperations")
    return {
        "symbol": symbol, "market": "TWSE", "currency": "TWD", "unit": "thousand",
        "scope": filing.statement_scope, "source": "MOPS_IXBRL", "flow_basis": "YTD",
        "period_end": end, "available_at": filing.available_at,
        "revision_id": filing.content_sha256 or f"filing-{filing.filing_id}-r{filing.revision_number}",
        "revenue": revenue, "cost_of_revenue": revenue - gross if revenue is not None and gross is not None else None,
        "operating_income": value("ifrs-full:ProfitLossFromOperatingActivities"),
        "net_income": value("ifrs-full:ProfitLoss"),
        "basic_eps": value("ifrs-full:BasicEarningsLossPerShare", eps=True),
        "basic_eps_quarter": value("ifrs-full:BasicEarningsLossPerShare", eps=True, quarter=True),
        "current_assets": current, "total_assets": assets,
        "non_current_assets": assets - current if assets is not None and current is not None else None,
        "current_liabilities": value("ifrs-full:CurrentLiabilities", instant=True),
        "total_liabilities": value("ifrs-full:Liabilities", instant=True),
        "total_equity": value("ifrs-full:Equity", instant=True),
    }

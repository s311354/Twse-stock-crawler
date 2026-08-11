"""Transport-independent filing and financial snapshot records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from domain.data_provenance import LEGACY_UNVERIFIED


@dataclass(frozen=True)
class FilingCandidate:
    symbol: str
    fiscal_year: int
    quarter: int
    report_id: str
    source_url: str
    available_at: str | None = None
    period_end: str | None = None
    statement_scope: str = "consolidated"
    audit_status: str | None = None
    availability_source: str = LEGACY_UNVERIFIED


@dataclass(frozen=True)
class FilingDocument:
    """Raw filing content and its first observation timestamp."""

    candidate: FilingCandidate
    content: bytes
    fetched_at: str

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True)
class XbrlContext:
    context_id: str
    start_date: str | None = None
    end_date: str | None = None
    instant: str | None = None
    dimensions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class NormalizedFact:
    concept: str
    context_id: str
    unit: str | None
    numeric_value: float | None
    text_value: str | None
    start_date: str | None
    end_date: str | None
    instant: str | None
    dimensions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class FilingRecord:
    filing_id: int
    symbol: str
    fiscal_year: int
    quarter: int
    report_id: str
    available_at: str
    fetched_at: str
    revision_number: int
    content_sha256: str
    source_url: str
    raw_path: str
    period_end: str | None = None
    statement_scope: str = "consolidated"
    audit_status: str | None = None
    correction_description: str | None = None
    availability_source: str = LEGACY_UNVERIFIED
    correction_announced_on: str | None = None


@dataclass(frozen=True)
class FilingCorrection:
    symbol: str
    fiscal_year: int
    quarter: int
    announced_on: str
    description: str


@dataclass(frozen=True)
class FilingAnnouncement:
    symbol: str
    fiscal_year: int
    quarter: int
    available_at: str
    title: str


@dataclass(frozen=True)
class FundamentalSnapshot:
    """Latest matching income/balance snapshot for one listed company."""

    symbol: str
    company_name: str
    fiscal_year: int
    quarter: int
    observed_at: str
    revenue_ytd: float | None
    gross_profit_ytd: float | None
    operating_income_ytd: float | None
    net_income_ytd: float | None
    parent_net_income_ytd: float | None
    basic_eps_ytd: float | None
    current_assets: float | None
    total_assets: float | None
    current_liabilities: float | None
    total_liabilities: float | None
    parent_equity: float | None
    total_equity: float | None

    @property
    def period(self) -> tuple[int, int]:
        return self.fiscal_year, self.quarter

    @property
    def content_sha256(self) -> str:
        values = {
            key: value
            for key, value in self.as_snapshot_fields().items()
            if key != "twse_financial_observed_at"
        }
        encoded = json.dumps(
            values,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def as_snapshot_fields(self) -> dict[str, object]:
        """Expose only mathematically compatible YTD/latest-period evidence."""

        return {
            "twse_financial_source": "TWSE_OPENAPI_LATEST",
            "twse_financial_observed_at": self.observed_at,
            "twse_financial_fiscal_year": self.fiscal_year,
            "twse_financial_quarter": self.quarter,
            "twse_revenue_ytd": self.revenue_ytd,
            "twse_gross_profit_ytd": self.gross_profit_ytd,
            "twse_operating_income_ytd": self.operating_income_ytd,
            "twse_net_income_ytd": self.net_income_ytd,
            "twse_parent_net_income_ytd": self.parent_net_income_ytd,
            "twse_basic_eps_ytd": self.basic_eps_ytd,
            "twse_current_assets": self.current_assets,
            "twse_total_assets": self.total_assets,
            "twse_current_liabilities": self.current_liabilities,
            "twse_total_liabilities": self.total_liabilities,
            "twse_parent_equity": self.parent_equity,
            "twse_total_equity": self.total_equity,
            "twse_gross_margin_ytd": _safe_div(
                self.gross_profit_ytd,
                self.revenue_ytd,
            ),
            "twse_operating_margin_ytd": _safe_div(
                self.operating_income_ytd,
                self.revenue_ytd,
            ),
            "twse_roa_ytd_end_balance": _safe_div(
                self.net_income_ytd,
                self.total_assets,
            ),
            "twse_roe_ytd_end_balance": _safe_div(
                self.parent_net_income_ytd,
                self.parent_equity,
            ),
            "twse_debt_ratio": _safe_div(
                self.total_liabilities,
                self.total_assets,
            ),
            "twse_current_ratio": _safe_div(
                self.current_assets,
                self.current_liabilities,
            ),
        }


def _safe_div(
    numerator: float | None,
    denominator: float | None,
) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator

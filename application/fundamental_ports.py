"""Ports used by the point-in-time fundamental refresh application service."""

from __future__ import annotations

import datetime as dt
from types import TracebackType
from typing import Protocol
from typing import Self

from domain.fundamental_derivation import FiledFinancials
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_models import FilingAnnouncement
from domain.fundamental_models import FilingCandidate
from domain.fundamental_models import FilingCorrection
from domain.fundamental_models import FilingDocument
from domain.fundamental_models import FilingRecord
from domain.fundamental_models import FundamentalSnapshot
from domain.fundamental_models import NormalizedFact


class FundamentalUnitOfWork(Protocol):
    """Transaction boundary for all persistence work on one company."""

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class FundamentalRepositoryPort(Protocol):
    def unit_of_work(self) -> FundamentalUnitOfWork:
        """Open one atomic persistence boundary for one company."""

    def load_filings(
        self,
        symbol: str,
        *,
        as_of: str | None = None,
        uow: FundamentalUnitOfWork | None = None,
    ) -> list[FilingRecord]: ...

    def load_filed_financials(
        self,
        symbol: str,
        as_of: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
    ) -> list[FiledFinancials]: ...

    def last_full_refresh_epoch(
        self,
        symbol: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
    ) -> int | None: ...

    def mark_full_refresh(
        self,
        symbol: str,
        refreshed_at: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
    ) -> None: ...

    def load_financial_batch_state(
        self,
        symbol: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
    ) -> tuple[tuple[int, int], str] | None: ...

    def save_financial_batch_state(
        self,
        symbol: str,
        fiscal_year: int,
        quarter: int,
        content_sha256: str,
        observed_at: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
    ) -> None: ...

    def latest_filing(
        self,
        symbol: str,
        fiscal_year: int,
        quarter: int,
        report_id: str = "C",
        *,
        uow: FundamentalUnitOfWork | None = None,
    ) -> FilingRecord | None: ...

    def update_availability_provenance(
        self,
        filing: FilingRecord,
        *,
        available_at: str,
        availability_source: str,
        uow: FundamentalUnitOfWork | None = None,
    ) -> FilingRecord: ...

    def save_revision(
        self,
        candidate: FilingCandidate,
        content: bytes,
        facts: list[NormalizedFact],
        *,
        available_at: str,
        fetched_at: str,
        availability_source: str | None = None,
        minimum_revision_number: int = 1,
        correction_description: str | None = None,
        correction_announced_on: str | None = None,
        uow: FundamentalUnitOfWork | None = None,
    ) -> FilingRecord: ...

    def filing_set_hash(self, filings: list[FiledFinancials]) -> str: ...

    def load_fundamental_bundle(
        self,
        symbol: str,
        filing_set_sha256: str,
        sector: str,
        *,
        uow: FundamentalUnitOfWork | None = None,
    ) -> FundamentalBundle | None: ...

    def save_fundamental_bundle(
        self,
        symbol: str,
        filing_set_sha256: str,
        sector: str,
        bundle: FundamentalBundle,
        *,
        uow: FundamentalUnitOfWork | None = None,
    ) -> None: ...


class FilingCatalogPort(Protocol):
    def discover(self, symbol: str) -> list[FilingCandidate]: ...

    def discover_announcements(
        self,
        symbol: str,
        calendar_years: list[int],
    ) -> list[FilingAnnouncement]: ...

    def latest_correction(
        self,
        candidate: FilingCandidate,
    ) -> FilingCorrection | None: ...

    def load_company_sectors(self) -> dict[str, str]:
        """Load official TWSE company-to-industry mappings."""
        ...


class FilingDocumentPort(Protocol):
    def download(self, candidate: FilingCandidate) -> FilingDocument: ...


class FilingParserPort(Protocol):
    def parse(self, document: FilingDocument) -> list[NormalizedFact]: ...


class FinancialBatchPort(Protocol):
    def load(
        self,
        *,
        observed_at: dt.datetime,
    ) -> dict[str, FundamentalSnapshot]: ...

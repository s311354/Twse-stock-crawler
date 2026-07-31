import hashlib

import pytest

from application.filing_ingestion import FilingIngestionService
from domain.data_provenance import FIRST_OBSERVED_DOWNLOAD
from domain.data_provenance import MOPS_PUBLIC_ANNOUNCEMENT
from domain.data_provenance import OFFICIAL_FILING_TIMESTAMP
from domain.fundamental_models import FilingCandidate
from domain.fundamental_models import FilingDocument
from domain.fundamental_models import FilingRecord
from domain.fundamental_models import NormalizedFact
from infrastructure.fundamentals.mops_repository import MopsRepository


def test_ingestion_runs_download_parse_and_save_through_separate_ports() -> None:
    events: list[str] = []
    candidate = _candidate(
        availability_source=MOPS_PUBLIC_ANNOUNCEMENT
    )
    document = FilingDocument(
        candidate=candidate,
        content=b"filing-v1",
        fetched_at="2026-07-21T06:00:00+00:00",
    )
    repository = _Repository(events)
    service = FilingIngestionService(
        _Catalog(events),
        _Documents(events, document),
        _Parser(events),
        repository,
    )

    filing = service.ingest(
        candidate,
        fallback_available_at="2026-02-10T10:00:00+00:00",
        uow=_UnitOfWork(),
    )

    assert events == [
        "download",
        "latest",
        "correction",
        "parse",
        "save",
    ]
    assert filing.content_sha256 == document.content_sha256
    assert repository.save_kwargs["available_at"] == document.fetched_at
    assert (
        repository.save_kwargs["availability_source"]
        == FIRST_OBSERVED_DOWNLOAD
    )


def test_ingestion_reuses_matching_hash_without_parse_or_write() -> None:
    events: list[str] = []
    candidate = _candidate(
        availability_source=OFFICIAL_FILING_TIMESTAMP
    )
    document = FilingDocument(
        candidate=candidate,
        content=b"unchanged",
        fetched_at="2026-07-21T06:00:00+00:00",
    )
    existing = _filing(
        candidate,
        content_sha256=document.content_sha256,
        availability_source=OFFICIAL_FILING_TIMESTAMP,
    )
    repository = _Repository(events, existing=existing)
    service = FilingIngestionService(
        _Catalog(events),
        _Documents(events, document),
        _Parser(events),
        repository,
    )

    result = service.ingest(
        candidate,
        fallback_available_at=document.fetched_at,
        uow=_UnitOfWork(),
    )

    assert result is existing
    assert events == ["download", "latest"]


def test_sqlite_unit_of_work_commits_or_rolls_back_all_company_writes(
    tmp_path,
) -> None:
    repository = MopsRepository(tmp_path / "mops")
    candidate = _candidate(
        availability_source=OFFICIAL_FILING_TIMESTAMP
    )

    with pytest.raises(RuntimeError, match="abort refresh"):
        with repository.unit_of_work() as uow:
            repository.save_revision(
                candidate,
                b"rolled-back",
                [_fact()],
                available_at=candidate.available_at or "",
                fetched_at="2026-02-11T00:00:00+00:00",
                uow=uow,
            )
            repository.mark_full_refresh(
                candidate.symbol,
                "2026-07-21T00:00:00+00:00",
                uow=uow,
            )
            raise RuntimeError("abort refresh")

    assert repository.load_filings(candidate.symbol) == []
    assert repository.last_full_refresh_epoch(candidate.symbol) is None

    with repository.unit_of_work() as uow:
        repository.save_revision(
            candidate,
            b"committed",
            [_fact()],
            available_at=candidate.available_at or "",
            fetched_at="2026-02-11T00:00:00+00:00",
            uow=uow,
        )

    assert len(repository.load_filings(candidate.symbol)) == 1


class _UnitOfWork:
    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


class _Catalog:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def latest_correction(self, candidate: FilingCandidate):
        self.events.append("correction")
        return None


class _Documents:
    def __init__(
        self,
        events: list[str],
        document: FilingDocument,
    ) -> None:
        self.events = events
        self.document = document

    def download(self, candidate: FilingCandidate) -> FilingDocument:
        self.events.append("download")
        return self.document


class _Parser:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def parse(self, document: FilingDocument) -> list[NormalizedFact]:
        self.events.append("parse")
        return [_fact()]


class _Repository:
    def __init__(
        self,
        events: list[str],
        *,
        existing: FilingRecord | None = None,
    ) -> None:
        self.events = events
        self.existing = existing
        self.save_kwargs: dict[str, object] = {}

    def latest_filing(self, *args, **kwargs) -> FilingRecord | None:
        self.events.append("latest")
        return self.existing

    def update_availability_provenance(self, *args, **kwargs):
        raise AssertionError("provenance update was not expected")

    def save_revision(
        self,
        candidate: FilingCandidate,
        content: bytes,
        facts: list[NormalizedFact],
        **kwargs,
    ) -> FilingRecord:
        self.events.append("save")
        self.save_kwargs = dict(kwargs)
        return _filing(
            candidate,
            content_sha256=hashlib.sha256(content).hexdigest(),
            availability_source=str(kwargs["availability_source"]),
        )


def _candidate(*, availability_source: str) -> FilingCandidate:
    return FilingCandidate(
        symbol="2330",
        fiscal_year=2025,
        quarter=4,
        report_id="C",
        source_url="https://example.test/2330-2025Q4.html",
        available_at="2026-02-10T18:16:19+08:00",
        availability_source=availability_source,
    )


def _filing(
    candidate: FilingCandidate,
    *,
    content_sha256: str,
    availability_source: str,
) -> FilingRecord:
    return FilingRecord(
        filing_id=1,
        symbol=candidate.symbol,
        fiscal_year=candidate.fiscal_year,
        quarter=candidate.quarter,
        report_id=candidate.report_id,
        available_at=candidate.available_at or "",
        fetched_at="2026-02-11T00:00:00+00:00",
        revision_number=1,
        content_sha256=content_sha256,
        source_url=candidate.source_url,
        raw_path="raw/2330/2025Q4/test.html",
        availability_source=availability_source,
    )


def _fact() -> NormalizedFact:
    return NormalizedFact(
        concept="ifrs-full:Revenue",
        context_id="FY",
        unit="TWD",
        numeric_value=100.0,
        text_value=None,
        start_date="2025-01-01",
        end_date="2025-12-31",
        instant=None,
        dimensions=(),
    )

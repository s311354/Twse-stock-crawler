"""Application orchestration for one MOPS filing revision."""

from __future__ import annotations

from application.fundamental_ports import FilingCatalogPort
from application.fundamental_ports import FilingDocumentPort
from application.fundamental_ports import FilingParserPort
from application.fundamental_ports import FundamentalRepositoryPort
from application.fundamental_ports import FundamentalUnitOfWork
from application.daily_performance import count_work, measure_stage
from domain.data_provenance import FIRST_OBSERVED_DOWNLOAD
from domain.data_provenance import LEGACY_UNVERIFIED
from domain.data_provenance import OFFICIAL_FILING_TIMESTAMP
from domain.fundamental_models import FilingCandidate
from domain.fundamental_models import FilingRecord


class FilingIngestionService:
    """Download, parse, and version a filing through explicit ports."""

    def __init__(
        self,
        catalog: FilingCatalogPort,
        documents: FilingDocumentPort,
        parser: FilingParserPort,
        repository: FundamentalRepositoryPort,
    ) -> None:
        self.catalog = catalog
        self.documents = documents
        self.parser = parser
        self.repository = repository

    def ingest(
        self,
        candidate: FilingCandidate,
        *,
        fallback_available_at: str,
        uow: FundamentalUnitOfWork,
    ) -> FilingRecord:
        document = self.documents.download(candidate)
        if document.candidate != candidate:
            raise ValueError(
                "Downloaded filing identity does not match the candidate"
            )

        existing = self.repository.latest_filing(
            candidate.symbol,
            candidate.fiscal_year,
            candidate.quarter,
            candidate.report_id,
            uow=uow,
        )
        if (
            existing is not None
            and document.content_sha256 == existing.content_sha256
        ):
            if existing.availability_source == LEGACY_UNVERIFIED:
                return self.repository.update_availability_provenance(
                    existing,
                    available_at=existing.fetched_at,
                    availability_source=FIRST_OBSERVED_DOWNLOAD,
                    uow=uow,
                )
            return existing

        available_at = candidate.available_at or fallback_available_at
        availability_source = candidate.availability_source
        minimum_revision_number = 1
        correction_description: str | None = None
        correction_announced_on: str | None = None
        correction = self.catalog.latest_correction(candidate)

        if existing is not None:
            available_at = document.fetched_at
            availability_source = FIRST_OBSERVED_DOWNLOAD
            if correction is not None:
                correction_description = correction.description
                correction_announced_on = correction.announced_on
        else:
            if correction is not None:
                minimum_revision_number = 2
                correction_description = correction.description
                correction_announced_on = correction.announced_on
            if (
                candidate.available_at is None
                or candidate.availability_source
                != OFFICIAL_FILING_TIMESTAMP
            ):
                available_at = document.fetched_at
                availability_source = FIRST_OBSERVED_DOWNLOAD

        with measure_stage("filing_parse"):
            count_work("filing_parse_calls")
            facts = self.parser.parse(document)
        return self.repository.save_revision(
            candidate,
            document.content,
            facts,
            available_at=available_at,
            fetched_at=document.fetched_at,
            availability_source=availability_source,
            minimum_revision_number=minimum_revision_number,
            correction_description=correction_description,
            correction_announced_on=correction_announced_on,
            uow=uow,
        )


__all__ = ["FilingIngestionService"]

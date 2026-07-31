from __future__ import annotations

import datetime as dt

import pytest

from application.fundamental_service import MopsFundamentalService
from domain.data_provenance import MOPS_PUBLIC_ANNOUNCEMENT
from domain.fundamental_models import FilingAnnouncement
from domain.fundamental_models import FilingCandidate
from domain.fundamental_models import FundamentalSnapshot


def test_service_requires_an_explicit_repository_port() -> None:
    with pytest.raises(ValueError, match="repository port is required"):
        MopsFundamentalService(_CatalogPort())


def test_daily_batch_uses_domain_snapshot_through_financial_port() -> None:
    observed_at = dt.datetime(2026, 7, 26, tzinfo=dt.timezone.utc)
    snapshot = _snapshot(observed_at)
    batch = _FinancialBatchPort(snapshot)
    service = MopsFundamentalService(
        _CatalogPort(),
        repository=object(),  # prepare_daily_batch does not access persistence.
        filing_documents=_DocumentPort(),
        filing_parser=_ParserPort(),
        financial_batch_client=batch,
    )

    assert service.prepare_daily_batch(observed_at) == 1
    assert service._daily_financials == {"2330": snapshot}
    assert batch.observed_at == observed_at


def test_company_sector_is_loaded_once_through_catalog_port() -> None:
    catalog = _CatalogPort(sectors={"2330": "24"})
    service = MopsFundamentalService(
        catalog,
        repository=object(),
        filing_documents=_DocumentPort(),
        filing_parser=_ParserPort(),
    )

    assert service._resolve_sector("2330", "ELEC") == "24"
    assert service._resolve_sector("2317", "25") == "25"
    assert service._resolve_sector("2317", "ELEC") == ""
    assert catalog.sector_loads == 1


def test_availability_mapping_uses_domain_dtos() -> None:
    candidate = FilingCandidate(
        symbol="2330",
        fiscal_year=2025,
        quarter=4,
        report_id="C",
        source_url="https://example.test/2330-2025Q4.html",
    )
    announcement = FilingAnnouncement(
        symbol="2330",
        fiscal_year=2025,
        quarter=4,
        available_at="2026-02-10T18:16:19+08:00",
        title="財務報告",
    )

    result = MopsFundamentalService._apply_availability(
        [candidate],
        [],
        [announcement],
    )

    assert result[0].available_at == announcement.available_at
    assert result[0].availability_source == MOPS_PUBLIC_ANNOUNCEMENT


class _CatalogPort:
    def __init__(self, sectors: dict[str, str] | None = None) -> None:
        self.sectors = sectors or {}
        self.sector_loads = 0

    def load_company_sectors(self) -> dict[str, str]:
        self.sector_loads += 1
        return dict(self.sectors)


class _DocumentPort:
    pass


class _ParserPort:
    pass


class _FinancialBatchPort:
    def __init__(self, snapshot: FundamentalSnapshot) -> None:
        self.snapshot = snapshot
        self.observed_at: dt.datetime | None = None

    def load(
        self,
        *,
        observed_at: dt.datetime,
    ) -> dict[str, FundamentalSnapshot]:
        self.observed_at = observed_at
        return {self.snapshot.symbol: self.snapshot}


def _snapshot(observed_at: dt.datetime) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol="2330",
        company_name="台積電",
        fiscal_year=2026,
        quarter=1,
        observed_at=observed_at.isoformat(),
        revenue_ytd=1_000.0,
        gross_profit_ytd=400.0,
        operating_income_ytd=200.0,
        net_income_ytd=180.0,
        parent_net_income_ytd=170.0,
        basic_eps_ytd=10.0,
        current_assets=500.0,
        total_assets=1_000.0,
        current_liabilities=250.0,
        total_liabilities=350.0,
        parent_equity=600.0,
        total_equity=650.0,
    )

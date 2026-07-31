from __future__ import annotations

import datetime as dt
from dataclasses import replace

import pandas as pd
import pytest

import application.fundamental_service as fundamental_service_module
from application.filing_refresh_planner import FilingRefreshPlanner
from application.fundamental_service import MopsFundamentalService
from domain.data_provenance import FIRST_OBSERVED_DOWNLOAD
from domain.data_provenance import MOPS_PUBLIC_ANNOUNCEMENT
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_models import FilingDocument
from infrastructure.fundamentals.ixbrl_parser import NormalizedFact
from infrastructure.fundamentals.models import MopsFiling
from infrastructure.fundamentals.models import MopsFilingCandidate
from infrastructure.fundamentals.mops_client import FilingAnnouncement
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.fundamentals.twse_financial_batch import TwseFinancialSnapshot
from infrastructure.crawler.twse_http_client import TwseNoDataError


def test_refresh_planner_seeds_five_years_then_verifies_two_recent_periods() -> None:
    planner = FilingRefreshPlanner()
    candidates = [_candidate(year, 4) for year in range(2020, 2026)] + [
        _candidate(2025, 1),
        _candidate(2026, 1),
    ]

    first_run = planner.plan(candidates, [], history_years=5)

    assert _candidate_periods(first_run.download) == [
        (2021, 4),
        (2022, 4),
        (2023, 4),
        (2024, 4),
        (2025, 1),
        (2025, 4),
        (2026, 1),
    ]
    assert first_run.reuse == []
    assert first_run.verify_revision == []

    cached = [_filing(candidate) for candidate in first_run.download]
    daily = planner.plan(
        candidates,
        cached,
        history_years=5,
        corrected_periods={(2022, 4)},
    )

    assert _candidate_periods(daily.verify_revision) == [
        (2022, 4),
        (2025, 4),
        (2026, 1),
    ]
    assert {(item.fiscal_year, item.quarter) for item in daily.reuse} == {
        (2021, 4),
        (2023, 4),
        (2024, 4),
        (2025, 1),
    }
    assert daily.download == []


def test_monthly_refresh_is_timezone_aware_and_due_once_per_month() -> None:
    planner = FilingRefreshPlanner()
    signal_time = dt.datetime(2026, 7, 21, 13, 30, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    prior_same_month = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)
    prior_month = dt.datetime(2026, 6, 30, 15, 59, tzinfo=dt.timezone.utc)

    assert planner.is_monthly_refresh_due(None, signal_time) is True
    assert planner.is_monthly_refresh_due(int(prior_same_month.timestamp()), signal_time) is False
    assert planner.is_monthly_refresh_due(int(prior_month.timestamp()), signal_time) is True

    with pytest.raises(ValueError, match="timezone-aware"):
        planner.is_monthly_refresh_due(None, dt.datetime(2026, 7, 21))


def test_sector_fallback_does_not_promote_market_label_to_official_code(tmp_path) -> None:
    repository = MopsRepository(tmp_path / "mops")
    client = _FakeMopsClient(repository, [], [])
    service = _fundamental_service(
        client,
        repository,
    )
    service._company_sectors = {}

    assert service._resolve_sector("2330", "ELEC") == ""
    assert service._resolve_sector("2330", "24") == "24"


def test_repository_uses_epoch_index_single_join_and_atomic_fact_write(tmp_path) -> None:
    repository = MopsRepository(tmp_path / "mops")
    candidate = _candidate(2025, 4)
    fact = _fact(100.0)
    repository.save_revision(
        candidate,
        b"valid-filing",
        [fact],
        available_at=candidate.available_at or "",
        fetched_at="2026-02-11T00:00:00+00:00",
    )

    traced: list[str] = []
    with repository.company_connection() as connection:
        connection.set_trace_callback(traced.append)
        loaded = repository.load_filed_financials(
            "2330",
            "2026-02-12T00:00:00+00:00",
            connection=connection,
        )
        indexes = {
            str(row[1])
            for row in connection.execute("PRAGMA index_list(filings)").fetchall()
        }
        epoch = connection.execute(
            "SELECT available_at_epoch FROM filings WHERE id = ?",
            (loaded[0].filing_id,),
        ).fetchone()[0]

    join_queries = [query for query in traced if "LEFT JOIN facts" in query]
    assert len(join_queries) == 1
    assert "julianday" not in join_queries[0].lower()
    assert loaded[0].facts[0].numeric_value == 100.0
    assert "filings_symbol_available_epoch" in indexes
    assert epoch == int(dt.datetime.fromisoformat(candidate.available_at or "").timestamp())

    invalid_fact = _fact(200.0)
    object.__setattr__(invalid_fact, "dimensions", ((object(), "invalid"),))
    with pytest.raises(TypeError):
        repository.save_revision(
            _candidate(2024, 4, symbol="2317"),
            b"invalid-filing",
            [invalid_fact],
            available_at="2025-02-10T18:16:19+08:00",
            fetched_at="2025-02-11T00:00:00+00:00",
        )
    assert repository.latest_filing("2317", 2024, 4) is None


def test_fundamental_service_reuses_hash_snapshot_and_limits_daily_downloads(
    monkeypatch,
    tmp_path,
) -> None:
    repository = MopsRepository(tmp_path / "mops")
    candidates = [_candidate(year, 4) for year in range(2021, 2026)]
    announcements = [
        FilingAnnouncement(
            symbol="2330",
            fiscal_year=candidate.fiscal_year,
            quarter=4,
            available_at=candidate.available_at or "",
            title="財務報告",
        )
        for candidate in candidates
    ]
    client = _FakeMopsClient(repository, candidates, announcements)
    service = _fundamental_service(client, repository)
    service._company_sectors = {"2330": "24"}
    derivation_calls = 0
    real_derivation = fundamental_service_module.derive_fundamentals

    def counting_derivation(*args, **kwargs):
        nonlocal derivation_calls
        derivation_calls += 1
        return real_derivation(*args, **kwargs)

    monkeypatch.setattr(
        fundamental_service_module,
        "derive_fundamentals",
        counting_derivation,
    )

    first = service.load(
        "2330",
        signal_date="2026-07-21T13:30:00+08:00",
        sector="ELEC",
        history_years=5,
    )
    service.load(
        "2330",
        signal_date="2026-07-22T13:30:00+08:00",
        sector="ELEC",
        history_years=5,
    )

    assert isinstance(first, FundamentalBundle)
    assert len(client.downloaded_periods) == 7
    assert client.downloaded_periods[-2:] == [(2024, 4), (2025, 4)]
    assert client.announcement_years[-1] == [2026]
    assert derivation_calls == 1


def test_twse_batch_skips_unchanged_mops_and_rechecks_changed_values(
    tmp_path,
) -> None:
    repository = MopsRepository(tmp_path / "mops")
    candidates = [_candidate(year, 4) for year in range(2021, 2026)]
    announcements = [
        FilingAnnouncement(
            symbol="2330",
            fiscal_year=candidate.fiscal_year,
            quarter=4,
            available_at=candidate.available_at or "",
            title="財務報告",
        )
        for candidate in candidates
    ]
    client = _FakeMopsClient(repository, candidates, announcements)
    batch_client = _FakeFinancialBatchClient(_batch_snapshot(revenue=1_000.0))
    service = _fundamental_service(
        client,
        repository,
        financial_batch_client=batch_client,
    )
    service._company_sectors = {"2330": "24"}
    observed = dt.datetime(2026, 7, 21, 5, 30, tzinfo=dt.timezone.utc)

    assert service.prepare_daily_batch(observed) == 1
    first = service.load(
        "2330",
        signal_date="2026-07-21T13:30:00+08:00",
        sector="ELEC",
        history_years=5,
    )
    discovery_calls_after_seed = client.discovery_calls
    downloads_after_seed = list(client.downloaded_periods)

    unchanged = service.load(
        "2330",
        signal_date="2026-07-22T13:30:00+08:00",
        sector="ELEC",
        history_years=5,
    )

    assert client.discovery_calls == discovery_calls_after_seed
    assert client.downloaded_periods == downloads_after_seed
    assert unchanged.snapshot["twse_operating_margin_ytd"] == pytest.approx(0.2)
    assert first.snapshot["twse_financial_source"] == "TWSE_OPENAPI_LATEST"

    batch_client.snapshot = _batch_snapshot(revenue=1_100.0)
    service.prepare_daily_batch(observed + dt.timedelta(days=1))
    service.load(
        "2330",
        signal_date="2026-07-23T13:30:00+08:00",
        sector="ELEC",
        history_years=5,
    )

    assert client.discovery_calls == discovery_calls_after_seed + 1
    assert client.downloaded_periods[-2:] == [(2024, 4), (2025, 4)]

    batch_client.snapshot = replace(
        _batch_snapshot(revenue=300.0),
        fiscal_year=2026,
        quarter=1,
    )
    service.prepare_daily_batch(observed + dt.timedelta(days=2))
    service.load(
        "2330",
        signal_date="2026-07-24T13:30:00+08:00",
        sector="ELEC",
        history_years=5,
    )

    with repository.company_connection() as connection:
        batch_state = repository.load_financial_batch_state(
            "2330",
            connection=connection,
        )
    assert batch_state is not None
    assert batch_state[0] == (2025, 4)


def test_fundamental_service_uses_point_in_time_cache_when_catalog_is_unavailable(
    tmp_path,
) -> None:
    repository = MopsRepository(tmp_path / "mops")
    candidate = _candidate(2025, 4)
    filing = repository.save_revision(
        candidate,
        b"cached-annual",
        [_fact(100.0)],
        available_at=candidate.available_at or "",
        fetched_at="2026-02-11T00:00:00+00:00",
        availability_source=candidate.availability_source,
    )
    client = _CatalogNoDataClient(repository)
    service = _fundamental_service(
        client,
        repository,
    )
    service._company_sectors = {"2330": "24"}

    bundle = service.load(
        "2330",
        signal_date="2026-07-21T13:30:00+08:00",
        sector="ELEC",
        history_years=5,
    )

    assert bundle.snapshot["filing_id"] == filing.filing_id
    with repository.company_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM refresh_state").fetchone()[0] == 0
        snapshot_hash = connection.execute(
            "SELECT filing_set_sha256 FROM fundamental_snapshots WHERE symbol = '2330'"
        ).fetchone()[0]
    assert snapshot_hash != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_fundamental_service_does_not_store_empty_refresh_or_snapshot(tmp_path) -> None:
    repository = MopsRepository(tmp_path / "mops")
    client = _CatalogNoDataClient(repository)
    service = _fundamental_service(
        client,
        repository,
    )
    service._company_sectors = {"2330": "24"}

    with pytest.raises(TwseNoDataError, match="explicitly returned no data"):
        service.load(
            "2330",
            signal_date="2026-07-21T13:30:00+08:00",
            sector="ELEC",
            history_years=5,
        )

    with repository.company_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM refresh_state").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM fundamental_snapshots").fetchone()[0] == 0


def test_live_fundamental_service_uses_filings_observed_during_same_run(tmp_path) -> None:
    repository = MopsRepository(tmp_path / "mops")
    candidates = [
        replace(_candidate(year, 4), available_at=None)
        for year in (2023, 2024, 2025)
    ]
    announcements: list[FilingAnnouncement] = []
    evaluation_started_at = "2026-07-21T06:00:00+00:00"
    filing_observed_at = "2026-07-21T06:00:05+00:00"
    client = _FirstObservedMopsClient(
        repository,
        candidates,
        announcements,
        observed_at=filing_observed_at,
    )
    service = _fundamental_service(
        client,
        repository,
        include_current_observations=True,
        now_provider=lambda: dt.datetime.fromisoformat(evaluation_started_at),
    )
    service._company_sectors = {"2330": "24"}

    bundle = service.load(
        "2330",
        signal_date="2026-07-21T05:30:00+00:00",
        sector="ELEC",
        history_years=3,
    )

    assert len(bundle.annual_fundamentals) == 3
    assert bundle.as_of == filing_observed_at
    assert client.downloaded_periods == [(2023, 4), (2024, 4), (2025, 4)]
    stored_filings = repository.load_filings("2330")
    assert {
        filing.availability_source for filing in stored_filings
    } == {FIRST_OBSERVED_DOWNLOAD}
    assert {filing.available_at for filing in stored_filings} == {
        filing_observed_at
    }


def test_historical_service_does_not_seed_undated_catalog_filings(tmp_path) -> None:
    repository = MopsRepository(tmp_path / "mops")
    candidates = [replace(_candidate(2025, 4), available_at=None)]
    client = _FirstObservedMopsClient(
        repository,
        candidates,
        [],
        observed_at="2026-07-21T06:00:00+00:00",
    )
    service = _fundamental_service(client, repository)

    with pytest.raises(TwseNoDataError, match="no point-in-time eligible filings"):
        service.load(
            "2330",
            signal_date="2026-01-15T05:30:00+00:00",
            sector="ELEC",
            history_years=3,
        )

    assert client.downloaded_periods == []
    assert repository.load_filings("2330") == []


class _FakeMopsClient:
    def __init__(
        self,
        repository: MopsRepository,
        candidates: list[MopsFilingCandidate],
        announcements: list[FilingAnnouncement],
    ) -> None:
        self.repository = repository
        self.candidates = candidates
        self.announcements = announcements
        self.downloaded_periods: list[tuple[int, int]] = []
        self.announcement_years: list[list[int]] = []
        self.discovery_calls = 0

    def discover(self, symbol: str) -> list[MopsFilingCandidate]:
        self.discovery_calls += 1
        return list(self.candidates)

    def discover_announcements(
        self,
        symbol: str,
        calendar_years: list[int],
    ) -> list[FilingAnnouncement]:
        self.announcement_years.append(list(calendar_years))
        return [
            announcement
            for announcement in self.announcements
            if dt.datetime.fromisoformat(announcement.available_at).year in calendar_years
        ]

    def latest_correction(self, candidate: MopsFilingCandidate):
        return None

    def download(
        self,
        candidate: MopsFilingCandidate,
    ) -> FilingDocument:
        self.downloaded_periods.append((candidate.fiscal_year, candidate.quarter))
        return FilingDocument(
            candidate=candidate,
            content="{}Q{}".format(
                candidate.fiscal_year,
                candidate.quarter,
            ).encode(),
            fetched_at="2026-07-21T05:00:00+00:00",
        )


class _FakeFinancialBatchClient:
    def __init__(self, snapshot: TwseFinancialSnapshot) -> None:
        self.snapshot = snapshot
        self.calls = 0

    def load(self, *, observed_at: dt.datetime):
        self.calls += 1
        return {"2330": replace(self.snapshot, observed_at=observed_at.isoformat())}


def _batch_snapshot(*, revenue: float) -> TwseFinancialSnapshot:
    return TwseFinancialSnapshot(
        symbol="2330",
        company_name="台積電",
        fiscal_year=2025,
        quarter=4,
        observed_at="2026-07-21T05:30:00+00:00",
        revenue_ytd=revenue,
        gross_profit_ytd=400.0,
        operating_income_ytd=200.0,
        net_income_ytd=150.0,
        parent_net_income_ytd=150.0,
        basic_eps_ytd=10.0,
        current_assets=400.0,
        total_assets=1_000.0,
        current_liabilities=200.0,
        total_liabilities=300.0,
        parent_equity=600.0,
        total_equity=700.0,
    )


class _CatalogNoDataClient:
    def __init__(self, repository: MopsRepository) -> None:
        self.repository = repository

    def discover(self, symbol: str) -> list[MopsFilingCandidate]:
        raise TwseNoDataError(
            "MOPS catalog explicitly returned no data for {}".format(symbol),
            retryable=False,
        )


class _FirstObservedMopsClient(_FakeMopsClient):
    def __init__(
        self,
        repository: MopsRepository,
        candidates: list[MopsFilingCandidate],
        announcements: list[FilingAnnouncement],
        *,
        observed_at: str,
    ) -> None:
        super().__init__(repository, candidates, announcements)
        self.observed_at = observed_at

    def download(
        self,
        candidate: MopsFilingCandidate,
    ) -> FilingDocument:
        self.downloaded_periods.append((candidate.fiscal_year, candidate.quarter))
        return FilingDocument(
            candidate=candidate,
            content="{}Q{}".format(
                candidate.fiscal_year,
                candidate.quarter,
            ).encode(),
            fetched_at=self.observed_at,
        )


class _FakeFilingParser:
    def parse(self, document: FilingDocument) -> list[NormalizedFact]:
        return [_fact(float(document.candidate.fiscal_year))]


def _fundamental_service(
    client,
    repository: MopsRepository,
    **kwargs,
) -> MopsFundamentalService:
    return MopsFundamentalService(
        client,
        repository=repository,
        filing_documents=client,
        filing_parser=_FakeFilingParser(),
        **kwargs,
    )


def _candidate(
    year: int,
    quarter: int,
    *,
    symbol: str = "2330",
) -> MopsFilingCandidate:
    available_year = year + 1 if quarter == 4 else year
    return MopsFilingCandidate(
        symbol=symbol,
        fiscal_year=year,
        quarter=quarter,
        report_id="C",
        source_url="https://example.test/{}-{}Q{}.html".format(symbol, year, quarter),
        available_at="{}-02-10T18:16:19+08:00".format(available_year),
        period_end="{}-{:02d}-30".format(year, quarter * 3),
        availability_source=MOPS_PUBLIC_ANNOUNCEMENT,
    )


def _filing(candidate: MopsFilingCandidate) -> MopsFiling:
    return MopsFiling(
        filing_id=candidate.fiscal_year * 10 + candidate.quarter,
        symbol=candidate.symbol,
        fiscal_year=candidate.fiscal_year,
        quarter=candidate.quarter,
        report_id=candidate.report_id,
        available_at=candidate.available_at or "",
        fetched_at=candidate.available_at or "",
        revision_number=1,
        content_sha256="{}Q{}".format(candidate.fiscal_year, candidate.quarter),
        source_url=candidate.source_url,
        raw_path="raw/test.html",
        availability_source=candidate.availability_source,
    )


def _candidate_periods(
    candidates: list[MopsFilingCandidate],
) -> list[tuple[int, int]]:
    return [(candidate.fiscal_year, candidate.quarter) for candidate in candidates]


def _fact(value: float) -> NormalizedFact:
    return NormalizedFact(
        concept="ifrs-full:Revenue",
        context_id="FY",
        unit="TWD",
        numeric_value=value,
        text_value=None,
        start_date="2025-01-01",
        end_date="2025-12-31",
        instant=None,
        dimensions=(),
    )

"""SQLite integration contracts for command/query-separated MOPS storage."""

from __future__ import annotations

import pandas as pd
import pytest

from domain.data_provenance import OFFICIAL_FILING_TIMESTAMP
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_models import FundamentalSnapshot
from domain.fundamental_models import NormalizedFact
from infrastructure.fundamentals.models import MopsFilingCandidate
from infrastructure.fundamentals.mops_commands import MopsRevisionWriter
from infrastructure.fundamentals.mops_queries import MopsQueryRepository
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.fundamentals.mops_schema import MopsSchemaMigrator


def test_unit_of_work_rolls_back_revision_and_refresh_state(tmp_path) -> None:
    repository = MopsRepository(tmp_path / "mops")

    with pytest.raises(RuntimeError, match="abort"):
        with repository.unit_of_work() as uow:
            repository.save_revision(
                _candidate(),
                b"rolled-back-revision",
                [_fact(100.0)],
                available_at="2026-02-10T10:00:00+00:00",
                fetched_at="2026-02-10T10:01:00+00:00",
                uow=uow,
            )
            repository.mark_full_refresh(
                "2330",
                "2026-02-10T10:02:00+00:00",
                uow=uow,
            )
            raise RuntimeError("abort")

    assert repository.queries.load_filings("2330") == []
    assert repository.queries.last_full_refresh_epoch("2330") is None


def test_revision_writer_orders_revisions_and_query_is_point_in_time(
    tmp_path,
) -> None:
    repository = MopsRepository(tmp_path / "mops")
    assert isinstance(repository.revision_writer, MopsRevisionWriter)
    assert isinstance(repository.queries, MopsQueryRepository)
    assert isinstance(repository.schema_migrator, MopsSchemaMigrator)

    first = repository.revision_writer.save_revision(
        _candidate(),
        b"revision-one",
        [_fact(100.0)],
        available_at="2026-02-10T10:00:00+00:00",
        fetched_at="2026-02-10T10:01:00+00:00",
    )
    second = repository.revision_writer.save_revision(
        _candidate(),
        b"revision-two",
        [_fact(110.0)],
        available_at="2026-03-01T10:00:00+00:00",
        fetched_at="2026-03-01T10:01:00+00:00",
    )

    assert (first.revision_number, second.revision_number) == (1, 2)
    assert repository.queries.latest_filing("2330", 2025, 4) == second
    before_revision = repository.queries.load_filed_financials(
        "2330",
        "2026-02-28T23:59:59+00:00",
    )
    after_revision = repository.queries.load_filed_financials(
        "2330",
        "2026-03-01T10:00:00+00:00",
    )
    assert [filing.revision_number for filing in before_revision] == [1]
    assert [filing.revision_number for filing in after_revision] == [1, 2]
    assert before_revision[0].facts[0].numeric_value == 100.0


def test_snapshot_cache_hit_uses_one_query_without_reading_facts(
    monkeypatch,
    tmp_path,
) -> None:
    repository = MopsRepository(tmp_path / "mops")
    repository.save_revision(
        _candidate(),
        b"cache-source",
        [_fact(100.0)],
        available_at="2026-02-10T10:00:00+00:00",
        fetched_at="2026-02-10T10:01:00+00:00",
    )
    filings = repository.load_filed_financials(
        "2330",
        "2026-07-01T00:00:00+00:00",
    )
    filing_set_sha256 = repository.filing_set_hash(filings)
    repository.mark_full_refresh("2330", "2026-07-01T00:00:00+00:00")
    repository.save_fundamental_bundle(
        "2330",
        filing_set_sha256,
        "24",
        FundamentalBundle(
            annual_fundamentals=pd.DataFrame(
                [{"fiscal_year": 2025, "revenue": 100.0}]
            ),
            snapshot={"symbol": "2330", "source": "MOPS"},
        ),
    )
    repository.mark_active_fundamental_bundle(
        "2330",
        filing_set_sha256=filing_set_sha256,
        snapshot_cache_key=filing_set_sha256,
        derivation_version="integration-v1",
        available_at="2026-02-10T10:00:00+00:00",
        latest_period=(2025, 4),
        annual_history_years=1,
    )

    traced: list[str] = []
    real_connect = repository._connect

    def traced_connect():
        connection = real_connect()
        connection.set_trace_callback(traced.append)
        return connection

    monkeypatch.setattr(repository, "_connect", traced_connect)
    cached = repository.queries.load_cached_bundles(
        {"2330": "24"},
        as_of="2026-07-01T00:00:00+00:00",
        derivation_version="integration-v1",
    )

    selects = [sql for sql in traced if sql.lstrip().upper().startswith("SELECT")]
    assert list(cached) == ["2330"]
    assert cached["2330"].bundle.snapshot["source"] == "MOPS"
    assert len(selects) == 1
    assert "facts" not in selects[0].lower()


def test_bulk_financial_batch_queries_match_single_symbol_results(
    tmp_path,
) -> None:
    repository = MopsRepository(tmp_path / "mops")
    snapshots = [
        _snapshot("2330", 2025, 4, "2026-02-10T00:00:00+00:00"),
        _snapshot("2330", 2026, 1, "2026-05-10T00:00:00+00:00"),
        _snapshot("2317", 2025, 4, "2026-02-11T00:00:00+00:00"),
    ]
    with repository.unit_of_work() as uow:
        for snapshot in snapshots:
            repository.save_financial_batch_snapshot(snapshot, uow=uow)
        repository.save_financial_batch_state(
            "2330",
            2026,
            1,
            snapshots[1].content_sha256,
            snapshots[1].observed_at,
            uow=uow,
        )

    symbols = ["2330", "2317", "9999"]
    as_of = "2026-06-01T00:00:00+00:00"
    bulk_history = repository.load_financial_batch_snapshots_many(
        symbols,
        as_of=as_of,
    )
    bulk_states = repository.load_financial_batch_states(symbols)

    assert bulk_history == {
        symbol: repository.load_financial_batch_snapshots(
            symbol,
            as_of=as_of,
        )
        for symbol in symbols
    }
    assert bulk_states == {
        "2330": repository.load_financial_batch_state("2330")
    }


def _candidate() -> MopsFilingCandidate:
    return MopsFilingCandidate(
        symbol="2330",
        fiscal_year=2025,
        quarter=4,
        report_id="C",
        source_url="https://example.test/2330-2025Q4.html",
        availability_source=OFFICIAL_FILING_TIMESTAMP,
    )


def _fact(value: float) -> NormalizedFact:
    return NormalizedFact(
        concept="ifrs-full:Revenue",
        context_id="FY2025",
        unit="TWD",
        numeric_value=value,
        text_value=None,
        start_date="2025-01-01",
        end_date="2025-12-31",
        instant=None,
        dimensions=(),
    )


def _snapshot(
    symbol: str,
    fiscal_year: int,
    quarter: int,
    observed_at: str,
) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol=symbol,
        company_name=symbol,
        fiscal_year=fiscal_year,
        quarter=quarter,
        observed_at=observed_at,
        revenue_ytd=1_000.0,
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

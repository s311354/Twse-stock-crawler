from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from dataclasses import replace
from urllib.parse import urlsplit

import pytest

from application.filing_ingestion import FilingIngestionService
from domain.data_provenance import FIRST_OBSERVED_DOWNLOAD
from domain.data_provenance import LEGACY_UNVERIFIED
from domain.data_provenance import MOPS_PUBLIC_ANNOUNCEMENT
from domain.data_provenance import OFFICIAL_FILING_TIMESTAMP
from domain.fundamental_derivation import FiledFact
from domain.fundamental_derivation import FiledFinancials
from domain.fundamental_derivation import derive_fundamentals
from infrastructure.fundamentals.ixbrl_parser import NormalizedFact
from infrastructure.fundamentals.ixbrl_parser import IxbrlFilingParser
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.models import MopsFilingCandidate
from infrastructure.fundamentals.mops_client import MopsClient
import infrastructure.fundamentals.mops_repository as mops_repository_module
from infrastructure.fundamentals.mops_repository import EMPTY_FILING_SET_SHA256
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.fundamentals.mops_repository import MopsStorageError
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseHttpResponse
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.fundamentals.mops_client import TWSE_COMPANY_PROFILE_URL


def test_ixbrl_parser_resolves_period_scale_and_text() -> None:
    document = b"""
    <html><body><ix:header><ix:resources>
      <xbrli:context id="FY2025"><xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period></xbrli:context>
    </ix:resources></ix:header>
    <ix:nonFraction name="ifrs-full:Revenue" contextRef="FY2025" unitRef="TWD" scale="3">1,234</ix:nonFraction>
    <ix:nonNumeric name="tifrs-ar:AuditReport" contextRef="FY2025">\xe7\x84\xa1\xe4\xbf\x9d\xe7\x95\x99\xe6\x84\x8f\xe8\xa6\x8b</ix:nonNumeric>
    </body></html>
    """

    facts = parse_ixbrl(document)

    assert facts[0].numeric_value == 1_234_000
    assert facts[0].start_date == "2025-01-01"
    assert facts[0].end_date == "2025-12-31"
    assert facts[1].text_value == "無保留意見"


def test_mops_repository_preserves_changed_revisions_and_is_idempotent(tmp_path) -> None:
    repository = MopsRepository(tmp_path / "mops")
    candidate = MopsFilingCandidate(
        symbol="2330",
        fiscal_year=2025,
        quarter=4,
        report_id="C",
        source_url="https://example.test/2330-2025Q4.html",
        available_at="2026-02-10T18:16:19+08:00",
        availability_source=OFFICIAL_FILING_TIMESTAMP,
    )
    facts = [
        NormalizedFact(
            concept="ifrs-full:Revenue",
            context_id="FY2025",
            unit="TWD",
            numeric_value=100.0,
            text_value=None,
            start_date="2025-01-01",
            end_date="2025-12-31",
            instant=None,
            dimensions=(),
        )
    ]

    first = repository.save_revision(
        candidate,
        b"revision-one",
        facts,
        available_at=candidate.available_at or "",
        fetched_at="2026-02-10T11:00:00+00:00",
    )
    duplicate = repository.save_revision(
        candidate,
        b"revision-one",
        facts,
        available_at=candidate.available_at or "",
        fetched_at="2026-02-11T11:00:00+00:00",
    )
    second = repository.save_revision(
        candidate,
        b"revision-two",
        facts,
        available_at="2026-03-01T23:59:59+08:00",
        fetched_at="2026-03-02T01:00:00+00:00",
        correction_announced_on="20260301",
    )

    assert duplicate.filing_id == first.filing_id
    assert second.revision_number == 2
    assert len(repository.load_filings("2330")) == 2
    assert second.correction_announced_on == "20260301"
    assert repository.raw_path(first).read_bytes() == b"revision-one"
    assert repository.load_facts(second.filing_id)[0].numeric_value == 100.0
    assert first.availability_source == OFFICIAL_FILING_TIMESTAMP

    corrected_candidate = MopsFilingCandidate(
        symbol="2317",
        fiscal_year=2025,
        quarter=4,
        report_id="C",
        source_url="https://example.test/2317-2025Q4.html",
    )
    corrected = repository.save_revision(
        corrected_candidate,
        b"already-corrected",
        facts,
        available_at="2026-03-10T23:59:59+08:00",
        fetched_at="2026-03-11T01:00:00+00:00",
        minimum_revision_number=2,
    )
    assert corrected.revision_number == 2
    assert repository.load_filings(
        "2317",
        as_of="2026-03-10T15:00:00+00:00",
    ) == []
    assert len(
        repository.load_filings(
            "2317",
            as_of="2026-03-10T16:00:00+00:00",
        )
    ) == 1


def test_mops_repository_preserves_individual_statement_scope(tmp_path) -> None:
    repository = MopsRepository(tmp_path / "mops")
    candidate = MopsFilingCandidate(
        symbol="6281",
        fiscal_year=2025,
        quarter=4,
        report_id="A",
        source_url="https://example.test/6281-2025Q4.html",
        statement_scope="individual",
    )
    facts = [
        NormalizedFact(
            concept="ifrs-full:Revenue",
            context_id="FY2025",
            unit="TWD",
            numeric_value=100.0,
            text_value=None,
            start_date="2025-01-01",
            end_date="2025-12-31",
            instant=None,
            dimensions=(),
        )
    ]

    repository.save_revision(
        candidate,
        b"individual-filing",
        facts,
        available_at="2026-02-10T10:00:00+08:00",
        fetched_at="2026-02-10T02:00:00+00:00",
    )

    stored = repository.load_filings("6281")[0]
    assert stored.report_id == "A"
    assert stored.statement_scope == "individual"


def test_mops_repository_retries_transient_sqlite_open(monkeypatch, tmp_path) -> None:
    real_connect = sqlite3.connect
    calls = 0

    def flaky_connect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("external disk temporarily unavailable")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(mops_repository_module.sqlite3, "connect", flaky_connect)
    repository = MopsRepository(tmp_path / "mops", sleeper=lambda seconds: None)

    assert repository.latest_filing("2330", 2025, 4) is None
    assert calls >= 3


def test_mops_repository_reports_persistent_storage_failure(monkeypatch, tmp_path) -> None:
    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(mops_repository_module.sqlite3, "connect", unavailable)
    repository = MopsRepository(tmp_path / "mops", sleeper=lambda seconds: None)

    with pytest.raises(MopsStorageError, match="MOPS SQLite cache"):
        repository.latest_filing("2330", 2025, 4)


def test_mops_repository_promotes_legacy_time_to_first_observed(tmp_path) -> None:
    repository = MopsRepository(tmp_path / "mops")
    candidate = MopsFilingCandidate(
        symbol="2330",
        fiscal_year=2025,
        quarter=4,
        report_id="C",
        source_url="https://example.test/2330-2025Q4.html",
    )
    legacy = repository.save_revision(
        candidate,
        b"legacy",
        [],
        available_at="2026-02-10T18:16:19+08:00",
        fetched_at="2026-07-21T03:00:00+00:00",
    )

    promoted = repository.update_availability_provenance(
        legacy,
        available_at=legacy.fetched_at,
        availability_source=FIRST_OBSERVED_DOWNLOAD,
    )

    assert promoted.available_at == "2026-07-21T03:00:00+00:00"
    assert promoted.availability_source == FIRST_OBSERVED_DOWNLOAD
    assert repository.latest_filing("2330", 2025, 4) == promoted


def test_mops_repository_repairs_only_orphaned_empty_state(tmp_path) -> None:
    root = tmp_path / "mops"
    repository = MopsRepository(root)
    candidate = MopsFilingCandidate(
        symbol="2330",
        fiscal_year=2025,
        quarter=4,
        report_id="C",
        source_url="https://example.test/2330-2025Q4.html",
    )
    repository.save_revision(
        candidate,
        b"valid-filing",
        [],
        available_at="2026-02-10T18:16:19+08:00",
        fetched_at="2026-02-11T00:00:00+00:00",
    )
    with sqlite3.connect(repository.database_path) as connection:
        connection.executemany(
            "INSERT INTO refresh_state "
            "(symbol, full_catalog_checked_at_epoch) VALUES (?, ?)",
            [("1471", 1), ("2330", 1)],
        )
        connection.executemany(
            """
            INSERT INTO fundamental_snapshots
                (symbol, filing_set_sha256, sector, annual_json, snapshot_json,
                 created_at_epoch)
            VALUES (?, ?, '24', '[]', '{}', 1)
            """,
            [("1471", EMPTY_FILING_SET_SHA256), ("2330", EMPTY_FILING_SET_SHA256)],
        )

    reopened = MopsRepository(root)
    assert reopened.last_full_refresh_epoch("1471") is None
    with reopened.company_connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM fundamental_snapshots WHERE symbol = '1471'"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM fundamental_snapshots WHERE symbol = '2330'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM refresh_state WHERE symbol = '2330'"
        ).fetchone()[0] == 1


def test_mops_repository_migrates_legacy_rows_as_unverified(tmp_path) -> None:
    root = tmp_path / "mops"
    root.mkdir()
    database_path = root / "mops.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE filings (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                report_id TEXT NOT NULL,
                available_at TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                revision_number INTEGER NOT NULL,
                content_sha256 TEXT NOT NULL,
                source_url TEXT NOT NULL,
                raw_path TEXT NOT NULL,
                period_end TEXT,
                statement_scope TEXT NOT NULL DEFAULT 'consolidated',
                audit_status TEXT,
                correction_description TEXT,
                UNIQUE(symbol, fiscal_year, quarter, report_id, content_sha256)
            );
            INSERT INTO filings VALUES (
                1, '2330', 2025, 4, 'C',
                '2026-02-10T18:16:19+08:00', '2026-07-21T03:00:00+00:00',
                1, 'legacy-hash', 'https://example.test/legacy',
                'raw/legacy.html', '2025-12-31', 'consolidated', 'audited', NULL
            );
            """
        )

    filing = MopsRepository(root).latest_filing("2330", 2025, 4)

    assert filing is not None
    assert filing.availability_source == LEGACY_UNVERIFIED


def test_refresh_state_migration_backfills_daily_revision_check(tmp_path) -> None:
    root = tmp_path / "mops"
    root.mkdir()
    checked_at = int(
        dt.datetime(2026, 8, 2, 1, 0, tzinfo=dt.timezone.utc).timestamp()
    )
    with sqlite3.connect(root / "mops.sqlite3") as connection:
        connection.executescript(
            """
            CREATE TABLE refresh_state (
                symbol TEXT PRIMARY KEY,
                full_catalog_checked_at_epoch INTEGER NOT NULL
            );
            CREATE TABLE fundamental_snapshots (
                symbol TEXT NOT NULL,
                filing_set_sha256 TEXT NOT NULL,
                sector TEXT NOT NULL,
                annual_json TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                created_at_epoch INTEGER NOT NULL,
                PRIMARY KEY(symbol, filing_set_sha256, sector)
            );
            """
        )
        connection.execute(
            "INSERT INTO refresh_state VALUES (?, ?)",
            ("2330", checked_at),
        )
        connection.execute(
            "INSERT INTO fundamental_snapshots VALUES (?, ?, ?, ?, ?, ?)",
            ("2330", "non-empty", "24", "[]", "{}", checked_at),
        )

    repository = MopsRepository(root)

    assert repository.last_recent_revision_check_epoch("2330") == checked_at
    with sqlite3.connect(repository.database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(refresh_state)")
        }
    assert "recent_revision_checked_at_epoch" in columns


def test_first_ixbrl_observation_does_not_backdate_to_board_announcement(tmp_path) -> None:
    ixbrl = b"""
    <html><body><ix:header><ix:resources>
      <xbrli:context id="FY2025"><xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-12-31</xbrli:endDate></xbrli:period></xbrli:context>
    </ix:resources></ix:header>
    <ix:nonFraction name="ifrs-full:Revenue" contextRef="FY2025" unitRef="TWD">100</ix:nonFraction>
    </body></html>
    """

    class FakeHttpClient:
        def request_bytes(self, method, url, **kwargs):
            content = ixbrl if "example.test" in url else b"<table></table>"
            return type("Response", (), {"content": content})()

    candidate = MopsFilingCandidate(
        symbol="2330",
        fiscal_year=2025,
        quarter=4,
        report_id="C",
        source_url="https://example.test/2330-2025Q4.html",
        available_at="2026-02-10T18:16:19+08:00",
        availability_source=MOPS_PUBLIC_ANNOUNCEMENT,
    )
    observed_at = dt.datetime(2026, 7, 21, 3, 0, tzinfo=dt.timezone.utc)
    repository = MopsRepository(tmp_path / "mops")
    client = MopsClient(
        http_client=FakeHttpClient(),
        now_provider=lambda: observed_at,
    )
    ingestion = FilingIngestionService(
        client,
        client,
        IxbrlFilingParser(),
        repository,
    )

    with repository.unit_of_work() as uow:
        filing = ingestion.ingest(
            candidate,
            fallback_available_at=observed_at.isoformat(),
            uow=uow,
        )

    assert filing.available_at == observed_at.isoformat()
    assert filing.availability_source == FIRST_OBSERVED_DOWNLOAD


def test_mops_client_discovers_consolidated_filings_and_filing_time() -> None:
    catalog = b"""
    <table><tr><th>\xe5\xb9\xb4\xe5\xba\xa6\xe5\xad\xa3\xe5\x88\xa5</th><th>\xe8\xb2\xa1\xe5\xa0\xb1\xe9\xa1\x9e\xe5\x88\xa5</th><th>\xe6\x9f\xa5\xe8\xa9\xa2</th></tr>
    <tr><td>114Q4</td><td>\xe5\x90\x88\xe4\xbd\xb5</td><td><input onclick="window.open('/server-java/t164sb01?step=3&amp;SYEAR=2025&amp;file_name=tifrs-fr1-m1-ci-cr-2330-2025Q4.html','x');"></td></tr></table>
    """
    announcements = b"""
    <table><tr><td>2330</td><td>\xe5\x8f\xb0\xe7\xa9\x8d\xe9\x9b\xbb</td><td>115/02/10</td><td>18:16:19</td>
    <td>\xe6\x9c\xac\xe5\x85\xac\xe5\x8f\xb8\xe6\xb0\x91\xe5\x9c\x8b114\xe5\xb9\xb4\xe5\xba\xa6\xe5\x90\x88\xe4\xbd\xb5\xe8\xb2\xa1\xe5\x8b\x99\xe5\xa0\xb1\xe5\x91\x8a\xe6\xa5\xad\xe7\xb6\x93\xe8\x91\xa3\xe4\xba\x8b\xe6\x9c\x83\xe6\xb1\xba\xe8\xad\xb0\xe9\x80\x9a\xe9\x81\x8e</td>
    <td><input onclick="document.fm.spoke_time.value='181619';document.fm.spoke_date.value='20260210';"></td></tr></table>
    """

    class FakeHttpClient:
        def __init__(self) -> None:
            self.contents = [catalog, announcements]

        def request_bytes(self, *args, **kwargs):
            return type("Response", (), {"content": self.contents.pop(0)})()

    client = MopsClient(http_client=FakeHttpClient())
    filings = client.discover_filings("2330")
    discovered = client.discover_announcements("2330", [2026])

    assert [(filing.fiscal_year, filing.quarter) for filing in filings] == [(2025, 4)]
    assert discovered[0].available_at == "2026-02-10T18:16:19+08:00"


def test_mops_client_loads_company_sectors_through_twse_openapi() -> None:
    payload = [
        {"公司代號": "2330", "產業別": "24"},
        {"公司代號": "2317", "產業別": "25"},
    ]

    class FakeHttpClient:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get_json(self, url, *, payload_validator, **kwargs):
            self.urls.append(url)
            payload_validator(payload)
            return TwseHttpResponse(
                payload=payload,
                status_code=200,
                attempts=1,
            )

    http_client = FakeHttpClient()
    client = MopsClient(
        http_client=http_client,  # type: ignore[arg-type]
        min_request_interval_seconds=0,
    )

    assert client.load_company_sectors() == {"2330": "24", "2317": "25"}
    assert http_client.urls == [TWSE_COMPANY_PROFILE_URL]


def test_mops_client_discovers_market_wide_corrections_once_per_period() -> None:
    correction_html = """
    <table>
      <tr><td>2330</td><td>台積電</td><td>20260301</td><td>合併</td>
          <td>更正財務報告</td></tr>
      <tr><td>6281</td><td>啟碁</td><td>20260302</td><td>個別</td>
          <td>更正附註</td></tr>
    </table>
    """.encode()

    class FakeHttpClient:
        def __init__(self) -> None:
            self.calls = 0

        def request_bytes(self, *args, **kwargs):
            self.calls += 1
            return type("Response", (), {"content": correction_html})()

    http_client = FakeHttpClient()
    client = MopsClient(
        http_client=http_client,  # type: ignore[arg-type]
        min_request_interval_seconds=0,
    )
    corrections = client.discover_corrections([(2025, 4), (2025, 4)])
    candidate = MopsFilingCandidate(
        symbol="2330",
        fiscal_year=2025,
        quarter=4,
        report_id="C",
        source_url="https://example.test/2330-2025Q4.html",
    )

    assert [(item.symbol, item.announced_on) for item in corrections] == [
        ("2330", "20260301"),
        ("6281", "20260302"),
    ]
    assert client.latest_correction(candidate) == corrections[0]
    assert http_client.calls == 1


def test_mops_client_prefers_consolidated_and_falls_back_to_individual(
    caplog,
) -> None:
    catalog = b"""
    <table>
    <tr><th>period</th><th>scope</th><th>query</th></tr>
    <tr><td>114Q4</td><td>\xe5\x80\x8b\xe5\x88\xa5</td><td>
    <input onclick="window.open('/server-java/t164sb01?step=3&amp;SYEAR=2025&amp;file_name=tifrs-fr1-m1-ci-ir-6281-2025Q4.html','x');">
    </td></tr>
    <tr><td>114Q4</td><td>\xe5\x90\x88\xe4\xbd\xb5</td><td>
    <input onclick="window.open('/server-java/t164sb01?step=3&amp;SYEAR=2025&amp;file_name=tifrs-fr1-m1-ci-cr-6281-2025Q4.html','x');">
    </td></tr>
    <tr><td>114Q3</td><td>\xe5\x80\x8b\xe5\x88\xa5</td><td>
    <input onclick="window.open('/server-java/t164sb01?step=3&amp;SYEAR=2025&amp;file_name=tifrs-fr1-m1-ci-ir-6281-2025Q3.html','x');">
    </td></tr>
    </table>
    """

    class FakeHttpClient:
        def request_bytes(self, *args, **kwargs):
            return type("Response", (), {"content": catalog})()

    caplog.set_level("DEBUG")
    client = MopsClient(
        http_client=FakeHttpClient(),
        min_request_interval_seconds=0,
    )

    filings = client.discover_filings("6281")

    assert [
        (
            filing.fiscal_year,
            filing.quarter,
            filing.report_id,
            filing.statement_scope,
        )
        for filing in filings
    ] == [
        (2025, 3, "A", "individual"),
        (2025, 4, "C", "consolidated"),
    ]
    assert filings[0].source_url.endswith(
        "tifrs-fr1-m1-ci-ir-6281-2025Q3.html"
    )
    assert filings[1].source_url.endswith(
        "tifrs-fr1-m1-ci-cr-6281-2025Q4.html"
    )
    assert "detected_scopes=consolidated:1,individual:2" in caplog.text
    assert "fallback_scope=individual period_count=1" in caplog.text
    assert "first_period=2025Q3 latest_period=2025Q3" in caplog.text
    fallback_record = next(
        record
        for record in caplog.records
        if getattr(record, "status", None)
        == "INDIVIDUAL_SCOPE_FALLBACK"
    )
    assert fallback_record.levelno == logging.DEBUG
    catalog_record = next(
        record
        for record in caplog.records
        if "selected_filings=2" in record.getMessage()
    )
    assert catalog_record.symbol == "6281"
    assert catalog_record.source == "MOPS"
    assert catalog_record.status == "CATALOG_OK"
    assert catalog_record.attempt == 1
    assert catalog_record.cache_hit is False


def test_mops_client_rechecks_explicit_no_data_with_fresh_session() -> None:
    no_data = b"<html><body><font>\xe6\x9f\xa5\xe7\x84\xa1\xe8\xb3\x87\xe6\x96\x99</font></body></html>"
    catalog = b"""
    <table><tr><td>114Q4</td><td>\xe5\x90\x88\xe4\xbd\xb5</td><td>
    <input onclick="window.open('/server-java/t164sb01?file_name=tifrs-fr1-m1-ci-cr-2330-2025Q4.html','x');">
    </td></tr></table>
    """

    class Cookies:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear(self) -> None:
            self.clear_calls += 1

    class FakeHttpClient:
        def __init__(self) -> None:
            self.contents = [no_data, catalog]
            self.session = type("Session", (), {"cookies": Cookies()})()

        def request_bytes(self, *args, **kwargs):
            return type("Response", (), {"content": self.contents.pop(0)})()

    sleeps: list[float] = []
    http_client = FakeHttpClient()
    client = MopsClient(
        http_client=http_client,
        catalog_empty_attempts=2,
        catalog_empty_backoff_seconds=3,
        min_request_interval_seconds=0,
        sleeper=sleeps.append,
    )

    filings = client.discover_filings("2330")

    assert [(filing.fiscal_year, filing.quarter) for filing in filings] == [(2025, 4)]
    assert sleeps == [3]
    assert http_client.session.cookies.clear_calls == 1


def test_mops_client_classifies_unrecognized_catalog_html_as_schema_changed() -> None:
    class FakeHttpClient:
        def request_bytes(self, *args, **kwargs):
            return type("Response", (), {"content": b"<html><table><tr><td>changed</td></tr></table></html>"})()

    client = MopsClient(
        http_client=FakeHttpClient(),
        catalog_empty_backoff_seconds=0,
        min_request_interval_seconds=0,
    )

    with pytest.raises(TwseSchemaChangedError, match="without filing rows"):
        client.discover_filings("2330")


def test_mops_client_resets_session_after_transient_catalog_guard_page() -> None:
    guard_page = b"<html><body>request guard</body></html>"
    catalog = b"""
    <table><tr><td>114Q4</td><td>\xe5\x90\x88\xe4\xbd\xb5</td><td>
    <input onclick="window.open('/server-java/t164sb01?file_name=tifrs-fr1-m1-ci-cr-2330-2025Q4.html','x');">
    </td></tr></table>
    """

    class RecoveringHttpClient:
        def __init__(self) -> None:
            self.contents = [guard_page, catalog]
            self.reset_calls = 0

        def request_bytes(self, *args, **kwargs):
            return type("Response", (), {"content": self.contents.pop(0)})()

        def reset_session(self) -> None:
            self.reset_calls += 1

    sleeps: list[float] = []
    http_client = RecoveringHttpClient()
    client = MopsClient(
        http_client=http_client,
        catalog_empty_attempts=2,
        catalog_empty_backoff_seconds=3,
        min_request_interval_seconds=0,
        sleeper=sleeps.append,
    )

    filings = client.discover_filings("2330")

    assert [(filing.fiscal_year, filing.quarter) for filing in filings] == [(2025, 4)]
    assert http_client.reset_calls == 1
    assert sleeps == [3]


def test_mops_client_fails_over_to_alternate_host_on_network_error() -> None:
    catalog = b"""
    <table><tr><td>114Q4</td><td>\xe5\x90\x88\xe4\xbd\xb5</td><td>
    <input onclick="window.open('/server-java/t164sb01?file_name=tifrs-fr1-m1-ci-cr-2330-2025Q4.html','x');">
    </td></tr></table>
    """

    class FailingPrimaryHttpClient:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def request_bytes(self, method, url, **kwargs):
            self.urls.append(url)
            if "mopsov.twse.com.tw" in url:
                raise TwseNetworkError("primary DNS failure")
            return type("Response", (), {"content": catalog})()

    http_client = FailingPrimaryHttpClient()
    client = MopsClient(http_client=http_client)

    filings = client.discover_filings("2330")

    assert [urlsplit(url).netloc for url in http_client.urls] == [
        "mopsov.twse.com.tw",
        "mops.twse.com.tw",
    ]
    assert filings[0].source_url.startswith("https://mops.twse.com.tw/")


def test_mops_client_retries_all_hosts_after_dns_backoff() -> None:
    catalog = b"""
    <table><tr><td>114Q4</td><td>\xe5\x90\x88\xe4\xbd\xb5</td><td>
    <input onclick="window.open('/server-java/t164sb01?file_name=tifrs-fr1-m1-ci-cr-2312-2025Q4.html','x');">
    </td></tr></table>
    """

    class RecoveringDnsHttpClient:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def request_bytes(self, method, url, **kwargs):
            self.urls.append(url)
            if len(self.urls) <= 2:
                raise TwseNetworkError("temporary DNS failure", retryable=True)
            return type("Response", (), {"content": catalog})()

    sleeps: list[float] = []
    http_client = RecoveringDnsHttpClient()
    client = MopsClient(
        http_client=http_client,
        network_rounds=2,
        network_backoff_seconds=5,
        sleeper=sleeps.append,
    )

    filings = client.discover_filings("2312")

    assert sleeps == [5]
    assert [urlsplit(url).netloc for url in http_client.urls] == [
        "mopsov.twse.com.tw",
        "mops.twse.com.tw",
        "mopsov.twse.com.tw",
    ]
    assert [(filing.fiscal_year, filing.quarter) for filing in filings] == [(2025, 4)]


def test_mops_client_does_not_repeat_hosts_for_non_retryable_network_error() -> None:
    class PermanentFailureHttpClient:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def request_bytes(self, method, url, **kwargs):
            self.urls.append(url)
            raise TwseNetworkError("certificate rejected", retryable=False)

    sleeps: list[float] = []
    http_client = PermanentFailureHttpClient()
    client = MopsClient(
        http_client=http_client,
        network_rounds=2,
        network_backoff_seconds=5,
        sleeper=sleeps.append,
    )

    with pytest.raises(TwseNetworkError):
        client.discover_filings("2312")

    assert sleeps == []
    assert len(http_client.urls) == 2


def test_ttm_roic_and_fcf_derivation_are_point_in_time() -> None:
    annual_2024 = _annual_filing(2024, available_at="2025-02-01T10:00:00+08:00")
    latest_q1 = FiledFinancials(
        fiscal_year=2025,
        quarter=1,
        available_at="2025-05-01T10:00:00+08:00",
        revision_number=1,
        facts=tuple(
            _flow("ifrs-full:DilutedEarningsLossPerShare", 3, 2025, "2025-03-31")
            + _flow("ifrs-full:DilutedEarningsLossPerShare", 2, 2024, "2024-03-31")
            + _flow("ifrs-full:ProfitLossAttributableToOwnersOfParent", 18, 2025, "2025-03-31")
            + _flow("ifrs-full:ProfitLossAttributableToOwnersOfParent", 10, 2024, "2024-03-31")
            + _flow("ifrs-full:ProfitLossFromOperatingActivities", 30, 2025, "2025-03-31")
            + _flow("ifrs-full:ProfitLossFromOperatingActivities", 20, 2024, "2024-03-31")
            + _flow("ifrs-full:CashFlowsFromUsedInOperatingActivities", 30, 2025, "2025-03-31")
            + _flow("ifrs-full:CashFlowsFromUsedInOperatingActivities", 20, 2024, "2024-03-31")
            + _flow("ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", 10, 2025, "2025-03-31")
            + _flow("ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", 8, 2024, "2024-03-31")
            + _flow("ifrs-full:AdjustmentsForDepreciationExpense", 5, 2025, "2025-03-31")
            + _flow("ifrs-full:AdjustmentsForDepreciationExpense", 4, 2024, "2024-03-31")
            + _flow("ifrs-full:AdjustmentsForAmortisationExpense", 1, 2025, "2025-03-31")
            + _flow("ifrs-full:AdjustmentsForAmortisationExpense", 1, 2024, "2024-03-31")
            + _flow("ifrs-full:FinanceCosts", 2, 2025, "2025-03-31")
            + _flow("ifrs-full:FinanceCosts", 1, 2024, "2024-03-31")
            + _instant_facts(2025)
            + [FiledFact("tifrs-ar:UnqualifiedOpinionAbstract", None, "無保留意見", None, None, None)]
        ),
    )

    before_publication = derive_fundamentals(
        [annual_2024, latest_q1],
        signal_date="2025-04-30T13:30:00+08:00",
        sector="ELEC",
    )
    available = derive_fundamentals(
        [annual_2024, latest_q1],
        signal_date="2025-05-02T13:30:00+08:00",
        sector="ELEC",
    )

    assert before_publication.snapshot["ttm_eps"] == 10
    assert available.snapshot["ttm_eps"] == pytest.approx(58 / 5.25)
    assert available.snapshot["free_cash_flow_ttm"] == 68
    assert available.snapshot["ebit_ttm"] == 110
    assert available.snapshot["ebitda_ttm"] == 136
    assert available.snapshot["going_concern"] is False
    assert available.snapshot["revision_number"] == 1
    assert available.snapshot["availability_source"] == "LEGACY_UNVERIFIED"
    assert available.annual_fundamentals.iloc[-1]["free_cash_flow"] == 60
    assert available.annual_fundamentals.iloc[-1]["roic"] == pytest.approx(80 / 780)

    material_revision = derive_fundamentals(
        [annual_2024, replace(latest_q1, revision_number=2, correction_description="重編損益")],
        signal_date="2025-05-02T13:30:00+08:00",
        sector="ELEC",
    )
    assert material_revision.snapshot["material_accounting_issue"] is True
    assert (
        material_revision.snapshot["accounting_concern_classification"]
        == "MATERIAL_RESTATEMENT"
    )
    assert material_revision.snapshot["revision_number"] == 2


def test_audit_report_scope_limitation_does_not_become_material_issue() -> None:
    filing = _annual_filing(2024, available_at="2025-02-01T10:00:00+08:00")
    facts = tuple(
        fact
        for fact in filing.facts
        if fact.concept != "tifrs-ar:UnqualifiedOpinionAbstract"
    ) + (
        FiledFact(
            "tifrs-ar:AccountantsReportBody",
            None,
            (
                "範圍：除保留結論之基礎段所述者外，本會計師依核閱準則執行。"
                "保留結論之基礎：部分非重要子公司未經會計師核閱。"
                "保留結論：除上述事項外，並未發現財務報告未允當表達。"
            ),
            None,
            None,
            None,
        ),
    )

    bundle = derive_fundamentals(
        [replace(filing, facts=facts)],
        signal_date="2025-02-02T13:30:00+08:00",
        sector="24",
    )

    assert bundle.snapshot["audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert bundle.snapshot["going_concern"] is False
    assert bundle.snapshot["material_accounting_issue"] is False
    assert (
        bundle.snapshot["accounting_concern_classification"]
        == "QUALIFIED_SCOPE_LIMITATION"
    )


def test_current_unmodified_abstract_overrides_prior_qualified_narrative() -> None:
    filing = _annual_filing(
        2024,
        available_at="2025-02-01T10:00:00+08:00",
    )
    facts = filing.facts + (
        FiledFact(
            "tifrs-ar:AccountantsReportBody",
            None,
            (
                "本期核閱結論：並未發現財務報告未允當表達。"
                "其他事項：前一年度保留結論之基礎為部分子公司"
                "未經會計師核閱。"
            ),
            None,
            None,
            None,
        ),
    )

    bundle = derive_fundamentals(
        [replace(filing, facts=facts)],
        signal_date="2025-02-02T13:30:00+08:00",
        sector="24",
    )

    assert bundle.snapshot["audit_opinion"] == "UNMODIFIED"
    assert bundle.snapshot["current_audit_opinion"] == "UNMODIFIED"
    assert (
        bundle.snapshot["comparative_audit_opinion"]
        == "QUALIFIED_SCOPE_LIMITATION"
    )
    assert bundle.snapshot["audit_opinion_source"] == "ABSTRACT"
    assert (
        bundle.snapshot["comparative_audit_opinion_source"]
        == "REPORT_BODY_COMPARATIVE"
    )
    assert bundle.snapshot["material_accounting_issue"] is False


def test_report_body_fallback_ignores_prior_period_audit_markers() -> None:
    filing = _annual_filing(
        2024,
        available_at="2025-02-01T10:00:00+08:00",
    )
    facts = tuple(
        fact
        for fact in filing.facts
        if fact.concept != "tifrs-ar:UnqualifiedOpinionAbstract"
    ) + (
        FiledFact(
            "tifrs-ar:AccountantsReportBody",
            None,
            (
                "本期核閱結論：並未發現財務報告未允當表達。"
                "其他事項：前一年度保留結論之基礎為部分子公司"
                "未經會計師核閱；前一年度繼續經營重大不確定性"
                "亦已於前期財務報告揭露。"
            ),
            None,
            None,
            None,
        ),
    )

    bundle = derive_fundamentals(
        [replace(filing, facts=facts)],
        signal_date="2025-02-02T13:30:00+08:00",
        sector="24",
    )

    assert bundle.snapshot["audit_opinion"] == "UNMODIFIED"
    assert bundle.snapshot["current_audit_opinion"] == "UNMODIFIED"
    assert (
        bundle.snapshot["comparative_audit_opinion"]
        == "QUALIFIED_SCOPE_LIMITATION"
    )
    assert bundle.snapshot["audit_opinion_source"] == "REPORT_BODY_FALLBACK"
    assert bundle.snapshot["going_concern"] is False


def test_comparative_year_heading_is_separated_from_current_opinion() -> None:
    filing = _annual_filing(
        2024,
        available_at="2025-02-01T10:00:00+08:00",
    )
    facts = filing.facts + (
        FiledFact(
            "tifrs-ar:AccountantsReportBody",
            None,
            (
                "本期無保留意見。民國112年度保留意見之基礎:"
                "部分非重要子公司未經會計師查核。"
            ),
            None,
            None,
            None,
        ),
    )

    bundle = derive_fundamentals(
        [replace(filing, facts=facts)],
        signal_date="2025-02-02T13:30:00+08:00",
        sector="24",
    )

    assert bundle.snapshot["current_audit_opinion"] == "UNMODIFIED"
    assert (
        bundle.snapshot["comparative_audit_opinion"]
        == "QUALIFIED_SCOPE_LIMITATION"
    )
    assert bundle.snapshot["material_accounting_issue"] is False


def test_scope_limitation_classifier_ignores_ixbrl_whitespace() -> None:
    filing = _annual_filing(
        2024,
        available_at="2025-02-01T10:00:00+08:00",
    )
    raw_report_text = (
        "保留結論之基礎：部分非重要 \n 子公司"
        "未經會計師 \n 核閱。"
    )
    facts = tuple(
        fact
        for fact in filing.facts
        if fact.concept != "tifrs-ar:UnqualifiedOpinionAbstract"
    ) + (
        FiledFact(
            "tifrs-ar:QualifiedOpinionAbstract",
            None,
            "Y",
            None,
            None,
            None,
        ),
        FiledFact(
            "tifrs-ar:AccountantsReportBody",
            None,
            raw_report_text,
            None,
            None,
            None,
        ),
    )

    bundle = derive_fundamentals(
        [replace(filing, facts=facts)],
        signal_date="2025-02-02T13:30:00+08:00",
        sector="24",
    )

    assert bundle.snapshot["audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert bundle.snapshot["material_accounting_issue"] is False
    assert bundle.snapshot["audit_report_text_raw"] == raw_report_text
    assert (
        bundle.snapshot["audit_report_text_normalized"]
        == "保留結論之基礎:部分非重要子公司未經會計師核閱。"
    )


def test_audit_report_material_misstatement_remains_hard_evidence() -> None:
    filing = _annual_filing(2024, available_at="2025-02-01T10:00:00+08:00")
    facts = tuple(
        fact
        for fact in filing.facts
        if fact.concept != "tifrs-ar:UnqualifiedOpinionAbstract"
    ) + (
        FiledFact(
            "tifrs-ar:AccountantsReportBody",
            None,
            "保留意見之基礎：公司對存貨會計處理不適當，造成重大錯報。保留意見。",
            None,
            None,
            None,
        ),
    )

    bundle = derive_fundamentals(
        [replace(filing, facts=facts)],
        signal_date="2025-02-02T13:30:00+08:00",
        sector="24",
    )

    assert bundle.snapshot["audit_opinion"] == "QUALIFIED_MATERIAL_MISSTATEMENT"
    assert bundle.snapshot["material_accounting_issue"] is True


def test_unclassifiable_qualified_basis_remains_unknown() -> None:
    filing = _annual_filing(2024, available_at="2025-02-01T10:00:00+08:00")
    facts = tuple(
        fact
        for fact in filing.facts
        if fact.concept != "tifrs-ar:UnqualifiedOpinionAbstract"
    ) + (
        FiledFact(
            "tifrs-ar:QualifiedOpinionAbstract",
            None,
            "Y",
            None,
            None,
            None,
        ),
    )

    bundle = derive_fundamentals(
        [replace(filing, facts=facts)],
        signal_date="2025-02-02T13:30:00+08:00",
        sector="24",
    )

    assert bundle.snapshot["audit_opinion"] == "UNKNOWN"
    assert bundle.snapshot["material_accounting_issue"] is None


def test_non_material_correction_does_not_become_material_restatement() -> None:
    filing = replace(
        _annual_filing(2024, available_at="2025-02-01T10:00:00+08:00"),
        correction_description="更正財務報告附註頁碼誤植",
    )

    bundle = derive_fundamentals(
        [filing],
        signal_date="2025-02-02T13:30:00+08:00",
        sector="24",
    )

    assert bundle.snapshot["material_accounting_issue"] is False


def _annual_filing(year: int, available_at: str) -> FiledFinancials:
    facts = (
        _flow("ifrs-full:Revenue", 200, year, "{}-12-31".format(year))
        + _flow("ifrs-full:ProfitLossAttributableToOwnersOfParent", 50, year, "{}-12-31".format(year))
        + _flow("ifrs-full:CashFlowsFromUsedInOperatingActivities", 100, year, "{}-12-31".format(year))
        + _flow("ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", 40, year, "{}-12-31".format(year))
        + _flow("ifrs-full:ProfitLossFromOperatingActivities", 100, year, "{}-12-31".format(year))
        + _flow("ifrs-full:ProfitLossBeforeTax", 100, year, "{}-12-31".format(year))
        + _flow("ifrs-full:IncomeTaxExpenseContinuingOperations", 20, year, "{}-12-31".format(year))
        + _flow("ifrs-full:DilutedEarningsLossPerShare", 10, year, "{}-12-31".format(year))
        + _flow("ifrs-full:AdjustmentsForDepreciationExpense", 20, year, "{}-12-31".format(year))
        + _flow("ifrs-full:AdjustmentsForAmortisationExpense", 5, year, "{}-12-31".format(year))
        + _flow("ifrs-full:FinanceCosts", 5, year, "{}-12-31".format(year))
        + _instant_facts(year)
        + [FiledFact("tifrs-ar:UnqualifiedOpinionAbstract", None, "無保留意見", None, None, None)]
    )
    return FiledFinancials(year, 4, available_at, 1, tuple(facts))


def _flow(concept: str, value: float, year: int, end_date: str) -> list[FiledFact]:
    return [FiledFact(concept, value, None, "{}-01-01".format(year), end_date, None)]


def _instant_facts(year: int) -> list[FiledFact]:
    instant = "{}-03-31".format(year) if year == 2025 else "{}-12-31".format(year)
    return [
        FiledFact("ifrs-full:CashAndCashEquivalents", 20, None, None, None, instant),
        FiledFact("ifrs-full:EquityAttributableToOwnersOfParent", 700, None, None, None, instant),
        FiledFact("ifrs-full:LongtermBorrowings", 100, None, None, None, instant),
    ]

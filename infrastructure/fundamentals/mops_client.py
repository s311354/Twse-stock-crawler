"""Official MOPS filing discovery and iXBRL download client."""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo

from config.settings import AppSettings
from infrastructure.crawler.twse_http_client import TwseAccessBlockedError
from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseNoDataError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.fundamentals.ixbrl_parser import decode_mops_content
from infrastructure.fundamentals.models import MopsCorrection
from domain.fundamental_models import FilingAnnouncement
from domain.fundamental_models import FilingDocument
from infrastructure.fundamentals.models import MopsFilingCandidate
from infrastructure.observability import structured_extra


MOPS_BASE_URL = "https://mopsov.twse.com.tw"
MOPS_FALLBACK_BASE_URL = "https://mops.twse.com.tw"
MOPS_BASE_URLS = (MOPS_BASE_URL, MOPS_FALLBACK_BASE_URL)
MOPS_CATALOG_URL = MOPS_BASE_URL + "/mops/web/ajax_t203sb01"
MOPS_ANNOUNCEMENT_URL = MOPS_BASE_URL + "/mops/web/ajax_t05st01"
MOPS_CORRECTION_URL = MOPS_BASE_URL + "/mops/web/ajax_t56sb31_q1"
TWSE_COMPANY_PROFILE_URL = (
    "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
)
HTML_HEADERS = {
    "Accept": "text/html,application/xhtml+xml",
    "Referer": MOPS_BASE_URL + "/mops/web/index",
}
TAIPEI = ZoneInfo("Asia/Taipei")
MOPS_ACCESS_BLOCKED_MARKERS = (
    "因為安全性考量",
    "所執行的頁面無法呈現",
)


@dataclass
class _HtmlRow:
    cells: list[str]
    actions: list[str]


@dataclass(frozen=True)
class _CatalogParseResult:
    filings: list[MopsFilingCandidate]
    detected_scope_counts: dict[str, int]
    filing_link_rows: int
    table_rows: int

    @property
    def scope_summary(self) -> str:
        if not self.detected_scope_counts:
            return "NONE"
        return ",".join(
            "{}:{}".format(scope, self.detected_scope_counts[scope])
            for scope in sorted(self.detected_scope_counts)
        )


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[_HtmlRow] = []
        self._row: _HtmlRow | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.lower(): value or "" for name, value in attrs}
        if tag.lower() == "tr":
            self._row = _HtmlRow(cells=[], actions=[])
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell_parts = []
        elif tag.lower() == "input" and self._row is not None:
            action = attributes.get("onclick")
            if action:
                self._row.actions.append(action)

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"td", "th"} and self._row is not None and self._cell_parts is not None:
            self._row.cells.append(_normalize_text("".join(self._cell_parts)))
            self._cell_parts = None
        elif tag.lower() == "tr" and self._row is not None:
            if self._row.cells:
                self.rows.append(self._row)
            self._row = None


class MopsClient:
    """Discover and download official MOPS filings without persistence."""

    def __init__(
        self,
        http_client: TwseHttpClient | None = None,
        now_provider: Callable[[], dt.datetime] | None = None,
        base_urls: tuple[str, ...] = MOPS_BASE_URLS,
        network_rounds: int | None = None,
        network_backoff_seconds: float | None = None,
        catalog_empty_attempts: int | None = None,
        catalog_empty_backoff_seconds: float | None = None,
        min_request_interval_seconds: float | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not base_urls:
            raise ValueError("base_urls must contain at least one MOPS host")
        settings = (
            AppSettings.from_env(
                mops_enabled=False,
                validate_paths=False,
            )
            if any(
                value is None
                for value in (
                    network_rounds,
                    network_backoff_seconds,
                    catalog_empty_attempts,
                    catalog_empty_backoff_seconds,
                    min_request_interval_seconds,
                )
            )
            else None
        )
        self.http_client = http_client or TwseHttpClient()
        self._now_provider = now_provider or (lambda: dt.datetime.now(dt.timezone.utc))
        self._correction_cache: dict[tuple[int, int], list[MopsCorrection]] = {}
        self._base_urls = tuple(url.rstrip("/") for url in base_urls)
        self._active_base_url = self._base_urls[0]
        self._network_rounds = (
            (
                settings.mops_network_retry_rounds
                if settings is not None
                else 2
            )
            if network_rounds is None
            else network_rounds
        )
        self._network_backoff_seconds = (
            (
                settings.mops_network_round_backoff_seconds
                if settings is not None
                else 5.0
            )
            if network_backoff_seconds is None
            else network_backoff_seconds
        )
        self._catalog_empty_attempts = (
            (
                settings.mops_catalog_empty_attempts
                if settings is not None
                else 2
            )
            if catalog_empty_attempts is None
            else catalog_empty_attempts
        )
        self._catalog_empty_backoff_seconds = (
            (
                settings.mops_catalog_empty_backoff_seconds
                if settings is not None
                else 3.0
            )
            if catalog_empty_backoff_seconds is None
            else catalog_empty_backoff_seconds
        )
        self._min_request_interval_seconds = (
            (
                settings.mops_min_request_interval_seconds
                if settings is not None
                else 1.0
            )
            if min_request_interval_seconds is None
            else min_request_interval_seconds
        )
        self._sleep = sleeper
        self._monotonic = monotonic
        self._last_request_at: float | None = None
        if self._network_rounds < 1:
            raise ValueError("network_rounds must be at least one")
        if self._network_backoff_seconds < 0:
            raise ValueError("network_backoff_seconds cannot be negative")
        if self._catalog_empty_attempts < 1:
            raise ValueError("catalog_empty_attempts must be at least one")
        if self._catalog_empty_backoff_seconds < 0:
            raise ValueError("catalog_empty_backoff_seconds cannot be negative")
        if self._min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds cannot be negative")

    def discover(self, symbol: str) -> list[MopsFilingCandidate]:
        normalized_symbol = _normalize_symbol(symbol)
        context = "MOPS iXBRL catalog {}".format(normalized_symbol)
        for attempt in range(1, self._catalog_empty_attempts + 1):
            response, response_url = self._request_bytes(
                "POST",
                MOPS_CATALOG_URL,
                data={
                    "step": "1",
                    "firstin": "1",
                    "off": "1",
                    "TYPEK": "all",
                    "co_id": normalized_symbol,
                },
                context=context,
            )
            parse_result = self._parse_catalog(
                normalized_symbol,
                response.content,
                response_url,
            )
            filings = parse_result.filings
            if filings:
                selected_scopes = sorted(
                    {filing.statement_scope for filing in filings}
                )
                logging.debug(
                    "MOPS catalog symbol=%s detected_scopes=%s "
                    "filing_link_rows=%s selected_filings=%s "
                    "selected_scopes=%s policy=CONSOLIDATED_FIRST",
                    normalized_symbol,
                    parse_result.scope_summary,
                    parse_result.filing_link_rows,
                    len(filings),
                    ",".join(selected_scopes),
                    extra=structured_extra(
                        symbol=normalized_symbol,
                        source="MOPS",
                        status="CATALOG_OK",
                        attempt=attempt,
                        cache_hit=False,
                    ),
                )
                individual_periods = [
                    "{}Q{}".format(filing.fiscal_year, filing.quarter)
                    for filing in filings
                    if filing.statement_scope == "individual"
                ]
                if individual_periods:
                    logging.debug(
                        "MOPS catalog symbol=%s detected_scopes=%s "
                        "fallback_scope=individual period_count=%s "
                        "first_period=%s latest_period=%s "
                        "reason=consolidated_filing_unavailable",
                        normalized_symbol,
                        parse_result.scope_summary,
                        len(individual_periods),
                        individual_periods[0],
                        individual_periods[-1],
                        extra=structured_extra(
                            symbol=normalized_symbol,
                            source="MOPS",
                            status="INDIVIDUAL_SCOPE_FALLBACK",
                            attempt=attempt,
                            cache_hit=False,
                        ),
                    )
                return filings

            content_text = decode_mops_content(response.content)
            explicit_no_data = "查無資料" in content_text
            access_blocked = any(
                marker in content_text
                for marker in MOPS_ACCESS_BLOCKED_MARKERS
            )
            parse_detail = (
                "detected_scopes={} filing_link_rows={} table_rows={}".format(
                    parse_result.scope_summary,
                    parse_result.filing_link_rows,
                    parse_result.table_rows,
                )
            )
            if access_blocked:
                response_status = "ACCESS_BLOCKED"
                response_detail = "MOPS security guard page; {}".format(
                    parse_detail
                )
            elif explicit_no_data:
                response_status = "NO_DATA"
                response_detail = "explicit no-data response; {}".format(
                    parse_detail
                )
            else:
                response_status = "SCHEMA_CHANGED"
                response_detail = (
                    "HTML without filing rows or no-data marker; "
                    "supported_scope_detail={}".format(parse_detail)
                )
            if attempt < self._catalog_empty_attempts:
                delay = self._catalog_empty_backoff_seconds * attempt
                log_retry = logging.debug if access_blocked else logging.warning
                log_retry(
                    "MOPS status=%s context=%s attempt=%s/%s retrying=true "
                    "reset_session=true backoff_seconds=%s detail=%s",
                    response_status,
                    context,
                    attempt,
                    self._catalog_empty_attempts,
                    delay,
                    response_detail,
                    extra=structured_extra(
                        symbol=normalized_symbol,
                        source="MOPS",
                        status=response_status,
                        attempt=attempt,
                        cache_hit=False,
                    ),
                )
                self._reset_mops_session()
                self._sleep(delay)
                continue
            if explicit_no_data:
                raise TwseNoDataError(
                    "MOPS catalog explicitly returned no data for {} after {} attempt(s)".format(
                        normalized_symbol,
                        self._catalog_empty_attempts,
                    ),
                    retryable=False,
                )
            if access_blocked:
                raise TwseAccessBlockedError(
                    "{} returned the MOPS security guard page; {}".format(
                        context,
                        parse_detail,
                    ),
                    retryable=False,
                )
            preview = " ".join(content_text[:120].split())
            raise TwseSchemaChangedError(
                "{} returned HTML without filing rows or an explicit no-data "
                "marker; supported_scope_detail={}; body_prefix={!r}".format(
                    context,
                    parse_detail,
                    preview,
                )
            )

        raise AssertionError("unreachable catalog retry state")

    def discover_filings(
        self,
        symbol: str,
    ) -> list[MopsFilingCandidate]:
        """Backward-compatible alias for callers outside the application."""

        return self.discover(symbol)

    @staticmethod
    def _parse_catalog(
        symbol: str,
        content: bytes,
        response_url: str,
    ) -> _CatalogParseResult:
        parser = _parse_table(content)
        filings: dict[tuple[int, int], tuple[int, MopsFilingCandidate]] = {}
        scope_counts: dict[str, int] = {}
        filing_link_rows = 0
        for row in parser.rows:
            if len(row.cells) < 2:
                continue
            scope = _catalog_statement_scope(row.cells[1])
            if scope is None:
                continue
            statement_scope, report_id, priority = scope
            scope_counts[statement_scope] = (
                scope_counts.get(statement_scope, 0) + 1
            )
            action = next(
                (value for value in row.actions if "file_name=" in value),
                "",
            )
            match = re.search(
                r"window\.open\('(?P<path>[^']*file_name=[^']*-"
                r"(?P<year>\d{4})Q(?P<quarter>[1-4])\.html)'",
                action,
                re.IGNORECASE,
            )
            if match is None:
                continue
            filing_link_rows += 1
            year = int(match.group("year"))
            quarter = int(match.group("quarter"))
            candidate = MopsFilingCandidate(
                symbol=symbol,
                fiscal_year=year,
                quarter=quarter,
                report_id=report_id,
                source_url=urljoin(_origin(response_url), match.group("path")),
                period_end="{}-{:02d}-{}".format(
                    year,
                    quarter * 3,
                    31 if quarter in {1, 4} else 30,
                ),
                statement_scope=statement_scope,
                audit_status="audited" if quarter == 4 else "reviewed",
            )
            period = (year, quarter)
            selected = filings.get(period)
            if selected is None or priority > selected[0]:
                filings[period] = (priority, candidate)
        return _CatalogParseResult(
            filings=[filings[key][1] for key in sorted(filings)],
            detected_scope_counts=scope_counts,
            filing_link_rows=filing_link_rows,
            table_rows=len(parser.rows),
        )

    def load_company_sectors(self) -> dict[str, str]:
        """Load the official listed-company industry code mapping."""

        response = self.http_client.get_json(
            TWSE_COMPANY_PROFILE_URL,
            context="TWSE listed company profiles",
            payload_validator=self._validate_company_profiles,
        )
        payload = response.payload
        if not isinstance(payload, list):
            raise TwseSchemaChangedError(
                "TWSE company profile payload schema changed"
            )
        return {
            str(row.get("公司代號", "")).strip(): str(
                row.get("產業別", "")
            ).strip()
            for row in payload
            if isinstance(row, dict)
            and row.get("公司代號")
            and row.get("產業別")
        }

    @staticmethod
    def _validate_company_profiles(payload: object) -> None:
        if not isinstance(payload, list) or any(
            not isinstance(row, dict) for row in payload
        ):
            raise TwseSchemaChangedError(
                "TWSE company profile payload schema changed"
            )

    def discover_announcements(
        self,
        symbol: str,
        calendar_years: list[int],
    ) -> list[FilingAnnouncement]:
        normalized_symbol = _normalize_symbol(symbol)
        announcements: dict[tuple[int, int], FilingAnnouncement] = {}
        for calendar_year in sorted(set(calendar_years)):
            response, _ = self._request_bytes(
                "POST",
                MOPS_ANNOUNCEMENT_URL,
                data={
                    "step": "1",
                    "firstin": "1",
                    "off": "1",
                    "TYPEK": "all",
                    "co_id": normalized_symbol,
                    "year": str(calendar_year - 1911),
                    "month": "all",
                    "b_date": "",
                    "e_date": "",
                },
                context="MOPS announcements {} {}".format(normalized_symbol, calendar_year),
            )
            for row in _parse_table(response.content).rows:
                announcement = self._parse_announcement_row(normalized_symbol, row)
                if announcement is None:
                    continue
                key = (announcement.fiscal_year, announcement.quarter)
                current = announcements.get(key)
                if current is None or announcement.available_at < current.available_at:
                    announcements[key] = announcement
        return [announcements[key] for key in sorted(announcements)]

    def download(
        self,
        candidate: MopsFilingCandidate,
    ) -> FilingDocument:
        fetched_at = self._now_provider().astimezone(dt.timezone.utc).isoformat()
        response, _ = self._request_bytes(
            "GET",
            candidate.source_url,
            context="MOPS iXBRL {} {}Q{}".format(
                candidate.symbol,
                candidate.fiscal_year,
                candidate.quarter,
            ),
            content_validator=_validate_ixbrl,
        )
        return FilingDocument(
            candidate=candidate,
            content=response.content,
            fetched_at=fetched_at,
        )

    def discover_corrections(
        self,
        periods: list[tuple[int, int]],
    ) -> list[MopsCorrection]:
        """Load each market-wide correction period once per client run."""

        corrections: dict[tuple[str, int, int, str], MopsCorrection] = {}
        for period in sorted(set(periods)):
            for correction in self._load_corrections(period):
                key = (
                    correction.symbol,
                    correction.fiscal_year,
                    correction.quarter,
                    correction.announced_on,
                )
                corrections[key] = correction
        logging.debug(
            "MOPS correction scan periods=%s events=%s scope=ALL_MARKET",
            ",".join("{}Q{}".format(*period) for period in sorted(set(periods))),
            len(corrections),
            extra=structured_extra(
                source="MOPS",
                status="CORRECTION_SCAN_COMPLETE",
                cache_hit=False,
            ),
        )
        return [corrections[key] for key in sorted(corrections)]

    def latest_correction(self, candidate: MopsFilingCandidate) -> MopsCorrection | None:
        key = (candidate.fiscal_year, candidate.quarter)
        corrections = [
            correction
            for correction in self._load_corrections(key)
            if correction.symbol == candidate.symbol
        ]
        if not corrections:
            return None
        return max(corrections, key=lambda correction: correction.announced_on)

    def _load_corrections(
        self,
        key: tuple[int, int],
    ) -> list[MopsCorrection]:
        if key not in self._correction_cache:
            fiscal_year, quarter = key
            response, _ = self._request_bytes(
                "POST",
                MOPS_CORRECTION_URL,
                data={
                    "step": "1",
                    "firstin": "1",
                    "off": "1",
                    "isQuery": "Y",
                    "TYPEK": "sii",
                    "year": str(fiscal_year - 1911),
                    "season": "{:02d}".format(quarter),
                },
                context="MOPS corrections {}Q{}".format(fiscal_year, quarter),
            )
            corrections: list[MopsCorrection] = []
            for row in _parse_table(response.content).rows:
                if len(row.cells) < 5 or not re.fullmatch(r"\d{4}", row.cells[0]):
                    continue
                if not any(
                    scope in row.cells[3] for scope in ("合併", "個別")
                ) or not re.fullmatch(r"\d{8}", row.cells[2]):
                    continue
                corrections.append(
                    MopsCorrection(
                        symbol=row.cells[0],
                        fiscal_year=fiscal_year,
                        quarter=quarter,
                        announced_on=row.cells[2],
                        description=row.cells[4],
                    )
                )
            self._correction_cache[key] = corrections
        return list(self._correction_cache[key])

    def _request_bytes(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """Retry a MOPS request on the alternate official host after network failure."""

        last_error: TwseNetworkError | None = None
        for network_round in range(1, self._network_rounds + 1):
            candidates = self._request_candidates(url)
            for index, (request_url, base_url) in enumerate(candidates):
                request_headers = dict(HTML_HEADERS)
                request_headers["Referer"] = base_url + "/mops/web/index"
                try:
                    self._wait_for_mops_rate_limit()
                    response = self.http_client.request_bytes(
                        method,
                        request_url,
                        headers=request_headers,
                        **kwargs,
                    )
                except TwseNetworkError as error:
                    last_error = error
                    if index + 1 < len(candidates):
                        logging.warning(
                            "MOPS status=NETWORK_ERROR host=%s; "
                            "retrying alternate host=%s.",
                            base_url,
                            candidates[index + 1][1],
                        )
                    continue

                self._last_request_at = self._monotonic()
                if base_url in self._base_urls:
                    self._active_base_url = base_url
                return response, request_url

            if (
                network_round < self._network_rounds
                and last_error is not None
                and last_error.retryable
            ):
                delay = self._network_backoff_seconds * (2 ** (network_round - 1))
                logging.warning(
                    "MOPS status=NETWORK_ERROR all_hosts_unavailable=true "
                    "network_round=%s/%s retrying=true backoff_seconds=%s.",
                    network_round,
                    self._network_rounds,
                    delay,
                )
                self._sleep(delay)
                continue
            break

        assert last_error is not None
        raise last_error

    def _wait_for_mops_rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = self._monotonic() - self._last_request_at
        remaining = self._min_request_interval_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)

    def _reset_mops_session(self) -> None:
        reset_session = getattr(self.http_client, "reset_session", None)
        if callable(reset_session):
            reset_session()
            self._last_request_at = None
            return
        session = getattr(self.http_client, "session", None)
        cookies = getattr(session, "cookies", None)
        if cookies is not None:
            cookies.clear()

    def _request_candidates(self, url: str) -> list[tuple[str, str]]:
        parsed = urlsplit(url)
        known_hosts = {urlsplit(base_url).netloc for base_url in self._base_urls}
        if parsed.netloc and parsed.netloc not in known_hosts:
            return [(url, _origin(url))]

        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        ordered_bases = [self._active_base_url]
        ordered_bases.extend(base_url for base_url in self._base_urls if base_url != self._active_base_url)
        return [(urljoin(base_url + "/", path.lstrip("/")), base_url) for base_url in ordered_bases]

    @staticmethod
    def _parse_announcement_row(symbol: str, row: _HtmlRow) -> FilingAnnouncement | None:
        if len(row.cells) < 5 or row.cells[0] != symbol:
            return None
        title = row.cells[4]
        if "財務報告業經" not in title or "決議通過" not in title:
            return None
        period_match = re.search(
            r"(?:民國)?(?P<year>\d{3})年度(?:(?:第)?(?P<quarter>[一二三四1-4])季|(?P<half>上半年度))?",
            title,
        )
        action = next((value for value in row.actions if "spoke_date" in value), "")
        timestamp_match = re.search(
            r"spoke_time\.value='(?P<time>\d{6})'.*spoke_date\.value='(?P<date>\d{8})'",
            action,
        )
        if period_match is None or timestamp_match is None:
            return None
        quarter_text = period_match.group("quarter")
        quarter_map = {"一": 1, "二": 2, "三": 3, "四": 4}
        if period_match.group("half"):
            quarter = 2
        elif quarter_text in quarter_map:
            quarter = quarter_map[quarter_text]
        elif quarter_text:
            quarter = int(quarter_text)
        else:
            quarter = 4
        available_at = dt.datetime.strptime(
            timestamp_match.group("date") + timestamp_match.group("time"),
            "%Y%m%d%H%M%S",
        ).replace(tzinfo=TAIPEI)
        return FilingAnnouncement(
            symbol=symbol,
            fiscal_year=int(period_match.group("year")) + 1911,
            quarter=quarter,
            available_at=available_at.isoformat(),
            title=title,
        )


def _parse_table(content: bytes) -> _TableParser:
    parser = _TableParser()
    parser.feed(decode_mops_content(content))
    return parser


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _catalog_statement_scope(
    label: str,
) -> tuple[str, str, int] | None:
    normalized = _normalize_text(label)
    if "合併" in normalized:
        return "consolidated", "C", 2
    if "個別" in normalized:
        return "individual", "A", 1
    return None


def _normalize_symbol(symbol: str) -> str:
    normalized = str(symbol).strip()
    if not re.fullmatch(r"[0-9A-Za-z-]{2,12}", normalized):
        raise ValueError("Invalid MOPS stock symbol: {!r}".format(symbol))
    return normalized


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    return "{}://{}".format(parsed.scheme, parsed.netloc)


def _validate_ixbrl(content: bytes) -> None:
    lower_content = content.lower()
    if b"<ix:header" not in lower_content or b"<ix:nonfraction" not in lower_content:
        raise TwseSchemaChangedError("MOPS response is not an Inline XBRL document")


__all__ = [
    "FilingAnnouncement",
    "MOPS_ANNOUNCEMENT_URL",
    "MOPS_BASE_URLS",
    "MOPS_CATALOG_URL",
    "MOPS_CORRECTION_URL",
    "MopsClient",
]

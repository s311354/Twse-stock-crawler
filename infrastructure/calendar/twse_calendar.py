"""Year-aware trading calendar backed by the official TWSE schedule."""

from __future__ import annotations

import datetime
import copy
from collections.abc import Callable
from zoneinfo import ZoneInfo

from domain.trading_calendar import (OfficialAnnualCalendar, TradingDayStatus,
    TwseCalendarError, TwseCalendarUnavailableError, TwseCalendarSchemaError)

from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError


TWSE_CALENDAR_URL = "https://www.twse.com.tw/holidaySchedule/holidaySchedule"
TAIPEI_TIMEZONE = ZoneInfo("Asia/Taipei")
REGULAR_MARKET_OPEN = datetime.time(9, 0)
REGULAR_MARKET_CLOSE = datetime.time(13, 30)


JsonFetcher = Callable[[], object]


class TwseCalendar(OfficialAnnualCalendar):
    """Resolve TWSE trading days from the official open/close schedule.

    `is_trading_day()` returns a bool when the official schedule covers the
    requested year and raises `TwseCalendarError` otherwise. Call
    `trading_day_status()` when a tri-state OPEN/CLOSED/UNKNOWN result is more
    convenient for rendering.
    """

    def __init__(
        self,
        fetcher: JsonFetcher | None = None,
        http_client: TwseHttpClient | None = None,
        *,
        emergency_announcements=None,
        annual_archive=None,
    ) -> None:
        self._fetcher = fetcher
        self._http_client = http_client or TwseHttpClient()
        self._schedule: dict[datetime.date, bool] = {}
        self._covered_years: set[int] = set()
        self._failed_years: dict[int, tuple[type[TwseCalendarError], str]] = {}
        self._annual_evidence: dict[str, dict] = {}
        self._legacy_schedule_loaded = False
        # An injected raw annual payload must remain raw: strict calendar_at()
        # applies its own cutoff-filtered announcements, never today's archive.
        if emergency_announcements is None and fetcher is None:
            from infrastructure.calendar.emergency_announcements import ReviewedEmergencyAnnouncements
            emergency_announcements = ReviewedEmergencyAnnouncements()
        self._emergency_announcements = emergency_announcements
        # Only native current analysis defaults to observed-cache fallback.
        # Raw replay adapters and explicitly injected transports stay isolated.
        if annual_archive is None and fetcher is None and http_client is None:
            from infrastructure.calendar.annual_observations import ObservedAnnualSchedules
            annual_archive = ObservedAnnualSchedules()
        self._annual_archive = annual_archive if annual_archive is not False else None

    def is_trading_day(self, target_date: datetime.date) -> bool:
        """Return whether the date trades, or raise when it cannot be known."""

        if isinstance(target_date, datetime.datetime):
            target_date = target_date.date()
        if not isinstance(target_date, datetime.date):
            raise TypeError("target_date must be a datetime.date")

        if self._emergency_announcements is not None:
            try:
                status = self._emergency_announcements.status(target_date)
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise TwseCalendarSchemaError(f"Emergency calendar evidence invalid: {error}") from error
            if status == "UNKNOWN":
                raise TwseCalendarUnavailableError("Emergency calendar evidence unverified")
            if status is not None:
                return status == "OPEN"

        self._ensure_year_schedule(target_date.year)
        if target_date.year not in self._covered_years:
            raise TwseCalendarUnavailableError(
                f"TWSE calendar does not cover year {target_date.year}"
            )

        scheduled_state = self._schedule.get(target_date)
        if scheduled_state is not None:
            return scheduled_state

        return target_date.weekday() < 5

    def emergency_evidence(self) -> dict | None:
        """Public verified supplement for immutable current-analysis snapshots."""
        return self._emergency_announcements.evidence() if self._emergency_announcements is not None else None

    def annual_failures(self) -> dict[str, str]:
        """Observable source failures for a consumer's preparation diagnostics."""
        return {str(year): message for year, (_, message) in self._failed_years.items()}

    def annual_evidence(self) -> dict[str, dict]:
        """Selected current-calendar sources; no claim of PIT publication proof."""
        return copy.deepcopy(self._annual_evidence)

    def is_market_open_at(self, target_datetime: datetime.datetime) -> bool:
        """Return whether the TWSE regular session is open at the given instant."""

        if not isinstance(target_datetime, datetime.datetime):
            raise TypeError("target_datetime must be a datetime.datetime")

        if target_datetime.tzinfo is None:
            local_datetime = target_datetime.replace(tzinfo=TAIPEI_TIMEZONE)
        else:
            local_datetime = target_datetime.astimezone(TAIPEI_TIMEZONE)

        local_time = local_datetime.time().replace(tzinfo=None)
        if not REGULAR_MARKET_OPEN <= local_time < REGULAR_MARKET_CLOSE:
            return False

        return self.is_trading_day(local_datetime.date())

    def _ensure_year_schedule(self, year: int) -> None:
        if year in self._failed_years:
            kind, message = self._failed_years[year]
            raise kind(message)
        try:
            self._load_year_schedule(year)
        except TwseCalendarError as error:
            # One job must not retry the same failed annual HTTP request for
            # every calendar day. A new instance or explicit retry resets this.
            self._failed_years[year] = (type(error), str(error))
            raise

    def retry_year(self, year: int) -> None:
        """Explicitly retry an unavailable annual schedule; no fabricated dates."""
        self._failed_years.pop(year, None)
        self._ensure_year_schedule(year)

    def _load_year_schedule(self, year: int) -> None:
        if year in self._covered_years:
            return

        if self._fetcher is not None:
            if not self._legacy_schedule_loaded:
                payload = self._fetch_json()
                schedule, covered_years = self._parse_schedule(payload)
                self._schedule.update(schedule)
                self._covered_years.update(covered_years)
                self._legacy_schedule_loaded = True
        else:
            provenance = dict(source="LIVE_OFFICIAL_JSON", source_url=TWSE_CALENDAR_URL)
            try:
                payload = self._fetch_year_json(year)
                provenance["retrieved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            except TwseCalendarUnavailableError as error:
                if self._annual_archive is None:
                    raise
                try:
                    observation = self._annual_archive.get(year)
                except (OSError, ValueError, KeyError, TypeError) as archive_error:
                    raise TwseCalendarUnavailableError(
                        f"{error}; annual archive invalid: {archive_error}"
                    ) from archive_error
                if observation is None:
                    raise
                payload, provenance = observation
                provenance["live_failure"] = str(error)
            schedule, covered_years = self._parse_schedule(payload)
            if year not in covered_years:
                raise TwseCalendarUnavailableError(
                    f"TWSE calendar response does not cover requested year {year}"
                )
            if covered_years != {year}:
                raise TwseCalendarSchemaError("TWSE annual calendar includes unexpected years")
            self._schedule.update(schedule)
            self._covered_years.update(covered_years)
            self._annual_evidence[str(year)] = provenance

    def _fetch_json(self) -> object:
        if self._fetcher is None:
            raise TwseCalendarUnavailableError("TWSE calendar fetcher is not configured")
        try:
            return self._fetcher()
        except Exception as error:
            raise TwseCalendarUnavailableError(
                "TWSE calendar fetcher failed"
            ) from error

    def _fetch_year_json(self, year: int) -> object:
        """Fetch one Gregorian year's official schedule through the shared client."""

        try:
            return self._http_client.get_json(
                TWSE_CALENDAR_URL,
                params={
                    "response": "json",
                    "date": f"{year:04d}0101",
                },
                context=f"TWSE calendar {year}",
            ).payload
        except TwseNetworkError as error:
            raise TwseCalendarUnavailableError(
                f"TWSE calendar API unavailable for {year}: {error}"
            ) from error
        except TwseSchemaChangedError as error:
            raise TwseCalendarSchemaError(
                f"TWSE calendar API schema changed for {year}: {error}"
            ) from error

"""Year-aware trading calendar backed by the official TWSE schedule."""

from __future__ import annotations

import datetime
import enum
from collections.abc import Callable
from zoneinfo import ZoneInfo

from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError


TWSE_CALENDAR_URL = "https://www.twse.com.tw/holidaySchedule/holidaySchedule"
TAIPEI_TIMEZONE = ZoneInfo("Asia/Taipei")
REGULAR_MARKET_OPEN = datetime.time(9, 0)
REGULAR_MARKET_CLOSE = datetime.time(13, 30)


class TradingDayStatus(str, enum.Enum):
    """A tri-state result that never maps API failures to a closed market."""

    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class TwseCalendarError(RuntimeError):
    """Base error for calendar lookup failures."""


class TwseCalendarUnavailableError(TwseCalendarError):
    """Raised when the TWSE calendar cannot be fetched or lacks year coverage."""


class TwseCalendarSchemaError(TwseCalendarError):
    """Raised when the TWSE calendar payload cannot be interpreted safely."""


JsonFetcher = Callable[[], object]


class TwseCalendar:
    """Resolve TWSE trading days from the official open/close schedule.

    `is_trading_day()` returns a bool when the official schedule covers the
    requested year and raises `TwseCalendarError` otherwise. Call
    `trading_day_status()` when a tri-state OPEN/CLOSED/UNKNOWN result is more
    convenient for rendering.
    """

    _OPEN_MARKERS = ("開始交易", "最後交易", "補行交易")
    _CLOSED_MARKERS = ("放假", "補假", "無交易", "休市", "停止交易")

    def __init__(
        self,
        fetcher: JsonFetcher | None = None,
        http_client: TwseHttpClient | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._http_client = http_client or TwseHttpClient()
        self._schedule: dict[datetime.date, bool] = {}
        self._covered_years: set[int] = set()
        self._legacy_schedule_loaded = False

    def is_trading_day(self, target_date: datetime.date) -> bool:
        """Return whether the date trades, or raise when it cannot be known."""

        if isinstance(target_date, datetime.datetime):
            target_date = target_date.date()
        if not isinstance(target_date, datetime.date):
            raise TypeError("target_date must be a datetime.date")

        self._ensure_year_schedule(target_date.year)
        if target_date.year not in self._covered_years:
            raise TwseCalendarUnavailableError(
                f"TWSE calendar does not cover year {target_date.year}"
            )

        scheduled_state = self._schedule.get(target_date)
        if scheduled_state is not None:
            return scheduled_state

        return target_date.weekday() < 5

    def trading_day_status(self, target_date: datetime.date) -> TradingDayStatus:
        """Return OPEN/CLOSED/UNKNOWN without collapsing errors into CLOSED."""

        try:
            return TradingDayStatus.OPEN if self.is_trading_day(target_date) else TradingDayStatus.CLOSED
        except TwseCalendarError:
            return TradingDayStatus.UNKNOWN

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
            payload = self._fetch_year_json(year)
            schedule, covered_years = self._parse_schedule(payload)
            if year not in covered_years:
                raise TwseCalendarUnavailableError(
                    f"TWSE calendar response does not cover requested year {year}"
                )
            self._schedule.update(schedule)
            self._covered_years.update(covered_years)

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

    @classmethod
    def _parse_schedule(
        cls,
        payload: object,
    ) -> tuple[dict[datetime.date, bool], set[int]]:
        if isinstance(payload, list):
            return cls._parse_openapi_schedule(payload)
        if isinstance(payload, dict):
            return cls._parse_historical_schedule(payload)
        raise TwseCalendarSchemaError(
            "TWSE calendar payload must be a list or object"
        )

    @classmethod
    def _parse_openapi_schedule(
        cls,
        payload: list[object],
    ) -> tuple[dict[datetime.date, bool], set[int]]:
        if not payload:
            raise TwseCalendarSchemaError("TWSE calendar payload must be a non-empty list")

        schedule: dict[datetime.date, bool] = {}
        covered_years: set[int] = set()

        for index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise TwseCalendarSchemaError(
                    f"TWSE calendar row {index} must be an object"
                )

            date_value = record.get("Date")
            name = record.get("Name")
            description = record.get("Description", "")
            if not isinstance(date_value, str) or not isinstance(name, str):
                raise TwseCalendarSchemaError(
                    f"TWSE calendar row {index} is missing Date or Name"
                )
            if not isinstance(description, str):
                raise TwseCalendarSchemaError(
                    f"TWSE calendar row {index} has an invalid Description"
                )

            target_date = cls._parse_roc_date(date_value)
            is_open = cls._parse_open_state(name, description)
            previous = schedule.get(target_date)
            if previous is not None and previous != is_open:
                raise TwseCalendarSchemaError(
                    f"TWSE calendar has conflicting states for {target_date.isoformat()}"
                )

            schedule[target_date] = is_open
            covered_years.add(target_date.year)

        return schedule, covered_years

    @classmethod
    def _parse_historical_schedule(
        cls,
        payload: dict[object, object],
    ) -> tuple[dict[datetime.date, bool], set[int]]:
        stat = payload.get("stat")
        if not isinstance(stat, str) or stat.lower() != "ok":
            raise TwseCalendarSchemaError(
                f"TWSE historical calendar returned invalid status: {stat!r}"
            )

        fields = payload.get("fields")
        rows = payload.get("data")
        if not isinstance(fields, list) or not all(
            isinstance(field, str) for field in fields
        ):
            raise TwseCalendarSchemaError(
                "TWSE historical calendar fields must be a string list"
            )
        if not isinstance(rows, list) or not rows:
            raise TwseCalendarSchemaError(
                "TWSE historical calendar data must be a non-empty list"
            )

        required_fields = ("日期", "名稱", "說明")
        missing_fields = [field for field in required_fields if field not in fields]
        if missing_fields:
            raise TwseCalendarSchemaError(
                f"TWSE historical calendar is missing fields: {missing_fields}"
            )
        date_index, name_index, description_index = (
            fields.index(field) for field in required_fields
        )
        minimum_width = max(date_index, name_index, description_index) + 1

        schedule: dict[datetime.date, bool] = {}
        covered_years: set[int] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) < minimum_width:
                raise TwseCalendarSchemaError(
                    f"TWSE historical calendar row {index} has invalid width"
                )

            date_value = row[date_index]
            name = row[name_index]
            description = row[description_index]
            if not all(isinstance(value, str) for value in (date_value, name, description)):
                raise TwseCalendarSchemaError(
                    f"TWSE historical calendar row {index} has invalid values"
                )

            target_date = cls._parse_iso_date(date_value)
            is_open = cls._parse_open_state(name, description)
            previous = schedule.get(target_date)
            if previous is not None and previous != is_open:
                raise TwseCalendarSchemaError(
                    f"TWSE calendar has conflicting states for {target_date.isoformat()}"
                )
            schedule[target_date] = is_open
            covered_years.add(target_date.year)

        return schedule, covered_years

    @staticmethod
    def _parse_roc_date(value: str) -> datetime.date:
        normalized = value.strip()
        if len(normalized) != 7 or not normalized.isdigit():
            raise TwseCalendarSchemaError(f"Invalid TWSE ROC date: {value!r}")

        try:
            return datetime.date(
                int(normalized[:3]) + 1911,
                int(normalized[3:5]),
                int(normalized[5:7]),
            )
        except ValueError as error:
            raise TwseCalendarSchemaError(f"Invalid TWSE ROC date: {value!r}") from error

    @staticmethod
    def _parse_iso_date(value: str) -> datetime.date:
        try:
            return datetime.date.fromisoformat(value.strip())
        except ValueError as error:
            raise TwseCalendarSchemaError(
                f"Invalid TWSE Gregorian date: {value!r}"
            ) from error

    @classmethod
    def _parse_open_state(cls, name: str, description: str) -> bool:
        text = f"{name} {description}"
        if any(marker in text for marker in cls._OPEN_MARKERS):
            return True
        if any(marker in text for marker in cls._CLOSED_MARKERS):
            return False

        raise TwseCalendarSchemaError(
            f"Unrecognized TWSE calendar event: {name!r}"
        )

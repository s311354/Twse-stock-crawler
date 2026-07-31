"""TWSE OpenAPI-backed trading calendar."""

from __future__ import annotations

import datetime
import enum
from collections.abc import Callable
from zoneinfo import ZoneInfo

from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError


TWSE_CALENDAR_URL = "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule"
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
        self._schedule: dict[datetime.date, bool] | None = None
        self._covered_years: set[int] = set()

    def is_trading_day(self, target_date: datetime.date) -> bool:
        """Return whether the date trades, or raise when it cannot be known."""

        if isinstance(target_date, datetime.datetime):
            target_date = target_date.date()
        if not isinstance(target_date, datetime.date):
            raise TypeError("target_date must be a datetime.date")

        self._ensure_schedule()
        if target_date.year not in self._covered_years:
            raise TwseCalendarUnavailableError(
                f"TWSE calendar does not cover year {target_date.year}"
            )

        scheduled_state = self._schedule.get(target_date) if self._schedule is not None else None
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

    def _ensure_schedule(self) -> None:
        if self._schedule is not None:
            return

        payload = self._fetch_json()
        self._schedule, self._covered_years = self._parse_schedule(payload)

    def _fetch_json(self) -> object:
        if self._fetcher is not None:
            try:
                return self._fetcher()
            except Exception as error:
                raise TwseCalendarUnavailableError(
                    "TWSE calendar fetcher failed"
                ) from error

        try:
            return self._http_client.get_json(
                TWSE_CALENDAR_URL,
                context="TWSE calendar",
            ).payload
        except TwseNetworkError as error:
            raise TwseCalendarUnavailableError(
                f"TWSE calendar API unavailable: {error}"
            ) from error
        except TwseSchemaChangedError as error:
            raise TwseCalendarSchemaError(
                f"TWSE calendar API schema changed: {error}"
            ) from error

    @classmethod
    def _parse_schedule(
        cls,
        payload: object,
    ) -> tuple[dict[datetime.date, bool], set[int]]:
        if not isinstance(payload, list) or not payload:
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

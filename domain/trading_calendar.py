"""Pure official annual-schedule interpretation, without transport or storage.

The annual payload is supplied by a source-verifying adapter. Weekday defaults
are used only after a successfully parsed official payload covers that year;
unavailable/malformed evidence remains UNKNOWN, never a closed-market guess.
"""
from __future__ import annotations

import datetime
import enum
import re


ANNUAL_PARSER_VERSION = "OFFICIAL_ANNUAL_CURRENT_SESSION_R2"


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


class OfficialAnnualCalendar:
    """Interpret an already-selected raw official annual schedule.

    This is not an official-source authenticator. The caller remains responsible
    for publication/content/version validation before supplying the payload.
    There is no fetch fallback, observed archive or emergency-announcement store.
    Infrastructure can inherit the identical parser while supplying those ports.
    """

    _OPEN_MARKERS = ("開始交易", "最後交易", "補行交易")
    _CLOSED_MARKERS = ("放假", "補假", "無交易", "休市", "停止交易", "不交易")

    def __init__(self, payload: object) -> None:
        # Match the existing injected-payload adapter's lazy/error semantics:
        # malformed raw evidence becomes UNKNOWN on status lookup, not an
        # exception at construction that bypasses the tri-state boundary.
        self._payload = payload
        self._schedule: dict[datetime.date, bool] = {}
        self._covered_years: set[int] = set()
        self._loaded = False
        self._failure: tuple[type[TwseCalendarError], str] | None = None

    def _ensure_loaded(self) -> None:
        if self._failure is not None:
            kind, message = self._failure
            raise kind(message)
        if not self._loaded:
            try:
                self._schedule, self._covered_years = self._parse_schedule(self._payload)
            except TwseCalendarError as error:
                self._failure = type(error), str(error)
                raise
            self._loaded = True

    def is_trading_day(self, target_date: datetime.date) -> bool:
        """Return a covered-year status; never infer missing year coverage."""
        if isinstance(target_date, datetime.datetime):
            target_date = target_date.date()
        if not isinstance(target_date, datetime.date):
            raise TypeError("target_date must be a datetime.date")
        self._ensure_loaded()
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
            is_open = cls._parse_open_state(name, description, target_date=target_date)
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
            is_open = cls._parse_open_state(name, description, target_date=target_date)
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
    def _parse_open_state(cls, name: str, description: str, *,
                          target_date: datetime.date | None = None) -> bool:
        """Bind dated explanatory text to this row, not another session.

        Some 2021/2022 originals label settlement-only days as "最後交易日".
        Their same-date description explicitly says the market does not trade.
        Conversely, the real 2023 last-trading-day row describes the *next*
        two settlement days too. Neither OPEN-first nor CLOSED-first is safe.
        Only the explicit same-day settlement statement overrides that generic
        label; any other contradictory current-session claims remain unknown.
        """
        def states(text):
            return ({True} if any(m in text for m in cls._OPEN_MARKERS) else set()) | (
                {False} if any(m in text for m in cls._CLOSED_MARKERS) else set())

        current, undated = set(), set()
        same_day_settlement = False
        # Both literal escaped newlines and HTML breaks occur in raw schedules.
        sentences = re.split(r"(?:[。\r\n；;]|\\r\\n|\\r|\\n|<br\s*/?>)+",
                             description, flags=re.IGNORECASE)
        for sentence in sentences:
            mentions = list(re.finditer(r"(?:(\d{3,4})年)?(\d{1,2})月(\d{1,2})日", sentence))
            if not mentions or target_date is None:
                undated.update(states(sentence))
                continue
            dates = []
            for match in mentions:
                year = int(match[1]) if match[1] else target_date.year
                if year < 1911:
                    year += 1911
                try:
                    dates.append(datetime.date(year, int(match[2]), int(match[3])))
                except ValueError as error:
                    raise TwseCalendarSchemaError("Invalid date in TWSE calendar explanation") from error
            applies = target_date in dates
            for index in range(len(dates) - 1):
                connector = sentence[mentions[index].end():mentions[index + 1].start()].strip()
                if connector == "至" and dates[index] <= target_date <= dates[index + 1]:
                    applies = True
            if applies:
                current.update(states(sentence))
                same_day_settlement |= bool(re.search(
                    r"市場無交易[，,]\s*僅辦理結算交割作業", sentence))

        if len(current) > 1:
            raise TwseCalendarSchemaError("Conflicting current-session TWSE calendar explanation")
        label = states(name)
        if (same_day_settlement and current == {False} and label == {True}
                and "最後交易日" in name and not undated.difference({False})):
            return False
        resolved = label | current | undated
        if len(resolved) == 1:
            return next(iter(resolved))
        if len(resolved) > 1:
            raise TwseCalendarSchemaError("Conflicting current-session TWSE calendar states")
        raise TwseCalendarSchemaError(f"Unrecognized TWSE calendar event: {name!r}")

"""Current exchange-bound restriction snapshots, never historical PIT evidence."""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from zoneinfo import ZoneInfo

from domain.trading_restrictions import CURRENT_RESTRICTION_SCOPE, RESTRICTION_ENDPOINTS, TPEX_RESTRICTION_ENDPOINTS
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError


ENDPOINTS = RESTRICTION_ENDPOINTS


def roc_date(value: str) -> dt.date:
    digits = re.sub(r"\D", "", value)
    if len(digits) != 7:
        raise ValueError("INVALID_RESTRICTION_DATE")
    return dt.date(int(digits[:3]) + 1911, int(digits[3:5]), int(digits[5:]))


class TradingRestrictionClient:
    def __init__(self, http_client):
        self.http_client = http_client

    def load(self, symbols, *, session_date: str, observed_at: dt.datetime) -> dict:
        # OpenAPI serves current lists, not historical point-in-time lists.
        if observed_at.tzinfo is None or session_date != observed_at.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat():
            return {}
        today = dt.date.fromisoformat(session_date)
        restricted = set()
        for endpoint in ENDPOINTS:
            rows = self.http_client.get_json(
                "https://openapi.twse.com.tw/v1/" + endpoint,
                context="TWSE trading restrictions",
            ).payload
            if not isinstance(rows, list) or any(not isinstance(row, dict) or not row.get("Code") for row in rows):
                raise TwseSchemaChangedError("Invalid trading restrictions payload", retryable=False)
            for row in rows:
                try:
                    active = True
                    if endpoint.endswith("TWTAWU"):
                        start = roc_date(row["TradingHaltDate"])
                        resume = roc_date(row["TradingResumptionDate"]) if row.get("TradingResumptionDate") else None
                        # Conservatively block a resumption day too; fresh execution
                        # must independently confirm the instrument is tradable.
                        active = start <= today and (resume is None or today <= resume)
                    elif endpoint.endswith("punish"):
                        bounds = re.findall(r"\d{3}/\d{2}/\d{2}", row["DispositionPeriod"])
                        if len(bounds) != 2:
                            raise ValueError("INVALID_DISPOSITION_PERIOD")
                        active = roc_date(bounds[0]) <= today <= roc_date(bounds[1])
                except (KeyError, ValueError) as error:
                    raise TwseSchemaChangedError(str(error), retryable=False) from error
                if active:
                    restricted.add(str(row["Code"]).strip())
        # False is supported by successfully reading ALL official lists.
        return {str(symbol): {"restricted": str(symbol) in restricted, "session_date": session_date,
                              "observation_date": session_date, "scope": CURRENT_RESTRICTION_SCOPE,
                              "source": "TWSE_OPENAPI", "available_at": observed_at.isoformat(),
                              "endpoints": list(ENDPOINTS)} for symbol in symbols}


def _tpex_date(value: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError("INVALID_RESTRICTION_DATE")
    digits = re.sub(r"[/-]", "", value.strip())
    if not digits.isascii() or not digits.isdigit() or len(digits) not in (7, 8):
        raise ValueError("INVALID_RESTRICTION_DATE")
    return (roc_date(digits) if len(digits) == 7 else
            dt.date(int(digits[:4]), int(digits[4:6]), int(digits[6:])))


class TpexTradingRestrictionClient:
    """Require current halt, event history, trading-mode and disposition feeds.

    TPEx history has separate halt/resumption rows. Merely checking today's
    announcements misses a halt that started earlier and remains unresolved.
    """

    def __init__(self, http_client):
        self.http_client = http_client

    def load(self, symbols, *, session_date: str, observed_at: dt.datetime) -> dict:
        if observed_at.tzinfo is None or session_date != observed_at.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat():
            return {}
        today = dt.date.fromisoformat(session_date)
        restricted, events, event_keys = set(), {}, set()
        for endpoint in TPEX_RESTRICTION_ENDPOINTS:
            rows = self.http_client.get_json(
                "https://www.tpex.org.tw/openapi/v1/" + endpoint,
                context="TPEx trading restrictions",
            ).payload
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise TwseSchemaChangedError("Invalid TPEx trading restrictions payload", retryable=False)
            # Unlike an empty history, today's feed must explicitly certify its
            # date; the official empty-day representation is a dated sentinel.
            if endpoint == "tpex_spendi_today" and not rows:
                raise TwseSchemaChangedError("Missing TPEx current halt date", retryable=False)
            for row in rows:
                try:
                    symbol = row["SecuritiesCompanyCode"]
                    if not isinstance(symbol, str):
                        raise ValueError("INVALID_RESTRICTION_SYMBOL")
                    symbol = symbol.strip()
                    if endpoint == "tpex_spendi_today":
                        if _tpex_date(row["Date"]) != today:
                            raise ValueError("STALE_TPEX_CURRENT_HALT_DATE")
                        halt, resume = row["暫停交易"], row["恢復交易"]
                        if not isinstance(halt, str) or not isinstance(resume, str):
                            raise ValueError("INVALID_TPEX_CURRENT_HALT")
                        if not symbol:
                            if len(rows) != 1 or row.get("CompanyName", "").strip() or halt.strip() or resume.strip():
                                raise ValueError("INVALID_TPEX_EMPTY_HALT_SENTINEL")
                            continue
                        if not (halt.strip() or resume.strip()):
                            raise ValueError("MISSING_TPEX_CURRENT_HALT_STATUS")
                        restricted.add(symbol)  # Conservatively include resumption day.
                        continue
                    if not symbol:
                        raise ValueError("MISSING_RESTRICTION_SYMBOL")
                    if endpoint == "tpex_spendi_history":
                        if row["Date"] != str(today.year - 1911):
                            raise ValueError("STALE_TPEX_HALT_HISTORY_YEAR")
                        found = False
                        for field, kind in (("Suspended", "halt"), ("Resumed", "resume")):
                            value = row["DateOf" + field + "Trading"]
                            clock = row["TimeOf" + field + "Trading"]
                            if not isinstance(value, str) or not isinstance(clock, str):
                                raise ValueError("INVALID_TPEX_HALT_EVENT")
                            if not value:
                                if clock:
                                    raise ValueError("MISSING_TPEX_HALT_EVENT_DATE")
                                continue
                            date = _tpex_date(value)
                            if date.year != today.year:
                                raise ValueError("INVALID_TPEX_HALT_EVENT_YEAR")
                            time = dt.datetime.strptime(clock, "%H%M%S").time() if clock else dt.time()
                            event_key = (symbol, date, time)
                            if event_key in event_keys:
                                raise ValueError("DUPLICATE_TPEX_HALT_EVENT")
                            event_keys.add(event_key)
                            found = True
                            if date <= today:
                                events.setdefault(symbol, []).append((date, time, kind))
                        if not found:
                            raise ValueError("MISSING_TPEX_HALT_EVENT")
                    elif endpoint == "tpex_cmode":
                        if _tpex_date(row["Date"]) > today:
                            raise ValueError("FUTURE_TPEX_TRADING_MODE")
                        flags = []
                        for field in ("AlteredTrading", "PeriodicTrading", "ManagedStock", "SuspensionOfTrading"):
                            value = row[field]
                            if not isinstance(value, str):
                                raise ValueError("INVALID_TPEX_TRADING_MODE")
                            flag = unicodedata.normalize("NFKC", value).strip().upper()
                            if flag not in ("", "Y", "N"):
                                raise ValueError("INVALID_TPEX_TRADING_MODE")
                            flags.append(flag == "Y")
                        if any(flags):
                            restricted.add(symbol)
                    else:
                        if _tpex_date(row["Date"]) > today:
                            raise ValueError("FUTURE_TPEX_DISPOSITION_PUBLICATION")
                        bounds = row["DispositionPeriod"].split("~")
                        if len(bounds) != 2:
                            raise ValueError("INVALID_DISPOSITION_PERIOD")
                        start, end = map(_tpex_date, bounds)
                        if start > end:
                            raise ValueError("INVALID_DISPOSITION_PERIOD")
                        if start <= today <= end:
                            restricted.add(symbol)
                except (KeyError, TypeError, ValueError, AttributeError) as error:
                    raise TwseSchemaChangedError(str(error), retryable=False) from error
        for symbol, history in events.items():
            latest_date, _, kind = max(history)
            if latest_date == today or kind == "halt":
                restricted.add(symbol)
        return {str(symbol): {"restricted": str(symbol) in restricted, "market": "TPEX",
                              "session_date": session_date, "observation_date": session_date,
                              "scope": CURRENT_RESTRICTION_SCOPE, "source": "TPEX_OPENAPI",
                              "available_at": observed_at.isoformat(),
                              "endpoints": list(TPEX_RESTRICTION_ENDPOINTS)} for symbol in symbols}

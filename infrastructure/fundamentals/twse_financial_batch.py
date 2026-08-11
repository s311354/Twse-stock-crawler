"""One-shot TWSE financial statement batch used to gate MOPS refreshes."""

from __future__ import annotations

import datetime as dt
from typing import Any

from domain.fundamental_models import FundamentalSnapshot
from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.crawler.twse_http_client import TwseNoDataError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError


TWSE_INCOME_STATEMENT_URL = (
    "https://openapi.twse.com.tw/v1/opendata/t187ap06_L_ci"
)
TWSE_BALANCE_SHEET_URL = (
    "https://openapi.twse.com.tw/v1/opendata/t187ap07_L_ci"
)

_IDENTITY_COLUMNS = frozenset({"年度", "季別", "公司代號", "公司名稱"})
_INCOME_COLUMNS = _IDENTITY_COLUMNS | {
    "營業收入",
    "營業毛利（毛損）",
    "營業利益（損失）",
    "本期淨利（淨損）",
    "淨利（淨損）歸屬於母公司業主",
    "基本每股盈餘（元）",
}
_BALANCE_COLUMNS = _IDENTITY_COLUMNS | {
    "流動資產",
    "流動負債",
    "歸屬於母公司業主之權益合計",
}
_BALANCE_ALIASES = {
    "資產總額": ("資產總額", "資產總計"),
    "負債總額": ("負債總額", "負債總計"),
    "權益總額": ("權益總額", "權益總計"),
}


TwseFinancialSnapshot = FundamentalSnapshot


class TwseFinancialBatchClient:
    """Fetch two all-company endpoints once and join matching periods."""

    def __init__(self, http_client: TwseHttpClient) -> None:
        self.http_client = http_client

    def load(
        self,
        *,
        observed_at: dt.datetime,
    ) -> dict[str, TwseFinancialSnapshot]:
        if observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")

        income_payload = self.http_client.get_json(
            TWSE_INCOME_STATEMENT_URL,
            context="TWSE all-company income statements",
            payload_validator=lambda payload: self._validate_payload(
                payload,
                _INCOME_COLUMNS,
                "income statement",
            ),
        ).payload
        balance_payload = self.http_client.get_json(
            TWSE_BALANCE_SHEET_URL,
            context="TWSE all-company balance sheets",
            payload_validator=lambda payload: self._validate_payload(
                payload,
                _BALANCE_COLUMNS,
                "balance sheet",
                aliases=_BALANCE_ALIASES,
            ),
        ).payload

        income = self._latest_by_symbol(income_payload)
        balance = self._latest_by_symbol(balance_payload)
        observed = observed_at.astimezone(dt.timezone.utc).isoformat()
        snapshots: dict[str, TwseFinancialSnapshot] = {}
        for symbol in sorted(income.keys() & balance.keys()):
            income_period, income_row = income[symbol]
            balance_period, balance_row = balance[symbol]
            if income_period != balance_period:
                continue
            snapshots[symbol] = self._build_snapshot(
                symbol,
                income_period,
                income_row,
                balance_row,
                observed,
            )
        if not snapshots:
            raise TwseNoDataError(
                "TWSE financial batches had no matching company periods",
                retryable=False,
            )
        return snapshots

    @staticmethod
    def _validate_payload(
        payload: object,
        required_columns: frozenset[str],
        label: str,
        *,
        aliases: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        if not isinstance(payload, list):
            raise TwseSchemaChangedError(
                "TWSE {} payload is not a list".format(label),
                retryable=False,
            )
        if not payload:
            raise TwseNoDataError(
                "TWSE {} payload is empty".format(label),
                retryable=True,
            )
        if any(not isinstance(row, dict) for row in payload):
            raise TwseSchemaChangedError(
                "TWSE {} payload contains a non-object row".format(label),
                retryable=False,
            )
        columns = set().union(*(row.keys() for row in payload if isinstance(row, dict)))
        missing = sorted(required_columns - columns)
        for canonical, accepted in (aliases or {}).items():
            if not any(column in columns for column in accepted):
                missing.append("{} ({})".format(canonical, " or ".join(accepted)))
        if missing:
            raise TwseSchemaChangedError(
                "TWSE {} schema is missing columns={}".format(label, missing),
                retryable=False,
            )

    @classmethod
    def _latest_by_symbol(
        cls,
        payload: object,
    ) -> dict[str, tuple[tuple[int, int], dict[str, Any]]]:
        latest: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
        if not isinstance(payload, list):
            return latest
        for row in payload:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("公司代號", "")).strip()
            period = cls._period(row)
            if not symbol or period is None:
                continue
            existing = latest.get(symbol)
            if existing is None or period > existing[0]:
                latest[symbol] = (period, row)
        return latest

    @staticmethod
    def _period(row: dict[str, Any]) -> tuple[int, int] | None:
        try:
            fiscal_year = int(str(row.get("年度", "")).strip())
            quarter = int(str(row.get("季別", "")).strip())
        except ValueError:
            return None
        if fiscal_year < 1911:
            fiscal_year += 1911
        if quarter not in {1, 2, 3, 4}:
            return None
        return fiscal_year, quarter

    @staticmethod
    def _build_snapshot(
        symbol: str,
        period: tuple[int, int],
        income: dict[str, Any],
        balance: dict[str, Any],
        observed_at: str,
    ) -> TwseFinancialSnapshot:
        return TwseFinancialSnapshot(
            symbol=symbol,
            company_name=str(income.get("公司名稱", "")).strip(),
            fiscal_year=period[0],
            quarter=period[1],
            observed_at=observed_at,
            revenue_ytd=_number(income.get("營業收入")),
            gross_profit_ytd=_number(income.get("營業毛利（毛損）")),
            operating_income_ytd=_number(income.get("營業利益（損失）")),
            net_income_ytd=_number(income.get("本期淨利（淨損）")),
            parent_net_income_ytd=_number(
                income.get("淨利（淨損）歸屬於母公司業主")
            ),
            basic_eps_ytd=_number(income.get("基本每股盈餘（元）")),
            current_assets=_number(balance.get("流動資產")),
            total_assets=_number(_first_value(balance, _BALANCE_ALIASES["資產總額"])),
            current_liabilities=_number(balance.get("流動負債")),
            total_liabilities=_number(
                _first_value(balance, _BALANCE_ALIASES["負債總額"])
            ),
            parent_equity=_number(balance.get("歸屬於母公司業主之權益合計")),
            total_equity=_number(_first_value(balance, _BALANCE_ALIASES["權益總額"])),
        )


def _first_value(row: dict[str, Any], columns: tuple[str, ...]) -> object:
    for column in columns:
        if column in row:
            return row[column]
    return None


def _number(value: object) -> float | None:
    text = str(value if value is not None else "").strip().replace(",", "")
    if text in {"", "--", "-", "N/A", "nan"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    try:
        parsed = float(text)
    except ValueError:
        return None
    return -parsed if negative else parsed


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


__all__ = [
    "TWSE_BALANCE_SHEET_URL",
    "TWSE_INCOME_STATEMENT_URL",
    "TwseFinancialBatchClient",
    "TwseFinancialSnapshot",
]

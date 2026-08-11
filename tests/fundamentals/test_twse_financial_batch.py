from __future__ import annotations

import datetime as dt

import pytest

from infrastructure.crawler.twse_http_client import TwseHttpResponse
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.fundamentals.twse_financial_batch import (
    TWSE_BALANCE_SHEET_URL,
)
from infrastructure.fundamentals.twse_financial_batch import TWSE_INCOME_STATEMENT_URL
from infrastructure.fundamentals.twse_financial_batch import TwseFinancialBatchClient


def test_financial_batch_fetches_two_endpoints_and_derives_safe_ratios() -> None:
    http_client = _FakeHttpClient(
        income=[_income_row()],
        balance=[_balance_row()],
    )
    client = TwseFinancialBatchClient(http_client)  # type: ignore[arg-type]

    snapshots = client.load(
        observed_at=dt.datetime(2026, 7, 21, 13, 30, tzinfo=dt.timezone.utc),
    )

    snapshot = snapshots["2330"]
    fields = snapshot.as_snapshot_fields()
    assert http_client.urls == [
        TWSE_INCOME_STATEMENT_URL,
        TWSE_BALANCE_SHEET_URL,
    ]
    assert snapshot.period == (2026, 1)
    assert fields["twse_gross_margin_ytd"] == pytest.approx(0.4)
    assert fields["twse_operating_margin_ytd"] == pytest.approx(0.2)
    assert fields["twse_roa_ytd_end_balance"] == pytest.approx(0.04)
    assert fields["twse_roe_ytd_end_balance"] == pytest.approx(0.1)
    assert fields["twse_debt_ratio"] == pytest.approx(0.4)
    assert fields["twse_current_ratio"] == pytest.approx(2.0)
    assert fields["twse_total_assets"] == 5000
    assert fields["twse_total_liabilities"] == 2000
    assert fields["twse_parent_equity"] == 2000
    assert fields["twse_total_equity"] == 3000


def test_financial_batch_rejects_missing_official_columns() -> None:
    income = _income_row()
    income.pop("營業收入")
    client = TwseFinancialBatchClient(  # type: ignore[arg-type]
        _FakeHttpClient(income=[income], balance=[_balance_row()])
    )

    with pytest.raises(TwseSchemaChangedError, match="營業收入"):
        client.load(
            observed_at=dt.datetime(2026, 7, 21, tzinfo=dt.timezone.utc),
        )


def test_financial_batch_accepts_current_total_column_aliases() -> None:
    balance = _balance_row()
    balance["資產總計"] = balance.pop("資產總額")
    balance["負債總計"] = balance.pop("負債總額")
    balance["權益總計"] = balance.pop("權益總額")
    client = TwseFinancialBatchClient(  # type: ignore[arg-type]
        _FakeHttpClient(income=[_income_row()], balance=[balance])
    )

    snapshot = client.load(
        observed_at=dt.datetime(2026, 8, 9, tzinfo=dt.timezone.utc),
    )["2330"]

    assert snapshot.total_assets == 5000
    assert snapshot.total_liabilities == 2000
    assert snapshot.total_equity == 3000


def test_financial_hash_ignores_observation_time_but_detects_value_change() -> None:
    first_client = TwseFinancialBatchClient(  # type: ignore[arg-type]
        _FakeHttpClient(income=[_income_row()], balance=[_balance_row()])
    )
    second_income = _income_row()
    second_income["營業收入"] = "1,100"
    second_client = TwseFinancialBatchClient(  # type: ignore[arg-type]
        _FakeHttpClient(income=[second_income], balance=[_balance_row()])
    )

    first = first_client.load(
        observed_at=dt.datetime(2026, 7, 21, tzinfo=dt.timezone.utc),
    )["2330"]
    same = first_client.load(
        observed_at=dt.datetime(2026, 7, 22, tzinfo=dt.timezone.utc),
    )["2330"]
    changed = second_client.load(
        observed_at=dt.datetime(2026, 7, 22, tzinfo=dt.timezone.utc),
    )["2330"]

    assert first.content_sha256 == same.content_sha256
    assert first.content_sha256 != changed.content_sha256


class _FakeHttpClient:
    def __init__(self, *, income: list[dict], balance: list[dict]) -> None:
        self.payloads = {
            TWSE_INCOME_STATEMENT_URL: income,
            TWSE_BALANCE_SHEET_URL: balance,
        }
        self.urls: list[str] = []

    def get_json(self, url, *, payload_validator, **kwargs):
        self.urls.append(url)
        payload = self.payloads[url]
        payload_validator(payload)
        return TwseHttpResponse(payload=payload, status_code=200, attempts=1)


def _income_row() -> dict[str, str]:
    return {
        "年度": "115",
        "季別": "1",
        "公司代號": "2330",
        "公司名稱": "台積電",
        "營業收入": "1,000",
        "營業毛利（毛損）": "400",
        "營業利益（損失）": "200",
        "本期淨利（淨損）": "200",
        "淨利（淨損）歸屬於母公司業主": "200",
        "基本每股盈餘（元）": "10.0",
    }


def _balance_row() -> dict[str, str]:
    return {
        "年度": "115",
        "季別": "1",
        "公司代號": "2330",
        "公司名稱": "台積電",
        "流動資產": "400",
        "資產總額": "5,000",
        "流動負債": "200",
        "負債總額": "2,000",
        "歸屬於母公司業主之權益合計": "2,000",
        "權益總額": "3,000",
    }

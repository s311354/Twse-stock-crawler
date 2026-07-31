import pytest

from infrastructure.crawler.twse_http_client import TwseErrorCategory
from infrastructure.crawler.twse_http_client import TwseHttpConfig
from infrastructure.crawler.twse_http_client import TwseHttpResponse
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseNoDataError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from twstockcrawler import TwStockCrawler


class EmptyPayloadResponse:
    ok = True
    status_code = 200

    def json(self) -> dict:
        return {
            "date": "20260515",
            "groups": [],
            "stat": "OK",
            "tables": [{"fields": ["證券代號", "證券名稱"], "data": []}],
        }


class TaiexPayloadResponse:
    ok = True
    status_code = 200

    def json(self) -> dict:
        return {
            "date": "20260709",
            "stat": "OK",
            "tables": [
                {
                    # Deliberately reordered to verify schema-based lookup.
                    "fields": ["漲跌點數", "收盤指數", "指數"],
                    "data": [
                        ["10.0", "50,000.00", "寶島股價指數"],
                        ["379.80", "45,354.61", "發行量加權股價指數"],
                    ],
                }
            ],
        }


class EndpointStubClient:
    def __init__(self, outcomes: list[TwseHttpResponse | Exception]) -> None:
        self.config = TwseHttpConfig(
            timeout_seconds=1,
            max_attempts=1,
            backoff_seconds=0,
            min_request_interval_seconds=0,
        )
        self.outcomes = list(outcomes)
        self.urls: list[str] = []

    def get_json(self, url: str, **kwargs: object) -> TwseHttpResponse:
        self.urls.append(url)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        payload_validator = kwargs.get("payload_validator")
        if callable(payload_validator):
            payload_validator(outcome.payload)
        return outcome


def test_get_stocktype_data_returns_empty_list_when_twse_tables_have_no_rows(monkeypatch) -> None:
    crawler = TwStockCrawler()

    def request_empty(query_params, date_time, payload_validator=None):
        payload = EmptyPayloadResponse().json()
        if payload_validator is not None:
            payload_validator(payload)
        return payload

    monkeypatch.setattr(crawler, "_request_stocktype_data", request_empty)

    assert crawler.get_stocktype_data("20260515", 13) == []


def test_get_taiex_close_uses_one_index_only_request_and_schema_lookup(monkeypatch) -> None:
    crawler = TwStockCrawler()
    calls: list[dict[str, str | int]] = []

    def request_taiex(
        query_params: dict[str, str | int],
        date_time: str,
        payload_validator=None,
    ) -> dict:
        calls.append(query_params)
        assert date_time == "20260709"
        payload = TaiexPayloadResponse().json()
        if payload_validator is not None:
            payload_validator(payload)
        return payload

    monkeypatch.setattr(crawler, "_request_stocktype_data", request_taiex)

    assert crawler.get_taiex_close("20260709") == 45354.61
    assert len(calls) == 1
    assert calls[0]["type"] == "IND"


def test_get_stocktype_data_raises_schema_changed_for_unknown_payload(monkeypatch) -> None:
    crawler = TwStockCrawler()

    def request_unknown(query_params, date_time, payload_validator=None):
        payload = {
            "stat": "OK",
            "tables": [{"fields": ["未知欄位"], "data": [["value"]]}],
        }
        if payload_validator is not None:
            payload_validator(payload)
        return payload

    monkeypatch.setattr(crawler, "_request_stocktype_data", request_unknown)

    with pytest.raises(TwseSchemaChangedError) as error:
        crawler.get_stocktype_data("20260720", 13)

    assert error.value.category is TwseErrorCategory.SCHEMA_CHANGED


def test_market_data_network_error_fails_over_to_official_rwd_endpoint() -> None:
    stock_row = ["2330", "台積電", "1,000", "1", "1", "100", "102", "99", "101"]
    payload = {
        "stat": "OK",
        "tables": [{"fields": ["證券代號", "證券名稱"], "data": [stock_row]}],
    }
    client = EndpointStubClient(
        [
            TwseNetworkError("legacy endpoint returned HTML", retryable=True),
            TwseHttpResponse(payload=payload, status_code=200, attempts=1),
        ]
    )
    crawler = TwStockCrawler(http_client=client)  # type: ignore[arg-type]

    result = crawler.get_stocktype_data("20260402", 13)

    assert result == [stock_row]
    assert client.urls == [crawler.url, crawler.fallback_url]


@pytest.mark.parametrize(
    "data_error",
    [
        TwseSchemaChangedError("unexpected fields"),
        TwseNoDataError("empty stock table", retryable=True),
    ],
    ids=["schema_changed", "no_data"],
)
def test_market_data_data_errors_do_not_fail_over(data_error: Exception) -> None:
    client = EndpointStubClient([data_error])
    crawler = TwStockCrawler(http_client=client)  # type: ignore[arg-type]

    with pytest.raises(type(data_error)):
        crawler._request_stocktype_data(
            query_params={"date": "20260402", "response": "json", "type": 13},
            date_time="20260402",
        )

    assert client.urls == [crawler.url]

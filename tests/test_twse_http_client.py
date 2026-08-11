from __future__ import annotations

import logging

import pytest
from requests import exceptions as requests_exceptions

from infrastructure.crawler.twse_http_client import TwseErrorCategory
from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.crawler.twse_http_client import TwseHttpConfig
from infrastructure.crawler.twse_http_client import TwseNetworkError
from infrastructure.crawler.twse_http_client import TwseNoDataError
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from application.contracts import StockAnalysisRequest
from infrastructure.observability import StructuredRunObserver


class FakeResponse:
    def __init__(
        self,
        payload: object = None,
        status_code: int = 200,
        json_error: Exception | None = None,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error
        self.content = content
        self.headers = headers or {}

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeSession:
    def __init__(self, outcomes: list[FakeResponse | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.headers: dict[str, str] = {}
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        return self.get(url, **kwargs)


def config(**overrides: object) -> TwseHttpConfig:
    values = {
        "timeout_seconds": 5.0,
        "max_attempts": 2,
        "backoff_seconds": 0.5,
        "min_request_interval_seconds": 0.0,
        "allow_insecure_ssl_fallback": False,
    }
    values.update(overrides)
    return TwseHttpConfig(**values)


def test_http_client_reuses_session_and_retries_network_error() -> None:
    session = FakeSession(
        [
            requests_exceptions.ConnectionError("offline"),
            FakeResponse({"stat": "OK"}),
        ]
    )
    sleeps: list[float] = []
    client = TwseHttpClient(
        config=config(),
        session=session,
        sleeper=sleeps.append,
    )

    response = client.get_json("https://example.test", context="test request")

    assert response.payload == {"stat": "OK"}
    assert response.attempts == 2
    assert sleeps == [0.5]
    assert len(session.calls) == 2
    assert session.calls[0]["timeout"] == 5.0
    assert session.headers["Accept"] == "application/json"


def test_http_retry_log_carries_structured_request_context(caplog) -> None:
    observer = StructuredRunObserver("run-http-test")
    observer.start(
        StockAnalysisRequest(
            stocklist=["2330"],
            stocktype="ELEC",
            output_file_names="shirong",
            endbacktrack=3,
        )
    )
    client = TwseHttpClient(
        config=config(),
        session=FakeSession(
            [
                requests_exceptions.ConnectionError("offline"),
                FakeResponse({"stat": "OK"}),
            ]
        ),
        sleeper=lambda _: None,
    )

    with caplog.at_level(logging.DEBUG):
        client.get_json(
            "https://example.test",
            context="MOPS iXBRL 2330 20260720",
        )

    retry = next(
        record
        for record in caplog.records
        if record.getMessage().startswith("TWSE status=")
    )
    assert retry.run_id == "run-http-test"
    assert retry.symbol == "2330"
    assert retry.trade_date == "20260720"
    assert retry.source == "MOPS"
    assert retry.status == "NETWORK_ERROR"
    assert retry.attempt == 1
    assert retry.levelno == logging.DEBUG
    assert retry.cache_hit is None


def test_http_client_enforces_rate_limit_between_requests() -> None:
    session = FakeSession([FakeResponse({}), FakeResponse({})])
    now = [10.0]
    sleeps: list[float] = []

    def sleep_and_advance(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    client = TwseHttpClient(
        config=config(max_attempts=1, min_request_interval_seconds=1.0),
        session=session,
        sleeper=sleep_and_advance,
        monotonic=lambda: now[0],
    )

    client.get_json("https://example.test/first")
    client.get_json("https://example.test/second")

    assert sleeps == [1.0]


def test_http_client_uses_only_targeted_ssl_fallback() -> None:
    session = FakeSession(
        [
            requests_exceptions.SSLError("Missing Subject Key Identifier"),
            FakeResponse({"stat": "OK"}),
        ]
    )
    client = TwseHttpClient(
        config=config(max_attempts=1, allow_insecure_ssl_fallback=True),
        session=session,
    )

    assert client.get_json("https://example.test").payload == {"stat": "OK"}
    assert [call["verify"] for call in session.calls] == [True, False]


def test_http_client_rejects_unrecognized_ssl_error_without_fallback() -> None:
    session = FakeSession([requests_exceptions.SSLError("certificate expired")])
    client = TwseHttpClient(
        config=config(max_attempts=1, allow_insecure_ssl_fallback=True),
        session=session,
    )

    with pytest.raises(TwseNetworkError):
        client.get_json("https://example.test")

    assert [call["verify"] for call in session.calls] == [True]


def test_http_client_classifies_invalid_json_as_schema_changed() -> None:
    client = TwseHttpClient(
        config=config(max_attempts=1),
        session=FakeSession(
            [
                FakeResponse(
                    json_error=ValueError("bad json"),
                    content=b'{"stat":',
                    headers={"Content-Type": "application/json;charset=UTF-8"},
                )
            ]
        ),
    )

    with pytest.raises(TwseSchemaChangedError) as error:
        client.get_json("https://example.test")

    assert error.value.category is TwseErrorCategory.SCHEMA_CHANGED


def test_http_client_retries_transient_html_then_returns_json() -> None:
    session = FakeSession(
        [
            FakeResponse(
                json_error=ValueError("not json"),
                content=b"<!doctype html><title>Service unavailable</title>",
                headers={"Content-Type": "text/html;charset=UTF-8"},
            ),
            FakeResponse({"stat": "OK"}),
        ]
    )
    sleeps: list[float] = []
    client = TwseHttpClient(
        config=config(),
        session=session,
        sleeper=sleeps.append,
    )

    response = client.get_json("https://example.test", context="market data")

    assert response.payload == {"stat": "OK"}
    assert response.attempts == 2
    assert len(session.calls) == 2
    assert sleeps == [0.5]


def test_http_client_classifies_repeated_html_as_network_error() -> None:
    responses = [
        FakeResponse(
            json_error=ValueError("not json"),
            content=b"<html>temporary gateway response</html>",
            headers={"Content-Type": "text/html"},
        )
        for _ in range(2)
    ]
    session = FakeSession(responses)
    client = TwseHttpClient(config=config(), session=session, sleeper=lambda _: None)

    with pytest.raises(TwseNetworkError) as error:
        client.get_json("https://example.test", context="market data")

    assert error.value.category is TwseErrorCategory.NETWORK_ERROR
    assert "non-JSON response" in str(error.value)
    assert len(session.calls) == 2


def test_http_client_retries_malformed_json_before_schema_error() -> None:
    responses = [
        FakeResponse(
            json_error=ValueError("bad json"),
            content=b'{"stat":',
            headers={"Content-Type": "application/json"},
        )
        for _ in range(2)
    ]
    session = FakeSession(responses)
    client = TwseHttpClient(config=config(), session=session, sleeper=lambda _: None)

    with pytest.raises(TwseSchemaChangedError) as error:
        client.get_json("https://example.test", context="market data")

    assert error.value.category is TwseErrorCategory.SCHEMA_CHANGED
    assert "malformed JSON" in str(error.value)
    assert len(session.calls) == 2


def test_http_client_retries_retryable_no_data_validation(caplog) -> None:
    session = FakeSession([FakeResponse({}), FakeResponse({})])
    sleeps: list[float] = []
    client = TwseHttpClient(
        config=config(),
        session=session,
        sleeper=sleeps.append,
    )

    def reject_empty_payload(payload: object) -> None:
        raise TwseNoDataError("empty", retryable=True)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TwseNoDataError) as error:
            client.get_json(
                "https://example.test",
                context="TWSE market data at 20251006",
                payload_validator=reject_empty_payload,
            )

    assert error.value.category is TwseErrorCategory.NO_DATA
    assert len(session.calls) == 2
    assert sleeps == [0.5]
    retries = [
        record
        for record in caplog.records
        if "retrying=true" in record.getMessage()
    ]
    assert len(retries) == 1
    assert retries[0].levelno == logging.DEBUG
    assert retries[0].status == "NO_DATA"


def test_http_client_classifies_http_failure_as_network_error() -> None:
    client = TwseHttpClient(
        config=config(max_attempts=1),
        session=FakeSession([FakeResponse({}, status_code=503)]),
    )

    with pytest.raises(TwseNetworkError) as error:
        client.get_json("https://example.test")

    assert error.value.category is TwseErrorCategory.NETWORK_ERROR


def test_http_client_posts_form_and_returns_raw_bytes() -> None:
    session = FakeSession([FakeResponse(content=b"<html>MOPS</html>")])
    client = TwseHttpClient(config=config(max_attempts=1), session=session)

    response = client.request_bytes(
        "POST",
        "https://example.test/mops",
        data={"co_id": "2330"},
        headers={"Accept": "text/html"},
    )

    assert response.content == b"<html>MOPS</html>"
    assert session.calls[0]["data"] == {"co_id": "2330"}
    assert session.calls[0]["headers"] == {"Accept": "text/html"}

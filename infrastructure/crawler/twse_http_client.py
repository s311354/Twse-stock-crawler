"""Shared HTTP transport and error taxonomy for TWSE endpoints."""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from urllib.parse import urljoin
from collections.abc import Callable, Mapping

import requests
from requests import Response
from requests import exceptions as requests_exceptions
from config.settings import AppSettings
from domain.errors import AccessBlockedError
from domain.errors import DataErrorCategory
from domain.errors import DataSourceError
from domain.errors import MarketClosedError
from domain.errors import NetworkError
from domain.errors import NoDataError
from domain.errors import SchemaChangedError
from infrastructure.observability import structured_extra

TwseErrorCategory = DataErrorCategory
TwseClientError = DataSourceError
TwseMarketClosedError = MarketClosedError
TwseNoDataError = NoDataError
TwseSchemaChangedError = SchemaChangedError
TwseNetworkError = NetworkError
TwseAccessBlockedError = AccessBlockedError


@dataclasses.dataclass(frozen=True)
class TwseHttpResponse:
    """Decoded response plus transport metadata useful for diagnostics."""

    payload: object
    status_code: int
    attempts: int


@dataclasses.dataclass(frozen=True)
class TwseBinaryResponse:
    """Raw response bytes plus transport metadata for HTML/iXBRL endpoints."""

    content: bytes
    status_code: int
    attempts: int


@dataclasses.dataclass(frozen=True)
class TwseHttpConfig:
    timeout_seconds: float = 30.0
    max_attempts: int = 2
    backoff_seconds: float = 1.0
    min_request_interval_seconds: float = 0.2
    allow_insecure_ssl_fallback: bool = True

    @classmethod
    def from_environment(cls) -> "TwseHttpConfig":
        settings = AppSettings.from_env(
            mops_enabled=False,
            validate_paths=False,
        )
        return cls(
            timeout_seconds=settings.http_timeout_seconds,
            max_attempts=settings.http_max_attempts,
            backoff_seconds=settings.http_backoff_seconds,
            min_request_interval_seconds=settings.http_min_interval_seconds,
            allow_insecure_ssl_fallback=(
                settings.allow_insecure_ssl_fallback
            ),
        )

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")
        if self.min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds cannot be negative")


class TwseHttpClient:
    """Rate-limited JSON/HTML client backed by one requests.Session."""

    _SSL_COMPATIBILITY_MARKER = "Missing Subject Key Identifier"
    _logged_insecure_ssl_fallback = False

    def __init__(
        self,
        config: TwseHttpConfig | None = None,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or TwseHttpConfig.from_environment()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Twse-stock-crawler/1.0",
            }
        )
        self._sleep = sleeper
        self._monotonic = monotonic
        self._last_request_at: float | None = None

    def reset_session(self) -> None:
        """Replace a throttled or poisoned keep-alive session in place."""

        headers = dict(self.session.headers)
        self.session.close()
        self.session = requests.Session()
        self.session.headers.update(headers)
        self._last_request_at = None

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        context: str = "TWSE request",
        payload_validator: Callable[[object], None] | None = None,
    ) -> TwseHttpResponse:
        """GET and decode JSON using centralized retry and error handling."""

        last_error: TwseClientError | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self._request(
                    method="GET",
                    url=url,
                    params=params,
                    data=None,
                    headers=None,
                    context=context,
                )
                self._validate_http_status(response=response, context=context)
                try:
                    payload = response.json()
                except (ValueError, requests_exceptions.JSONDecodeError) as error:
                    raise self._classify_json_decode_error(
                        response=response,
                        context=context,
                        retryable=attempt < self.config.max_attempts,
                    ) from error
                if payload_validator is not None:
                    payload_validator(payload)

                return TwseHttpResponse(
                    payload=payload,
                    status_code=response.status_code,
                    attempts=attempt,
                )
            except TwseClientError as error:
                last_error = error
                if not error.retryable or attempt >= self.config.max_attempts:
                    raise
                logging.log(
                    self._retry_log_level(error),
                    "TWSE status=%s context=%s attempt=%s/%s retrying=true detail=%s",
                    error.category.value.upper(),
                    context,
                    attempt,
                    self.config.max_attempts,
                    error,
                    extra=self._request_log_extra(
                        context,
                        status=error.category.value.upper(),
                        attempt=attempt,
                    ),
                )
                self._sleep(self.config.backoff_seconds * (2 ** (attempt - 1)))

        raise TwseNetworkError(f"{context} failed", retryable=False) from last_error

    def request_bytes(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        data: Mapping[str, str | int] | None = None,
        headers: Mapping[str, str] | None = None,
        context: str = "TWSE request",
        content_validator: Callable[[bytes], None] | None = None,
        before_attempt: Callable[[], None] | None = None,
    ) -> TwseBinaryResponse:
        """Request raw bytes with the same retry, SSL, and rate-limit policy."""

        normalized_method = method.upper()
        if normalized_method not in {"GET", "POST"}:
            raise ValueError("method must be GET or POST")

        last_error: TwseClientError | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self._request(
                    method=normalized_method,
                    url=url,
                    params=params,
                    data=data,
                    headers=headers,
                    context=context,
                    before_attempt=before_attempt,
                )
                self._validate_http_status(response=response, context=context)
                content = bytes(response.content)
                if content_validator is not None:
                    content_validator(content)
                return TwseBinaryResponse(
                    content=content,
                    status_code=response.status_code,
                    attempts=attempt,
                )
            except TwseClientError as error:
                last_error = error
                if not error.retryable or attempt >= self.config.max_attempts:
                    raise
                logging.log(
                    self._retry_log_level(error),
                    "TWSE status=%s context=%s attempt=%s/%s retrying=true detail=%s",
                    error.category.value.upper(),
                    context,
                    attempt,
                    self.config.max_attempts,
                    error,
                    extra=self._request_log_extra(
                        context,
                        status=error.category.value.upper(),
                        attempt=attempt,
                    ),
                )
                self._sleep(self.config.backoff_seconds * (2 ** (attempt - 1)))

        raise TwseNetworkError(f"{context} failed", retryable=False) from last_error

    def _request(
        self,
        *,
        method: str,
        url: str,
        params: Mapping[str, str | int] | None,
        data: Mapping[str, str | int] | None,
        headers: Mapping[str, str] | None,
        context: str,
        before_attempt: Callable[[], None] | None = None,
    ) -> Response:
        try:
            return self._session_request(
                method=method,
                url=url,
                params=params,
                data=data,
                headers=headers,
                verify=True,
                before_attempt=before_attempt,
            )
        except requests_exceptions.SSLError as error:
            if not self._should_use_ssl_fallback(error):
                raise TwseNetworkError(
                    f"{context} SSL verification failed",
                    retryable=False,
                ) from error

            if not TwseHttpClient._logged_insecure_ssl_fallback:
                logging.warning(
                    "TWSE SSL verification failed with the known compatibility issue; "
                    "retrying once with verification disabled.",
                    extra=self._request_log_extra(
                        context,
                        status="SSL_FALLBACK",
                        attempt=1,
                    ),
                )
                TwseHttpClient._logged_insecure_ssl_fallback = True
            requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
            try:
                return self._session_request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=headers,
                    verify=False,
                    before_attempt=before_attempt,
                )
            except requests_exceptions.RequestException as fallback_error:
                raise TwseNetworkError(
                    f"{context} failed during SSL compatibility fallback",
                    retryable=True,
                ) from fallback_error
        except requests_exceptions.TooManyRedirects as error:
            # TWSE can poison a long-lived keep-alive session with a
            # self-redirecting 308 response after many historical requests.
            # A fresh session recovers the same bounded request without
            # weakening TLS or changing the requested market dataset.
            self.reset_session()
            raise TwseNetworkError(
                f"{context} exceeded redirects; HTTP session reset",
                retryable=True,
            ) from error
        except requests_exceptions.RequestException as error:
            raise TwseNetworkError(
                f"{context} network request failed: {error}",
                retryable=True,
            ) from error

    def _session_request(
        self,
        *,
        method: str,
        url: str,
        params: Mapping[str, str | int] | None,
        data: Mapping[str, str | int] | None,
        headers: Mapping[str, str] | None,
        verify: bool,
        before_attempt: Callable[[], None] | None = None,
    ) -> Response:
        request = self.session.get if method == "GET" else self.session.post
        kwargs: dict[str, object] = {
            "params": params,
            "timeout": self.config.timeout_seconds,
            "verify": verify,
        }
        if data is not None:
            kwargs["data"] = data
        if headers is not None:
            kwargs["headers"] = headers
        if before_attempt is None:
            self._wait_for_rate_limit()
            self._last_request_at = self._monotonic()
            return request(url, **kwargs)
        # requests' automatic redirects would bypass the per-attempt budget.
        kwargs["allow_redirects"] = False
        for _ in range(6):
            before_attempt()
            self._wait_for_rate_limit()
            self._last_request_at = self._monotonic()
            response = request(url, **kwargs)
            location = getattr(response, "headers", {}).get("Location")
            if response.status_code not in (301, 302, 303, 307, 308) or not location:
                return response
            url = urljoin(url, location)
            kwargs.pop("params", None)
            if response.status_code == 303 or (response.status_code in (301, 302) and method == "POST"):
                method = "GET"
                request = self.session.get
                kwargs.pop("data", None)
        raise requests_exceptions.TooManyRedirects("bounded redirect limit exceeded")

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is None:
            return

        elapsed = self._monotonic() - self._last_request_at
        remaining = self.config.min_request_interval_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)

    @staticmethod
    def _retry_log_level(error: TwseClientError) -> int:
        """Keep all intermediate retry details in debug mode."""

        return logging.DEBUG

    @staticmethod
    def _request_log_extra(
        context: str,
        *,
        status: str,
        attempt: int,
    ) -> dict[str, object]:
        date_match = re.search(r"\b(20\d{6})\b", context)
        symbol_match = re.search(r"\b([0-9]{4}[A-Z]?)\b", context)
        return structured_extra(
            symbol=symbol_match.group(1) if symbol_match else None,
            trade_date=date_match.group(1) if date_match else None,
            source=("MOPS" if context.startswith("MOPS") else "TPEX" if context.startswith("TPEx") else "TWSE"),
            status=status,
            attempt=attempt,
            cache_hit=None,
        )

    def _should_use_ssl_fallback(self, error: requests_exceptions.SSLError) -> bool:
        return (
            self.config.allow_insecure_ssl_fallback
            and self._SSL_COMPATIBILITY_MARKER in str(error)
        )

    @staticmethod
    def _classify_json_decode_error(
        response: Response,
        context: str,
        *,
        retryable: bool,
    ) -> TwseClientError:
        """Separate transient non-JSON gateway pages from malformed JSON."""

        headers = getattr(response, "headers", {})
        content_type = str(headers.get("Content-Type", "")).split(";", 1)[0]
        content = bytes(getattr(response, "content", b"") or b"")
        stripped = content.lstrip()

        if not stripped or stripped[:1] not in {b"{", b"["}:
            preview = " ".join(
                stripped[:120].decode("utf-8", errors="replace").split()
            )
            detail = f" content_type={content_type or 'unknown'}"
            if preview:
                detail += f" body_prefix={preview!r}"
            return TwseNetworkError(
                f"{context} returned a transient non-JSON response;{detail}",
                retryable=True,
            )

        return TwseSchemaChangedError(
            f"{context} returned malformed JSON "
            f"(content_type={content_type or 'unknown'}, bytes={len(content)})",
            retryable=retryable,
        )

    @staticmethod
    def _validate_http_status(response: Response, context: str) -> None:
        if 200 <= response.status_code < 300:
            return

        retryable = response.status_code == 429 or response.status_code >= 500
        raise TwseNetworkError(
            f"{context} returned HTTP {response.status_code}",
            retryable=retryable,
        )

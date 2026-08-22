"""Immutable runtime configuration loaded from environment variables."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from domain.models import PortfolioPosition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHART_PATH = PROJECT_ROOT / "max_profit_analysis_chart.png"
ENV_FILES = (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example")
QUERY_WAIT_SECONDS = 12


class SettingsError(ValueError):
    """Raised when runtime configuration is invalid or unsafe to use."""


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(
        "{} must be one of 1/0, true/false, yes/no, or on/off".format(name)
    )


def _parse_int(
    values: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    try:
        return int(values.get(name, str(default)))
    except ValueError as error:
        raise SettingsError("{} must be an integer".format(name)) from error


def _parse_float(
    values: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    try:
        return float(values.get(name, str(default)))
    except ValueError as error:
        raise SettingsError("{} must be numeric".format(name)) from error


def _parse_optional_float(
    values: Mapping[str, str],
    name: str,
) -> float | None:
    raw = values.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as error:
        raise SettingsError("{} must be numeric".format(name)) from error


def _parse_pae_positions(values: Mapping[str, str]) -> tuple[PortfolioPosition, ...]:
    raw = values.get("TWSE_PAE_HOLDINGS_JSON", "").strip()
    if not raw:
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SettingsError(
            "TWSE_PAE_HOLDINGS_JSON must be a valid JSON array"
        ) from error
    if not isinstance(payload, list):
        raise SettingsError("TWSE_PAE_HOLDINGS_JSON must be a JSON array")

    allowed = {
        "symbol",
        "shares",
        "average_cost",
        "entry_les",
        "entry_date",
        "sector",
        "correlation_cluster",
        "maximum_weight",
    }
    positions: list[PortfolioPosition] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise SettingsError(
                "TWSE_PAE_HOLDINGS_JSON item {} must be an object".format(index)
            )
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise SettingsError(
                "TWSE_PAE_HOLDINGS_JSON item {} has unknown fields: {}".format(
                    index,
                    ", ".join(unknown),
                )
            )
        if "symbol" not in item or "shares" not in item:
            raise SettingsError(
                "TWSE_PAE_HOLDINGS_JSON item {} requires symbol and shares".format(
                    index
                )
            )
        try:
            positions.append(
                PortfolioPosition(
                    symbol=str(item["symbol"]),
                    shares=float(item["shares"]),
                    average_cost=(
                        float(item["average_cost"])
                        if item.get("average_cost") is not None
                        else None
                    ),
                    entry_les=(
                        float(item["entry_les"])
                        if item.get("entry_les") is not None
                        else None
                    ),
                    entry_date=item.get("entry_date"),
                    sector=item.get("sector"),
                    correlation_cluster=item.get("correlation_cluster"),
                    maximum_weight=(
                        float(item["maximum_weight"])
                        if item.get("maximum_weight") is not None
                        else None
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise SettingsError(
                "TWSE_PAE_HOLDINGS_JSON item {} is invalid: {}".format(index, error)
            ) from error
    return tuple(positions)


@dataclass(frozen=True)
class AppSettings:
    """Validated configuration shared by every runtime adapter."""

    http_timeout_seconds: float = 30.0
    http_max_attempts: int = 2
    http_backoff_seconds: float = 1.0
    http_min_interval_seconds: float = 0.2
    allow_insecure_ssl_fallback: bool = True
    market_history_cache_path: Path = DATA_DIR / "market_history.sqlite3"
    run_lock_path: Path = DATA_DIR / "stockdataanalysis.lock"

    mops_cache_dir: Path = DATA_DIR / "mops"
    mops_sqlite_connect_attempts: int = 3
    mops_network_retry_rounds: int = 2
    mops_network_round_backoff_seconds: float = 5.0
    mops_catalog_empty_attempts: int = 2
    mops_catalog_empty_backoff_seconds: float = 3.0
    mops_min_request_interval_seconds: float = 1.0
    mops_history_years: int = 5

    account_equity: float = 1_000_000.0
    pae_positions: tuple[PortfolioPosition, ...] = ()
    pae_current_cash: float | None = None
    pae_persistence_path: Path = DATA_DIR / "portfolio_allocation.sqlite3"

    smtp_email: str = ""
    smtp_password: str = field(default="", repr=False)
    smtp_receivers: tuple[str, ...] = ()
    smtp_cc: str = ""
    smtp_host: str = ""
    smtp_port: int = 587

    runtime_mode: str = "RELEASE"
    log_level: str = "INFO"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        mail_enabled: bool = False,
        mops_enabled: bool = True,
        account_equity: float | None = None,
        mops_history_years: int | None = None,
        runtime_mode: str | None = None,
        validate_paths: bool = True,
    ) -> "AppSettings":
        values = os.environ if environ is None else environ
        receivers = tuple(
            receiver.strip()
            for receiver in values.get("TWSE_SMTP_RECEIVERS", "").split(",")
            if receiver.strip()
        )
        configured_equity = (
            account_equity
            if account_equity is not None
            else _parse_float(
                values,
                "TWSE_ACCOUNT_EQUITY",
                1_000_000.0,
            )
        )
        configured_history_years = (
            mops_history_years
            if mops_history_years is not None
            else _parse_int(values, "MOPS_HISTORY_YEARS", 5)
        )
        settings = cls(
            http_timeout_seconds=_parse_float(
                values,
                "TWSE_HTTP_TIMEOUT_SECONDS",
                30.0,
            ),
            http_max_attempts=_parse_int(
                values,
                "TWSE_HTTP_MAX_ATTEMPTS",
                2,
            ),
            http_backoff_seconds=_parse_float(
                values,
                "TWSE_HTTP_BACKOFF_SECONDS",
                1.0,
            ),
            http_min_interval_seconds=_parse_float(
                values,
                "TWSE_HTTP_MIN_INTERVAL_SECONDS",
                0.2,
            ),
            allow_insecure_ssl_fallback=_parse_bool(
                values.get("TWSE_ALLOW_INSECURE_SSL_FALLBACK", "1"),
                "TWSE_ALLOW_INSECURE_SSL_FALLBACK",
            ),
            market_history_cache_path=Path(
                values.get(
                    "TWSE_MARKET_HISTORY_CACHE_PATH",
                    str(DATA_DIR / "market_history.sqlite3"),
                )
            ).expanduser(),
            run_lock_path=Path(
                values.get(
                    "TWSE_RUN_LOCK_PATH",
                    str(DATA_DIR / "stockdataanalysis.lock"),
                )
            ).expanduser(),
            mops_cache_dir=Path(
                values.get("MOPS_CACHE_DIR", str(DATA_DIR / "mops"))
            ).expanduser(),
            mops_sqlite_connect_attempts=_parse_int(
                values,
                "MOPS_SQLITE_CONNECT_ATTEMPTS",
                3,
            ),
            mops_network_retry_rounds=_parse_int(
                values,
                "MOPS_NETWORK_RETRY_ROUNDS",
                2,
            ),
            mops_network_round_backoff_seconds=_parse_float(
                values,
                "MOPS_NETWORK_ROUND_BACKOFF_SECONDS",
                5.0,
            ),
            mops_catalog_empty_attempts=_parse_int(
                values,
                "MOPS_CATALOG_EMPTY_ATTEMPTS",
                2,
            ),
            mops_catalog_empty_backoff_seconds=_parse_float(
                values,
                "MOPS_CATALOG_EMPTY_BACKOFF_SECONDS",
                3.0,
            ),
            mops_min_request_interval_seconds=_parse_float(
                values,
                "MOPS_MIN_REQUEST_INTERVAL_SECONDS",
                1.0,
            ),
            mops_history_years=configured_history_years,
            account_equity=configured_equity,
            pae_positions=_parse_pae_positions(values),
            pae_current_cash=_parse_optional_float(
                values,
                "TWSE_PAE_CURRENT_CASH",
            ),
            pae_persistence_path=Path(
                values.get(
                    "TWSE_PAE_PERSISTENCE_PATH",
                    str(DATA_DIR / "portfolio_allocation.sqlite3"),
                )
            ).expanduser(),
            smtp_email=values.get("TWSE_SMTP_EMAIL", "").strip(),
            smtp_password=values.get("TWSE_SMTP_PASSWORD", "").strip(),
            smtp_receivers=receivers,
            smtp_cc=values.get("TWSE_SMTP_CC", "").strip(),
            smtp_host=values.get("TWSE_SMTP_HOST", "").strip(),
            smtp_port=_parse_int(values, "TWSE_SMTP_PORT", 587),
            runtime_mode=(
                runtime_mode
                if runtime_mode is not None
                else values.get("TWSE_RUNTIME_MODE", "release")
            ).strip().upper(),
            log_level=values.get("TWSE_LOG_LEVEL", "INFO").strip().upper(),
        )
        settings.validate(
            mail_enabled=mail_enabled,
            mops_enabled=mops_enabled,
            validate_paths=validate_paths,
        )
        return settings

    def validate(
        self,
        *,
        mail_enabled: bool,
        mops_enabled: bool,
        validate_paths: bool = True,
    ) -> None:
        if self.http_timeout_seconds <= 0:
            raise SettingsError("TWSE_HTTP_TIMEOUT_SECONDS must be greater than zero")
        if self.http_max_attempts < 1:
            raise SettingsError("TWSE_HTTP_MAX_ATTEMPTS must be at least one")
        if self.http_backoff_seconds < 0:
            raise SettingsError("TWSE_HTTP_BACKOFF_SECONDS cannot be negative")
        if self.http_min_interval_seconds < 0:
            raise SettingsError(
                "TWSE_HTTP_MIN_INTERVAL_SECONDS cannot be negative"
            )
        if self.mops_sqlite_connect_attempts < 1:
            raise SettingsError("MOPS_SQLITE_CONNECT_ATTEMPTS must be at least one")
        if self.mops_network_retry_rounds < 1:
            raise SettingsError("MOPS_NETWORK_RETRY_ROUNDS must be at least one")
        if self.mops_network_round_backoff_seconds < 0:
            raise SettingsError(
                "MOPS_NETWORK_ROUND_BACKOFF_SECONDS cannot be negative"
            )
        if self.mops_catalog_empty_attempts < 1:
            raise SettingsError("MOPS_CATALOG_EMPTY_ATTEMPTS must be at least one")
        if self.mops_catalog_empty_backoff_seconds < 0:
            raise SettingsError(
                "MOPS_CATALOG_EMPTY_BACKOFF_SECONDS cannot be negative"
            )
        if self.mops_min_request_interval_seconds < 0:
            raise SettingsError(
                "MOPS_MIN_REQUEST_INTERVAL_SECONDS cannot be negative"
            )
        if self.mops_history_years < 3:
            raise SettingsError("MOPS_HISTORY_YEARS must be at least 3")
        if self.account_equity <= 0:
            raise SettingsError("TWSE_ACCOUNT_EQUITY must be greater than zero")
        if len(self.pae_positions) > 4:
            raise SettingsError("PAE holdings support at most 4 positions")
        symbols = [position.symbol for position in self.pae_positions]
        if len(symbols) != len(set(symbols)):
            raise SettingsError("PAE holdings contain duplicate symbols")
        if self.pae_current_cash is not None:
            if self.pae_current_cash < 0:
                raise SettingsError("TWSE_PAE_CURRENT_CASH cannot be negative")
            if self.pae_current_cash > self.account_equity:
                raise SettingsError(
                    "TWSE_PAE_CURRENT_CASH cannot exceed TWSE_ACCOUNT_EQUITY"
                )
        if self.smtp_port < 1 or self.smtp_port > 65535:
            raise SettingsError("TWSE_SMTP_PORT must be between 1 and 65535")
        if self.runtime_mode not in {"RELEASE", "DEBUG"}:
            raise SettingsError("TWSE_RUNTIME_MODE must be release or debug")
        if self.log_level not in {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        }:
            raise SettingsError("TWSE_LOG_LEVEL is invalid")

        if mops_enabled and validate_paths:
            self._validate_mops_cache()
        if validate_paths:
            self._validate_market_history_cache()
            self._validate_pae_persistence()
            self._validate_run_lock()
        if mail_enabled:
            self._validate_smtp()

    def _validate_mops_cache(self) -> None:
        try:
            self.mops_cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=self.mops_cache_dir,
                prefix=".settings-write-test-",
            ):
                pass
        except OSError as error:
            raise SettingsError(
                "MOPS_CACHE_DIR is not writable: {}".format(
                    self.mops_cache_dir
                )
            ) from error

    def _validate_market_history_cache(self) -> None:
        path = self.market_history_cache_path
        try:
            if path.exists() and not path.is_file():
                raise OSError("cache path is not a file")
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=".market-history-write-test-",
            ):
                pass
        except OSError as error:
            raise SettingsError(
                "TWSE_MARKET_HISTORY_CACHE_PATH is not writable: {}".format(
                    path
                )
            ) from error

    def _validate_pae_persistence(self) -> None:
        path = self.pae_persistence_path
        try:
            if path.exists() and not path.is_file():
                raise OSError("persistence path is not a file")
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=".pae-write-test-",
            ):
                pass
        except OSError as error:
            raise SettingsError(
                "TWSE_PAE_PERSISTENCE_PATH is not writable: {}".format(path)
            ) from error

    def _validate_run_lock(self) -> None:
        path = self.run_lock_path
        try:
            if path.exists() and not path.is_file():
                raise OSError("run lock path is not a file")
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=".run-lock-write-test-",
            ):
                pass
        except OSError as error:
            raise SettingsError(
                "TWSE_RUN_LOCK_PATH is not writable: {}".format(path)
            ) from error

    def _validate_smtp(self) -> None:
        if (
            "@" not in self.smtp_email
            or "." not in self.smtp_email.rsplit("@", 1)[-1]
        ):
            raise SettingsError("TWSE_SMTP_EMAIL must be a valid email address")
        if not self.smtp_password:
            raise SettingsError("TWSE_SMTP_PASSWORD is required when mail is enabled")
        if not self.smtp_receivers:
            raise SettingsError(
                "TWSE_SMTP_RECEIVERS is required when mail is enabled"
            )
        if not self.smtp_server_host:
            raise SettingsError(
                "TWSE_SMTP_HOST is required for an unsupported email provider"
            )

    @property
    def smtp_server_host(self) -> str:
        if self.smtp_host:
            return self.smtp_host
        domain = self.smtp_email.rsplit("@", 1)[-1].lower()
        if domain == "gmail.com":
            return "smtp.gmail.com"
        if domain in {"hotmail.com", "outlook.com", "live.com"}:
            return "smtp.live.com"
        return ""

    @property
    def effective_log_level(self) -> str:
        """Make the runtime mode authoritative for diagnostic output."""

        if self.runtime_mode == "DEBUG":
            return "DEBUG"
        return "INFO" if self.log_level == "DEBUG" else self.log_level

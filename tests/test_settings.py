import dataclasses

import pytest

from config.settings import AppSettings
from config.settings import SettingsError
from infrastructure.bootstrap import build_stock_analysis_runtime
from infrastructure.storage.market_history_cache import SQLiteMarketHistoryCache


def test_app_settings_are_immutable_and_redact_smtp_password(tmp_path) -> None:
    settings = AppSettings.from_env(
        {
            "MOPS_CACHE_DIR": str(tmp_path),
            "TWSE_SMTP_PASSWORD": "secret-value",
        }
    )

    assert "secret-value" not in repr(settings)
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.account_equity = 2_000_000


def test_app_settings_validate_and_normalize_all_runtime_groups(tmp_path) -> None:
    settings = AppSettings.from_env(
        {
            "TWSE_HTTP_TIMEOUT_SECONDS": "12.5",
            "TWSE_HTTP_MAX_ATTEMPTS": "4",
            "TWSE_ALLOW_INSECURE_SSL_FALLBACK": "false",
            "TWSE_MARKET_HISTORY_CACHE_PATH": str(
                tmp_path / "market_history.sqlite3"
            ),
            "TWSE_RUN_LOCK_PATH": str(tmp_path / "analysis.lock"),
            "MOPS_CACHE_DIR": str(tmp_path / "mops"),
            "MOPS_SQLITE_CONNECT_ATTEMPTS": "5",
            "MOPS_HISTORY_YEARS": "6",
            "TWSE_ACCOUNT_EQUITY": "2500000",
            "TWSE_PAE_HOLDINGS_JSON": (
                '[{"symbol":"2330","shares":1000,'
                '"average_cost":900,"entry_les":82,'
                '"entry_date":"2026-08-05",'
                '"sector":"24","correlation_cluster":"FOUNDRY"}]'
            ),
            "TWSE_PAE_CURRENT_CASH": "1500000",
            "TWSE_PAE_PERSISTENCE_PATH": str(tmp_path / "pae.sqlite3"),
            "TWSE_SMTP_EMAIL": "investor@gmail.com",
            "TWSE_SMTP_PASSWORD": "app-password",
            "TWSE_SMTP_RECEIVERS": "one@example.com,two@example.com",
            "TWSE_SMTP_PORT": "2525",
            "TWSE_RUNTIME_MODE": "debug",
            "TWSE_LOG_LEVEL": "debug",
        },
        mail_enabled=True,
    )

    assert settings.http_timeout_seconds == 12.5
    assert settings.http_max_attempts == 4
    assert settings.allow_insecure_ssl_fallback is False
    assert settings.market_history_cache_path == (
        tmp_path / "market_history.sqlite3"
    )
    assert settings.run_lock_path == tmp_path / "analysis.lock"
    assert settings.mops_sqlite_connect_attempts == 5
    assert settings.mops_history_years == 6
    assert settings.account_equity == 2_500_000
    assert settings.pae_positions[0].symbol == "2330"
    assert settings.pae_positions[0].entry_les == 82
    assert settings.pae_positions[0].entry_date.isoformat() == "2026-08-05"
    assert settings.pae_current_cash == 1_500_000
    assert settings.pae_persistence_path == tmp_path / "pae.sqlite3"
    assert settings.smtp_receivers == (
        "one@example.com",
        "two@example.com",
    )
    assert settings.smtp_server_host == "smtp.gmail.com"
    assert settings.smtp_port == 2525
    assert settings.runtime_mode == "DEBUG"
    assert settings.log_level == "DEBUG"
    assert settings.effective_log_level == "DEBUG"
    assert settings.mops_cache_dir.is_dir()


def test_release_mode_suppresses_debug_even_with_legacy_debug_level(tmp_path) -> None:
    settings = AppSettings.from_env(
        {
            "MOPS_CACHE_DIR": str(tmp_path),
            "TWSE_RUNTIME_MODE": "release",
            "TWSE_LOG_LEVEL": "debug",
        }
    )

    assert settings.runtime_mode == "RELEASE"
    assert settings.effective_log_level == "INFO"


def test_runtime_mode_argument_overrides_environment(tmp_path) -> None:
    settings = AppSettings.from_env(
        {
            "MOPS_CACHE_DIR": str(tmp_path),
            "TWSE_RUNTIME_MODE": "release",
        },
        runtime_mode="debug",
    )

    assert settings.runtime_mode == "DEBUG"
    assert settings.effective_log_level == "DEBUG"


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"TWSE_HTTP_TIMEOUT_SECONDS": "0"}, "must be greater than zero"),
        ({"TWSE_HTTP_MAX_ATTEMPTS": "0"}, "must be at least one"),
        ({"TWSE_ALLOW_INSECURE_SSL_FALLBACK": "maybe"}, "must be one of"),
        ({"TWSE_ACCOUNT_EQUITY": "-1"}, "must be greater than zero"),
        ({"MOPS_HISTORY_YEARS": "2"}, "must be at least 3"),
        ({"TWSE_RUNTIME_MODE": "verbose"}, "must be release or debug"),
        ({"TWSE_PAE_HOLDINGS_JSON": "{}"}, "must be a JSON array"),
        (
            {
                "TWSE_PAE_HOLDINGS_JSON": (
                    '[{"symbol":"2330","shares":100},'
                    '{"symbol":"2330","shares":200}]'
                )
            },
            "duplicate symbols",
        ),
    ],
)
def test_app_settings_reject_invalid_values(
    values,
    message,
    tmp_path,
) -> None:
    environment = {"MOPS_CACHE_DIR": str(tmp_path)}
    environment.update(values)

    with pytest.raises(SettingsError, match=message):
        AppSettings.from_env(environment)


def test_smtp_is_required_only_when_mail_is_enabled(tmp_path) -> None:
    values = {"MOPS_CACHE_DIR": str(tmp_path)}

    AppSettings.from_env(values, mail_enabled=False)
    with pytest.raises(SettingsError, match="TWSE_SMTP_EMAIL"):
        AppSettings.from_env(values, mail_enabled=True)


def test_mops_cache_path_must_be_writable_when_enabled(tmp_path) -> None:
    invalid_path = tmp_path / "not-a-directory"
    invalid_path.write_text("file")

    with pytest.raises(SettingsError, match="MOPS_CACHE_DIR is not writable"):
        AppSettings.from_env(
            {"MOPS_CACHE_DIR": str(invalid_path)},
            mops_enabled=True,
        )

    AppSettings.from_env(
        {"MOPS_CACHE_DIR": str(invalid_path)},
        mops_enabled=False,
    )


def test_composition_uses_the_validated_http_and_mops_settings(tmp_path) -> None:
    settings = AppSettings.from_env(
        {
            "MOPS_CACHE_DIR": str(tmp_path / "mops"),
            "TWSE_HTTP_TIMEOUT_SECONDS": "7",
            "TWSE_HTTP_MAX_ATTEMPTS": "3",
            "TWSE_ALLOW_INSECURE_SSL_FALLBACK": "0",
            "TWSE_MARKET_HISTORY_CACHE_PATH": str(
                tmp_path / "market_history.sqlite3"
            ),
        }
    )

    runtime = build_stock_analysis_runtime(
        settings=settings,
        run_id_provider=lambda: "run-settings-test",
    )

    assert runtime.http_client.config.timeout_seconds == 7
    assert runtime.http_client.config.max_attempts == 3
    assert runtime.http_client.config.allow_insecure_ssl_fallback is False
    assert (
        runtime.fundamental_service.repository.root
        == settings.mops_cache_dir
    )
    crawler = runtime.crawler_factory(1)
    assert isinstance(crawler.market_history_cache, SQLiteMarketHistoryCache)
    assert (
        crawler.market_history_cache.database_path
        == settings.market_history_cache_path
    )


def test_market_history_cache_path_must_be_writable(tmp_path) -> None:
    invalid_path = tmp_path / "cache-directory"
    invalid_path.mkdir()

    with pytest.raises(
        SettingsError,
        match="TWSE_MARKET_HISTORY_CACHE_PATH is not writable",
    ):
        AppSettings.from_env(
            {
                "MOPS_CACHE_DIR": str(tmp_path / "mops"),
                "TWSE_MARKET_HISTORY_CACHE_PATH": str(invalid_path),
            }
        )

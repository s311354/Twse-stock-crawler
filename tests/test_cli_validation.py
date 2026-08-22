"""Tests for CLI backtrack parameter validation."""

import json

import pytest

import interface.cli as cli
from interface.cli import build_request
from interface.cli import main
from interface.cli import non_negative_int
from infrastructure.run_lock import RunAlreadyActiveError
import argparse


def test_non_negative_int_accepts_zero():
    """Verify that zero is a valid backtrack value."""
    assert non_negative_int("0") == 0


def test_non_negative_int_accepts_positive_integers():
    """Verify that positive integers are accepted."""
    assert non_negative_int("1") == 1
    assert non_negative_int("500") == 500
    assert non_negative_int("999999") == 999999


def test_non_negative_int_rejects_negative_integers():
    """Verify that negative integers are rejected with clear error."""
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        non_negative_int("-1")
    
    assert "must be non-negative" in str(exc_info.value)
    assert "-1" in str(exc_info.value)


def test_non_negative_int_rejects_negative_large_values():
    """Verify that large negative values are rejected."""
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        non_negative_int("-999")
    
    assert "must be non-negative" in str(exc_info.value)
    assert "-999" in str(exc_info.value)


def test_non_negative_int_rejects_non_integers():
    """Verify that non-integer strings are rejected."""
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        non_negative_int("abc")
    
    assert "not a valid integer" in str(exc_info.value)


def test_non_negative_int_rejects_floats():
    """Verify that floating point strings are rejected."""
    with pytest.raises(argparse.ArgumentTypeError) as exc_info:
        non_negative_int("3.14")
    
    assert "not a valid integer" in str(exc_info.value)


def test_cli_rejects_negative_endbacktrack(capsys):
    """Verify CLI rejects negative --endbacktrack at parse time."""
    with pytest.raises(SystemExit) as exc_info:
        main(["-e", "-5", "-b", "0", "2330"])
    
    assert exc_info.value.code == 2  # argparse exits with code 2 for usage errors
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "must be non-negative" in captured.err
    assert "-5" in captured.err


def test_cli_rejects_negative_beginbacktrack(capsys):
    """Verify CLI rejects negative --beginbacktrack at parse time."""
    with pytest.raises(SystemExit) as exc_info:
        main(["-e", "10", "-b", "-3", "2330"])
    
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "must be non-negative" in captured.err
    assert "-3" in captured.err


def test_cli_rejects_reversed_backtrack_order(capsys):
    """Verify CLI rejects when endbacktrack <= beginbacktrack."""
    with pytest.raises(SystemExit) as exc_info:
        main(["-e", "5", "-b", "10", "2330"])
    
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "must be greater than" in captured.err
    assert "Example:" in captured.err  # Helpful usage example


def test_cli_rejects_equal_backtrack_values(capsys):
    """Verify CLI rejects when endbacktrack == beginbacktrack."""
    with pytest.raises(SystemExit) as exc_info:
        main(["-e", "5", "-b", "5", "2330"])
    
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "must be greater than" in captured.err


def test_cli_accepts_valid_backtrack_order():
    """Verify CLI accepts valid backtrack parameters."""
    # Use --list-strategies to avoid needing market data
    exit_code = main(["-e", "10", "-b", "0", "--list-strategies"])
    
    assert exit_code == 0


def test_cli_accepts_large_valid_backtrack_range():
    """Verify CLI accepts large valid backtrack ranges."""
    exit_code = main(["-e", "500", "-b", "0", "--list-strategies"])
    
    assert exit_code == 0


def test_build_request_validates_backtrack_parameters():
    """Verify build_request raises SystemExit for invalid parameters."""
    with pytest.raises(SystemExit) as exc_info:
        build_request(["-e", "5", "-b", "10", "2330"])
    
    assert exc_info.value.code == 2  # argparse usage error


def test_error_message_includes_helpful_example(capsys):
    """Verify error messages include actionable usage examples."""
    with pytest.raises(SystemExit) as exc_info:
        main(["-e", "5", "-b", "10", "2330"])
    
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    # Should include an example of correct usage
    assert "-e 500 -b 0" in captured.err or "Example" in captured.err


def test_validation_happens_before_market_data_fetch(capsys):
    """Verify validation fails fast without attempting market operations."""
    with pytest.raises(SystemExit) as exc_info:
        main(["-e", "-10", "-b", "0", "2330"])
    
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    # Should not see any market-related log messages
    assert "TWSE" not in captured.out or "error:" in captured.err
    # Should fail with usage error, not runtime error
    assert exc_info.value.code == 2


def test_check_config_prints_safe_summary_without_building_service(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    def fail_if_built(**kwargs):
        raise AssertionError("preflight must not build the analysis service")

    monkeypatch.setattr(cli, "build_stock_analysis_service", fail_if_built)
    monkeypatch.setenv(
        "TWSE_MARKET_HISTORY_CACHE_PATH",
        str(tmp_path / "market.sqlite3"),
    )
    monkeypatch.setenv(
        "TWSE_PAE_PERSISTENCE_PATH",
        str(tmp_path / "pae.sqlite3"),
    )

    exit_code = cli.main(
        [
            "--check-config",
            "--skip-mops",
            "-e",
            "30",
            "-b",
            "0",
            "-t",
            "SEMI",
            "2330",
            "2317",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "OK"
    assert summary["stock_type"] == "SEMI"
    assert summary["symbol_count"] == 2
    assert summary["mops_enabled"] is False
    assert summary["mail_enabled"] is False
    assert summary["run_lock"].endswith("stockdataanalysis.lock")
    assert "smtp_password" not in summary


def test_cli_rejects_overlapping_run_before_building_service(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    class BusyLock:
        def __init__(self, path) -> None:
            self.path = path

        def __enter__(self):
            raise RunAlreadyActiveError(
                "Another stock analysis run is active "
                "(pid=1234, host=test, started_at=2026-08-16T00:00:00+00:00)"
            )

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

    def fail_if_built(**kwargs):
        raise AssertionError("busy runs must not build the analysis service")

    monkeypatch.setattr(cli, "ProcessRunLock", BusyLock)
    monkeypatch.setattr(cli, "build_stock_analysis_service", fail_if_built)
    monkeypatch.setenv(
        "TWSE_MARKET_HISTORY_CACHE_PATH",
        str(tmp_path / "market.sqlite3"),
    )
    monkeypatch.setenv(
        "TWSE_PAE_PERSISTENCE_PATH",
        str(tmp_path / "pae.sqlite3"),
    )
    monkeypatch.setenv(
        "TWSE_RUN_LOCK_PATH",
        str(tmp_path / "analysis.lock"),
    )

    exit_code = cli.main(["--skip-mops", "-e", "3", "2330"])

    assert exit_code == 4
    error = capsys.readouterr().err
    assert "TWSE run status: ALREADY_RUNNING" in error
    assert "pid=1234" in error

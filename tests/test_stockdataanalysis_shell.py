from pathlib import Path
import subprocess
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("replay_mail", [False, True])
def test_historical_wrapper_preserves_date_range_and_opt_in_mail(tmp_path: Path, replay_mail: bool) -> None:
    captured = tmp_path / "argv.txt"
    holdings = tmp_path / "historical.txt"
    holdings.write_text("# explicit historical all-cash account\n", encoding="utf-8")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/bin/sh\nif [ "${1:-}" = "-c" ]; then exit 0; fi\n'
        'printf "%s\\n" "$@" > "$CAPTURED_ARGS"\n', encoding="utf-8",
    )
    fake_python.chmod(0o755)
    result = subprocess.run(
        ["./stockdataanalysis.sh", "--profile", "elec", "-b", "2024-10-25",
         "-e", "2026-09-16", "--pae-holdings-file", str(holdings),
         "--pae-current-cash", "1000000", "--account-equity", "1000000", "--check"]
        + (["--replay-mail"] if replay_mail else []),
        cwd=PROJECT_ROOT, env={"PATH": "/usr/bin:/bin", "PYTHON": str(fake_python), "CAPTURED_ARGS": str(captured)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    argv = captured.read_text().splitlines()
    assert argv[argv.index("-b") + 1] == "2024-10-25"
    assert argv[argv.index("-e") + 1] == "2026-09-16"
    assert "--as-of" not in argv
    assert argv[argv.index("--pae-holdings-file") + 1] == str(holdings)
    assert "-m" not in argv and "--check-config" in argv
    assert ("--replay-mail" in argv) is replay_mail


def test_empty_holdings_file_uses_pae_cold_start(tmp_path: Path) -> None:
    captured = tmp_path / "argv.txt"
    holdings = tmp_path / "holdings.txt"
    holdings.write_text("# no current positions\n", encoding="utf-8")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = \"-c\" ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURED_ARGS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["./stockdataanalysis.sh"],
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHON": str(fake_python),
            "CAPTURED_ARGS": str(captured),
            "TWSE_PAE_HOLDINGS_FILE": str(holdings),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "using COLD_START_ALL_CASH" in result.stderr
    arguments = captured.read_text(encoding="utf-8").splitlines()
    assert "--pae-holdings-file" not in arguments
    assert str(PROJECT_ROOT / "stocklist_elec") in arguments
    assert arguments[arguments.index("-t") + 1] == "ELEC"


def test_profile_selects_matching_market_stocklist_and_holdings(
    tmp_path: Path,
) -> None:
    captured = tmp_path / "argv.txt"
    holdings = tmp_path / "semi_holdings.txt"
    holdings.write_text("3105:1000\n", encoding="utf-8")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = \"-c\" ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURED_ARGS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["./stockdataanalysis.sh", "--profile", "semi"],
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHON": str(fake_python),
            "CAPTURED_ARGS": str(captured),
            "TWSE_PAE_HOLDINGS_FILE": str(holdings),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "TPEx analysis profile: SEMI" in result.stdout
    arguments = captured.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("-t") + 1] == "SEMI"
    assert "ALL" in arguments
    assert str(PROJECT_ROOT / "stocklist_semi") not in arguments
    assert arguments[arguments.index("--pae-holdings-file") + 1] == str(
        holdings
    )


def test_environment_profile_selects_matching_market(tmp_path: Path) -> None:
    captured = tmp_path / "argv.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = \"-c\" ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURED_ARGS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["./stockdataanalysis.sh"],
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHON": str(fake_python),
            "CAPTURED_ARGS": str(captured),
            "TWSE_ANALYSIS_PROFILE": "chemi",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    arguments = captured.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("-t") + 1] == "CHEMI"
    assert str(PROJECT_ROOT / "stocklist_chemi") in arguments
    assert "--pae-holdings-file" not in arguments


def test_unknown_profile_fails_before_analysis() -> None:
    result = subprocess.run(
        ["./stockdataanalysis.sh", "--profile", "unknown"],
        cwd=PROJECT_ROOT,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Unsupported analysis profile: unknown" in result.stderr
    assert "Profiles: elec, semi, chemi, bio, comp, air, veh, eleccomp" in result.stderr


def test_check_mode_forwards_preflight_without_changing_profile(
    tmp_path: Path,
) -> None:
    captured = tmp_path / "argv.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = \"-c\" ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURED_ARGS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["./stockdataanalysis.sh", "--check", "--profile", "semi"],
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHON": str(fake_python),
            "CAPTURED_ARGS": str(captured),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    arguments = captured.read_text(encoding="utf-8").splitlines()
    assert "--check-config" in arguments
    assert "-m" in arguments
    assert arguments[arguments.index("-t") + 1] == "SEMI"
    assert "ALL" in arguments
    assert str(PROJECT_ROOT / "stocklist_semi") not in arguments

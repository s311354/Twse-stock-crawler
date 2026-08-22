from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    holdings.write_text("2330:1000\n", encoding="utf-8")
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
    assert "TWSE analysis profile: SEMI" in result.stdout
    arguments = captured.read_text(encoding="utf-8").splitlines()
    assert arguments[arguments.index("-t") + 1] == "SEMI"
    assert str(PROJECT_ROOT / "stocklist_semi") in arguments
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
    assert "Profiles: elec, semi, chemi, bio, comp, air, veh" in result.stderr


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
    assert str(PROJECT_ROOT / "stocklist_semi") in arguments

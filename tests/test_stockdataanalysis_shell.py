from pathlib import Path
import subprocess
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("explicit_end", [None, "700"])
@pytest.mark.parametrize("check", [False, True])
def test_offline_research_wrapper_defaults_to_mail_except_preflight(tmp_path, explicit_end, check):
    captured = tmp_path / "argv.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/sh\nif [ "${1:-}" = "-c" ]; then exit 0; fi\n'
                           'printf "%s\\n" "$@" > "$CAPTURED_ARGS"\n')
    fake_python.chmod(0o755)
    bundle = tmp_path / "native archive" / "frozen_bundle.json"
    command = ["./stockdataanalysis.sh", "--profile", "elec", "--les-v7-research-bundle", str(bundle)]
    if check:
        command += ["--check"]
    if explicit_end:
        command += ["-e", explicit_end]
    result = subprocess.run(command, cwd=PROJECT_ROOT, env={"PATH": "/usr/bin:/bin", "PYTHON": str(fake_python),
        "CAPTURED_ARGS": str(captured)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = captured.read_text().splitlines()
    assert args[args.index("--les-v7-research-bundle") + 1] == str(bundle)
    assert args[args.index("-e") + 1] == (explicit_end or "1100")
    assert ("-m" in args) is not check
    assert "--replay-mail" not in args
    assert ("--check-config" in args) is check


def test_offline_research_wrapper_explicitly_suppresses_default_mail(tmp_path):
    captured = tmp_path / "argv.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/sh\nif [ "${1:-}" = "-c" ]; then exit 0; fi\n'
                           'printf "%s\\n" "$@" > "$CAPTURED_ARGS"\n')
    fake_python.chmod(0o755)
    bundle = tmp_path / "native archive" / "frozen_bundle.json"
    result = subprocess.run(["./stockdataanalysis.sh", "--profile", "elec",
        "--les-v7-research-bundle", str(bundle), "--no-mail"], cwd=PROJECT_ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHON": str(fake_python), "CAPTURED_ARGS": str(captured)},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = captured.read_text().splitlines()
    assert "--no-mail" in args and "-m" not in args


def test_research_wrapper_forwards_fixed_research_id_without_changing_bundle_or_scope(tmp_path):
    captured = tmp_path / "argv.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/sh\nif [ "${1:-}" = "-c" ]; then exit 0; fi\n'
                           'printf "%s\\n" "$@" > "$CAPTURED_ARGS"\n')
    fake_python.chmod(0o755)
    bundle = tmp_path / "SYNTHETIC archive" / "frozen_bundle.json"
    command = ["./stockdataanalysis.sh", "--profile", "elec",
        "--les-v7-research-bundle", str(bundle), "--research-id", "elec-study_01", "--check"]
    result = subprocess.run(command, cwd=PROJECT_ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHON": str(fake_python), "CAPTURED_ARGS": str(captured)},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = captured.read_text().splitlines()
    assert args[args.index("--research-id") + 1] == "elec-study_01"
    assert args[args.index("--les-v7-research-bundle") + 1] == str(bundle)
    assert "--check-config" in args and "-m" not in args
    assert args[args.index("-e") + 1] == "1100"


def test_research_wrapper_rejects_fixed_id_without_bundle_before_python(tmp_path):
    captured = tmp_path / "argv.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/sh\nif [ "${1:-}" = "-c" ]; then exit 0; fi\n'
                           'printf "%s\\n" "$@" > "$CAPTURED_ARGS"\n')
    fake_python.chmod(0o755)
    result = subprocess.run(["./stockdataanalysis.sh", "--profile", "elec",
        "--research-id", "elec-study_01", "--check"], cwd=PROJECT_ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHON": str(fake_python), "CAPTURED_ARGS": str(captured)},
        capture_output=True, text=True)
    assert result.returncode != 0
    assert not captured.exists()


@pytest.mark.parametrize("configured,expected", [(None, "v7"), ("v6", "v6")])
def test_daily_wrapper_selects_v7_with_explicit_v6_compatibility(tmp_path, configured, expected):
    captured = tmp_path / "model.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/sh\nif [ "${1:-}" = "-c" ]; then exit 0; fi\n'
                           'printf "%s" "$TWSE_DAILY_ENTRY_MODEL" > "$CAPTURED_MODEL"\n')
    fake_python.chmod(0o755)
    env = {"PATH": "/usr/bin:/bin", "PYTHON": str(fake_python), "CAPTURED_MODEL": str(captured)}
    if configured is not None:
        env["TWSE_DAILY_ENTRY_MODEL"] = configured
    result = subprocess.run(["./stockdataanalysis.sh", "--check"], cwd=PROJECT_ROOT,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert captured.read_text() == expected


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

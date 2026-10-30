"""Isolated shell-selection tests; fake interpreters cannot analyze or email."""
from pathlib import Path
import shlex
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = ("stockanalysis.sh", "stockdataanalysis.sh")


def sandbox(tmp_path):
    root = tmp_path / "wrapper sandbox with spaces"
    root.mkdir()
    for name in WRAPPERS:
        target = root / name
        target.write_bytes((PROJECT_ROOT / name).read_bytes())
        target.chmod(0o755)
    # Never source the checkout's environment files or use its account data.
    (root / ".env.example").write_text("# isolated test configuration\n", encoding="utf-8")
    (root / "stocklist_elec").write_text("2330\n", encoding="utf-8")
    (root / "stocklist_elec_list").write_text("2330\n", encoding="utf-8")
    (root / "path-bin").mkdir()
    return root


def fake_python(path, label, *, ready=True, version=(3, 15)):
    path.parent.mkdir(parents=True, exist_ok=True)
    quoted_label = shlex.quote(label)
    path.write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = "-c" ]; then\n'
        f'  printf "%s\\n" {quoted_label} >> "$CAPTURED_PROBES"\n'
        f"  if [ {int(ready)} -eq 0 ]; then exit 1; fi\n"
        '  case "${2:-}" in\n'
        '    *"sys.version_info[:2] == (3, 15)"*) '
        f"exit {0 if version == (3, 15) else 1} ;;\n"
        "  esac\n"
        "  exit 0\n"
        "fi\n"
        f'printf "%s" {quoted_label} > "$CAPTURED_SELECTED"\n'
        'printf "%s\\n" "$@" > "$CAPTURED_ARGS"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def run_wrapper(root, wrapper, *, explicit_python=None):
    env = {
        "PATH": f"{root / 'path-bin'}:/usr/bin:/bin",
        "CAPTURED_SELECTED": str(root / "selected.txt"),
        "CAPTURED_ARGS": str(root / "argv.txt"),
        "CAPTURED_PROBES": str(root / "probes.txt"),
    }
    if explicit_python is not None:
        env["PYTHON"] = str(explicit_python)
    argv = [str(root / wrapper)]
    if wrapper == "stockdataanalysis.sh":
        argv.append("--check")
    return subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True)


def assert_selected(root, result, expected, wrapper):
    assert result.returncode == 0, result.stderr
    assert (root / "selected.txt").read_text(encoding="utf-8") == expected
    argv = (root / "argv.txt").read_text(encoding="utf-8").splitlines()
    assert argv[0] == str(root / "main.py")
    assert "-m" in argv  # Preserve the wrappers' existing live-mail argument.
    if wrapper == "stockdataanalysis.sh":
        assert "--check-config" in argv
        assert argv[argv.index("-t") + 1] == "ELEC"
        assert argv[-1] == str(root / "stocklist_elec")
    else:
        assert argv[-1] == str(root / "stocklist_elec_list")


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_ready_python315_is_preferred_over_legacy_and_path(tmp_path, wrapper):
    root = sandbox(tmp_path)
    fake_python(root / ".venv-py315/bin/python", "python315")
    fake_python(root / ".venv/bin/python", "python314", version=(3, 14))
    fake_python(root / "path-bin/python3", "path-python", version=(3, 14))
    result = run_wrapper(root, wrapper)
    assert_selected(root, result, "python315", wrapper)
    assert set((root / "probes.txt").read_text().splitlines()) == {"python315"}


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("ready,version", [(False, (3, 15)), (True, (3, 14))])
def test_broken_or_mismatched_python315_falls_back_to_legacy(tmp_path, wrapper, ready, version):
    root = sandbox(tmp_path)
    fake_python(root / ".venv-py315/bin/python", "python315", ready=ready, version=version)
    fake_python(root / ".venv/bin/python", "python314", version=(3, 14))
    fake_python(root / "path-bin/python3", "path-python", version=(3, 14))
    result = run_wrapper(root, wrapper)
    assert_selected(root, result, "python314", wrapper)
    assert "path-python" not in (root / "probes.txt").read_text().splitlines()


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_legacy_environment_is_preferred_over_path(tmp_path, wrapper):
    root = sandbox(tmp_path)
    fake_python(root / ".venv/bin/python", "python314", version=(3, 14))
    fake_python(root / "path-bin/python3", "path-python", version=(3, 15))
    result = run_wrapper(root, wrapper)
    assert_selected(root, result, "python314", wrapper)
    assert set((root / "probes.txt").read_text().splitlines()) == {"python314"}


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_unready_local_environments_fall_back_to_path(tmp_path, wrapper):
    root = sandbox(tmp_path)
    fake_python(root / ".venv-py315/bin/python", "python315", ready=False)
    fake_python(root / ".venv/bin/python", "python314", ready=False, version=(3, 14))
    fake_python(root / "path-bin/python3", "path-python", version=(3, 14))
    assert_selected(root, run_wrapper(root, wrapper), "path-python", wrapper)


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_explicit_python_precedes_automatic_selection_with_spaces(tmp_path, wrapper):
    root = sandbox(tmp_path)
    explicit = fake_python(root / "chosen interpreter/python with spaces", "explicit", version=(3, 14))
    fake_python(root / ".venv-py315/bin/python", "python315")
    fake_python(root / ".venv/bin/python", "python314", version=(3, 14))
    fake_python(root / "path-bin/python3", "path-python", version=(3, 14))
    result = run_wrapper(root, wrapper, explicit_python=explicit)
    assert_selected(root, result, "explicit", wrapper)
    assert (root / "probes.txt").read_text().splitlines() == ["explicit"]


@pytest.mark.parametrize("wrapper", WRAPPERS)
@pytest.mark.parametrize("missing", [False, True])
def test_failed_explicit_python_does_not_fall_back(tmp_path, wrapper, missing):
    root = sandbox(tmp_path)
    explicit = root / "chosen interpreter/python with spaces"
    if not missing:
        fake_python(explicit, "explicit", ready=False)
    fake_python(root / ".venv-py315/bin/python", "python315")
    fake_python(root / ".venv/bin/python", "python314", version=(3, 14))
    fake_python(root / "path-bin/python3", "path-python", version=(3, 14))
    result = run_wrapper(root, wrapper, explicit_python=explicit)
    assert result.returncode == 1
    assert f"Selected Python interpreter failed its dependency check: {explicit}" in result.stderr
    assert not (root / "selected.txt").exists()
    probes = root / "probes.txt"
    if missing:
        assert not probes.exists()
    else:
        assert probes.read_text().splitlines() == ["explicit"]

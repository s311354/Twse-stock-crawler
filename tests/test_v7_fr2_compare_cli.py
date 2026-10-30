"""SYNTHETIC manual comparison publication tests; no mail or account writes."""
import json
from pathlib import Path

import pytest

from interface.les_v7_fr2_compare import main
from tests.strategies.test_v7_fr2_comparison import comparison


def test_cli_writes_honest_four_arm_comparison_idempotently(tmp_path, capsys):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(comparison()))
    output = tmp_path / "output"
    args = ["--input", str(source), "--output-dir", str(output)]
    assert main(args) == 2
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "NOT_ASSESSABLE"
    folder = Path(first["output"])
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert main(args) == 2
    assert json.loads(capsys.readouterr().out) == first
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    note = (folder / "comparison_support_note.md").read_text()
    assert all(arm in note for arm in ("M0_P0", "M1_P0", "M0_P1", "M1_P1"))
    assert "未變更" in note and "NOT_ASSESSABLE" in note


def test_cli_refuses_to_save_into_original_model_archive(tmp_path):
    archive = tmp_path / "archive" / "frozen_bundle.json"
    archive.parent.mkdir()
    archive.write_text("{}")
    with pytest.raises(SystemExit):
        main(["--archive", str(archive), "--output-dir", str(archive.parent / "new")])
    assert list(archive.parent.iterdir()) == [archive]


def test_cli_never_overwrites_completed_report(tmp_path, capsys):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(comparison()))
    args = ["--input", str(source), "--output-dir", str(tmp_path / "output")]
    assert main(args) == 2
    folder = Path(json.loads(capsys.readouterr().out)["output"])
    target = folder / "comparison.json"
    target.write_text("SYNTHETIC_CORRUPTION")
    with pytest.raises(SystemExit):
        main(args)
    assert target.read_text() == "SYNTHETIC_CORRUPTION"

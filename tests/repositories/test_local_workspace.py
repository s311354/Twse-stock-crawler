"""Synthetic storage boundaries, never official evidence/account writes."""
import hashlib
import sqlite3

import pytest

from application.daily_performance import DailyPerformance
from config.settings import AppSettings, SettingsError
from infrastructure.storage import local_workspace
from infrastructure.storage.evidence_graph import EvidenceGraphReader, EvidenceGraphWriter
from infrastructure.storage.report_evidence import ReportEvidenceReader, ReportEvidenceWriter


def value(label="original"):
    return dict(symbol="TEST", signal_date="2026-10-15", model_version="7.0.0",
                parameter_version="SYNTHETIC_ONLY", proof=[dict(source=label, text="x" * 2000)] * 20)


def test_local_work_commits_closed_verified_store_before_publication(tmp_path):
    target, root = tmp_path / "archive" / "report_evidence.sqlite3", tmp_path / "local"
    probe = DailyPerformance()
    with probe.activate(), ReportEvidenceWriter(target, workspace_root=root) as writer:
        assert writer.path == target
        assert writer.connection.execute("PRAGMA database_list").fetchone()[2].startswith(str(root))
        reference = writer.retain(value())
        assert not target.exists()
    assert reference["store"] == target.name
    assert writer._workspace.receipt["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert probe.counters["evidence_workspace_publications"] == 1
    with ReportEvidenceReader(target.parent) as reader:
        assert reader.resolve(reference) == value()
    with pytest.raises(sqlite3.ProgrammingError):
        writer.connection.execute("SELECT 1")
    assert list(root.iterdir()) == []


def test_seed_reuses_full_old_graph_and_retry_has_identical_references(tmp_path):
    target, root = tmp_path / "archive" / "evidence.sqlite3", tmp_path / "local"
    with EvidenceGraphWriter(target) as writer:
        old = writer.retain(value())
    with EvidenceGraphWriter(target, workspace_root=root) as writer:
        assert writer.retain(value()) == old
        new = writer.retain(value("new"))
    with EvidenceGraphReader(target.parent) as reader:
        assert reader.resolve(old) == value()
        assert reader.resolve(new) == value("new")
    with EvidenceGraphWriter(target, workspace_root=root) as writer:
        assert writer.retain(value("new")) == new
    assert writer.stats["nodes_written"] == 0


@pytest.mark.parametrize("stage", ["write", "verify", "copy", "hash", "replace"])
def test_interruption_never_publishes_partial_or_overwrites_old_data(tmp_path, monkeypatch, stage):
    target, root = tmp_path / "archive" / "evidence.sqlite3", tmp_path / "local"
    with EvidenceGraphWriter(target) as writer:
        old = writer.retain(value())
    before = target.read_bytes()
    original_hash = local_workspace._hash
    def fail(*args, **kwargs):
        raise RuntimeError("interrupted " + stage)
    with pytest.raises(RuntimeError, match="interrupted"):
        with EvidenceGraphWriter(target, workspace_root=root) as writer:
            writer.retain(value("new"))
            if stage == "write":
                fail()
            elif stage == "verify":
                monkeypatch.setattr(local_workspace, "_verify_database", fail)
            elif stage == "copy":
                monkeypatch.setattr(local_workspace.shutil, "copyfileobj", fail)
            elif stage == "hash":
                def broken_hash(path):
                    if path.name.startswith(".evidence-publish-"):
                        fail()
                    return original_hash(path)
                monkeypatch.setattr(local_workspace, "_hash", broken_hash)
            else:
                monkeypatch.setattr(local_workspace.os, "replace", fail)
    assert target.read_bytes() == before
    assert list(root.iterdir()) == []
    assert not list(target.parent.glob(".evidence-publish-*"))
    with EvidenceGraphReader(target.parent) as reader:
        assert reader.resolve(old) == value()


def test_direct_and_staged_writers_share_nonblocking_lock(tmp_path):
    target = tmp_path / "evidence.sqlite3"
    with EvidenceGraphWriter(target) as writer:
        writer.retain(value())
        with pytest.raises(ValueError, match="WRITER_BUSY"):
            with EvidenceGraphWriter(target, workspace_root=tmp_path / "local", workspace_lock_timeout=0):
                pass
    with EvidenceGraphWriter(target, workspace_root=tmp_path / "local"):
        with pytest.raises(ValueError, match="WRITER_BUSY"):
            with EvidenceGraphWriter(target, workspace_lock_timeout=0):
                pass


def test_changed_destination_rejects_publication(tmp_path):
    target = tmp_path / "evidence.sqlite3"
    with EvidenceGraphWriter(target) as writer:
        writer.retain(value())
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        with EvidenceGraphWriter(target, workspace_root=tmp_path / "local") as writer:
            writer.retain(value("new"))
            # Simulate an out-of-contract foreign writer ignoring our lock.
            with sqlite3.connect(target) as foreign:
                foreign.execute("PRAGMA user_version=9")
    with sqlite3.connect(target) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 9


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_open_sqlite_sidecars_are_not_copied(tmp_path, suffix):
    target = tmp_path / "evidence.sqlite3"
    target.with_name(target.name + suffix).write_bytes(b"SYNTHETIC")
    with pytest.raises(ValueError, match="SOURCE_NOT_CLOSED"):
        with EvidenceGraphWriter(target, workspace_root=tmp_path / "local"):
            pass
    assert not target.exists()


def test_corrupt_existing_evidence_is_not_hidden_by_staging(tmp_path):
    target = tmp_path / "evidence.sqlite3"
    with EvidenceGraphWriter(target) as writer:
        reference = writer.retain(value())
    with sqlite3.connect(target) as connection:
        sha = connection.execute("SELECT sha256 FROM report_objects LIMIT 1").fetchone()[0]
        connection.execute("UPDATE report_objects SET body=? WHERE sha256=?", (b"corrupt", sha))
    before = target.read_bytes()
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        with EvidenceGraphWriter(target, workspace_root=tmp_path / "local") as writer:
            writer.retain(value())
    assert target.read_bytes() == before


def test_workspace_settings_default_off_and_bad_configuration(tmp_path):
    assert AppSettings().local_workspace_dir is None
    assert AppSettings.from_env({}, validate_paths=False).local_workspace_dir.is_absolute()
    assert AppSettings.from_env({"TWSE_LOCAL_WORKSPACE_DIR": "off"}, validate_paths=False).local_workspace_dir is None
    assert AppSettings.from_env({"TWSE_LOCAL_WORKSPACE_DIR": str(tmp_path)}, validate_paths=False).local_workspace_dir == tmp_path
    for bad in ("", "relative/path"):
        with pytest.raises(SettingsError, match="TWSE_LOCAL_WORKSPACE_DIR"):
            AppSettings.from_env({"TWSE_LOCAL_WORKSPACE_DIR": bad}, validate_paths=False)


def test_daily_csv_and_full_native_report_are_identical_with_workspace(tmp_path):
    import json
    import pandas as pd
    from infrastructure.storage.csv_repository import CsvRepository
    from infrastructure.storage.report_evidence import load_report_bundle
    from tests.strategies.test_daily_v6_pipeline import build
    builder, decisions = build()
    dataset = builder.project_decisions(decisions)
    before = dataset.copy(deep=True)
    paths = []
    for mode in ("direct", "local"):
        output = tmp_path / mode
        repository = CsvRepository(output, workspace_root=tmp_path / "ssd" if mode == "local" else None)
        repository.write_analysis_dataset("sample", dataset)
        paths.append(output)
    pd.testing.assert_frame_equal(before, dataset, check_exact=True)
    assert (paths[0] / "sample_analysis_dataset.csv").read_bytes() == (paths[1] / "sample_analysis_dataset.csv").read_bytes()
    assert load_report_bundle(paths[0] / "sample_les_v6_reports.json") == load_report_bundle(paths[1] / "sample_les_v6_reports.json")
    assert load_report_bundle(paths[1] / "sample_les_v6_reports.json")["reports"] == [json.loads(decisions[0].value("LES v6 Report"))]


def test_existing_reader_reopens_after_atomic_publication(tmp_path):
    target = tmp_path / "evidence.sqlite3"
    with EvidenceGraphWriter(target) as writer:
        old = writer.retain(value())
    with EvidenceGraphReader(tmp_path) as reader:
        assert reader.resolve(old) == value()
        old_connection = reader._connections[target.name]
        with EvidenceGraphWriter(target, workspace_root=tmp_path / "local") as writer:
            new = writer.retain(value("new"))
        assert reader.resolve(new) == value("new")
        assert reader._connections[target.name] is not old_connection
        assert reader.resolve(old) == value()


def test_warm_reader_does_not_hide_corrupt_atomic_replacement(tmp_path):
    import os
    import shutil
    target = tmp_path / "evidence.sqlite3"
    with EvidenceGraphWriter(target) as writer:
        reference = writer.retain(value())
    with EvidenceGraphReader(tmp_path) as reader:
        assert reader.resolve(reference) == value()
        replacement = tmp_path / "corrupt.sqlite3"
        shutil.copyfile(target, replacement)
        with sqlite3.connect(replacement) as connection:
            connection.execute("UPDATE report_objects SET body=?", (b"corrupt",))
        os.replace(replacement, target)
        with pytest.raises(ValueError, match="HASH_MISMATCH"):
            reader.resolve(reference)

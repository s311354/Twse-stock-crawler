"""Synthetic source/environment fixtures only; no real packages are modified."""
from copy import deepcopy
import pickle
from types import SimpleNamespace

import pytest

from application.replay_contracts import ReplayVersions, digest
from application.work_version import WorkVersionChanged
from infrastructure import research_objects as objects


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "main.py").write_text("# SYNTHETIC source\n")
    (root / "requirements.txt").write_text("# SYNTHETIC dependencies\n")
    (root / "domain").mkdir()
    package_root = tmp_path / "site-packages"
    package_root.mkdir()
    metadata = package_root / "synthetic-1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: synthetic\nVersion: 1\n")
    parameters = {"policy": "SYNTHETIC_V1"}
    runtime = {"gil_enabled": True, "python_version": "SYNTHETIC"}
    deps = [SimpleNamespace(metadata={"Name": "synthetic"}, version="1", _path=metadata)]
    scans = []

    def distributions():
        scans.append(1)
        return list(deps)

    monkeypatch.setattr(objects.runtime, "ROOT", root)
    monkeypatch.setattr(objects.runtime, "parameter_manifest", lambda: deepcopy(parameters))
    monkeypatch.setattr(objects.runtime, "current_versions", lambda: ReplayVersions(
        "SYNTHETIC", digest(parameters), "SYNTHETIC_CODE"))
    monkeypatch.setattr(objects, "distributions", distributions)
    monkeypatch.setattr(objects, "current_runtime_identity", lambda: deepcopy(runtime))
    return SimpleNamespace(root=root, metadata=metadata, package_root=package_root,
                           parameters=parameters, runtime=runtime, deps=deps, scans=scans)


def test_unchanged_checks_do_not_rescan_distributions(fixture):
    guard = objects.ResearchVersionGuard()
    for _ in range(3):
        assert guard.check() == guard.versions
    assert len(fixture.scans) == 1


@pytest.mark.parametrize("change", ["modified", "added", "removed", "same_content_rewrite"])
def test_source_change_is_rejected_with_precise_paths(fixture, change):
    guard = objects.ResearchVersionGuard()
    path = fixture.root / "main.py"
    expected = change
    if change == "added":
        path = fixture.root / "domain" / "added.py"
        path.write_text("# SYNTHETIC new source\n")
    elif change == "removed":
        path.unlink()
    else:
        path.write_text(path.read_text() if change == "same_content_rewrite" else "# changed\n")
        expected = "modified"
    with pytest.raises(WorkVersionChanged, match="REPLAY_VERSION_CHANGED_DURING_JOB") as caught:
        guard.check()
    delta = caught.value.diagnostics["changes"]["source"][expected]
    assert delta == {"count": 1, "paths": [str(path.relative_to(fixture.root))]}
    assert caught.value.diagnostics["action"] == "RESTART_WITH_NEW_JOB"


def test_edit_during_initial_code_hash_scan_is_rejected(fixture, monkeypatch):
    original = objects.runtime.current_versions

    def versions():
        result = original()
        (fixture.root / "main.py").write_text("# edit during version scan\n")
        return result

    monkeypatch.setattr(objects.runtime, "current_versions", versions)
    with pytest.raises(WorkVersionChanged) as caught:
        objects.ResearchVersionGuard()
    assert "source" in caught.value.diagnostics["changes"]


def test_parameter_change_reports_keys_and_hashes_not_values(fixture):
    guard = objects.ResearchVersionGuard()
    fixture.parameters["policy"] = "SYNTHETIC_PRIVATE_VALUE_DO_NOT_LOG"
    with pytest.raises(WorkVersionChanged) as caught:
        guard.check()
    delta = caught.value.diagnostics["changes"]["parameters"]
    assert delta == {"keys": ["policy"], "expected": guard.versions.parameters,
                     "actual": digest(fixture.parameters)}
    assert "SYNTHETIC_PRIVATE_VALUE_DO_NOT_LOG" not in str(caught.value)


@pytest.mark.parametrize("change", ["modified", "removed", "nested_added"])
def test_package_metadata_drift_is_rejected_not_masked_by_missing_file(fixture, change):
    nested = fixture.metadata / "nested"
    nested.mkdir()
    guard = objects.ResearchVersionGuard()
    path = fixture.metadata / "METADATA"
    if change == "modified":
        path.write_text("Name: synthetic\nVersion: 2\n")
    elif change == "removed":
        path.unlink()
    else:
        (nested / "NEW").write_text("SYNTHETIC new metadata\n")
    with pytest.raises(WorkVersionChanged) as caught:
        guard.check()
    assert "environment_metadata" in caught.value.diagnostics["changes"]


def test_parent_timestamp_alone_checks_inventory_but_does_not_invalidate(fixture):
    guard = objects.ResearchVersionGuard()
    (fixture.package_root / "__pycache__").mkdir()
    assert guard.check() == guard.versions
    assert len(fixture.scans) == 2
    guard.check()
    assert len(fixture.scans) == 2


@pytest.mark.parametrize("name", ["new_module.py", "new_namespace", "new.pth"])
def test_parent_check_cannot_hide_new_importable_files_or_import_configuration(fixture, name):
    guard = objects.ResearchVersionGuard()
    path = fixture.package_root / name
    if name == "new_namespace":
        path.mkdir()
    else:
        path.write_text("# SYNTHETIC import change\n")
    with pytest.raises(WorkVersionChanged) as caught:
        guard.check()
    assert caught.value.diagnostics["changes"]["environment_inventory"]["added"]["paths"] == [str(path)]


def test_existing_import_configuration_edit_is_rejected_without_parent_timestamp_change(fixture):
    path = fixture.package_root / "synthetic.pth"
    path.write_text("# SYNTHETIC original import configuration\n")
    guard = objects.ResearchVersionGuard()
    path.write_text("# SYNTHETIC changed import configuration\n")
    with pytest.raises(WorkVersionChanged) as caught:
        guard.check()
    assert "environment_metadata" in caught.value.diagnostics["changes"]


def test_parent_check_still_rejects_new_distribution(fixture):
    guard = objects.ResearchVersionGuard()
    path = fixture.package_root / "new-2.dist-info"
    path.mkdir()
    fixture.deps.append(SimpleNamespace(metadata={"Name": "new"}, version="2", _path=path))
    with pytest.raises(WorkVersionChanged) as caught:
        guard.check()
    assert caught.value.diagnostics["changes"]["distributions"] == {"added": ["new"], "removed": []}


def test_runtime_drift_has_separate_diagnostics(fixture):
    guard = objects.ResearchVersionGuard()
    fixture.runtime["gil_enabled"] = False
    with pytest.raises(WorkVersionChanged) as caught:
        guard.check()
    assert caught.value.diagnostics["changes"] == {"runtime": ["gil_enabled"]}


def test_changed_path_list_is_bounded_but_count_is_complete(fixture):
    guard = objects.ResearchVersionGuard()
    for i in range(25):
        (fixture.root / "domain" / f"new_{i:02}.py").write_text("# SYNTHETIC\n")
    with pytest.raises(WorkVersionChanged) as caught:
        guard.check()
    delta = caught.value.diagnostics["changes"]["source"]["added"]
    assert delta["count"] == 25 and len(delta["paths"]) == 20


def test_exception_roundtrip_preserves_worker_diagnostics_and_legacy_error_code():
    original = WorkVersionChanged({"runtime": ["gil_enabled"]})
    original.diagnostics["checkpoint"] = "before_report"
    original.add_note("Version checkpoint: before_report")
    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored, ValueError)
    assert str(restored) == str(original)
    assert restored.diagnostics == original.diagnostics
    assert restored.__notes__ == original.__notes__

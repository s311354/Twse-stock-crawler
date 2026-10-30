"""SYNTHETIC manifest/binding tests; no trained archive or live account used."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from application.replay_contracts import ReplayVersions, digest
from infrastructure import research_series as series
from interface import cli


@pytest.fixture
def study(monkeypatch, tmp_path):
    version = ReplayVersions("SYNTHETIC_MODELS", "a" * 64, "b" * 64)
    monkeypatch.setattr(series, "current_versions", lambda: version)
    metadata = dict(research_id="synthetic-study", bundle_sha256="c" * 64,
        dataset_id="d" * 64, config_id="e" * 64, native_job_id="f" * 64,
        symbols=["3014", "6834"], admission_component={"version": "SYNTHETIC"},
        execution_component={"version": "SYNTHETIC"}, valuation_policy="SYNTHETIC_EPS",
        decision_thresholds={"buy": 75, "buy_small": 65})
    request = SimpleNamespace(stocklist=["3014", "6834"], market_source="TWSE",
        stocktype=SimpleNamespace(name="ELEC"), beginbacktrack=0, endbacktrack=1100,
        mops_enabled=True, mops_history_years=5)
    settings = SimpleNamespace(les_v7_native_evidence=True, les_v7_source_repair=True,
        les_v7_strategy_policy_json=None)
    identity = series.make_research_identity(metadata, request, settings)
    metadata.update(research_identity=identity, research_identity_id=digest(identity))
    return tmp_path / "les-v7-research" / "synthetic-study", identity, metadata, version


def test_manifest_publication_is_immutable_and_read_only_preflight_creates_nothing(study, monkeypatch):
    root, identity, _, _ = study
    assert series.check_study(root, identity) is None
    assert not root.exists()
    first = series.ensure_study(root, identity)
    path = root / "study.json"
    original = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(series, "utc_now", lambda: "2099-01-01T00:00:00Z")
    assert series.ensure_study(root, deepcopy(identity)) == first
    assert series.check_study(root, identity) == first
    assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    assert not (root / "portfolio.sqlite3").exists()
    assert first["identity_id"] == digest(identity)


@pytest.mark.parametrize("content", [b"{", b"[]", b"null", b'{"schema":"WRONG"}'])
def test_corrupt_manifest_is_never_replaced(study, content):
    root, identity, _, _ = study
    root.mkdir(parents=True)
    path = root / "study.json"
    path.write_bytes(content)
    with pytest.raises(series.ResearchSeriesError):
        series.ensure_study(root, identity)
    assert path.read_bytes() == content


def test_manifest_hash_and_expected_identity_both_checked(study):
    root, identity, _, _ = study
    series.ensure_study(root, identity)
    path = root / "study.json"
    record = json.loads(path.read_bytes())
    record["identity"]["symbols"].reverse()
    path.write_text(json.dumps(record))
    with pytest.raises(series.ResearchSeriesError, match="CORRUPT"):
        series.check_study(root, identity)
    record["identity_id"] = digest(record["identity"])
    path.write_text(json.dumps(record))
    with pytest.raises(series.ResearchSeriesError, match="IDENTITY_CHANGED"):
        series.check_study(root, identity)


@pytest.mark.parametrize("orphan", ["portfolio.sqlite3", "request.json", "completion.json", "runs/orphan/request.json"])
def test_missing_manifest_cannot_adopt_previous_state_or_attempts(study, orphan):
    root, identity, _, _ = study
    path = root / orphan
    path.parent.mkdir(parents=True)
    path.write_bytes(b"SYNTHETIC_EXISTING_KEEP_UNCHANGED")
    with pytest.raises(series.ResearchSeriesError, match="LEGACY_STATE_REQUIRES_NEW_ID"):
        series.ensure_study(root, identity)
    assert path.read_bytes() == b"SYNTHETIC_EXISTING_KEEP_UNCHANGED"
    assert not (root / "study.json").exists()


def test_version_drift_rejects_before_manifest_publication(study, monkeypatch):
    root, identity, _, version = study
    monkeypatch.setattr(series, "current_versions", lambda: replace(version, code="0" * 64))
    with pytest.raises(series.ResearchSeriesError, match="VERSIONS_CHANGED"):
        series.ensure_study(root, identity)
    assert not root.exists()


@pytest.mark.parametrize("field", ["bundle_sha256", "dataset_id", "config_id", "native_job_id", "symbols",
    "admission_component", "execution_component", "valuation_policy", "decision_thresholds"])
def test_enclosing_metadata_cannot_swap_identity_bound_inputs(study, field):
    _, identity, metadata, version = study
    assert series.metadata_identity(metadata) == identity
    assert identity["versions"] == asdict(version)
    changed = deepcopy(metadata)
    changed[field] = ["2330"] if field == "symbols" else "SYNTHETIC_OTHER"
    with pytest.raises(series.ResearchSeriesError, match="METADATA_IDENTITY_MISMATCH"):
        series.metadata_identity(changed)


def test_research_id_not_silently_ignored_by_historical_replay(monkeypatch):
    import interface.replay
    def unexpected(*_):
        raise AssertionError("named current-data study reached historical replay")
    monkeypatch.setattr(interface.replay, "run_from_args", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.main(["-b", "2024-10-25", "-e", "2026-09-16", "--research-id", "synthetic-study", "1"])
    assert error.value.code == 2

"""Document-reference integrity and legacy parity, using isolated synthetic data."""
import base64
from copy import deepcopy
from dataclasses import replace
import datetime as dt
import json
import zipfile

import pytest

from application.replay_contracts import canonical_json, digest
from application.research_evidence import materialize_filings, verified_document, verify_publication
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.replay_evidence import collect_evidence, load_evidence, save_evidence
from infrastructure.replay_runtime import build_replay_engine, current_versions
from infrastructure.research_archive import create_dataset, write_once
from infrastructure.research_documents import EvidenceDocuments
from infrastructure.research_job import ResearchJob
from infrastructure.research_objects import ResearchVersionGuard
from interface.research import main
from tests.test_research_archive import add_raw, proof, raw_fixture
from tests.test_strict_replay import request


def legacy_payload():
    payload = raw_fixture()
    row = add_raw(payload)
    proof_doc = row["publication_proof"]["document"]
    proof_doc["body_base64"] = base64.b64encode(proof_doc.pop("body_utf8").encode()).decode()
    return payload


def freeze(tmp_path):
    payload = legacy_payload()
    path = create_dataset(payload, tmp_path / "datasets")
    eid, frozen = load_evidence(path)
    documents = EvidenceDocuments.for_evidence(path, eid)
    return payload, path, frozen, documents


def test_deduplicate_across_datasets_preserve_provenance_and_original(tmp_path):
    payload, path, frozen, documents = freeze(tmp_path)
    original = deepcopy(payload)
    assert "body_base64" not in path.read_text()
    assert documents.verify(frozen)["referenced_documents"] == 2
    first_files = {p.name: p.stat().st_mtime_ns for p in documents.objects.root.iterdir()}
    changed = deepcopy(payload)
    changed["filings"]["2330"][0]["revision_number"] = 2
    changed["filings"]["2330"][0]["source_version"] = "test-source-r2"
    later = create_dataset(changed, tmp_path / "datasets", parent=path)
    assert later != path
    assert {p.name: p.stat().st_mtime_ns for p in documents.objects.root.iterdir()} == first_files
    assert payload == original
    row, frozen_row = payload["filings"]["2330"][0], frozen["filings"]["2330"][0]
    for field in ("available_at", "availability_source", "receipt", "source_version", "revision_number",
                  "period_end", "statement_scope", "content_sha256", "fiscal_year", "quarter"):
        assert frozen_row[field] == row[field]
    before = materialize_filings(payload, parse_filing=parse_ixbrl)
    after = materialize_filings(frozen, parse_filing=parse_ixbrl, document_reader=documents.read)
    assert before == after
    assert after[0]["2330"][0]["available_at"] == "2026-08-02T00:00:00+08:00"


def test_legacy_reader_and_migration_cli_are_immutable_and_idempotent(tmp_path, capsys):
    payload = legacy_payload()
    legacy = tmp_path / "legacy.json"
    write_once(legacy, dict(evidence_id=digest(payload), payload=payload))
    before = legacy.read_bytes()
    assert main(["dataset-verify", "--evidence", str(legacy)]) == 0
    assert json.loads(capsys.readouterr().out)["legacy_inline_documents"] == 2
    engine, eid = build_replay_engine(legacy)
    assert engine.evidence.select(request(tmp_path, payload)).evidence["filings"]["2330"]
    args = ["dataset-migrate", "--evidence", str(legacy), "--store", str(tmp_path / "new")]
    assert main(args) == 0
    migrated = json.loads(capsys.readouterr().out)
    assert migrated["dataset_id"] != eid
    assert migrated["parent_dataset_id"] == eid
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == migrated
    assert legacy.read_bytes() == before
    assert main(["dataset-verify", "--evidence", migrated["evidence"]]) == 0
    assert json.loads(capsys.readouterr().out)["referenced_documents"] == 2
    # Copy an already-converted archive to a different store without paths in JSON.
    assert main(["dataset-migrate", "--evidence", migrated["evidence"], "--store", str(tmp_path / "copy")]) == 0
    copied = json.loads(capsys.readouterr().out)
    assert copied["dataset_id"] == migrated["dataset_id"]
    assert copied["parent_dataset_id"] is None
    manifest = json.loads((tmp_path / "copy" / copied["dataset_id"] / "manifest.json").read_text())
    assert manifest["parent_dataset_id"] is None
    assert main(["dataset-verify", "--evidence", copied["evidence"]]) == 0


def test_inline_and_referenced_native_decisions_match(tmp_path):
    payload, path, frozen, documents = freeze(tmp_path)
    legacy = tmp_path / "legacy.json"
    write_once(legacy, dict(evidence_id=digest(payload), payload=payload))
    old, _ = build_replay_engine(legacy)
    new, eid = build_replay_engine(path)
    req = replace(request(tmp_path, payload), versions=current_versions())
    before = old.run(req)
    after = new.run(replace(req, evidence_id=eid, state_path=tmp_path / "new.sqlite3"))
    for key in ("rows", "decisions", "diagnostics", "account"):
        assert before[key] == after[key]
    assert old.evidence.select(req).evidence == new.evidence.select(replace(req, evidence_id=eid)).evidence


@pytest.mark.parametrize("change", [
    {"schema": "unsupported"}, {"sha256": "../outside"}, {"size_bytes": -1},
    {"size_bytes": True}, {"size_bytes": 1.0}, {"path": "/tmp/outside"},
])
def test_invalid_reference_cannot_supply_a_path_or_ambiguous_format(tmp_path, change):
    _, _, frozen, documents = freeze(tmp_path)
    doc = next(iter(frozen["raw_documents"].values()))
    doc["body_ref"].update(change)
    with pytest.raises(ValueError, match="INVALID_SOURCE_DOCUMENT_REFERENCE"):
        verified_document(doc, document_reader=documents.read)


def test_hash_binding_size_and_body_ambiguity_are_checked(tmp_path):
    _, _, frozen, documents = freeze(tmp_path)
    doc = next(iter(frozen["raw_documents"].values()))
    for changed, error in (
        (dict(doc, sha256="0" * 64), "HASH_MISMATCH"),
        (dict(doc, body_ref=dict(doc["body_ref"], size_bytes=0)), "SIZE_MISMATCH"),
        (dict(doc, body_base64=""), "AMBIGUOUS"),
    ):
        with pytest.raises(ValueError, match=error):
            verified_document(changed, document_reader=documents.read)
    with pytest.raises(ValueError, match="READER_REQUIRED"):
        verified_document(doc)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        verified_document(doc, document_reader=lambda ref: b"x" * ref["size_bytes"])


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_missing_or_tampered_bytes_rejected_before_model_or_export(tmp_path, mutation):
    payload, path, frozen, documents = freeze(tmp_path)
    doc = next(iter(frozen["raw_documents"].values()))
    source = documents.objects.path(doc["sha256"])
    req = replace(request(tmp_path, payload), evidence_id=path.parent.name, versions=current_versions())
    with ResearchJob(path, tmp_path / "job") as job:
        job.run_day(req)
        if mutation == "missing":
            source.unlink()
        else:
            source.write_bytes(b"x" * source.stat().st_size)
        # Even a same-day receipt/cache hit must revalidate source identity.
        with pytest.raises(ValueError, match="SOURCE_DOCUMENT_(UNAVAILABLE|CHANGED)"):
            job.run_day(req)
        with pytest.raises(ValueError, match="SOURCE_DOCUMENT_(UNAVAILABLE|CHANGED)"):
            job.export_bundle(tmp_path / "broken.zip")
    with pytest.raises(ValueError, match="(UNAVAILABLE|HASH_MISMATCH)"):
        build_replay_engine(path)


def test_export_replays_without_the_original_document_store(tmp_path):
    payload, path, frozen, documents = freeze(tmp_path)
    req = replace(request(tmp_path, payload), evidence_id=path.parent.name, versions=current_versions())
    with ResearchJob(path, tmp_path / "job") as job:
        expected = job.run_day(req)
        job.export_bundle(tmp_path / "portable.zip")
    with zipfile.ZipFile(tmp_path / "portable.zip") as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert "body_base64" not in archive.read("evidence.json").decode()
        documents.verify(frozen)
        for sha in documents.references:
            assert "objects/" + sha in names
        archive.extractall(tmp_path / "portable")
    documents.objects.root.rename(tmp_path / "original-objects-unavailable")
    with ResearchJob(tmp_path / "portable" / "evidence.json", tmp_path / "portable") as restored:
        assert restored.run_day(req).result == expected.result
        assert restored.reproduce()["equal"]


def test_export_keeps_raw_document_when_its_id_also_names_a_graph_root(tmp_path):
    payload = legacy_payload()
    environment = ResearchVersionGuard().environment
    body = canonical_json(environment).encode()
    sha = digest(environment)
    # Deliberately shared bytes in separate roles, not a publication assertion.
    payload["raw_documents"][sha] = dict(url="https://mops.twse.com.tw/SYNTHETIC_TEST_ONLY/environment.json",
        sha256=sha, retrieved_at="2026-09-01T00:00:00+08:00",
        body_base64=base64.b64encode(body).decode())
    path = create_dataset(payload, tmp_path / "datasets")
    req = replace(request(tmp_path, payload), evidence_id=path.parent.name, versions=current_versions())
    with ResearchJob(path, tmp_path / "job") as job:
        expected = job.run_day(req)
        assert expected.receipt["objects"]["environment"] == sha
        assert job.objects._index_path(sha).exists() and not job.objects.path(sha).exists()
        job.export_bundle(tmp_path / "portable.zip")
    with zipfile.ZipFile(tmp_path / "portable.zip") as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert archive.read("objects/" + sha) == body
        assert "objects/json-index/" + sha + ".json" in names
        archive.extractall(tmp_path / "portable")
    with ResearchJob(tmp_path / "portable" / "evidence.json", tmp_path / "portable") as restored:
        assert restored.run_day(req).result == expected.result
        assert restored.reproduce()["equal"]


def test_publication_hash_binding_still_applies_to_referenced_proof(tmp_path):
    _, _, frozen, documents = freeze(tmp_path)
    proof = frozen["filings"]["2330"][0]["publication_proof"]
    assert verify_publication(proof, binding=proof["binding"], document_reader=documents.read).startswith("2026-08-02")
    with pytest.raises(ValueError, match="BINDING_NOT_VERIFIED"):
        verify_publication(proof, binding=dict(proof["binding"], source_version="wrong-version"),
                           document_reader=documents.read)


def test_calendar_and_lifecycle_base64_proofs_resolve_offline(tmp_path):
    payload = raw_fixture()
    event = dict(symbol="2330", market="TWSE", kind="PRE_LISTING", start=payload["start"], end="2026-09-09")
    payload["security_lifecycle"] = {"2330": [dict(event, proof=proof(event))]}
    payload["bars"] = [bar for bar in payload["bars"] if bar["date"] >= "20260910"]
    proofs = [item["publication_proof"] for item in payload["calendar"]["documents"].values()]
    proofs.append(payload["security_lifecycle"]["2330"][0]["proof"])
    for item in proofs:
        doc = item["document"]
        doc["body_base64"] = base64.b64encode(doc.pop("body_utf8").encode()).decode()
    path = create_dataset(payload, tmp_path / "datasets")
    req = replace(request(tmp_path, payload), evidence_id=path.parent.name, versions=current_versions())
    with ResearchJob(path, tmp_path / "job") as job:
        for _ in range(2):
            selected = job.evidence.select(req)
            assert not selected.diagnostics["missing_bars"]["2330"]
            assert selected.diagnostics["bar_exemptions"]["2330"]
            assert not selected.diagnostics["calendar_unknown_dates"]


def test_failed_conversion_never_publishes_a_partial_dataset(tmp_path):
    payload = legacy_payload()
    broken = deepcopy(payload)
    next(iter(broken["raw_documents"].values()))["body_base64"] = base64.b64encode(b"corrupt").decode()
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        create_dataset(broken, tmp_path / "datasets")
    assert not list((tmp_path / "datasets").glob("*/evidence.json"))
    path = create_dataset(payload, tmp_path / "datasets")
    eid, frozen = load_evidence(path)
    assert EvidenceDocuments.for_evidence(path, eid).verify(frozen)["referenced_documents"] == 2


def test_flat_save_and_raw_collection_use_refs_without_base64_encoding(tmp_path, monkeypatch):
    from infrastructure.fundamentals.mops_repository import MopsRepository
    from tests.fundamentals.test_filing_refresh_planner import _candidate, _fact
    from tests.fundamentals.test_ixbrl_units import document
    repo = MopsRepository(tmp_path / "mops", connect_attempts=1)
    candidate = replace(_candidate(2025, 4), symbol="2330", available_at="2026-08-01T00:00:00Z",
                        source_url="https://mops.twse.com.tw/test-only")
    repo.save_revision(candidate, document(), [_fact(100.)], available_at=candidate.available_at,
                       fetched_at=candidate.available_at, availability_source="OFFICIAL_FILING_TIMESTAMP")
    original_db = repo.database_path.read_bytes()
    documents = EvidenceDocuments(tmp_path / "objects")
    def forbid(*args, **kwargs):
        raise AssertionError("Direct raw preparation must not encode Base64 or request HTTP")
    monkeypatch.setattr("base64.b64encode", forbid)
    monkeypatch.setattr("requests.sessions.Session.request", forbid)
    payload = collect_evidence(market_cache=tmp_path / "missing-market.sqlite3", mops_cache=repo.database_path,
        restriction_cache=None, restriction_file=None, calendar=dict(source="TWSE_OFFICIAL", days={}),
        symbols=("2330",), market="TWSE", sector="24", start=dt.date(2026, 8, 1), end=dt.date(2026, 8, 1),
        raw=True, document_store=documents.store)
    path = tmp_path / "evidence.json"
    eid = save_evidence(payload, path)
    assert "body_base64" not in path.read_text()
    assert EvidenceDocuments.for_evidence(path, eid).verify(load_evidence(path)[1])["referenced_documents"] == 1
    assert repo.database_path.read_bytes() == original_db

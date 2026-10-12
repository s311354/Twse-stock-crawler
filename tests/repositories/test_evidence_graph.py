"""Synthetic storage/parity tests, not publication proof or live research."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import sqlite3
import zlib

import pytest

from application.replay_contracts import canonical_json, digest
from infrastructure.research_archive import write_bytes_once
from infrastructure.research_objects import GRAPH_STORE_NAME, ResearchObjects
from infrastructure.storage.evidence_graph import EvidenceGraphReader, EvidenceGraphWriter
from infrastructure.storage.report_evidence import ReportEvidenceReader, ReportEvidenceWriter


def proof_tree(count=700):
    return dict(source="SYNTHETIC_TEST_ONLY", available_at="2026-07-31T15:00:00+08:00",
        revision="synthetic-v1", fields=[dict(name=f"field-{i}", value=i,
        original="synthetic-only-" + str(i) + "x" * 1200) for i in range(count)])


def stored_root(value):
    # Exceed the existing 512-byte inline threshold for root-corruption tests.
    return dict(value, fixture_binding="SYNTHETIC_TEST_ONLY_" + "p" * 600)


def rows(path):
    with sqlite3.connect(path) as connection:
        return dict(connection.execute("SELECT sha256, body FROM report_objects"))


def test_cross_day_roots_share_unchanged_nodes_and_public_ids(tmp_path):
    first = stored_root(dict(as_of="2026-08-03T13:30:00+08:00", cash=1_000_000, proof=proof_tree()))
    second = dict(first, as_of="2026-08-04T13:30:00+08:00", cash=900_000)
    original = deepcopy(first)
    store = ResearchObjects(tmp_path / "objects")
    first_id = store.put_json(first)
    baseline = rows(store.root / GRAPH_STORE_NAME)
    store = ResearchObjects(store.root)  # Reopen: no process-local dedup required.
    second_id = store.put_json(second)
    after = rows(store.root / GRAPH_STORE_NAME)
    assert first_id == digest(first) and second_id == digest(second) != first_id
    assert store.get(first_id) == canonical_json(first).encode()
    assert store.get_json(second_id) == second and first == original
    assert len(after) - len(baseline) == 1  # Only the changed ancestor/root.
    assert all(after[key] == body for key, body in baseline.items())
    assert store.stats["nodes_written"] == 1 and store.stats["nodes_reused"] >= 700
    assert store.stats["node_bytes_written"] < 4096
    assert store._index_path(first_id).stat().st_size < 512
    assert not store.path(first_id).exists()  # No full duplicate JSON object.
    stamps = {p: p.stat().st_mtime_ns for p in (store.root / "json-index").iterdir()}
    assert store.put_json(second) == second_id
    assert store.stats["json_roots_reused"] == 1 and rows(store.root / GRAPH_STORE_NAME) == after
    assert all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items())
    result = store.get_json(first_id)
    result["proof"]["fields"][0]["value"] = "mutated"
    assert store.get_json(first_id) == first


def test_daily_report_and_research_share_same_node_encoding(tmp_path):
    report = dict(symbol="TEST", signal_date="2026-08-03", model_version="6.0.0",
                  parameter_version="synthetic", proof=proof_tree(30))
    path = tmp_path / GRAPH_STORE_NAME
    with ReportEvidenceWriter(path) as daily:
        ref = daily.retain(report)
    before = rows(path)
    objects = ResearchObjects(tmp_path)
    assert objects.put_json(report) == digest(report)
    assert objects.stats["nodes_written"] == 0 and objects.stats["nodes_reused"] > 0
    assert rows(path) == before
    with ReportEvidenceReader(tmp_path) as reader:
        assert reader.resolve(ref) == objects.get_json(digest(report)) == report


def test_old_flat_objects_and_new_graphs_coexist_without_rewrite(tmp_path):
    objects = ResearchObjects(tmp_path)
    legacy = dict(proof=proof_tree(10), mode="legacy")
    body = canonical_json(legacy).encode()
    sha = objects.put(body)
    stamp = objects.path(sha).stat().st_mtime_ns
    assert objects.put_json(legacy) == sha
    assert objects.get_json(sha) == legacy
    assert objects.path(sha).stat().st_mtime_ns == stamp
    assert not objects._index_path(sha).exists()
    raw = b"synthetic raw iXBRL bytes"
    raw_id = objects.put(raw)
    assert objects.get(raw_id) == raw
    changed = dict(legacy, mode="new")
    assert objects.get_json(objects.put_json(changed)) == changed
    assert objects.get_json(sha) == legacy


def test_tuple_metadata_retains_legacy_json_array_semantics(tmp_path):
    objects = ResearchObjects(tmp_path)
    value = dict(dependencies=[("pandas", "test-version")], nested=(proof_tree(3),))
    sha = objects.put_json(value)
    assert sha == digest(value)
    assert objects.get(sha) == canonical_json(value).encode()
    assert objects.get_json(sha) == json.loads(canonical_json(value))
    assert objects.put_json(value) == sha


def test_legacy_numeric_keys_keep_original_bytes_and_identity(tmp_path):
    objects = ResearchObjects(tmp_path)
    value = {2: "two", 10: "ten"}
    sha = objects.put_json(value)
    assert sha == digest(value) and objects.get(sha) == canonical_json(value).encode()
    assert objects.get_json(sha) == {"2": "two", "10": "ten"}
    assert not objects._index_path(sha).exists()


@pytest.mark.parametrize("shape", ["oversized", "deep"])
def test_legacy_json_outside_bounded_node_shape_keeps_original_format(tmp_path, monkeypatch, shape):
    import infrastructure.storage.evidence_graph as module
    value = dict(original="SYNTHETIC_TEST_ONLY" + "x" * 2000)
    if shape == "oversized":
        monkeypatch.setattr(module, "MAX_NODE_BYTES", 1024)
    else:
        for _ in range(130):
            value = [value]
    objects = ResearchObjects(tmp_path)
    sha = objects.put_json(value)
    assert sha == digest(value) and objects.get(sha) == canonical_json(value).encode()
    assert objects.path(sha).exists() and not objects._index_path(sha).exists()
    assert rows(tmp_path / GRAPH_STORE_NAME) == {}


def test_flat_write_conflict_keeps_legacy_error_contract(tmp_path):
    objects = ResearchObjects(tmp_path)
    value = dict(original="SYNTHETIC_TEST_ONLY")
    body = canonical_json(value).encode()
    sha = objects.put(body)
    objects.path(sha).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        objects.put(body)
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        objects.put_json(value)
    with pytest.raises(ValueError, match="RESEARCH_OBJECT_HASH_MISMATCH"):
        objects.get(sha)
    assert objects.path(sha).read_bytes() == b"corrupt"


@pytest.mark.parametrize("target", ["missing_node", "corrupt_node", "root_binding", "root_hash", "root_type"])
def test_missing_or_tampered_evidence_is_not_repaired_silently(tmp_path, target):
    objects = ResearchObjects(tmp_path)
    value = stored_root(dict(proof=proof_tree(20)))
    sha = objects.put_json(value)
    ref = json.loads(objects._index_path(sha).read_bytes())
    if target in {"missing_node", "corrupt_node"}:
        with sqlite3.connect(tmp_path / GRAPH_STORE_NAME) as connection:
            if target == "missing_node":
                connection.execute("DELETE FROM report_objects WHERE sha256=?", (ref["root"][1],))
            else:
                connection.execute("UPDATE report_objects SET body=? WHERE sha256=?", (b"corrupt", ref["root"][1]))
    elif target == "root_type":
        objects._index_path(sha).write_text("[]")
    else:
        ref["value_sha256" if target == "root_hash" else "store"] = "0" * 64
        objects._index_path(sha).write_text(json.dumps(ref))
    with pytest.raises(ValueError, match="EVIDENCE|RESEARCH_OBJECT"):
        objects.get_json(sha)
    with pytest.raises(ValueError, match="EVIDENCE|RESEARCH_OBJECT"):
        objects.put_json(value)


def test_warm_reader_detects_external_database_corruption(tmp_path):
    with EvidenceGraphWriter(tmp_path / GRAPH_STORE_NAME) as writer:
        ref = writer.retain(stored_root(proof_tree(10)))
    with EvidenceGraphReader(tmp_path) as reader:
        assert reader.resolve(ref) == stored_root(proof_tree(10))
        with sqlite3.connect(tmp_path / GRAPH_STORE_NAME) as connection:
            connection.execute("UPDATE report_objects SET body=? WHERE sha256=?", (b"corrupt", ref["root"][1]))
        with pytest.raises(ValueError, match="HASH_MISMATCH"):
            reader.resolve(ref)


def test_graph_nodes_rollback_and_research_root_published_only_after_commit(tmp_path, monkeypatch):
    path = tmp_path / GRAPH_STORE_NAME
    with pytest.raises(RuntimeError, match="interrupted"), EvidenceGraphWriter(path) as writer:
        writer.retain(proof_tree())  # Flushes several bounded batches.
        raise RuntimeError("interrupted")
    assert rows(path) == {}
    objects = ResearchObjects(tmp_path)
    import infrastructure.research_objects as module
    native = module.write_once
    def interrupt(path, value):
        raise RuntimeError("before root publication")
    monkeypatch.setattr(module, "write_once", interrupt)
    value = dict(proof=proof_tree())
    with pytest.raises(RuntimeError, match="publication"):
        objects.put_json(value)
    assert rows(path) and not objects._index_path(digest(value)).exists()
    before = rows(path)
    monkeypatch.setattr(module, "write_once", native)
    assert objects.put_json(value) == digest(value)
    assert rows(path) == before and objects.stats["nodes_written"] == 0


def test_concurrent_identical_publish_has_one_immutable_root_and_graph(tmp_path):
    value = dict(proof=proof_tree(80))
    def publish(_):
        return ResearchObjects(tmp_path).put_json(value)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert set(pool.map(publish, range(4))) == {digest(value)}
    assert len(list((tmp_path / "json-index").iterdir())) == 1
    objects = ResearchObjects(tmp_path)
    before = rows(tmp_path / GRAPH_STORE_NAME)
    assert objects.put_json(value) == digest(value)
    assert rows(tmp_path / GRAPH_STORE_NAME) == before


def test_portable_nodes_are_deduplicated_reachable_and_can_continue_without_sqlite(tmp_path):
    objects = ResearchObjects(tmp_path / "source")
    values = [stored_root(dict(day=i, proof=proof_tree(40))) for i in range(2)]
    ids = [objects.put_json(value) for value in values]
    unrelated = objects.put_json(dict(secret="SYNTHETIC_UNRELATED", proof=proof_tree(3)))
    entries = list(objects.export_entries(ids))
    assert len(entries) == len({name for name, _ in entries})
    assert not any(name.endswith(".sqlite3") for name, _ in entries)
    assert not any(unrelated in name for name, _ in entries)
    target = tmp_path / "restored"
    for name, body in entries:
        write_bytes_once(target / name, body)
    restored = ResearchObjects(target)
    assert [restored.get_json(sha) for sha in ids] == values
    assert not (target / GRAPH_STORE_NAME).exists()
    third = dict(values[-1], day=2)
    third_id = restored.put_json(third)
    assert restored.stats["nodes_written"] == 1  # Portable old nodes not rewritten.
    assert restored.get_json(third_id) == third
    assert [restored.get_json(sha) for sha in ids] == values
    node_file = next((target / (GRAPH_STORE_NAME + ".nodes")).iterdir())
    node_file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        list(restored.export_entries(ids))


def test_foreign_database_depth_and_expansion_fail_closed(tmp_path):
    path = tmp_path / GRAPH_STORE_NAME
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE account (cash INTEGER)")
    with pytest.raises(ValueError, match="MUST_BE_SEPARATE"), EvidenceGraphWriter(path):
        pass
    with sqlite3.connect(path) as connection:
        assert set(connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone()) == {"account"}
    good = tmp_path / "good.sqlite3"
    with EvidenceGraphWriter(good) as writer:
        ref = writer.retain(dict(proof="x" * 2000))
    with EvidenceGraphReader(tmp_path, max_value_bytes=100) as reader:
        with pytest.raises(ValueError, match="EXPANSION_TOO_LARGE"):
            reader.resolve(ref)
    value = []
    for _ in range(130):
        value = [value]
    with pytest.raises(ValueError, match="TOO_DEEP"), EvidenceGraphWriter(good) as writer:
        writer.retain(value)


def test_existing_hash_collision_or_tampering_cannot_be_overwritten(tmp_path):
    with EvidenceGraphWriter(tmp_path / GRAPH_STORE_NAME) as writer:
        ref = writer.retain(stored_root(dict(proof=proof_tree(10))))
    with sqlite3.connect(tmp_path / GRAPH_STORE_NAME) as connection:
        connection.execute("UPDATE report_objects SET body=? WHERE sha256=?", (zlib.compress(b"poison"), ref["root"][1]))
    with pytest.raises(ValueError, match="HASH_MISMATCH"), EvidenceGraphWriter(tmp_path / GRAPH_STORE_NAME) as writer:
        writer.retain(stored_root(dict(proof=proof_tree(10))))


def test_corrupt_portable_node_cannot_trigger_flat_fallback(tmp_path):
    value = dict(proof="SYNTHETIC_TEST_ONLY" + "x" * 2000)
    source = ResearchObjects(tmp_path / "source")
    sha = source.put_json(value)
    ref = source._reference(sha)
    target = ResearchObjects(tmp_path / "target")
    node = target.root / (GRAPH_STORE_NAME + ".nodes") / ref["root"][1]
    node.parent.mkdir(parents=True)
    node.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="EVIDENCE_HASH_MISMATCH"):
        target.put_json(value)
    assert not target.path(sha).exists() and not target._index_path(sha).exists()


def test_portable_node_cannot_escape_its_store_even_with_correct_bytes(tmp_path):
    value = dict(proof="SYNTHETIC_TEST_ONLY" + "x" * 2000)
    source = ResearchObjects(tmp_path / "source")
    sha = source.put_json(value)
    ref = source._reference(sha)
    encoded = zlib.decompress(rows(source.root / GRAPH_STORE_NAME)[ref["root"][1]])
    outside = tmp_path / "outside.node"
    outside.write_bytes(encoded)
    target = ResearchObjects(tmp_path / "target")
    node = target.root / (GRAPH_STORE_NAME + ".nodes") / ref["root"][1]
    node.parent.mkdir(parents=True)
    node.symlink_to(outside)
    with pytest.raises(ValueError, match="INVALID_EVIDENCE_STORE"):
        target.put_json(value)
    assert not target.path(sha).exists() and not target._index_path(sha).exists()

"""Structural memo is lossless/bounded, never trusts mutable input identities."""
from copy import deepcopy
import json
import sqlite3

import pytest

from application.daily_performance import DailyPerformance
from infrastructure.storage.evidence_graph import EvidenceGraphWriter, EvidenceGraphReader
from tests.repositories.test_evidence_graph import proof_tree, rows


def encode(directory, values, capacity):
    probe = DailyPerformance()
    path = directory / "evidence.sqlite3"
    with probe.activate(), EvidenceGraphWriter(path, encoding_cache_bytes=capacity) as writer:
        refs = [writer.retain(value) for value in values]
        assert writer._encoding.bytes <= capacity
        assert len(writer._encoding.values) <= writer._encoding.MAX_ENTRIES
    with EvidenceGraphReader(directory) as reader:
        assert [reader.resolve(ref) for ref in refs] == json.loads(json.dumps(values))
    return refs, rows(path), probe


def test_cached_and_uncached_graphs_have_identical_bytes_and_ids(tmp_path):
    proof = proof_tree(100)
    values = [dict(day=i, evidence=[deepcopy(proof) for _ in range(5)]) for i in range(2)]
    before = deepcopy(values)
    reference, original, ref_probe = encode(tmp_path / "reference", values, 0)
    actual, stored, probe = encode(tmp_path / "optimized", values, 8 * 1024 * 1024)
    assert actual == reference and stored == original and values == before
    assert probe.counters["evidence_node_encodes"] < ref_probe.counters["evidence_node_encodes"] // 5
    assert probe.counters["evidence_encoding_hits"] > 0


def test_mutation_and_returned_reference_cannot_poison_memo(tmp_path):
    value = dict(proof=proof_tree(12), inline=dict(x=1))
    with EvidenceGraphWriter(tmp_path / "evidence.sqlite3") as writer:
        first = writer.retain(value)
        original = deepcopy(first)
        first["root"][1] = "caller-corrupted-reference"
        second = writer.retain(value)
        assert second == original
        value["proof"]["fields"][0]["value"] = 999
        third = writer.retain(value)
        assert third != original
    with EvidenceGraphReader(tmp_path) as reader:
        assert reader.resolve(original)["proof"]["fields"][0]["value"] == 0
        assert reader.resolve(third)["proof"]["fields"][0]["value"] == 999


def test_scalar_json_distinctions_and_tuple_semantics(tmp_path):
    values = [dict(value=x, proof=proof_tree(2)) for x in (True, 1, 1.0, 0.0, -0.0, None, "1", (1,))]
    actual, stored, _ = encode(tmp_path / "cached", values, 8 * 1024 * 1024)
    expected, original, _ = encode(tmp_path / "reference", values, 0)
    assert actual == expected and stored == original
    assert len({ref["value_sha256"] for ref in actual}) == len(values)


def test_eviction_does_not_recycle_tokens_or_change_graph(tmp_path):
    values = [proof_tree(80), proof_tree(80)]
    actual, stored, probe = encode(tmp_path / "tiny", values, 2048)
    expected, original, _ = encode(tmp_path / "reference", values, 0)
    assert actual == expected and stored == original
    assert probe.counters["evidence_encoding_evictions"] > 0


def test_warm_encoding_still_detects_corrupt_existing_node(tmp_path):
    value = dict(proof=proof_tree(3))
    with EvidenceGraphWriter(tmp_path / "evidence.sqlite3") as writer:
        writer.retain(value)
    with sqlite3.connect(tmp_path / "evidence.sqlite3") as connection:
        connection.execute("UPDATE report_objects SET body=?", (b"corrupt",))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        with EvidenceGraphWriter(tmp_path / "evidence.sqlite3") as writer:
            writer.retain(value)


@pytest.mark.parametrize("capacity", [-1, True, 65 * 1024 * 1024])
def test_cache_capacity_is_validated(tmp_path, capacity):
    with pytest.raises(ValueError, match="ENCODING_CACHE_BYTES"):
        EvidenceGraphWriter(tmp_path / "evidence.sqlite3", encoding_cache_bytes=capacity)


def test_new_storage_stage_metrics_do_not_change_native_ids(tmp_path):
    refs, _, probe = encode(tmp_path, [proof_tree(3)], 8 * 1024 * 1024)
    assert len(refs) == 1
    assert {"evidence_encode", "evidence_value_hash", "evidence_lookup", "evidence_compress",
            "evidence_insert", "evidence_commit"} <= probe.seconds.keys()


def test_storage_telemetry_is_batched_not_one_callback_per_node(tmp_path):
    events = []
    probe = DailyPerformance(stage_callback=events.append)
    with probe.activate(), EvidenceGraphWriter(tmp_path / "evidence.sqlite3") as writer:
        writer.retain(proof_tree(700))
    batches = probe.counters["evidence_batches"]
    assert probe.calls["evidence_compress"] == batches
    assert probe.calls["evidence_stored_verify"] == batches
    assert probe.counters["evidence_nodes_written"] > 700
    assert len(events) < 40  # Stable batch bound, not a fragile time assertion.

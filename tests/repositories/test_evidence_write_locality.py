"""Bounded physical-order SQL reads preserve all bytes and fail-closed checks."""
import sqlite3

import pytest

from application.daily_performance import DailyPerformance
from infrastructure.storage.evidence_graph import (
    EvidenceGraphReader, EvidenceGraphWriter, MAX_NODE_BYTES, MAX_WRITE_BATCH_NODES,
)
from tests.repositories.test_evidence_graph import proof_tree, rows, stored_root


def store(path, value, *, optimized):
    probe = DailyPerformance()
    with probe.activate(), EvidenceGraphWriter(path, batch_nodes=1024 if optimized else 256,
                                               ordered_lookup=optimized) as writer:
        ref = writer.retain(value)
        assert len(writer._pending) < writer._batch_nodes
        assert writer._pending_bytes <= MAX_NODE_BYTES
    return ref, probe


def test_larger_ordered_batches_keep_same_ids_bytes_and_retry(tmp_path):
    value = proof_tree(2200)
    reference, before = store(tmp_path / "reference" / "evidence.sqlite3", value, optimized=False)
    actual, after = store(tmp_path / "optimized" / "evidence.sqlite3", value, optimized=True)
    assert actual == reference
    assert rows(tmp_path / "reference" / "evidence.sqlite3") == rows(tmp_path / "optimized" / "evidence.sqlite3")
    assert after.counters["evidence_batches"] < before.counters["evidence_batches"]
    repeated, retry = store(tmp_path / "optimized" / "evidence.sqlite3", value, optimized=True)
    assert repeated == actual
    assert retry.counters["evidence_nodes_written"] == 0
    assert retry.counters["evidence_lookup_keys"] == before.counters["evidence_lookup_keys"]
    with EvidenceGraphReader(tmp_path / "optimized") as reader:
        assert reader.resolve(actual) == value


def test_query_uses_covering_index_then_integer_primary_key(tmp_path):
    value = proof_tree(30)
    path = tmp_path / "evidence.sqlite3"
    store(path, value, optimized=True)
    statements = []
    with EvidenceGraphWriter(path) as writer:
        writer.connection.set_trace_callback(statements.append)
        writer.retain(value)
        writer._flush()
        writer.connection.set_trace_callback(None)
        lookup = next(s for s in statements if "CROSS JOIN report_objects AS stored" in s)
        plan = writer.connection.execute("EXPLAIN QUERY PLAN " + lookup).fetchall()
        descriptions = "\n".join(row[3] for row in plan)
        assert "COVERING INDEX" in descriptions
        assert "INTEGER PRIMARY KEY" in descriptions
        assert "LIMIT -1" in lookup and "ORDER BY rowid" in lookup


def test_sql_variable_limit_is_respected(tmp_path, monkeypatch):
    original = sqlite3.connect
    def limited(*args, **kwargs):
        connection = original(*args, **kwargs)
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 17)
        return connection
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", limited)
        with EvidenceGraphWriter(tmp_path / "evidence.sqlite3") as writer:
            assert writer._batch_nodes == 17
            ref = writer.retain(proof_tree(100))
    with EvidenceGraphReader(tmp_path) as reader:
        assert reader.resolve(ref) == proof_tree(100)


@pytest.mark.parametrize("change", ["corrupt", "oversized"])
def test_ordered_retry_still_rejects_corruption_and_rolls_back(tmp_path, change):
    path = tmp_path / "evidence.sqlite3"
    value = stored_root(proof_tree(40))
    ref, _ = store(path, value, optimized=True)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE report_objects SET body=? WHERE sha256=?",
                           (b"bad" if change == "corrupt" else b"x" * (MAX_NODE_BYTES + 16384), ref["root"][1]))
    before = rows(path)
    with pytest.raises(ValueError, match="HASH_MISMATCH"), EvidenceGraphWriter(path) as writer:
        writer.retain(dict(value, extra=proof_tree(1100)))
        writer.retain(value)
    assert rows(path) == before


@pytest.mark.parametrize("size", [0, True, MAX_WRITE_BATCH_NODES + 1])
def test_write_capacity_rejects_invalid_values(tmp_path, size):
    with pytest.raises(ValueError, match="WRITE_BATCH_NODES"):
        EvidenceGraphWriter(tmp_path / "evidence.sqlite3", batch_nodes=size)

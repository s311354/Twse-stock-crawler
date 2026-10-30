"""Synthetic proof-tree checks for root-scoped SQLite path validation."""
import sqlite3
import zlib

import pytest

from application.replay_contracts import canonical_json
from infrastructure.storage.evidence_graph import (
    EvidenceGraphReader, EvidenceGraphWriter, MAX_READER_STORES, SQLITE_READER_CACHE_BYTES, _sha,
)


STORE = "evidence.sqlite3"


def retained(tmp_path):
    value = {"source": "SYNTHETIC_ONLY", "root_binding": "SYNTHETIC_ONLY_" + "p" * 600, "proof": [
        {"field": i, "original": str(i) + "x" * 800} for i in range(300)]}
    with EvidenceGraphWriter(tmp_path / STORE) as writer:
        ref = writer.retain(value)
    return value, ref


def test_open_readonly_handle_validates_path_per_root_not_per_proof_node(tmp_path):
    value, ref = retained(tmp_path)
    class Reader(EvidenceGraphReader):
        validations = 0
        def validate_store(self, store):
            self.validations += 1
            return super().validate_store(store)
    with Reader(tmp_path) as reader:
        assert reader.resolve(ref) == value
        assert reader.validations == 1
        assert reader._bytes <= reader.cache_bytes
        assert all(type(encoded) is bytes for encoded in reader._cache.values())
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader._connections[STORE].execute("DELETE FROM report_objects")
        assert reader.resolve(ref) == value
        assert reader.validations == 2


def test_next_root_still_rejects_symlink_escape_with_warm_handle(tmp_path):
    directory = tmp_path / "inside"
    directory.mkdir()
    value, ref = retained(directory)
    outside = tmp_path / "outside.sqlite3"
    with EvidenceGraphReader(directory) as reader:
        assert reader.resolve(ref) == value
        (directory / STORE).replace(outside)
        (directory / STORE).symlink_to(outside)
        with pytest.raises(ValueError, match="INVALID_EVIDENCE_STORE"):
            reader.resolve(ref)


def test_next_root_reopens_replacement_and_rejects_corruption(tmp_path):
    value, ref = retained(tmp_path)
    replacement = tmp_path / "replacement.sqlite3"
    with EvidenceGraphReader(tmp_path) as reader:
        assert reader.resolve(ref) == value
        replacement.write_bytes((tmp_path / STORE).read_bytes())
        with sqlite3.connect(replacement) as connection:
            connection.execute("UPDATE report_objects SET body=? WHERE sha256=?",
                               (b"corrupt", ref["root"][1]))
        replacement.replace(tmp_path / STORE)
        with pytest.raises(ValueError, match="HASH_MISMATCH"):
            reader.resolve(ref)


def test_portable_fallback_keeps_confinement_checks_with_open_database(tmp_path):
    directory = tmp_path / "inside"
    directory.mkdir()
    with EvidenceGraphWriter(directory / STORE) as writer:
        ref = writer.retain({"source": "SYNTHETIC_ONLY", "proof": "x" * 2000})
        other = writer.retain({"source": "SYNTHETIC_ONLY", "proof": "y" * 2000})
    with sqlite3.connect(directory / STORE) as connection:
        encoded = zlib.decompress(connection.execute("SELECT body FROM report_objects WHERE sha256=?",
                                                    (ref["root"][1],)).fetchone()[0])
        connection.execute("DELETE FROM report_objects WHERE sha256=?", (ref["root"][1],))
    outside = tmp_path / "outside.node"
    outside.write_bytes(encoded)
    portable = directory / (STORE + ".nodes") / ref["root"][1]
    portable.parent.mkdir()
    portable.symlink_to(outside)
    with EvidenceGraphReader(directory) as reader:
        reader.resolve(other)
        assert STORE in reader._connections
        with pytest.raises(ValueError, match="INVALID_EVIDENCE_STORE"):
            reader.resolve(ref)


def test_sqlite_page_cache_target_is_shared_across_open_replaced_and_evicted_handles(tmp_path):
    refs = []
    value = {"source": "SYNTHETIC_ONLY", "proof": "x" * 2000}
    for index in range(MAX_READER_STORES + 1):
        with EvidenceGraphWriter(tmp_path / f"store-{index}.sqlite3") as writer:
            refs.append(writer.retain(value))
    def budget(reader):
        sizes = [connection.execute("PRAGMA cache_size").fetchone()[0]
                 for connection in reader._connections.values()]
        assert all(size < 0 for size in sizes)
        assert sum(-size * 1024 for size in sizes) <= SQLITE_READER_CACHE_BYTES
        assert len(sizes) <= MAX_READER_STORES
        return sizes
    with EvidenceGraphReader(tmp_path) as reader:
        for ref in refs:
            assert reader.resolve(ref) == value
            budget(reader)
        assert refs[0]["store"] not in reader._connections
        assert reader.resolve(refs[0]) == value
        budget(reader)
        store = tmp_path / refs[0]["store"]
        replacement = tmp_path / "replacement.sqlite3"
        replacement.write_bytes(store.read_bytes())
        previous = reader._connections[refs[0]["store"]]
        replacement.replace(store)
        assert reader.resolve(refs[0]) == value
        assert reader._connections[refs[0]["store"]] is not previous
        budget(reader)


def test_prefetch_visits_bounded_body_rows_in_physical_order_and_checks_hashes(tmp_path):
    records = []
    for index in range(12):
        encoded = canonical_json(["dict", [["proof", ["value", f"SYNTHETIC_ONLY_{index}" + "x" * 1000]]]]).encode()
        records.append((_sha(encoded), encoded))
    physical = sorted(records, reverse=True)
    with sqlite3.connect(tmp_path / STORE) as connection:
        connection.execute("CREATE TABLE report_objects (sha256 TEXT PRIMARY KEY, body BLOB NOT NULL)")
        connection.executemany("INSERT INTO report_objects VALUES (?,?)",
                               [(sha, zlib.compress(encoded)) for sha, encoded in physical])
    with EvidenceGraphReader(tmp_path) as reader:
        reader.check_store(STORE)
        reader._prefetch([["ref", sha] for sha, _ in sorted(records)], STORE)
        assert [sha for store, sha in reader._cache] == [sha for sha, _ in physical]
        assert reader._bytes <= reader.cache_bytes
        assert all(reader._cache[(STORE, sha)] == encoded for sha, encoded in records)
    with sqlite3.connect(tmp_path / STORE) as connection:
        connection.execute("UPDATE report_objects SET body=? WHERE sha256=?", (b"corrupt", physical[4][0]))
    with EvidenceGraphReader(tmp_path) as reader:
        reader.check_store(STORE)
        with pytest.raises(ValueError, match="HASH_MISMATCH"):
            reader._prefetch([["ref", sha] for sha, _ in records], STORE)

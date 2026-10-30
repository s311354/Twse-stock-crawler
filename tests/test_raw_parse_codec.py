"""SYNTHETIC ONLY: disposable source-cache codec, never financial evidence."""
import hashlib
import sqlite3
import zlib

import pytest

from application.replay_contracts import aware_timestamp, canonical_json
from infrastructure.storage import research_cache as cache_module
from infrastructure.storage.research_cache import SQLiteRawParseCache
from tests.test_research_sqlite_cache import adapter, history


KEY = ("SYNTHETIC-parser-v1", "SYNTHETIC-original-sha")


def payload(index=0):
    return [{"facts": [{"concept": "SYNTHETIC ONLY", "number": 3.125,
        "missing": None, "flag": True, "text": "財報測試" * 500,
        "dimensions": [["axis", "member"]], "ordinal": index}]}]


def encode(value):
    return canonical_json(value).encode("utf-8")


def packet(raw, *, declared=None):
    return (cache_module._RAW_ZLIB_MAGIC +
            (len(raw) if declared is None else declared).to_bytes(8, "big") + zlib.compress(raw))


def reject_compute():
    pytest.fail("verified cache hit unexpectedly recomputed")


@pytest.mark.parametrize("compression", [False, True])
@pytest.mark.parametrize("kind", ["ixbrl", "financial_companion"])
def test_codec_on_off_full_values_warm_restart_and_detached_mutation(tmp_path, compression, kind):
    expected = payload()
    path = tmp_path / "SYNTHETIC-source-cache.sqlite3"
    with SQLiteRawParseCache(path, compression=compression) as disk:
        actual = disk.memo(kind, KEY, lambda: expected)
        assert actual == expected
        actual[0]["facts"][0]["dimensions"][0].append("MUTATED")
        assert disk.memo(kind, KEY, reject_compute) == expected
        raw, sha = disk.connection.execute("SELECT facts_json,facts_sha256 FROM research_parses").fetchone()
        assert isinstance(raw, bytes if compression else str)
        assert sha == hashlib.sha256(encode(expected)).hexdigest()
        assert disk.stats["stored_payload_bytes"] == len(raw if isinstance(raw, bytes) else raw.encode())
    # Codec configuration is storage policy, not a parser/derivation identity.
    with SQLiteRawParseCache(path, compression=not compression) as restarted:
        assert restarted.memo(kind, KEY, reject_compute) == expected
        assert restarted.stats["sqlite_ixbrl_hit"] == 1


def test_identical_small_budget_compression_prevents_fifo_scan_thrashing(tmp_path):
    budget, count = len(encode(payload())) * 2, 8
    measurements = {}
    for compression in (False, True):
        path = tmp_path / (str(compression) + ".sqlite3")
        builds, values = [], []
        for day in range(2):
            with SQLiteRawParseCache(path, max_bytes=budget, compression=compression) as disk:
                current = []
                for index in range(count):
                    def compute():
                        builds.append((day, index))
                        return payload(index)
                    current.append(disk.memo("ixbrl", (KEY[0], str(index)), compute))
                assert disk.bytes <= budget
                if day == 1:
                    hits = disk.counts["raw_parse_disk_hits"]
            values.append(current)
        assert values[0] == values[1]
        measurements[compression] = (len(builds), hits, values[0])
    assert measurements[False][:2] == (count * 2, 0)
    assert measurements[True][:2] == (count, count)
    assert measurements[False][2] == measurements[True][2]


def test_legacy_text_keeps_original_hash_and_is_reused_without_rewriting(tmp_path):
    expected = payload()
    with SQLiteRawParseCache(tmp_path / "legacy.sqlite3") as disk:
        raw = encode(expected).decode()
        with disk.connection:
            disk.connection.execute("INSERT INTO research_parses VALUES (?,?,?,?)",
                (*KEY, raw, hashlib.sha256(raw.encode()).hexdigest()))
        disk.bytes = len(raw.encode())
        assert disk.memo("ixbrl", KEY, reject_compute) == expected
        stored = disk.connection.execute("SELECT facts_json,typeof(facts_json) FROM research_parses").fetchone()
        assert stored == (raw, "text")


@pytest.mark.parametrize("damage", [
    "magic", "header", "body", "truncated", "trailing", "declared_short",
    "declared_long", "declared_over_limit", "inflate_bomb", "sha", "utf8", "json", "not_list", "oversized_text",
])
def test_corrupt_cache_is_recomputed_repaired_and_byte_budget_reconciled(tmp_path, damage):
    expected = [{"facts": "SYNTHETIC" * 180}]
    raw = encode(expected)
    valid_sha = hashlib.sha256(raw).hexdigest()
    blob, sha = packet(raw), valid_sha
    if damage == "magic":
        blob = b"UNKNOWN_CODEC" + blob
    elif damage == "header":
        blob = cache_module._RAW_ZLIB_MAGIC + b"\x00\x00\x00"
    elif damage == "body":
        blob = blob[:len(cache_module._RAW_ZLIB_MAGIC) + 8] + b"NOT_ZLIB"
    elif damage == "truncated":
        blob = blob[:-3]
    elif damage == "trailing":
        blob += b"TRAILING"
    elif damage == "declared_short":
        blob = packet(raw, declared=len(raw) - 1)
    elif damage == "declared_long":
        blob = packet(raw, declared=len(raw) + 1)
    elif damage == "declared_over_limit":
        blob = packet(raw, declared=4097)
    elif damage == "inflate_bomb":
        blob = packet(b"x" * (2 * 1024 * 1024), declared=64)
    elif damage == "sha":
        sha = "0" * 64
    elif damage in {"utf8", "json", "not_list"}:
        invalid = {"utf8": b"\xff", "json": b"[", "not_list": b"{}"}[damage]
        blob, sha = packet(invalid), hashlib.sha256(invalid).hexdigest()
    else:
        blob = "x" * 5000
    with SQLiteRawParseCache(tmp_path / "repair.sqlite3", max_bytes=8192, max_decoded_bytes=4096) as disk:
        with disk.connection:
            disk.connection.execute("INSERT INTO research_parses VALUES (?,?,?,?)", (*KEY, blob, sha))
        # Simulates external corruption while open: old owner byte count is
        # deliberately stale; the repair must re-establish exact accounting.
        assert disk.bytes == 0
        calls = []
        assert disk.memo("ixbrl", KEY, lambda: calls.append(1) or expected) == expected
        assert calls == [1]
        assert disk.stats["raw_parse_disk_miss_corrupt"] == 1
        assert disk.memo("ixbrl", KEY, reject_compute) == expected
        total = disk.connection.execute("SELECT SUM(length(CAST(facts_json AS BLOB))) FROM research_parses").fetchone()[0]
        assert disk.bytes == total and 0 < disk.bytes <= disk.max_bytes


def test_declared_over_limit_is_rejected_before_decompressor_creation(tmp_path, monkeypatch):
    invalid = packet(b"[]", declared=1025)
    with SQLiteRawParseCache(tmp_path / "bounded.sqlite3", max_decoded_bytes=1024) as disk:
        with disk.connection:
            disk.connection.execute("INSERT INTO research_parses VALUES (?,?,?,?)", (*KEY, invalid, "bad"))
        monkeypatch.setattr(cache_module.zlib, "decompressobj", lambda: pytest.fail("oversized header inflated"))
        assert disk.memo("ixbrl", KEY, lambda: []) == []
        assert disk.stats["raw_parse_disk_miss_corrupt"] == 1


@pytest.mark.parametrize("budget", [0, 1, 30])
def test_zero_tiny_capacity_bypasses_without_partial_rows_or_eviction(tmp_path, budget):
    expected = payload()
    with SQLiteRawParseCache(tmp_path / "tiny.sqlite3", max_bytes=budget) as disk:
        assert disk.memo("ixbrl", KEY, lambda: expected) == expected
        assert disk.bytes == 0
        assert disk.connection.execute("SELECT COUNT(*) FROM research_parses").fetchone()[0] == 0
        assert disk.counts["raw_parse_disk_evictions"] == 0
        assert disk.counts["raw_parse_disk_oversized_bypass"] == 1


def test_small_uncompressed_and_decoded_oversized_bypass_preserve_existing_rows(tmp_path):
    with SQLiteRawParseCache(tmp_path / "small.sqlite3", max_bytes=500, max_decoded_bytes=2048) as disk:
        expected = [{"small": "SYNTHETIC"}]
        disk.memo("ixbrl", KEY, lambda: expected)
        stored = disk.connection.execute("SELECT typeof(facts_json) FROM research_parses").fetchone()[0]
        assert stored == "text"
        too_big = payload()
        assert disk.memo("ixbrl", (KEY[0], "large"), lambda: too_big) == too_big
        assert disk.stats["source_cache_decoded_oversized_bypass"] == 1
        assert disk.memo("ixbrl", KEY, reject_compute) == expected
        assert disk.counts["raw_parse_disk_evictions"] == 0


def test_unprofitable_compression_keeps_plain_json_not_an_unknown_blob(tmp_path, monkeypatch):
    expected = payload()
    monkeypatch.setattr(cache_module.zlib, "compress", lambda data, **kwargs: data + b"EXPANDED")
    with SQLiteRawParseCache(tmp_path / "unprofitable.sqlite3") as disk:
        assert disk.memo("ixbrl", KEY, lambda: expected) == expected
        row = disk.connection.execute("SELECT facts_json,typeof(facts_json) FROM research_parses").fetchone()
        assert row == (encode(expected).decode(), "text")
        assert disk.counts["source_cache_compressed_admissions"] == 0
        assert disk.memo("ixbrl", KEY, reject_compute) == expected


def test_oversized_repair_removes_only_corrupt_entry_and_keeps_other_cache_rows(tmp_path):
    with SQLiteRawParseCache(tmp_path / "oversized-repair.sqlite3", max_bytes=500, max_decoded_bytes=2048) as disk:
        small = [{"valid": "SYNTHETIC"}]
        disk.memo("ixbrl", (KEY[0], "keep"), lambda: small)
        with disk.connection:
            disk.connection.execute("INSERT INTO research_parses VALUES (?,?,?,?)", (*KEY, b"BAD", "bad"))
        assert disk.memo("ixbrl", KEY, payload) == payload()
        assert disk.memo("ixbrl", (KEY[0], "keep"), reject_compute) == small
        rows = disk.connection.execute("SELECT content_sha256,length(CAST(facts_json AS BLOB)) FROM research_parses").fetchall()
        assert rows == [("keep", len(encode(small)))]
        assert disk.bytes == len(encode(small))
        assert disk.counts["source_cache_decoded_oversized_bypass"] == 1
        assert disk.counts["raw_parse_disk_evictions"] == 0


def test_different_original_parser_and_companion_namespace_never_share_values(tmp_path):
    with SQLiteRawParseCache(tmp_path / "keys.sqlite3") as disk:
        keys = [("ixbrl", KEY), ("ixbrl", (KEY[0], "other-original")),
                ("ixbrl", ("new-parser", KEY[1])), ("financial_companion", KEY)]
        for ordinal, (kind, key) in enumerate(keys):
            assert disk.memo(kind, key, lambda: payload(ordinal)) == payload(ordinal)
        for ordinal, (kind, key) in enumerate(keys):
            assert disk.memo(kind, key, reject_compute) == payload(ordinal)
        assert disk.connection.execute("SELECT COUNT(*) FROM research_parses").fetchone()[0] == 4


def test_indexed_research_facts_remain_text_and_raw_cache_rejects_that_database(tmp_path):
    source = history()
    indexed = adapter(tmp_path, source)
    at = aware_timestamp("2026-09-11T13:30:00+08:00")
    result = indexed.load(source, ("2330",), at)
    with sqlite3.connect(indexed.path) as connection:
        assert set(row[0] for row in connection.execute("SELECT typeof(facts_json) FROM research_parses")) == {"text"}
    before = indexed.path.read_bytes()
    with pytest.raises(ValueError, match="RAW_PARSE_CACHE_MUST_NOT_USE_INDEXED_RESEARCH_DATABASE"):
        with SQLiteRawParseCache(indexed.path):
            pytest.fail("indexed history database accepted as disposable cache")
    assert indexed.path.read_bytes() == before
    assert adapter(tmp_path, source).load(source, ("2330",), at) == result


def test_indexed_history_cannot_adopt_raw_compressed_database(tmp_path):
    source = history()
    indexed = adapter(tmp_path, source)
    with SQLiteRawParseCache(indexed.path) as raw:
        raw.memo("ixbrl", KEY, payload)
    before = indexed.path.read_bytes()
    with pytest.raises(ValueError, match="INDEXED_RESEARCH.*RAW|RAW.*INDEXED_RESEARCH"):
        indexed.load(source, ("2330",), aware_timestamp("2026-09-11T13:30:00+08:00"))
    assert indexed.path.read_bytes() == before


@pytest.mark.parametrize("kwargs", [
    {"compression": "yes"}, {"max_decoded_bytes": 0}, {"max_decoded_bytes": -1},
    {"max_decoded_bytes": True}, {"max_decoded_bytes": 1.5},
])
def test_invalid_codec_configuration_is_explicit(tmp_path, kwargs):
    with pytest.raises(ValueError, match="INVALID_RAW_PARSE_CODEC_CONFIGURATION"):
        SQLiteRawParseCache(tmp_path / "invalid.sqlite3", **kwargs)

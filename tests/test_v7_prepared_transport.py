"""Lossless transport only: no feature, label, prediction or gate alteration."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from domain.les_v7.contracts import canonical, identity
from infrastructure.storage.les_v7_prepared import (
    inspect_prepared_packet, read_prepared_packet, write_prepared_packet,
)
from infrastructure.storage.les_v7_training import MAX_DATASET_BYTES


def packet(version=2):
    value = dict(schema=f"les-v7-native-prepared-training-shard/v{version}",
        dataset_id="SYNTHETIC", config_id="SYNTHETIC", diagnostics=[],
        rows=[dict(row_id=str(i), values=[1., -0., None, True, "測試"],
                   labels=dict(fill=i % 2, rebound=None, joint=None)) for i in range(30)])
    if version == 2:
        value["feature_columns"] = ["close", "dfa512"]
    return value


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_legacy_and_columnar_packet_identity_and_bits_survive_transport(tmp_path, version, suffix):
    original = packet(version)
    logical = canonical(original)
    path = tmp_path / ("prepared" + suffix)
    written = write_prepared_packet(path, original)
    loaded = inspect_prepared_packet(path)
    assert read_prepared_packet(path) == original
    assert canonical(loaded.value) == logical
    assert identity(loaded.value) == identity(original)
    assert loaded.encoded_bytes == written.encoded_bytes == path.stat().st_size
    assert loaded.decoded_bytes == written.decoded_bytes == len(logical.encode())
    if suffix.endswith(".gz"):
        assert loaded.encoded_bytes < loaded.decoded_bytes
        assert gzip.decompress(path.read_bytes()) == logical.encode()


def test_deterministic_gzip_has_no_wall_time_or_filename_in_identity(tmp_path):
    first, second = tmp_path / "first.json.gz", tmp_path / "second.json.gz"
    write_prepared_packet(first, packet()); write_prepared_packet(second, packet())
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[4:8] == b"\x00\x00\x00\x00"


def test_atomic_retry_is_idempotent_and_conflicting_write_preserves_original(tmp_path):
    path = tmp_path / "prepared.json.gz"
    original = packet()
    write_prepared_packet(path, original)
    encoded = path.read_bytes()
    write_prepared_packet(path, original)
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        write_prepared_packet(path, dict(original, config_id="CHANGED"))
    assert path.read_bytes() == encoded
    assert not list(tmp_path.glob(".publish-*"))


@pytest.mark.parametrize("removed", [1, 4, 8, 20])
def test_truncated_gzip_is_rejected(tmp_path, removed):
    path = tmp_path / "truncated.json.gz"
    path.write_bytes(gzip.compress(canonical(packet()).encode(), mtime=0)[:-removed])
    with pytest.raises(ValueError, match="GZIP_INTEGRITY"):
        read_prepared_packet(path)


def test_payload_and_checksum_corruption_are_rejected(tmp_path):
    path = tmp_path / "damaged.json.gz"
    encoded = bytearray(gzip.compress(canonical(packet()).encode(), mtime=0))
    encoded[-8] ^= 1
    path.write_bytes(encoded)
    with pytest.raises(ValueError, match="GZIP_INTEGRITY"):
        read_prepared_packet(path)


def test_small_compression_bomb_is_bounded_before_json_parse(tmp_path, monkeypatch):
    path = tmp_path / "bomb.json.gz"
    path.write_bytes(gzip.compress(b'{"bomb":"' + b"x" * 8192 + b'"}', mtime=0))
    assert path.stat().st_size < 128
    monkeypatch.setattr(json, "loads", lambda *_: pytest.fail("oversized decoded JSON parsed"))
    with pytest.raises(ValueError, match="DECODED_SIZE_LIMIT"):
        read_prepared_packet(path, max_bytes=128)


def test_gzip_reader_requests_only_limit_plus_one_decoded_bytes(tmp_path, monkeypatch):
    path = tmp_path / "bounded.json.gz"
    path.write_bytes(gzip.compress(b'{"x":"' + b"x" * 8192 + b'"}', mtime=0))
    sizes, read = [], gzip.GzipFile.read
    def observed(stream, size=-1):
        sizes.append(size)
        return read(stream, size)
    monkeypatch.setattr(gzip.GzipFile, "read", observed)
    with pytest.raises(ValueError, match="DECODED_SIZE_LIMIT"):
        read_prepared_packet(path, max_bytes=128)
    assert sizes == [129]


def test_concatenated_gzip_members_cannot_bypass_decoded_bound(tmp_path):
    path = tmp_path / "members.json.gz"
    member = gzip.compress(b" " * 100, mtime=0)
    path.write_bytes(member + member)
    with pytest.raises(ValueError, match="DECODED_SIZE_LIMIT"):
        read_prepared_packet(path, max_bytes=128)


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_encoded_bound_is_independent_from_decoded_bound(tmp_path, suffix):
    path = tmp_path / ("oversized" + suffix)
    path.write_bytes(b"x" * 129)
    with pytest.raises(ValueError, match="ENCODED_SIZE_LIMIT"):
        read_prepared_packet(path, max_bytes=128)


def test_read_limit_catches_growth_even_if_initial_stat_was_small(tmp_path, monkeypatch):
    path = tmp_path / "growth.json"
    path.write_bytes(b"x" * 129)
    monkeypatch.setattr(Path, "stat", lambda *_args, **_kwargs: SimpleNamespace(st_size=1))
    with pytest.raises(ValueError, match="ENCODED_SIZE_LIMIT"):
        read_prepared_packet(path, max_bytes=128)


def test_exact_decoded_limit_is_allowed_and_oversized_write_not_published(tmp_path):
    value = dict(x="a" * 120)
    limit = len(canonical(value).encode())
    path = tmp_path / "exact.json.gz"
    write_prepared_packet(path, value, max_bytes=limit)
    assert read_prepared_packet(path, max_bytes=limit) == value
    rejected = tmp_path / "rejected.json.gz"
    with pytest.raises(ValueError, match="DECODED_SIZE_LIMIT"):
        write_prepared_packet(rejected, value, max_bytes=limit - 1)
    assert not rejected.exists()


@pytest.mark.parametrize("value", [[], None, 1, "text"])
def test_nonobject_json_is_rejected(tmp_path, value):
    path = tmp_path / "nonobject.json.gz"
    path.write_bytes(gzip.compress(json.dumps(value).encode(), mtime=0))
    with pytest.raises(ValueError, match="JSON_OBJECT_REQUIRED"):
        read_prepared_packet(path)


def test_nonfinite_json_and_malformed_utf8_are_rejected(tmp_path):
    path = tmp_path / "invalid.json.gz"
    path.write_bytes(gzip.compress(b'{"x": NaN}', mtime=0))
    with pytest.raises(ValueError):
        read_prepared_packet(path)
    path.write_bytes(gzip.compress(b'{"x":"\xff"}', mtime=0))
    with pytest.raises(ValueError, match="JSON_INVALID"):
        read_prepared_packet(path)


@pytest.mark.parametrize("limit", [0, -1, True, MAX_DATASET_BYTES + 1])
def test_caller_cannot_disable_artifact_bound(tmp_path, limit):
    with pytest.raises(ValueError, match="BOUNDED_ARTIFACT_LIMIT"):
        write_prepared_packet(tmp_path / "bad.json.gz", {}, max_bytes=limit)


def test_repacked_logical_tamper_changes_existing_canonical_identity(tmp_path):
    original = packet(); expected = identity(original)
    path = tmp_path / "logical-tamper.json.gz"
    changed = dict(original, config_id="CHANGED")
    write_prepared_packet(path, changed)
    # Archive consumers retain their existing canonical SHA comparison.
    assert identity(read_prepared_packet(path)) != expected

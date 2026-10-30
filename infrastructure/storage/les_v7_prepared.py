"""Lossless, bounded prepared-shard transport; logical JSON identity is unchanged.

New writers use deterministic gzip over canonical JSON. Existing plain JSON
archives are read in place. Compression is storage only, never a new feature,
label, permission or evidence identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import io
import json
from pathlib import Path
import zlib

from domain.les_v7.contracts import canonical
from infrastructure.research_archive import write_bytes_once
from infrastructure.storage.les_v7_training import MAX_DATASET_BYTES


@dataclass(frozen=True)
class PreparedPacket:
    value: dict
    encoded_bytes: int
    decoded_bytes: int


def _limit(max_bytes):
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_DATASET_BYTES:
        raise ValueError("PREPARED_BOUNDED_ARTIFACT_LIMIT_REQUIRED")
    return max_bytes


def inspect_prepared_packet(path, *, max_bytes=MAX_DATASET_BYTES):
    """Read at most limit+1 encoded AND decoded bytes, including gzip bombs."""
    path, limit = Path(path), _limit(max_bytes)
    if path.stat().st_size > limit:
        raise ValueError("PREPARED_ENCODED_SIZE_LIMIT")
    with path.open("rb") as stream:
        encoded = stream.read(limit + 1)
    if len(encoded) > limit:
        raise ValueError("PREPARED_ENCODED_SIZE_LIMIT")
    if path.suffix == ".gz":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(encoded), mode="rb") as stream:
                decoded = stream.read(limit + 1)
        except (OSError, EOFError, zlib.error) as error:
            raise ValueError("PREPARED_GZIP_INTEGRITY_INVALID") from error
    else:
        decoded = encoded
    if len(decoded) > limit:
        raise ValueError("PREPARED_DECODED_SIZE_LIMIT")
    try:
        packet = json.loads(decoded)
    except (ValueError, UnicodeError) as error:
        raise ValueError("PREPARED_JSON_INVALID") from error
    if not isinstance(packet, dict):
        raise ValueError("PREPARED_JSON_OBJECT_REQUIRED")
    canonical(packet)  # Reject nonfinite numbers before a caller binds identity.
    return PreparedPacket(packet, len(encoded), len(decoded))


def read_prepared_packet(path, *, max_bytes=MAX_DATASET_BYTES):
    return inspect_prepared_packet(path, max_bytes=max_bytes).value


def write_prepared_packet(path, packet, *, max_bytes=MAX_DATASET_BYTES):
    """Atomically publish canonical JSON or deterministic gzip, without clobber."""
    path, limit = Path(path), _limit(max_bytes)
    if not isinstance(packet, dict):
        raise ValueError("PREPARED_JSON_OBJECT_REQUIRED")
    decoded = canonical(packet).encode("utf-8")
    if len(decoded) > limit:
        raise ValueError("PREPARED_DECODED_SIZE_LIMIT")
    encoded = gzip.compress(decoded, compresslevel=6, mtime=0) if path.suffix == ".gz" else decoded
    if len(encoded) > limit:
        raise ValueError("PREPARED_ENCODED_SIZE_LIMIT")
    write_bytes_once(path, encoded)
    return PreparedPacket(packet, len(encoded), len(decoded))

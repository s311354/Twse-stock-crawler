"""Integrity-checked annual JSON observations for current-analysis fallback.

These are lossless canonical captures from an existing immutable dataset, not
transport bytes or proof of the historical publication date. Strict PIT replay
must continue to verify its own cutoff-bound annual calendar evidence.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path


SCHEMA = "twse-observed-annual-calendar/v1"
USE_POLICY = "OBSERVED_CURRENT_CALENDAR_ONLY_NOT_PIT"
MANIFEST_SHA256 = "4d9a10378518d3cd0c7f8dac6c501f7757b2a4f79cc3aef9ebbd2625ae1a4e8d"
ROOT = Path(__file__).with_name("evidence")
SOURCE_URL = "https://www.twse.com.tw/holidaySchedule/holidaySchedule"
MAX_BYTES = 512 * 1024


def read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("ANNUAL_CALENDAR_ARCHIVE_TOO_LARGE")
    return raw


class ObservedAnnualSchedules:
    def __init__(self, root: Path = ROOT, *, expected_sha256=MANIFEST_SHA256,
                 cutoff: dt.datetime | None = None):
        self.root = Path(root)
        self.expected_sha256 = expected_sha256
        self.cutoff = cutoff or dt.datetime.now(dt.timezone.utc)
        if self.cutoff.tzinfo is None:
            raise ValueError("CALENDAR_CUTOFF_MUST_BE_AWARE")
        self._manifest = None

    def _load(self):
        if self._manifest is None:
            raw = read_bounded(self.root / "annual-observations.json", 64 * 1024)
            if hashlib.sha256(raw).hexdigest() != self.expected_sha256:
                raise ValueError("ANNUAL_CALENDAR_MANIFEST_HASH_MISMATCH")
            manifest = json.loads(raw)
            if (manifest.get("schema") != SCHEMA or manifest.get("source") != "TWSE_OFFICIAL"
                    or manifest.get("use_policy") != USE_POLICY
                    or not isinstance(manifest.get("documents"), dict)):
                raise ValueError("INVALID_ANNUAL_CALENDAR_ARCHIVE")
            self._manifest = manifest
        return self._manifest

    def get(self, year: int):
        """Return a verified observation or None; never infer missing years."""
        document = self._load()["documents"].get(str(year))
        if document is None:
            return None
        sha = document.get("sha256", "")
        origin = document.get("origin_evidence_id", "")
        if (not re.fullmatch(r"[0-9a-f]{64}", sha)
                or not re.fullmatch(r"[0-9a-f]{64}", origin)
                or document.get("source_url") != SOURCE_URL
                or document.get("availability_policy") != "ORIGINAL_PUBLICATION_UNVERIFIED"
                or document.get("representation") != "CANONICAL_JSON_CAPTURE_NOT_TRANSPORT_BYTES"):
            raise ValueError("INVALID_ANNUAL_CALENDAR_PROVENANCE")
        retrieved = dt.datetime.fromisoformat(document["retrieved_at"])
        if retrieved.tzinfo is None:
            raise ValueError("ANNUAL_CALENDAR_RETRIEVED_AT_MUST_BE_AWARE")
        if retrieved > self.cutoff:
            return None
        raw = read_bounded(self.root / "objects" / sha, MAX_BYTES)
        if len(raw) != document.get("bytes") or hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("ANNUAL_CALENDAR_OBJECT_HASH_MISMATCH")
        payload = json.loads(raw)
        if (not isinstance(payload, dict) or payload.get("date") != f"{year:04d}0101"
                or payload.get("queryYear") != year
                or payload.get("title") != f"{year - 1911} 年市場開休市日期"):
            raise ValueError("ANNUAL_CALENDAR_YEAR_BINDING_MISMATCH")
        return payload, dict(document, year=year, source="VERIFIED_OBSERVED_ARCHIVE",
                             use_policy=USE_POLICY, manifest_sha256=self.expected_sha256,
                             verified_at=dt.datetime.now(dt.timezone.utc).isoformat())

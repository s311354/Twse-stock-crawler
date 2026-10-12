"""Reviewed official exceptions; raw PDF references are pinned by code version.

This small supplement is not a complete historical announcement archive. It
does not infer closures from missing bars, HTTP failures or weekday rules.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

from application.research_evidence import calendar_at, verify_publication


SCHEMA = "twse-reviewed-emergency-calendar/v1"
# Changing this reviewed manifest changes current_versions()'s source hash.
MANIFEST_SHA256 = "51724ef6ef8829d8f7b5d1fcad33c78cbd97f9156da1f829e425ce5bed6ae47d"
EVIDENCE_ROOT = Path(__file__).with_name("evidence")


class ReviewedEmergencyAnnouncements:
    def __init__(self, root: Path = EVIDENCE_ROOT, *,
                 expected_sha256: str = MANIFEST_SHA256,
                 cutoff: dt.datetime | None = None):
        self.root = Path(root)
        self.expected_sha256 = expected_sha256
        self.cutoff = cutoff or dt.datetime.now(dt.timezone.utc)
        if self.cutoff.tzinfo is None:
            raise ValueError("CALENDAR_CUTOFF_MUST_BE_AWARE")
        self._payload = None
        self._verified = None

    def _read(self, reference):
        return (self.root / "objects" / reference["sha256"]).read_bytes()

    def _load(self):
        if self._payload is None:
            raw = (self.root / "announcements.json").read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.expected_sha256:
                raise ValueError("CALENDAR_ANNOUNCEMENT_MANIFEST_HASH_MISMATCH")
            payload = json.loads(raw)
            if payload.get("schema") != SCHEMA or payload.get("source") != "TWSE_OFFICIAL":
                raise ValueError("INVALID_CALENDAR_ANNOUNCEMENT_SCHEMA")
            if not isinstance(payload.get("announcements"), list):
                raise ValueError("CALENDAR_ANNOUNCEMENTS_REQUIRED")
            self._payload = payload
        return self._payload

    def status(self, date: dt.date) -> str | None:
        payload = self._load()
        events = [e for e in payload["announcements"] if e.get("date") == date.isoformat()]
        if not events:
            return None
        days, excluded = calendar_at(
            dict(source="TWSE_OFFICIAL", days={}, announcements=events),
            self.cutoff, strict=False,
            verifier=lambda proof, **kwargs: verify_publication(
                proof, document_reader=self._read, **kwargs),
        )
        # Invalid evidence must not fall through to the annual weekday default.
        if any(e["reason"] != "CALENDAR_ANNOUNCEMENT_AFTER_CUTOFF" for e in excluded):
            return "UNKNOWN"
        return days.get(date.isoformat())

    def evidence(self) -> dict:
        """Verified provenance for freezing a current-analysis job, no PDF copies."""
        if self._verified is None:
            payload = self._load()
            entries = []
            for event in payload["announcements"]:
                binding = dict(kind="CALENDAR_EXCEPTION", market=event["market"],
                               date=event["date"], status=event["status"])
                if event["market"] != "TWSE" or event["status"] not in {"OPEN", "CLOSED", "UNKNOWN"}:
                    raise ValueError("INVALID_CALENDAR_ANNOUNCEMENT_MARKET_OR_STATUS")
                available = verify_publication(event["proof"], binding=binding,
                                               document_reader=self._read)
                entries.append(dict(date=event["date"], market=event["market"], status=event["status"],
                                    availableAt=available, proof=event["proof"]))
            self._verified = dict(schema=SCHEMA, source="TWSE_OFFICIAL", manifestSha256=self.expected_sha256,
                                  coverage=payload.get("coverage"), cutoff=self.cutoff.isoformat(),
                                  announcements=entries)
        return copy.deepcopy(self._verified)

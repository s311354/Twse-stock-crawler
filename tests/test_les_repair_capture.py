"""Synthetic source transport only; never official restriction clearance."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from application.replay_contracts import digest
from examples.capture_les_repair_sources import capture


class Transport:
    def __init__(self, payloads): self.calls = []; self.payloads = iter(payloads)
    def request_bytes(self, method, url, *, context):
        self.calls.append((method, url))
        return SimpleNamespace(content=json.dumps(next(self.payloads)).encode(), status_code=200, attempts=1)


def test_three_raw_sources_shared_by_whole_scope_and_completion_time_is_not_backdated(tmp_path):
    early = datetime(2026, 10, 28, 13, tzinfo=timezone.utc)
    late = datetime(2026, 10, 28, 13, 0, 4, tzinfo=timezone.utc)
    clocks = iter([early, early, early, early, late])
    client = Transport([[], [{"Code": "TEST1"}], []])
    root = tmp_path / "capture"
    result = capture(["TEST1", "TEST2"], root, transport=client, clock=lambda: next(clocks))
    assert len(client.calls) == 3
    assert result["by_symbol"]["TEST1"]["restriction_evidence"]["restricted"]
    assert not result["by_symbol"]["TEST2"]["restriction_evidence"]["restricted"]
    assert result["captured_at"] == late.isoformat()
    for node in result["by_symbol"].values():
        assert node["restriction_evidence"]["available_at"] == late.isoformat()
    for doc in result["metadata"]["documents"]:
        raw = (root / doc["path"]).read_bytes()
        assert sha256(raw).hexdigest() == doc["content_sha256"]
        assert digest(json.loads(raw)) == doc["payload_sha256"]
    assert digest({k: v for k, v in result.items() if k != "content_sha256"}) == result["content_sha256"]
    with pytest.raises(FileExistsError): capture(["TEST1"], root, transport=client)


def test_partial_or_invalid_official_response_never_publishes_complete_override(tmp_path):
    at = datetime(2026, 10, 28, 13, tzinfo=timezone.utc)
    root = tmp_path / "capture"
    with pytest.raises(Exception, match="Invalid trading restrictions"):
        capture(["TEST1"], root, transport=Transport([[], {}]), clock=lambda: at)
    assert not (root / "source-overrides.json").exists()

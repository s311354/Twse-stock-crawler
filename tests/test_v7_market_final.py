"""Synthetic transport/publication regressions, not historical-price proof."""
from datetime import date
import json
from types import SimpleNamespace

import pytest

from domain.errors import AccessBlockedError
from infrastructure.crawler.twse_http_client import TwseBinaryResponse
from infrastructure.les_v7_market_final import (
    STOCK_FIELDS, collect_final_market, parse_final_market, load_final_market, final_market_records,
)


def content(day):
    return json.dumps(dict(stat="OK", date=day.replace("-", ""),
        params=dict(date=day.replace("-", ""), type="ALLBUT0999"), tables=[
            dict(fields=STOCK_FIELDS, data=[["3014", "SYNTHETIC_TEST_ONLY", "1000", "1", "100000", "100", "101", "99", "100", "+", "0", "99.9", "1", "100", "1", ""]]),
            dict(fields=["指數", "收盤指數"], data=[["發行量加權股價指數", "10000"]])])).encode()


class Client:
    config = SimpleNamespace(allow_insecure_ssl_fallback=False, min_request_interval_seconds=1)
    def __init__(self, error=None):
        self.calls = []
        self.error = error
    def request_bytes(self, method, url, *, params, before_attempt, **kwargs):
        before_attempt()
        self.calls.append(params)
        if self.error:
            raise self.error
        raw = params["date"]
        return TwseBinaryResponse(content(raw[:4]+"-"+raw[4:6]+"-"+raw[6:]), 200, 1)


def calendar(monkeypatch):
    import infrastructure.les_v7_calendar_final as cal
    monkeypatch.setattr(cal, "load_final_calendar", lambda _: ({"SYNTHETIC_TEST_ONLY": True}, None))
    monkeypatch.setattr(cal, "select_final_calendar", lambda *a, **k: (
        dict(days={"2026-01-05":"OPEN", "2026-01-06":"OPEN", "2026-01-07":"UNKNOWN"}), {}))


def collect(tmp_path, monkeypatch, client, **kwargs):
    calendar(monkeypatch)
    return collect_final_market("SYNTHETIC-calendar", start=date(2026,1,5), end=date(2026,1,7),
        root=tmp_path, client=client, now=lambda:"2026-10-13T00:00:00+00:00", **kwargs)


def test_parse_exact_source_preserves_native_units_and_caller_bytes():
    raw = content("2026-01-05")
    rows, index = parse_final_market(raw, "2026-01-05")
    assert rows[0][8] == "100" and rows[0][2] == "1000" and index == 10000.
    assert raw == content("2026-01-05")
    with pytest.raises(ValueError, match="DATE_SCHEMA"):
        parse_final_market(raw, "2026-01-06")


def test_swapped_price_and_volume_columns_rejected():
    payload = json.loads(content("2026-01-05"))
    payload["tables"][0]["fields"][2], payload["tables"][0]["fields"][8] = "收盤價", "成交股數"
    with pytest.raises(ValueError, match="COLUMN_MAPPING"):
        parse_final_market(json.dumps(payload).encode(), "2026-01-05")


def test_completed_capture_reuses_raw_bytes_without_http_and_keeps_unknown(tmp_path, monkeypatch):
    first = Client()
    path = collect(tmp_path, monkeypatch, first)
    body, docs = load_final_market(path)
    assert len(first.calls) == 2 and body["status"] == "PARTIAL"
    assert body["gaps"] == [dict(date="2026-01-07", reason="CALENDAR_UNKNOWN")]
    assert all(x["historical_available_at"] is None for x in body["documents"])
    again = Client(AssertionError("SYNTHETIC redundant download"))
    assert collect(tmp_path, monkeypatch, again) == path and not again.calls
    rows = final_market_records(body, docs, ["3014"])
    assert len(rows) == 4 and rows[0]["data"][0][8] == "100"


def test_request_budget_and_access_block_do_not_become_closed_or_complete(tmp_path, monkeypatch):
    path = collect(tmp_path, monkeypatch, Client(), max_requests=1)
    body, _ = load_final_market(path)
    assert len(body["documents"]) == 1 and body["status"] == "PARTIAL"
    assert any(x["reason"] == "REQUEST_BUDGET_OR_CONSECUTIVE_FAILURE_LIMIT" for x in body["gaps"])
    blocked = Client(AccessBlockedError("SYNTHETIC ACCESS_BLOCKED", retryable=False))
    path = collect(tmp_path / "blocked", monkeypatch, blocked)
    body, _ = load_final_market(path)
    assert len(blocked.calls) == 1 and not body["documents"] and body["status"] == "PARTIAL"


def test_raw_source_corruption_is_not_refetched_or_ignored(tmp_path, monkeypatch):
    path = collect(tmp_path, monkeypatch, Client())
    body, docs = load_final_market(path)
    docs.objects.path(body["documents"][0]["document"]["sha256"]).write_bytes(b"SYNTHETIC_CORRUPTION")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        load_final_market(path)


def test_manual_entry_requires_explicit_optin_before_fetch_or_publication(tmp_path):
    from interface.les_v7_evidence_repair import main
    with pytest.raises(SystemExit) as error:
        main(["calendar", "--output-dir", str(tmp_path)])
    assert error.value.code == 2 and not list(tmp_path.iterdir())


def test_range_expansion_reuses_verified_parent_without_redownload(tmp_path, monkeypatch):
    calendar(monkeypatch)
    import infrastructure.les_v7_calendar_final as cal
    monkeypatch.setattr(cal, "select_final_calendar", lambda *a, **k: (
        dict(days={d:s for d,s in {"2026-01-05":"OPEN", "2026-01-06":"OPEN", "2026-01-07":"UNKNOWN"}.items()
                   if k["start"].isoformat() <= d <= k["end"].isoformat()}), {}))
    first = Client()
    parent = collect_final_market("SYNTHETIC-calendar", start=date(2026,1,5), end=date(2026,1,5),
        root=tmp_path / "first", client=first, now=lambda:"2026-10-13T00:00:00+00:00")
    earlier, _ = load_final_market(parent)
    original = parent.read_bytes()
    client = Client()
    expanded = collect_final_market("SYNTHETIC-calendar", start=date(2026,1,5), end=date(2026,1,7),
        root=tmp_path / "second", client=client, previous=parent, now=lambda:"2026-10-13T00:00:00+00:00")
    body, _ = load_final_market(expanded)
    assert len(client.calls) == 1 and client.calls[0]["date"] == "20260106"
    assert body["parent_manifest_id"]
    assert body["documents"][0] == earlier["documents"][0]
    assert parent.read_bytes() == original
    assert body["gaps"] == [dict(date="2026-01-07", reason="CALENDAR_UNKNOWN")]


def test_expansion_does_not_hide_parent_corruption_with_fresh_http(tmp_path, monkeypatch):
    parent = collect(tmp_path / "first", monkeypatch, Client())
    body, docs = load_final_market(parent)
    docs.objects.path(body["documents"][0]["document"]["sha256"]).write_bytes(b"SYNTHETIC_BAD")
    client = Client()
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        collect(tmp_path / "second", monkeypatch, client, previous=parent)
    assert not client.calls and not list((tmp_path / "second").glob("manifests/*/manifest.json"))


def test_new_calendar_rechecks_dates_instead_of_reusing_parent_open_claim(tmp_path, monkeypatch):
    parent = collect(tmp_path / "first", monkeypatch, Client())
    earlier, _ = load_final_market(parent)
    import infrastructure.les_v7_calendar_final as cal
    monkeypatch.setattr(cal, "load_final_calendar", lambda _: ({"SYNTHETIC_TEST_ONLY": "NEW_VERSION"}, None))
    monkeypatch.setattr(cal, "select_final_calendar", lambda *a, **k: (
        dict(days={"2026-01-05":"CLOSED", "2026-01-06":"OPEN", "2026-01-07":"UNKNOWN"}), {}))
    client = Client(AssertionError("SYNTHETIC_REDUNDANT_REQUEST"))
    path = collect_final_market("SYNTHETIC-calendar", start=date(2026,1,5), end=date(2026,1,7),
        root=tmp_path / "second", client=client, previous=parent, now=lambda:"2026-10-13T00:00:00+00:00")
    body, _ = load_final_market(path)
    assert body["calendar_sha256"] != earlier["calendar_sha256"]
    assert body["parent_calendar_sha256"] == earlier["calendar_sha256"]
    assert [row["date"] for row in body["documents"]] == ["2026-01-06"]
    assert body["gaps"] == [dict(date="2026-01-07", reason="CALENDAR_UNKNOWN")]
    assert not client.calls


def test_unchanged_previous_capture_returns_same_manifest_without_http_or_lineage_growth(tmp_path, monkeypatch):
    calendar(monkeypatch)
    import infrastructure.les_v7_calendar_final as cal
    monkeypatch.setattr(cal, 'select_final_calendar', lambda *a, **k: (
        dict(days={'2026-01-05': 'OPEN', '2026-01-06': 'OPEN', '2026-01-07': 'CLOSED'}), {}))
    first = collect_final_market('SYNTHETIC-calendar', start=date(2026, 1, 5), end=date(2026, 1, 7),
        root=tmp_path / 'first', client=Client(), now=lambda: '2026-10-13T00:00:00+00:00')
    original = first.read_bytes()
    original_body, _ = load_final_market(first)
    assert original_body['status'] == 'COMPLETE'
    for index in range(3):
        client = Client(AssertionError('SYNTHETIC unnecessary request'))
        replay = collect_final_market('SYNTHETIC-calendar', start=date(2026, 1, 5), end=date(2026, 1, 7),
            root=tmp_path / f'recheck-{index}', previous=first, client=client,
            now=lambda: '2026-10-14T00:00:00+00:00')
        assert replay == first and replay.read_bytes() == original
        assert not client.calls
        assert not list((tmp_path / f'recheck-{index}').glob('manifests/*/manifest.json'))


def test_unchanged_capture_with_existing_parent_keeps_original_lineage(tmp_path, monkeypatch):
    calendar(monkeypatch)
    import infrastructure.les_v7_calendar_final as cal
    monkeypatch.setattr(cal, 'select_final_calendar', lambda *a, **k: (
        dict(days={day: 'OPEN' for day in ['2026-01-05', '2026-01-06']
            if k['start'].isoformat() <= day <= k['end'].isoformat()}), {}))
    first = collect_final_market('SYNTHETIC-calendar', start=date(2026, 1, 5), end=date(2026, 1, 5),
        root=tmp_path / 'first', client=Client(), now=lambda: '2026-10-13T00:00:00+00:00')
    expanded = collect_final_market('SYNTHETIC-calendar', start=date(2026, 1, 5), end=date(2026, 1, 6),
        root=tmp_path / 'second', previous=first, client=Client(), now=lambda: '2026-10-13T00:00:00+00:00')
    body, _ = load_final_market(expanded)
    assert body['parent_manifest_id'] == json.loads(first.read_bytes())['manifest_id']
    original = expanded.read_bytes()
    client = Client(AssertionError('SYNTHETIC unnecessary request'))
    replay = collect_final_market('SYNTHETIC-calendar', start=date(2026, 1, 5), end=date(2026, 1, 6),
        root=tmp_path / 'third', previous=expanded, client=client, now=lambda: '2026-10-14T00:00:00+00:00')
    assert replay == expanded and replay.read_bytes() == original and not client.calls


def test_changed_calendar_hash_with_same_sessions_is_not_a_noop(tmp_path, monkeypatch):
    parent = collect(tmp_path / 'first', monkeypatch, Client())
    earlier, _ = load_final_market(parent)
    import infrastructure.les_v7_calendar_final as cal
    monkeypatch.setattr(cal, 'load_final_calendar', lambda _: ({'SYNTHETIC_TEST_ONLY': 'REFRESHED'}, None))
    client = Client(AssertionError('SYNTHETIC unnecessary request'))
    replay = collect_final_market('SYNTHETIC-calendar', start=date(2026, 1, 5), end=date(2026, 1, 7),
        root=tmp_path / 'second', previous=parent, client=client, now=lambda: '2026-10-14T00:00:00+00:00')
    body, _ = load_final_market(replay)
    assert replay != parent and body['calendar_sha256'] != earlier['calendar_sha256']
    assert body['documents'] == earlier['documents'] and not client.calls
    assert body['parent_manifest_id'] == json.loads(parent.read_bytes())['manifest_id']


def test_non_lineage_metadata_differences_are_not_ignored_by_noop_comparison(tmp_path, monkeypatch):
    from application.replay_contracts import digest
    from infrastructure.research_archive import write_once
    parent = collect(tmp_path / 'first', monkeypatch, Client())
    record = json.loads(parent.read_bytes())
    record['body']['SYNTHETIC_EXTRA_CAPTURE_METADATA'] = 'must-not-be-ignored'
    record['manifest_id'] = digest(record['body'])
    augmented = parent.parents[1] / record['manifest_id'] / 'manifest.json'
    write_once(augmented, record)
    load_final_market(augmented)
    replay = collect_final_market('SYNTHETIC-calendar', start=date(2026, 1, 5), end=date(2026, 1, 7),
        root=tmp_path / 'second', previous=augmented, client=Client(), now=lambda: '2026-10-14T00:00:00+00:00')
    assert replay != augmented
    body, _ = load_final_market(replay)
    assert body['parent_manifest_id'] == record['manifest_id']

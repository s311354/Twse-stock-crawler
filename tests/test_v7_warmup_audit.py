"""Synthetic lifecycle/warmup safety tests, not a live historical data claim."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from infrastructure.crawler.twse_http_client import TwseBinaryResponse
from infrastructure.les_v7_warmup_audit import (
    FIELDS, audit_normalized_warmup, audit_scope_warmup, collect_listing_final, load_listing_final, parse_listing_final,
    parse_listed_profiles_final, PROFILE_URL,
)


def source(symbol="3014", listing="114.12.31", remark=""):
    return json.dumps(dict(stat="OK", date=None, title="最近上市公司", fields=FIELDS, total=1,
        data=[[symbol, "SYNTHETIC_TEST_ONLY", "", "", "", "", "", "", "", listing, "", "", remark]])).encode()


class Client:
    config = SimpleNamespace(allow_insecure_ssl_fallback=False, min_request_interval_seconds=1)

    def __init__(self, content=None):
        self.content = content or source()
        self.calls = 0

    def request_bytes(self, method, url, *, before_attempt, **kwargs):
        before_attempt()
        self.calls += 1
        return TwseBinaryResponse(self.content, 200, 1)


def captured(tmp_path, content=None):
    client = Client(content)
    path = collect_listing_final(tmp_path, client=client, now=lambda: "2026-10-13T00:00:00+00:00")
    packet, documents = load_listing_final(path)
    return path, packet, documents, client


def bar(day, close=100, **extra):
    return dict(Date=day, Close=close, Complete=True,
                **{"Available At": day + "T13:30:00+08:00"}, **extra)


def history(symbol="3014", bars=None):
    return dict(symbol=symbol, market="TWSE", security_type="COMMON", bars=bars or [])


def calendar():
    return dict(days={"2025-12-31": "OPEN", "2026-01-01": "CLOSED", "2026-01-02": "OPEN",
                      "2026-01-03": "CLOSED", "2026-01-04": "CLOSED", "2026-01-05": "OPEN",
                      "2026-01-06": "OPEN"})


def test_exact_columns_roc_date_and_unknown_pending():
    parsed = parse_listing_final(source())
    assert parsed["entries"][0]["listing_date"] == "2025-12-31"
    assert parse_listing_final(source(listing=""))["entries"][0]["status"] == "PENDING_VERIFICATION"
    raw = json.loads(source())
    raw["fields"][9], raw["fields"][8] = raw["fields"][8], raw["fields"][9]
    with pytest.raises(ValueError, match="COLUMN_MAPPING"):
        parse_listing_final(json.dumps(raw).encode())
    with pytest.raises(ValueError, match="UNRECOGNIZED"):
        parse_listing_final(source(listing="2025-12-31"))


def test_source_identity_stable_and_first_observation_receipt_preserved(tmp_path):
    path, packet, _, client = captured(tmp_path)
    again = collect_listing_final(tmp_path, client=client, now=lambda: "2026-10-14T00:00:00+00:00")
    assert again == path and client.calls == 2
    assert load_listing_final(path)[0]["fetched_at"] == packet["fetched_at"]
    assert packet["historical_available_at"] is None and packet["historical_information_set_certified"] is False


def test_corrupt_original_and_rebound_source_rejected(tmp_path):
    _, packet, documents, _ = captured(tmp_path)
    changed = deepcopy(packet)
    changed["parsed"]["entries"][0]["listing_date"] = "1990-01-01"
    with pytest.raises(ValueError, match="ORIGINAL_AND_PARSED"):
        audit_normalized_warmup([], calendar(), changed, cutoff="2026-01-05T13:30:00+08:00",
                                document_reader=documents.read)
    documents.objects.path(packet["document"]["sha256"]).write_bytes(b"SYNTHETIC_CORRUPTION")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        load_listing_final(tmp_path / packet["evidence_id"] / "manifest.json")


def test_budget_redirect_and_insecure_transport_rejected_without_manifest(tmp_path):
    class Redirect(Client):
        def request_bytes(self, *args, before_attempt, **kwargs):
            before_attempt()
            before_attempt()

    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        collect_listing_final(tmp_path, client=Redirect())
    assert not list(tmp_path.rglob("manifest.json"))
    bad = Client()
    bad.config = SimpleNamespace(allow_insecure_ssl_fallback=True, min_request_interval_seconds=1)
    with pytest.raises(ValueError, match="SECURE_RATE_LIMITED"):
        collect_listing_final(tmp_path, client=bad)


def test_late_prices_not_used_and_short_listing_upper_bound_not_imputed(tmp_path):
    _, packet, docs, _ = captured(tmp_path)
    histories = [history(bars=[bar("2025-12-31"), bar("2026-01-02"), bar("2026-01-05"), bar("2026-01-06")])]
    original = deepcopy(histories)
    audit = audit_normalized_warmup(histories, calendar(), packet, cutoff="2026-01-05T13:30:00+08:00",
                                    required=4, document_reader=docs.read)
    row = audit["rows"][0]
    assert row["valid_closes_at_cutoff"] == 3 and row["missing_closes"] == 1
    assert row["status"] == "SHORT_TWSE_LISTED_HISTORY_UNFILLABLE_AT_CUTOFF"
    assert row["official_open_session_upper_bound_since_listing"] == 3
    assert row["first_required_close_date_in_archive"] == "2026-01-06"
    assert row["future_or_late_prices_excluded"] == 1 and histories == original


def test_unknown_calendar_missing_listing_and_first_quote_remain_unknown(tmp_path):
    _, packet, docs, _ = captured(tmp_path)
    days = calendar()
    days["days"]["2026-01-02"] = "UNKNOWN"
    audit = audit_normalized_warmup([history(bars=[bar("2026-01-05")]), history("9999", [bar("2026-01-05")])],
        days, packet, cutoff="2026-01-05T13:30:00+08:00", required=4, document_reader=docs.read)
    known, missing = audit["rows"]
    assert known["official_open_session_upper_bound_since_listing"] is None
    assert known["unknown_calendar_days_since_listing"] == 1
    assert missing["status"] == "PENDING_LISTING_VERIFICATION" and missing["listing_evidence"] is None


def test_close_conflict_and_incomplete_bars_do_not_count(tmp_path):
    _, packet, docs, _ = captured(tmp_path)
    incomplete = bar("2025-12-31")
    incomplete["Complete"] = False
    audit = audit_normalized_warmup([history(bars=[incomplete, bar("2026-01-02"), bar("2026-01-02", 101),
            bar("2026-01-05")])], calendar(), packet, cutoff="2026-01-05T13:30:00+08:00",
            required=3, document_reader=docs.read)
    row = audit["rows"][0]
    assert row["valid_closes_at_cutoff"] == 1 and row["conflicting_dates"] == ["2026-01-02"]
    assert row["excluded"]["INVALID_OR_INCOMPLETE_PRICE"] == 1


def test_future_listing_is_not_historical_membership_and_prelisting_requires_transition(tmp_path):
    _, packet, docs, _ = captured(tmp_path, source("5236", "115.07.16", "櫃轉市"))
    audit = audit_normalized_warmup([history("5236", [bar("2026-01-05")])], calendar(), packet,
        cutoff="2026-01-05T13:30:00+08:00", document_reader=docs.read)
    row = audit["rows"][0]
    assert row["status"] == "NOT_YET_TWSE_LISTED_AT_CUTOFF" and row["valid_closes_at_cutoff"] == 0
    assert row["excluded"]["PRICE_BEFORE_VERIFIED_TWSE_LISTING_REQUIRES_TRANSITION_EVIDENCE"] == 1


def test_all_supplied_stocks_retained_no_count_or_pass_fabrication():
    histories = [history(str(1000 + index), [bar("2026-01-05")]) for index in range(457)]
    audit = audit_normalized_warmup(histories, calendar(), None, cutoff="2026-01-05T13:30:00+08:00")
    assert audit["symbol_count"] == 457 and audit["status_counts"] == {"PENDING_LISTING_VERIFICATION": 457}
    assert all(row["missing_closes"] == 512 for row in audit["rows"])
    assert audit == audit_normalized_warmup(histories, calendar(), None, cutoff="2026-01-05T13:30:00+08:00")


def test_frozen_scope_public_audit_is_complete_idempotent_and_cutoff_bound(tmp_path):
    from tests.test_v7_dataset_scope import scope_inputs, freeze
    inputs = scope_inputs(tmp_path, count=3, absent_on_event=("1001",))
    frozen = freeze(inputs)
    listing, _, _, _ = captured(tmp_path / "listing")
    original_freeze = frozen.read_bytes()
    args = dict(cutoff="2026-08-03T13:30:00+08:00", root=tmp_path / "audits", required=2)
    path = audit_scope_warmup(frozen, listing, **args)
    before = path.read_bytes()
    record = json.loads(before)
    body = record["body"]
    assert [row["symbol"] for row in body["rows"]] == inputs["symbols"]
    assert body["symbol_count"] == 3 and body["ordinary_complete"] == 2
    missing = next(row for row in body["rows"] if row["symbol"] == "1001")
    assert missing["valid_closes_at_cutoff"] == 1 and missing["future_or_late_prices_excluded"] == 2
    assert body["fitting_performed"] is False and body["approved_for_live"] is False
    assert audit_scope_warmup(frozen, listing, **args) == path and path.read_bytes() == before
    assert frozen.read_bytes() == original_freeze
    with pytest.raises(ValueError, match="OUTSIDE_FROZEN_SCOPE"):
        audit_scope_warmup(frozen, listing, cutoff="2026-08-06T13:30:00+08:00", root=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_source_shard_corruption_fails_before_audit_completion(tmp_path):
    from tests.test_v7_dataset_scope import scope_inputs, freeze, evidence_for, frozen_body
    inputs = scope_inputs(tmp_path, count=3)
    frozen = freeze(inputs)
    listing, _, _, _ = captured(tmp_path / "listing")
    source_path = evidence_for(inputs, frozen_body(frozen)["chunks"][0])
    source_path.write_bytes(b'{"SYNTHETIC_CORRUPTION":true}')
    with pytest.raises((ValueError, KeyError)):
        audit_scope_warmup(frozen, listing, cutoff="2026-08-03T13:30:00+08:00", root=tmp_path / "audits")
    assert not (tmp_path / "audits").exists()


def test_exact_openapi_host_not_spoofed_or_plain_http():
    import hashlib
    from application.research_evidence import verified_document
    raw = 'SYNTHETIC_TEST_ONLY'
    document = dict(url="https://openapi.twse.com.tw/v1/opendata/t187ap03_L", body_utf8=raw,
        sha256=hashlib.sha256(raw.encode()).hexdigest(), retrieved_at="2026-10-13T00:00:00+00:00")
    assert verified_document(document) == raw.encode()
    for url in ("https://openapi.twse.com.tw.example.org/v1/opendata/t187ap03_L",
                "http://openapi.twse.com.tw/v1/opendata/t187ap03_L",
                "https://attacker@openapi.twse.com.tw/v1/opendata/t187ap03_L"):
        with pytest.raises(ValueError, match="OFFICIAL_EVIDENCE_SOURCE_REQUIRED"):
            verified_document(dict(document, url=url))


def test_audit_rejects_source_replacement_after_scope_verification(tmp_path, monkeypatch):
    from tests.test_v7_dataset_scope import scope_inputs, freeze
    import infrastructure.les_v7_training_source as source_module
    inputs = scope_inputs(tmp_path, count=3)
    frozen = freeze(inputs)
    listing, _, _, _ = captured(tmp_path / "listing")
    original = source_module.load_training_source
    def replaced(*args, **kwargs):
        value = original(*args, **kwargs)
        value["payload"]["source_evidence_ids"] = ["SYNTHETIC_CHANGED_SOURCE"]
        return value
    monkeypatch.setattr(source_module, "load_training_source", replaced)
    with pytest.raises(ValueError, match="SOURCE_CHANGED_AFTER_FREEZE"):
        audit_scope_warmup(frozen, listing, cutoff="2026-08-03T13:30:00+08:00", root=tmp_path / "audits")
    assert not (tmp_path / "audits").exists()


def profile_source(symbol="2321", listing="19911108"):
    return json.dumps([{"公司代號": symbol, "公司名稱": "SYNTHETIC_TEST_ONLY公司", "公司簡稱": "SYNTHETIC_TEST_ONLY",
        "上市日期": listing, "營利事業統一編號": "47224065", "實收資本額": "123456"}]).encode()


class ProfilesClient(Client):
    def __init__(self, profiles=None):
        super().__init__()
        self.profiles = profiles or profile_source()

    def request_bytes(self, method, url, *, before_attempt, **kwargs):
        before_attempt()
        self.calls += 1
        return TwseBinaryResponse(self.profiles if url == PROFILE_URL else self.content, 200, 1)


def test_profiles_explicit_optin_adds_missing_lifecycle_without_historical_time_claim(tmp_path):
    client = ProfilesClient()
    with pytest.raises(ValueError, match="REQUEST_BUDGET"):
        collect_listing_final(tmp_path, client=client, include_profiles=True)
    assert client.calls == 0
    path = collect_listing_final(tmp_path, client=client, include_profiles=True, max_requests=2,
                                 now=lambda: "2026-10-13T00:00:00+00:00")
    packet, documents = load_listing_final(path)
    assert client.calls == 2 and packet["profiles"]["historical_available_at"] is None
    assert packet["profiles"]["parsed"]["entries"][0]["listing_date"] == "1991-11-08"
    row = audit_normalized_warmup([history("2321", [bar("2026-01-05")])], calendar(), packet,
        cutoff="2026-01-05T13:30:00+08:00", document_reader=documents.read)["rows"][0]
    assert row["status"] == "VALID_PRICE_HISTORY_SHORT_EXTEND_WINDOW_OR_VERIFY_GAPS"
    assert row["listing_evidence"]["issuer_id"] == "47224065" and row["valid_closes_at_cutoff"] == 1
    assert row["listing_evidence"]["listing_date"] != row["first_valid_close_in_archive"]


def test_profile_duplicate_invalid_date_and_identity_rejected():
    values = json.loads(profile_source())
    with pytest.raises(ValueError, match="DUPLICATE_CONFLICT"):
        parse_listed_profiles_final(json.dumps(values + values).encode())
    with pytest.raises(ValueError, match="UNRECOGNIZED"):
        parse_listed_profiles_final(profile_source(listing="801108"))
    with pytest.raises(ValueError):
        parse_listed_profiles_final(profile_source(listing="19910230"))
    values[0]["營利事業統一編號"] = "UNKNOWN"
    with pytest.raises(ValueError, match="IDENTITY"):
        parse_listed_profiles_final(json.dumps(values).encode())
    assert parse_listed_profiles_final(profile_source(listing=""))["entries"][0]["status"] == "PENDING_VERIFICATION"


def test_positive_listing_date_conflict_never_silently_selects_either_source(tmp_path):
    client = ProfilesClient(profile_source("3014", "19911108"))
    path = collect_listing_final(tmp_path, client=client, include_profiles=True, max_requests=2,
                                 now=lambda: "2026-10-13T00:00:00+00:00")
    packet, documents = load_listing_final(path)
    result = audit_normalized_warmup([history("3014", [bar("2026-01-05")]), history("9999", [])],
        calendar(), packet, cutoff="2026-01-05T13:30:00+08:00", required=1,
        document_reader=documents.read)
    row = result["rows"][0]
    assert row["status"] == "CONFLICTING_LISTING_EVIDENCE"
    assert row["listing_evidence"]["listing_date"] is None
    assert {s["listing_date"] for s in row["listing_evidence"]["sources"]} == {"2025-12-31", "1991-11-08"}
    assert row["official_open_session_upper_bound_since_listing"] is None
    assert result["symbol_count"] == 2  # Unrelated stocks are not suppressed.


def test_matching_profile_retains_listing_remark_and_hash_bound_identity(tmp_path):
    client = ProfilesClient(profile_source("3014", "20251231"))
    client.content = source(remark="櫃轉市")
    path = collect_listing_final(tmp_path, client=client, include_profiles=True, max_requests=2,
                                 now=lambda: "2026-10-13T00:00:00+00:00")
    packet, documents = load_listing_final(path)
    row = audit_normalized_warmup([history("3014", [bar("2026-01-05")])], calendar(), packet,
        cutoff="2026-01-05T13:30:00+08:00", document_reader=documents.read)["rows"][0]
    assert row["listing_evidence"]["remark"] == "櫃轉市" and row["listing_evidence"]["profile_source_row_sha256"]
    changed = deepcopy(packet)
    changed["profiles"]["document"]["url"] = "https://example.invalid/rebound"
    with pytest.raises(ValueError, match="PROFILE_SOURCE"):
        audit_normalized_warmup([], calendar(), changed, cutoff="2026-01-05T13:30:00+08:00",
                                document_reader=documents.read)
    documents.objects.path(packet["profiles"]["document"]["sha256"]).write_bytes(b"SYNTHETIC_CORRUPTION")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        load_listing_final(path)


def test_profile_packet_identity_includes_full_original_version_not_collection_clock(tmp_path):
    client = ProfilesClient()
    first = collect_listing_final(tmp_path, client=client, include_profiles=True, max_requests=2,
                                  now=lambda: "2026-10-13T00:00:00+00:00")
    original = first.read_bytes()
    again = collect_listing_final(tmp_path, client=client, include_profiles=True, max_requests=2,
                                  now=lambda: "2026-10-14T00:00:00+00:00")
    assert again == first and first.read_bytes() == original
    raw = json.loads(client.profiles)
    raw[0]["實收資本額"] = "999999"
    client.profiles = json.dumps(raw).encode()
    changed = collect_listing_final(tmp_path, client=client, include_profiles=True, max_requests=2,
                                    now=lambda: "2026-10-14T00:00:00+00:00")
    assert changed != first and first.read_bytes() == original

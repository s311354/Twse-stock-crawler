"""Synthetic adapter regressions; never real publication or market evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from application.replay_contracts import canonical_json, digest
from infrastructure.les_v7_training_source import FINAL_CLOSE, FINAL_VINTAGE, load_training_source
from tests.test_research_archive import proof


CUTOFF = "2026-08-05T13:30:00+08:00"


def fixture():
    annual = dict(stat="ok", fields=["日期", "名稱", "說明"], data=[["2026-01-01", "元旦休市", "休市"]])
    binding = dict(kind="ANNUAL_CALENDAR", market="TWSE", year=2026, content_sha256=digest(annual))
    row = ["3014", "聯陽", "1,200", "20", "120,000", "100.00", "102.00", "99.00", "101.00",
           "+", "1.00", "100.50", "10", "101.00", "10", "20"]
    return dict(schema="replay-evidence/v2", purpose="SYNTHETIC_TEST_ONLY", market="TWSE", sector="13",
        symbols=["3014", "6834"], start="2026-08-03", end="2026-08-05", market_policy=FINAL_VINTAGE,
        price_unit="TWD", volume_unit="SHARES", amount_unit="TWD", universe_policy="EXPLICIT_TEST_UNIVERSE",
        calendar=dict(source="TWSE_OFFICIAL", days={"2026-08-03": "OPEN", "2026-08-04": "OPEN", "2026-08-05": "OPEN"},
            documents={"2026": dict(payload=annual, publication_proof=proof(binding, publication="2025-12-01"))},
            announcements=[]), bars=[dict(dataset="STOCK_TABLE", date="20260803", market_type="13", source="TWSE", data=[row]),
                dict(dataset="TAIEX", date="20260803", market_type="IND", source="TWSE_TAIEX", data="22,000")],
        raw_documents={}, filings={}, preparation_gaps=[])


def save(tmp_path, payload):
    p = tmp_path / "evidence.json"
    p.write_text(canonical_json(dict(evidence_id=digest(payload), payload=payload)))
    return p


def load(path, **kwargs):
    return load_training_source(path, evaluation_cutoff=CUTOFF, **kwargs)


def test_default_does_not_backdate_missing_bar_observation(tmp_path):
    record = load(save(tmp_path, fixture()))
    p = record["payload"]
    assert not any(h["bars"] for h in p["histories"])
    assert p["diagnostics"]["reason_counts"]["BAR_AVAILABILITY_UNVERIFIED"] == 1
    assert p["historical_information_set_certified"] is False
    assert p["dataset_kind"] == "SYNTHETIC_TEST_ONLY"
    assert p["calendars"]["TWSE"]["sessions"] == ["2026-08-03", "2026-08-04", "2026-08-05"]
    assert p["calendars"]["TWSE"]["availability_by_date"]["2026-08-03"] == "2025-12-02T00:00:00+08:00"


def test_explicit_final_reconstruction_preserves_real_scales_and_lineage(tmp_path):
    original = fixture()
    snapshot = deepcopy(original)
    path = save(tmp_path, original)
    record = load(path, availability_policy=FINAL_CLOSE)
    p = record["payload"]
    row = p["histories"][0]["bars"][0]
    assert (row["Close"], row["Volume"], row["Amount"]) == (101., 1200., 120000.)
    assert row["TAIEX Close"] == 22000.
    assert row["Available At"] == "2026-08-03T13:30:00+08:00"
    assert row["Availability Policy"] == FINAL_CLOSE
    trace = p["histories"][0]["bar_lineage"][0]
    assert trace["original_observed_at"] is trace["original_available_at"] is None
    assert trace["source_row_sha256"] == digest(original["bars"][0]["data"][0])
    assert trace["calendar_status"] == "OPEN"
    assert p["historical_information_set_certified"] is False
    assert record["sha256"] == digest(p)
    assert load(path, availability_policy=FINAL_CLOSE) == record
    assert original == snapshot


def test_unverified_calendar_keeps_unknown_not_weekday_inference(tmp_path):
    p = fixture()
    p["calendar"]["documents"]["2026"].pop("publication_proof")
    result = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    cal = result["calendars"]["TWSE"]
    assert cal["sessions"] == [] and cal["available_at"] is None
    assert set(cal["days"].values()) == {"UNKNOWN"}
    assert len(result["diagnostics"]["calendar"]["unknown_dates"]) == 3
    assert result["histories"][0]["bar_lineage"][0]["calendar_status"] == "UNKNOWN"


def test_date_only_calendar_publication_next_midnight(tmp_path):
    p = fixture()
    publication = p["calendar"]["documents"]["2026"]["publication_proof"]
    publication["published_at"] = "2026-08-05"
    publication["document"]["body_utf8"] += " 2026-08-05"
    publication["quotes"] = ["2026-08-05"]
    publication["document"]["sha256"] = hashlib.sha256(publication["document"]["body_utf8"].encode()).hexdigest()
    path = save(tmp_path, p)
    assert not load(path, availability_policy=FINAL_CLOSE)["payload"]["calendars"]["TWSE"]["sessions"]
    later = load_training_source(path, evaluation_cutoff="2026-08-06T00:00:00+08:00", availability_policy=FINAL_CLOSE)
    assert later["payload"]["calendars"]["TWSE"]["available_at"] == "2026-08-06T00:00:00+08:00"


def test_conflicting_market_and_benchmark_rows_not_silently_selected(tmp_path):
    p = fixture()
    second = deepcopy(p["bars"][0]); second["data"][0][8] = "100.00"
    p["bars"].append(second)
    conflict = deepcopy(p["bars"][1]); conflict["data"] = "23,000"
    p["bars"].append(conflict)
    out = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    assert not out["histories"][0]["bars"]
    assert out["diagnostics"]["reason_counts"]["CONFLICTING_MARKET_ROWS"] == 1
    assert out["diagnostics"]["reason_counts"]["CONFLICTING_BENCHMARK_ROWS"] == 1


def test_identical_duplicates_collapsed_but_all_source_rows_bound(tmp_path):
    p = fixture()
    duplicate = deepcopy(p["bars"][0]); duplicate["market_type"] = "ALLBUT0999"
    p["bars"].append(duplicate)
    out = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    assert len(out["histories"][0]["bars"]) == 1
    assert out["diagnostics"]["duplicate_rows"] == 1
    assert len(out["histories"][0]["bar_lineage"][0]["source_row_ids"]) == 2


@pytest.mark.parametrize("column,first_value,second_value", [
    (15, "", "0.00"),
    (1, "聯陽", "聯陽股份有限公司"),
    (9, "+", "<p style= color:red>+</p>"),
], ids=["missing-pe-versus-zero", "company-name", "display-only-change-sign"])
def test_non_technical_variants_keep_numeric_inputs_and_all_raw_bindings(
    tmp_path, monkeypatch, column, first_value, second_value,
):
    import infrastructure.les_v7_training_source as module

    p = fixture()
    p["bars"][0]["data"][0][column] = first_value
    baseline = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    duplicate = deepcopy(p["bars"][0])
    duplicate["market_type"] = "ALLBUT0999"
    duplicate["data"][0][column] = second_value
    p["bars"].append(duplicate)
    original = deepcopy(p)
    path = save(tmp_path, p)
    frozen_bytes = path.read_bytes()
    # Return the actual source object to detect accidental in-place edits, not
    # merely edits to a JSON-deserialized copy.
    monkeypatch.setattr(module, "load_evidence", lambda *a, **kw: (digest(p), p))
    out = load(path, availability_policy=FINAL_CLOSE)["payload"]
    history = out["histories"][0]
    assert history["bars"] == baseline["histories"][0]["bars"]
    assert len(history["bars"]) == 1
    assert "PE" not in history["bars"][0] and "Name" not in history["bars"][0]
    assert "CONFLICTING_MARKET_ROWS" not in out["diagnostics"]["reason_counts"]
    conflict = out["diagnostics"]["non_technical_conflicts"]
    raw_hashes = {digest(p["bars"][0]["data"][0]), digest(duplicate["data"][0])}
    assert len(conflict) == 1
    assert conflict[0]["differing_columns"] == [column]
    assert set(conflict[0]["source_row_sha256s"]) == raw_hashes
    assert conflict[0]["resolution"] == "NO_FINANCIAL_VALUE_SELECTED_TECHNICAL_INPUTS_EQUAL"
    trace = history["bar_lineage"][0]
    assert {variant["source_row_sha256"] for variant in trace["source_variants"]} == raw_hashes
    expected_row_ids = {
        digest(dict(header={key: item.get(key) for key in ("date", "dataset", "market_type", "source")}, row=item["data"][0]))
        for item in (p["bars"][0], duplicate)
    }
    assert set(trace["source_row_ids"]) == expected_row_ids
    assert {variant["source_header"]["market_type"] for variant in trace["source_variants"]} == {"13", "ALLBUT0999"}
    assert p == original and path.read_bytes() == frozen_bytes


@pytest.mark.parametrize("column,value", [
    (8, "100.00"), (2, "1,201"), (4, "120,001"),
    (11, "100.00"), (13, "101.50"),
], ids=["close", "volume", "amount", "bid", "ask"])
def test_required_technical_variant_conflicts_still_exclude_bar(tmp_path, column, value):
    p = fixture()
    duplicate = deepcopy(p["bars"][0])
    duplicate["market_type"] = "ALLBUT0999"
    duplicate["data"][0][column] = value
    p["bars"].append(duplicate)
    out = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    assert not out["histories"][0]["bars"]
    assert out["diagnostics"]["reason_counts"]["CONFLICTING_MARKET_ROWS"] == 1
    assert out["diagnostics"]["non_technical_conflicts"] == []


def final_reference_source(tmp_path, *, empty=False):
    from tests.test_v7_actions_final import packet as actions_packet
    from tests.test_v7_calendar_final import packet as calendar_packet

    value = fixture()
    value['calendar'], _ = calendar_packet(tmp_path)
    value['corporate_actions_final'] = actions_packet()
    value['reference_evidence_policy'] = 'OFFICIAL_FINAL_REFERENCE_RECONSTRUCTION_R1'
    if empty:
        value['bars'] = []
    return value


@pytest.mark.parametrize('empty', [False, True])
def test_final_action_originals_are_verified_once_with_identical_source_identity(tmp_path, monkeypatch, empty):
    import infrastructure.les_v7_actions_final as actions
    from infrastructure.les_v7_training_source import FINAL_REFERENCE

    path = save(tmp_path, final_reference_source(tmp_path, empty=empty))
    before = path.read_bytes()
    native_builder, native_verify = actions.build_final_coordinate_coverages, actions.verify_packet
    calls = []

    def verify(packet, **kwargs):
        calls.append(packet['evidence_id'])
        return native_verify(packet, **kwargs)

    def old_sequence(histories, packet, **kwargs):
        verify(packet, document_reader=kwargs.get('document_reader'))
        return native_builder(histories, packet, **kwargs)

    monkeypatch.setattr(actions, 'verify_packet', verify)
    monkeypatch.setattr(actions, 'build_final_coordinate_coverages', old_sequence)
    expected = load(path, availability_policy=FINAL_CLOSE, reference_evidence_policy=FINAL_REFERENCE)
    assert len(calls) == 2
    calls.clear()
    monkeypatch.setattr(actions, 'build_final_coordinate_coverages', native_builder)
    actual = load(path, availability_policy=FINAL_CLOSE, reference_evidence_policy=FINAL_REFERENCE)
    assert len(calls) == 1
    assert actual == expected and actual['sha256'] == digest(actual['payload'])
    assert path.read_bytes() == before


@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('change', ['bytes', 'parsed', 'packet_id'])
def test_final_action_source_and_packet_tampering_still_rejects_before_normalization_returns(
        tmp_path, empty, change):
    from infrastructure.les_v7_training_source import FINAL_REFERENCE

    value = final_reference_source(tmp_path, empty=empty)
    packet = value['corporate_actions_final']
    if change == 'bytes':
        packet['feeds'][0]['document']['body_utf8'] += ' '
    elif change == 'parsed':
        packet['feeds'][0]['parsed']['actions'] = [dict(symbol='3014', date='2026-08-03')]
    else:
        packet['evidence_id'] = '0' * 64
    path = save(tmp_path, value)  # Rehashing the outer source cannot hide inner source corruption.
    with pytest.raises(ValueError):
        load(path, availability_policy=FINAL_CLOSE, reference_evidence_policy=FINAL_REFERENCE)


def test_non_technical_reconciliation_does_not_bypass_cutoff_or_original_observation(tmp_path):
    p = fixture()
    p["bars"][0]["date"] = "20260805"
    p["bars"][0]["data"][0][15] = ""
    duplicate = deepcopy(p["bars"][0])
    duplicate["market_type"] = "ALLBUT0999"
    duplicate["data"][0][15] = "0.00"
    p["bars"].append(duplicate)
    path = save(tmp_path, p)
    before_close = load_training_source(path, evaluation_cutoff="2026-08-05T12:00:00+08:00", availability_policy=FINAL_CLOSE)["payload"]
    assert not before_close["histories"][0]["bars"]
    assert before_close["diagnostics"]["reason_counts"]["BAR_AFTER_EVALUATION_CUTOFF"] == 1
    original_observation = load(path)["payload"]
    assert not original_observation["histories"][0]["bars"]
    assert original_observation["diagnostics"]["reason_counts"]["BAR_AVAILABILITY_UNVERIFIED"] == 1
    assert len(original_observation["diagnostics"]["non_technical_conflicts"]) == 1
    assert original_observation["historical_information_set_certified"] is False


def test_missing_quotes_and_benchmark_remain_missing(tmp_path):
    p = fixture(); p["bars"].pop(); p["bars"][0]["data"][0][11] = "--"
    out = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    row = out["histories"][0]["bars"][0]
    assert row["Bid"] is None and row["TAIEX Close"] is None
    assert out["diagnostics"]["reason_counts"]["BID_UNAVAILABLE"] == 1
    assert out["diagnostics"]["reason_counts"]["BENCHMARK_UNAVAILABLE"] == 1


def test_bar_after_cutoff_and_incomplete_ohlc_excluded(tmp_path):
    p = fixture(); p["bars"][0]["date"] = "20260805"
    path = save(tmp_path, p)
    out = load_training_source(path, evaluation_cutoff="2026-08-05T12:00:00+08:00", availability_policy=FINAL_CLOSE)
    assert out["payload"]["diagnostics"]["reason_counts"]["BAR_AFTER_EVALUATION_CUTOFF"] == 1
    p["bars"][0]["data"][0][7] = "200"
    out = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)
    assert out["payload"]["diagnostics"]["reason_counts"]["OHLCV_UNAVAILABLE"] == 1


@pytest.mark.parametrize("change,reason", [
    ({"schema": "replay-evidence/v1"}, "RAW_REPLAY_EVIDENCE_V2_REQUIRED"),
    ({"price_unit": "TWD_1000"}, "TRAINING_SOURCE_UNITS_UNVERIFIED"),
    ({"market_policy": "SYNTHETIC_UNKNOWN_VINTAGE"}, "FINAL_CLOSE_REQUIRES_DECLARED_FINAL_ARCHIVE_VINTAGE"),
])
def test_source_contract_refuses_unknown_schema_units_vintage(tmp_path, change, reason):
    p = fixture(); p.update(change)
    with pytest.raises(ValueError, match=reason):
        load(save(tmp_path, p), availability_policy=FINAL_CLOSE)


def test_bounded_input_and_explicit_universe_subset(tmp_path):
    path = save(tmp_path, fixture())
    with pytest.raises(ValueError, match="TOO_LARGE"):
        load(path, max_bytes=10)
    with pytest.raises(ValueError, match="BOUNDED_UNIVERSE"):
        load(path, symbols=["2330"])
    out = load(path, symbols=["3014"], availability_policy=FINAL_CLOSE)["payload"]
    assert out["universe"] == [["3014", "TWSE"]]
    assert out["lineage"]["explicit_subset"] is True
    assert out["lineage"]["archive_universe"] == ["3014", "6834"]


def test_digest_corruption_and_calendar_proof_corruption_fail_closed(tmp_path):
    p = fixture(); path = save(tmp_path, p)
    doc = json.loads(path.read_text()); doc["payload"]["bars"][0]["data"][0][8] = "100"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="SIGNATURE_MISMATCH"):
        load(path)
    p["calendar"]["documents"]["2026"]["publication_proof"]["document"]["sha256"] = "0" * 64
    out = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    assert out["calendars"]["TWSE"]["sessions"] == []
    assert out["diagnostics"]["calendar"]["excluded"][0]["detail"] == "SOURCE_DOCUMENT_HASH_MISMATCH"


def test_source_record_order_does_not_change_numeric_output(tmp_path):
    p = fixture(); original = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    p["bars"].reverse()
    reordered = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    assert original["histories"][0]["bars"] == reordered["histories"][0]["bars"]
    # Archive ordering is business evidence content, therefore a different ID.
    assert original["source_evidence_ids"] != reordered["source_evidence_ids"]


def test_verified_temporary_closure_overrides_annual_and_excludes_quote(tmp_path):
    p = fixture()
    event = dict(date="2026-08-03", status="CLOSED", market="TWSE")
    binding = dict(kind="CALENDAR_EXCEPTION", **event)
    event["proof"] = proof(binding, publication="2026-08-02T18:00:00+08:00", precision="TIMESTAMP")
    p["calendar"]["announcements"] = [event]
    out = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    cal = out["calendars"]["TWSE"]
    assert cal["days"]["2026-08-03"] == "CLOSED"
    assert cal["availability_by_date"]["2026-08-03"] == "2026-08-02T18:00:00+08:00"
    assert not out["histories"][0]["bars"]
    assert out["diagnostics"]["reason_counts"]["BAR_ON_VERIFIED_CLOSED_SESSION"] == 1


def test_tpex_canonical_rows_keep_market_and_share_units(tmp_path):
    p = fixture(); p["market"] = "TPEX"
    p["calendar"]["source"] = "TPEX_OFFICIAL"
    annual = p["calendar"]["documents"]["2026"]
    binding = dict(kind="ANNUAL_CALENDAR", market="TPEX", year=2026, content_sha256=digest(annual["payload"]))
    annual["publication_proof"] = proof(binding, publication="2025-12-01")
    annual["publication_proof"]["document"]["url"] = "https://www.tpex.org.tw/test-only"
    p["bars"][0]["source"] = "TPEX"; p["bars"][0]["market_type"] = "TPEX:24:v1"
    out = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    assert out["universe"] == [["3014", "TPEX"], ["6834", "TPEX"]]
    assert out["histories"][0]["bars"][0]["Volume"] == 1200.
    assert out["histories"][0]["market"] == "TPEX"
    assert out["calendars"]["TPEX"]["source"] == "TPEX_OFFICIAL"


def test_crossed_quotes_kept_as_conflict_not_fake_healthy_spread(tmp_path):
    p = fixture(); p["bars"][0]["data"][0][11] = "102"
    out = load(save(tmp_path, p), availability_policy=FINAL_CLOSE)["payload"]
    row = out["histories"][0]["bars"][0]
    assert row["Bid"] == 102. and row["Ask"] == 101.
    assert out["diagnostics"]["reason_counts"]["CROSSED_SOURCE_QUOTES"] == 1


def test_pre_stat_does_not_bypass_bounded_reader(tmp_path, monkeypatch):
    import infrastructure.les_v7_training_source as module
    path = save(tmp_path, fixture())
    monkeypatch.setattr(module, "_stamp", lambda _: (1, 5, 1, 1))
    with pytest.raises(ValueError, match="EVIDENCE_READ_SIZE_LIMIT"):
        load(path, max_bytes=10)

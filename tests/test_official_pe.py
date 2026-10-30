"""SYNTHETIC ONLY: official dated P/E contract; no fabricated production data."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from domain.errors import NoDataError, SchemaChangedError
from domain.les_v7.contracts import identity
from domain.les_v7.official_pe import (AVAILABILITY_BASIS, ENDPOINTS, OfficialPeIndex,
    build_snapshot, join_pe_rows, select_pe, validate_pe_selection, validate_snapshot)
from infrastructure.crawler.official_pe import OfficialPeClient, OfficialPeRepository


DAY = "2026-10-13"
FETCH = "2026-10-14T04:00:00+00:00"
CUTOFF = "2026-10-14T04:30:00+00:00"


def payload(market="TWSE", *, pe="12.50", symbol="SYN1"):
    if market == "TWSE":
        return dict(stat="OK", date="20261013", selectType="ALL", total=1,
                    fields=["證券代號", "證券名稱", "收盤價", "本益比", "財報年/季"],
                    data=[[symbol, "SYNTHETIC ONLY", "100", pe, "115/2"]])
    return dict(stat="ok", date="20261013", tables=[dict(date="115/10/13", totalCount=1,
                stkCategory="全部產業", fields=["股票代號", "公司名稱", "本益比", "財報年/季"],
                data=[[symbol, "SYNTHETIC ONLY", pe, "115Q2"]])])


def snapshot(market="TWSE", *, pe="12.50", symbol="SYN1", fetched_at=FETCH):
    return build_snapshot(json.dumps(payload(market, pe=pe, symbol=symbol), ensure_ascii=False).encode(),
                          market=market, trade_date=DAY, fetched_at=fetched_at)


def selected(items, **kwargs):
    return select_pe(items, market="TWSE", symbol="SYN1", trade_date=DAY, cutoff=CUTOFF, **kwargs)


@pytest.mark.parametrize("market", ["TWSE", "TPEX"])
def test_official_schema_raw_hash_and_named_field_order(market):
    original = payload(market)
    table = original if market == "TWSE" else original["tables"][0]
    table["fields"].reverse()
    table["data"][0].reverse()
    record = build_snapshot(json.dumps(original).encode(), market=market, trade_date=DAY, fetched_at=FETCH)
    assert validate_snapshot(record, cutoff=CUTOFF) == record
    observed = select_pe([record], market=market, symbol="SYN1", trade_date=DAY, cutoff=CUTOFF)
    assert observed["pe"] == 12.5 and observed["status"] == "REPORTED"
    assert observed["source_record_ids"] == [record["evidence_id"]]
    assert record["source_url"] == ENDPOINTS[market]
    assert record["original_publication_at"] is None
    assert record["availability_basis"] == AVAILABILITY_BASIS
    assert record["available_at"] == FETCH
    assert observed["reported_close"] == (100. if market == "TWSE" else None)


@pytest.mark.parametrize("pe,status", [("-", "NOT_REPORTED"), ("--", "NOT_REPORTED"),
    ("N/A", "NOT_REPORTED"), ("", "MISSING_VALUE"), ("0", "NON_POSITIVE"),
    ("-1", "NON_POSITIVE"), ("1000", "REPORTED")])
def test_explicit_not_reported_missing_and_bad_values_are_distinct(pe, status):
    item = selected([snapshot(pe=pe)])
    assert item["status"] == status
    assert item["raw_pe"] == pe
    assert "EPS" not in json.dumps(item)
    rows, diagnostics = join_pe_rows([dict(Date=DAY, Close=100.)], [snapshot(pe=pe)],
                                      market="TWSE", symbol="SYN1", cutoff=CUTOFF)
    assert (rows[0]["PE"] is not None) == (status == "REPORTED")
    assert diagnostics[0]["status"] == status


@pytest.mark.parametrize("pe", ["NaN", "Inf", "twelve", "1,2,3", "1e2"])
def test_invalid_pe_schema_is_not_zero_or_pass(pe):
    with pytest.raises(SchemaChangedError):
        snapshot(pe=pe)


def test_duplicate_json_keys_and_oversized_raw_input_are_not_an_official_observation():
    with pytest.raises(SchemaChangedError):
        build_snapshot(b'{"date":"20261012","date":"20261013"}',
                       market="TWSE", trade_date=DAY, fetched_at=FETCH)
    with pytest.raises(ValueError, match="CAPACITY"):
        build_snapshot(b" " * (8 * 1024 * 1024 + 1), market="TWSE", trade_date=DAY, fetched_at=FETCH)


@pytest.mark.parametrize("market,damage", [("TWSE", "date"), ("TWSE", "scope"),
    ("TWSE", "count"), ("TWSE", "fields"), ("TWSE", "width"),
    ("TPEX", "date"), ("TPEX", "scope"), ("TPEX", "table_date"), ("TPEX", "count")])
def test_date_scope_schema_mismatches_never_join(market, damage):
    item = payload(market)
    table = item if market == "TWSE" else item["tables"][0]
    if damage == "date": item["date"] = "20261012"
    elif damage == "scope": table["selectType" if market == "TWSE" else "stkCategory"] = "SEMI"
    elif damage == "table_date": table["date"] = "115/10/12"
    elif damage == "count": table["total" if market == "TWSE" else "totalCount"] = 99
    elif damage == "fields": table["fields"][-1] = "本益比"
    elif damage == "width": table["data"][0].pop()
    with pytest.raises(SchemaChangedError):
        build_snapshot(json.dumps(item).encode(), market=market, trade_date=DAY, fetched_at=FETCH)


def test_no_data_is_not_market_closed_or_unrestricted():
    item = payload(); item["data"] = []; item["total"] = 0
    with pytest.raises(NoDataError):
        build_snapshot(json.dumps(item).encode(), market="TWSE", trade_date=DAY, fetched_at=FETCH)


def test_duplicate_symbol_retains_conflict_and_does_not_pick_positive():
    item = payload(); item["data"].append(deepcopy(item["data"][0])); item["data"][1][3] = "-"
    item["total"] = 2
    record = build_snapshot(json.dumps(item).encode(), market="TWSE", trade_date=DAY, fetched_at=FETCH)
    result = selected([record])
    assert result["status"] == "CONFLICT" and result["pe"] is None
    assert len(result["conflicting_records"]) == 2


@pytest.mark.parametrize("field", ["records", "raw_response_utf8", "content_sha256", "evidence_id",
    "source_url", "available_at", "original_publication_at", "availability_basis", "parser_version"])
def test_raw_mapping_endpoint_time_and_id_tampering_cannot_be_rehashed(field):
    record = snapshot()
    if field == "records": record[field][0]["pe"] = 99.
    else: record[field] = "TAMPERED_SYNTHETIC"
    if field != "evidence_id":
        record["evidence_id"] = identity({key: value for key, value in record.items() if key != "evidence_id"})
    with pytest.raises((ValueError, SchemaChangedError)):
        validate_snapshot(record, cutoff=CUTOFF)


def test_future_fetch_does_not_become_historical_publication():
    record = snapshot()
    with pytest.raises(ValueError, match="NOT_AVAILABLE"):
        validate_snapshot(record, cutoff="2026-10-13T13:00:00+08:00")
    assert select_pe([record], market="TWSE", symbol="SYN1", trade_date=DAY,
                     cutoff="2026-10-13T13:30:00+08:00")["status"] == "MISSING"
    with pytest.raises(ValueError, match="SESSION_NOT_COMPLETED"):
        snapshot(fetched_at="2026-10-13T13:29:00+08:00")
    with pytest.raises(ValueError, match="TIMEZONE"):
        snapshot(fetched_at="2026-10-14T12:00:00")


def test_exact_market_date_and_symbol_prevent_cross_join_and_stale_fallback():
    good = snapshot()
    assert select_pe([good], market="TPEX", symbol="SYN1", trade_date=DAY, cutoff=CUTOFF)["status"] == "MISSING"
    assert select_pe([good], market="TWSE", symbol="SYN1", trade_date="2026-10-12", cutoff=CUTOFF)["status"] == "MISSING"
    assert select_pe([good], market="TWSE", symbol="SYN2", trade_date=DAY, cutoff=CUTOFF)["status"] == "MISSING"
    latest = snapshot(pe="-", fetched_at="2026-10-14T04:01:00Z")
    assert selected([good, latest])["status"] == "NOT_REPORTED"
    disappearance = snapshot(symbol="SYN2", fetched_at="2026-10-14T04:02:00Z")
    assert selected([good, disappearance])["status"] == "MISSING"


def test_same_time_versions_conflict_but_future_versions_cannot_change_earlier_result():
    good = snapshot()
    bad = snapshot(pe="20.00")
    assert selected([good, bad])["status"] == "CONFLICT"
    future = snapshot(pe="20.00", fetched_at="2026-10-14T05:00:00Z")
    assert selected([good, future])["pe"] == 12.5


def test_index_validates_once_is_bounded_and_not_mutable_through_sources_or_results():
    record = snapshot(); before = deepcopy(record)
    with patch("domain.les_v7.official_pe.validate_snapshot", wraps=validate_snapshot) as validator:
        index = OfficialPeIndex([record])
        assert validator.call_count == 1
        for _ in range(10):
            result = index.select(market="TWSE", symbol="SYN1", trade_date=DAY, cutoff=CUTOFF)
            result["raw_cells"].clear(); result["pe"] = 999.
        assert validator.call_count == 1
    record["records"][0]["pe"] = 999.
    result = index.select(market="TWSE", symbol="SYN1", trade_date=DAY, cutoff=CUTOFF)
    assert result["pe"] == 12.5 and result["raw_cells"]
    with pytest.raises(ValueError, match="CAPACITY"):
        OfficialPeIndex([before], max_bytes=1)
    with pytest.raises(ValueError, match="CAPACITY"):
        OfficialPeIndex([before], max_observations=0)


def test_join_detects_quote_conflict_and_does_not_mutate_input_or_adopt_external_value():
    records = [snapshot()]
    rows = [dict(Date=DAY, PE=10., Close=100.)]
    before = deepcopy((rows, records))
    joined, diagnostics = join_pe_rows(rows, records, market="TWSE", symbol="SYN1", cutoff=CUTOFF)
    assert joined[0]["PE"] is None and diagnostics[0]["status"] == "CONFLICT"
    assert diagnostics[0]["quote_pe"] == 10.
    assert (rows, records) == before
    with pytest.raises(ValueError, match="IDENTITY"):
        join_pe_rows([dict(Date=DAY, market="TPEX")], records,
                     market="TWSE", symbol="SYN1", cutoff=CUTOFF)


def test_source_reported_close_is_exact_and_wrong_quote_close_is_conflict():
    rows = [dict(Date=DAY, PE=12.5, Close=101.)]
    joined, diagnostics = join_pe_rows(rows, [snapshot()], market="TWSE", symbol="SYN1", cutoff=CUTOFF)
    assert rows[0]["PE"] == 12.5 and joined[0]["PE"] is None
    assert diagnostics[0]["status"] == "CONFLICT"
    assert diagnostics[0]["reason"] == "OFFICIAL_PE_QUOTE_CLOSE_CONFLICT"
    assert diagnostics[0]["reported_close"] == 100.
    # TPEx P/E endpoint has no closing-price column. Nothing is inferred from
    # the P/E, dividend or price/book fields to manufacture a comparison.
    joined, diagnostics = join_pe_rows(rows, [snapshot("TPEX")], market="TPEX", symbol="SYN1", cutoff=CUTOFF)
    assert joined[0]["PE"] == 12.5 and diagnostics[0]["reported_close"] is None


@pytest.mark.parametrize("price", ["NaN", "1,2,3"])
def test_invalid_source_close_cannot_certify_quote_binding(price):
    item = payload(); item["data"][0][2] = price
    with pytest.raises(SchemaChangedError):
        build_snapshot(json.dumps(item).encode(), market="TWSE", trade_date=DAY, fetched_at=FETCH)


@pytest.mark.parametrize("price", ["0.00", "-1"])
def test_reported_nonpositive_close_is_retained_not_a_valid_quote_or_whole_batch_failure(price):
    item = payload(symbol="2321"); item["data"][0][2] = price
    # The shape reproduces a real official-table issue but values/names are
    # explicitly synthetic. Live 2026-10-13 rows 2321 / 5906 reported 0.00.
    item["data"].append(["SYN2", "SYNTHETIC ONLY", "100", "10.00", "115/2"])
    item["total"] = 2
    record = build_snapshot(json.dumps(item).encode(), market="TWSE", trade_date=DAY, fetched_at=FETCH)
    index = OfficialPeIndex([record])
    suspect = index.select(market="TWSE", symbol="2321", trade_date=DAY, cutoff=CUTOFF)
    assert suspect["reported_close"] == float(price)
    assert suspect["raw_close"] == price and suspect["close_status"] == "NON_POSITIVE"
    assert validate_pe_selection(suspect, market="TWSE", symbol="2321", trade_date=DAY,
                                 cutoff=CUTOFF) == suspect
    assert index.select(market="TWSE", symbol="SYN2", trade_date=DAY, cutoff=CUTOFF)["pe"] == 10.
    joined, diagnostic = join_pe_rows([dict(Date=DAY, Close=100., PE=None)], [record],
                                      market="TWSE", symbol="2321", cutoff=CUTOFF)
    assert joined[0]["PE"] is None and diagnostic[0]["status"] == "CONFLICT"
    assert diagnostic[0]["reason"] == "OFFICIAL_PE_QUOTE_CLOSE_CONFLICT"


def test_client_is_one_bounded_raw_request_and_clock_is_response_completion():
    http = Mock()
    content = json.dumps(payload("TPEX")).encode()
    http.request_bytes.return_value = SimpleNamespace(content=content, status_code=200, attempts=1)
    clock = Mock(return_value=datetime.fromisoformat(FETCH))
    result = OfficialPeClient(http, clock=clock).fetch(market="TPEX", trade_date=DAY)
    http.request_bytes.assert_called_once_with("GET", ENDPOINTS["TPEX"],
        params=dict(date="2026/10/13", cate="", response="json"),
        context="TPEX dated official P/E at 2026-10-13")
    assert result["available_at"] == FETCH and clock.call_count == 1


def test_repository_append_only_read_only_cutoff_bulk_and_conflicts(tmp_path):
    path = tmp_path / "synthetic-official-pe.sqlite3"
    repo = OfficialPeRepository(path)
    assert repo.load_as_of(market="TWSE", trade_dates=[DAY], cutoff=CUTOFF) == []
    assert not path.exists()
    first, second = snapshot(), snapshot(pe="20.00")
    assert repo.record(first) == repo.record(first)
    repo.record(second)
    loaded = repo.load_as_of(market="TWSE", trade_dates=[DAY], cutoff=CUTOFF)
    assert len(loaded) == 2 and selected(loaded)["status"] == "CONFLICT"
    assert repo.load_as_of(market="TPEX", trade_dates=[DAY], cutoff=CUTOFF) == []
    assert repo.load_as_of(market="TWSE", trade_dates=[DAY], cutoff="2026-10-13T13:30:00+08:00") == []
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM official_pe_observations").fetchone()[0] == 2
        connection.execute("UPDATE official_pe_observations SET evidence_id='changed' WHERE evidence_id=?", (first["evidence_id"],))
    with pytest.raises(ValueError, match="DATABASE_BINDING"):
        repo.load_as_of(market="TWSE", trade_dates=[DAY], cutoff=CUTOFF)


def test_repository_selects_latest_at_cutoff_without_loading_all_observation_versions(tmp_path):
    repository = OfficialPeRepository(tmp_path / "synthetic-many-versions.sqlite3")
    start = datetime.fromisoformat(FETCH)
    versions = [snapshot(pe=str(index + 1), fetched_at=(start + timedelta(seconds=index)).isoformat())
                for index in range(150)]
    for version in versions:
        repository.record(version)
    conflict = snapshot(pe="999", fetched_at=versions[-1]["fetched_at"])
    repository.record(conflict)
    early = repository.load_as_of(market="TWSE", trade_dates=[DAY],
                                  cutoff=(start + timedelta(seconds=70)).isoformat())
    assert len(early) == 1 and early[0] == versions[70]
    latest = repository.load_as_of(market="TWSE", trade_dates=[DAY], cutoff=CUTOFF)
    assert len(latest) == 2 and selected(latest)["status"] == "CONFLICT"
    # A bounded work index stays bounded even after many retries. All prior
    # immutable versions remain available to a query at an earlier cutoff.
    assert OfficialPeIndex(latest).select(market="TWSE", symbol="SYN1", trade_date=DAY,
                                         cutoff=CUTOFF)["status"] == "CONFLICT"
    with sqlite3.connect(repository.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM official_pe_observations").fetchone()[0] == 151


def multi_snapshot(market="TWSE", *, symbols=("SYN1", "SYN3", "SYN5", "SYN7", "SYN9")):
    item = payload(market)
    table = item if market == "TWSE" else item["tables"][0]
    table["data"] = [[symbol, *table["data"][0][1:]] for symbol in symbols]
    table["total" if market == "TWSE" else "totalCount"] = len(symbols)
    return build_snapshot(json.dumps(item).encode(), market=market, trade_date=DAY, fetched_at=FETCH)


@pytest.mark.parametrize("market", ["TWSE", "TPEX"])
@pytest.mark.parametrize("symbol", ["SYN0", "SYN1", "SYN2", "SYN5", "SYN9", "SYNA"])
def test_compact_membership_odd_tree_and_absence_proofs(market, symbol):
    index = OfficialPeIndex([multi_snapshot(market)])
    result = index.select(market=market, symbol=symbol, trade_date=DAY, cutoff=CUTOFF)
    assert validate_pe_selection(result, market=market, symbol=symbol,
                                 trade_date=DAY, cutoff=CUTOFF) == result
    assert (result["status"] == "REPORTED") == (symbol in ("SYN1", "SYN3", "SYN5", "SYN7", "SYN9"))
    assert "raw_response_utf8" not in json.dumps(result)
    if result["status"] == "REPORTED":
        assert len(result["selection_proofs"][0]["membership"]["siblings"]) == 3


@pytest.mark.parametrize("damage", ["parsed_pe", "raw_pe", "path", "index", "manifest", "source_time",
    "market", "date", "parser", "result", "extra_proof", "absence_neighbor"])
def test_compact_proof_rejects_changed_values_sources_and_absence(damage):
    symbol = "SYN2" if damage == "absence_neighbor" else "SYN1"
    result = OfficialPeIndex([multi_snapshot()]).select(market="TWSE", symbol=symbol,
                                                       trade_date=DAY, cutoff=CUTOFF)
    proof = result["selection_proofs"][0]
    if damage == "parsed_pe": proof["membership"]["record"]["pe"] = 999.
    elif damage == "raw_pe": proof["membership"]["record"]["raw_cells"][3] = "-"
    elif damage == "path": proof["membership"]["siblings"][0] = "0" * 64
    elif damage == "index": proof["membership"]["leaf_index"] = 4
    elif damage == "manifest": proof["manifest"]["content_sha256"] = "0" * 64
    elif damage == "source_time": proof["manifest"]["available_at"] = "2026-10-13T05:30:00Z"
    elif damage == "market": proof["manifest"]["market"] = "TPEX"
    elif damage == "date": proof["manifest"]["session_date"] = "2026-10-12"
    elif damage == "parser": proof["manifest"]["parser_version"] = "BAD"
    elif damage == "result": result["pe"] = 999.
    elif damage == "extra_proof": proof["unbound"] = "BAD"
    else: proof["absence"]["after"] = deepcopy(proof["absence"]["before"])
    with pytest.raises((ValueError, SchemaChangedError)):
        validate_pe_selection(result, market="TWSE", symbol=symbol, trade_date=DAY, cutoff=CUTOFF)


def test_compact_proof_cannot_move_known_not_reported_to_positive_or_remove_conflicting_version():
    first, second = snapshot(pe="-"), snapshot(pe="20.00")
    result = selected([first, second])
    assert result["status"] == "CONFLICT"
    assert validate_pe_selection(result, market="TWSE", symbol="SYN1", trade_date=DAY,
                                 cutoff=CUTOFF) == result
    result["status"], result["pe"] = "REPORTED", 20.
    with pytest.raises(ValueError, match="BINDING"):
        validate_pe_selection(result, market="TWSE", symbol="SYN1", trade_date=DAY, cutoff=CUTOFF)
    missing = selected([snapshot()])
    with pytest.raises(ValueError, match="NOT_AVAILABLE"):
        validate_pe_selection(missing, market="TWSE", symbol="SYN1", trade_date=DAY,
                              cutoff="2026-10-13T13:30:00+08:00")
    with pytest.raises(ValueError, match="PROOF_MISSING"):
        validate_pe_selection(selected([]), market="TWSE", symbol="SYN1", trade_date=DAY, cutoff=CUTOFF)

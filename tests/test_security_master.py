"""Synthetic official-shaped fixtures; not real evidence/publication receipts."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import pytest

from domain.security_master import content_id, verify_compact_security_selection
from infrastructure.crawler.security_master import (
    capture_security_master, clear_security_master_cache, collect_security_master,
    load_security_master, parse_isin_master, parse_issuer_master, prepare_security_master,
    select_security_master, validate_security_master, verify_security_selection,
)

OBSERVED = "2026-10-14T06:00:00+00:00"
CUTOFF = "2026-10-14T14:01:00+08:00"
HEADERS = ["有價證券代號及名稱", "國際證券辨識號碼(ISIN Code)", "上市日", "市場別", "產業別", "CFICode", "備註"]


def synthetic_instruments(*, market="TWSE", rows=None):
    rows = rows or [
        ("股票", ["2330　合成普通股", "TW0002330008", "1994/09/05", "上市", "半導體業", "ESVUFR", ""]),
        ("特別股", ["8112A　合成甲特", "TW0008112A08", "2022/03/07", "上市", "電子通路業", "EPRNAR", ""]),
        ("臺灣存託憑證(TDR)", ["9105　合成-DR", "TW0009105007", "2003/09/22", "上市", "", "EDSDDR", ""]),
    ]
    if market == "TPEX":
        rows = [(category, [*row[:3], "上櫃", *row[4:]]) for category, row in rows]
    title = "本國上市證券國際證券辨識號碼一覽表" if market == "TWSE" else "本國上櫃證券國際證券辨識號碼一覽表"
    html = f"<h2>{title}</h2><h2>最近更新日期:2026/10/14</h2><table><tr>" + "".join(f"<td>{cell}</td>" for cell in HEADERS) + "</tr>"
    for category, row in rows:
        html += f"<tr><td colspan=7>{category}</td></tr><tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    return (html + "</table>").encode("cp950")


def synthetic_issuers(*, market="TWSE", listing="19940905", rows=None):
    if rows is None:
        rows = [dict(公司代號="2330", 公司簡稱="合成普通股", 公司名稱="合成股份有限公司", 營利事業統一編號="12345678", 上市日期=listing),
                dict(公司代號="8112", 公司簡稱="合成", 公司名稱="合成至上股份有限公司", 營利事業統一編號="87654321", 上市日期="20071231")]
    if market == "TPEX":
        fields = dict(zip(["公司代號", "公司簡稱", "公司名稱", "營利事業統一編號", "上市日期"],
            ["SecuritiesCompanyCode", "CompanyAbbreviation", "CompanyName", "UnifiedBusinessNo.", "DateOfListing"]))
        rows = [{fields[key]: value for key, value in row.items()} for row in rows]
    return json.dumps(rows, ensure_ascii=False).encode("utf-8")


def packet(*, market="TWSE", listing="19940905", store=None):
    return capture_security_master(market=market, instrument_content=synthetic_instruments(market=market),
        issuer_content=synthetic_issuers(market=market, listing=listing), fetched_at=OBSERVED, document_store=store)


@pytest.fixture(autouse=True)
def clear_cache():
    clear_security_master_cache()
    yield
    clear_security_master_cache()


@pytest.mark.parametrize("market", ["TWSE", "TPEX"])
def test_official_raw_source_shapes_bind_common_security_and_issuer(market):
    master = packet(market=market)
    selection = select_security_master(master, "2330", CUTOFF)
    assert selection["status"] == "VERIFIED"
    assert selection["issuer_symbol"] == "2330"
    assert selection["security_type"] == "COMMON"
    assert selection["instrument"]["isin"] == "TW0002330008"
    assert selection["instrument"]["listing_date"] == "1994-09-05"
    assert verify_compact_security_selection(selection, cutoff=CUTOFF) == selection
    assert "body_base64" not in json.dumps(selection)
    assert selection["master"]["historical_information_set_certified"] is False


@pytest.mark.parametrize("symbol,kind", [("8112A", "PREFERRED"), ("9105", "DR")])
def test_preferred_and_dr_are_explicitly_out_of_scope_not_suffix_eps(symbol, kind):
    selection = select_security_master(packet(), symbol, CUTOFF)
    assert selection["status"] == "OUT_OF_SCOPE"
    assert selection["security_type"] == kind
    assert selection["issuer_symbol"] is None
    assert selection["issuer"] is None
    assert selection["low_entry_model_supported"] is False


def test_missing_security_does_not_prove_delisted_or_map_ordinary_issuer():
    selection = select_security_master(packet(), "3036A", CUTOFF)
    assert selection["status"] == "UNKNOWN"
    assert selection["reason"] == "SECURITY_MASTER_ROW_MISSING"
    assert selection["issuer_symbol"] is None


def test_today_master_cannot_certify_past_membership_even_with_1994_listing_date():
    selection = select_security_master(packet(), "2330", "2026-09-16T13:30:00+08:00")
    assert selection["status"] == "UNKNOWN"
    assert selection["reason"] == "SECURITY_MASTER_NOT_AVAILABLE_AT_CUTOFF"
    assert selection["instrument"] is None
    assert selection["issuer"] is None


def test_source_time_is_observation_not_listing_date_or_table_production_date():
    master = packet()
    assert master["available_at"] == master["fetched_at"] == OBSERVED
    assert master["original_publication_at"] is None
    assert master["time_policy"] == "FIRST_VERIFIED_OBSERVATION_NO_HISTORICAL_BACKDATE"
    altered = deepcopy(master); altered["available_at"] = "1994-09-05T00:00:00+08:00"
    with pytest.raises(ValueError, match="POLICY_OR_TIME"):
        validate_security_master(altered)


def test_two_source_observation_times_preserved_and_availability_is_latest_source():
    earlier = "2026-10-14T05:59:00+00:00"
    master = capture_security_master(market="TWSE", instrument_content=synthetic_instruments(),
        issuer_content=synthetic_issuers(), fetched_at=OBSERVED,
        instrument_fetched_at=earlier, issuer_fetched_at=OBSERVED)
    assert master["instruments_document"]["retrieved_at"] == earlier
    assert master["available_at"] == OBSERVED
    assert select_security_master(master, "2330", "2026-10-14T05:59:30+00:00")["status"] == "UNKNOWN"
    assert select_security_master(master, "2330", CUTOFF)["status"] == "VERIFIED"


def test_listing_dates_conflict_does_not_silently_choose_one_source():
    selection = select_security_master(packet(listing="19940906"), "2330", CUTOFF)
    assert selection["status"] == "CONFLICT"
    assert selection["reason"] == "SECURITY_ISSUER_LISTING_DATE_UNVERIFIED"
    assert not selection["low_entry_model_supported"]


def test_unknown_cfi_or_table_category_does_not_infer_common_from_four_digits():
    content = synthetic_instruments(rows=[("ETF", ["2330　合成", "TW0002330008", "1994/09/05", "上市", "", "CEOGEU", ""])])
    master = capture_security_master(market="TWSE", instrument_content=content,
        issuer_content=synthetic_issuers(), fetched_at=OBSERVED)
    selected = select_security_master(master, "2330", CUTOFF)
    assert selected["security_type"] == "OTHER"
    assert selected["status"] == "OUT_OF_SCOPE"


def test_innovation_board_exact_official_market_label_not_guessed_regular_board():
    # Live official ISIN table uses 上市臺灣創新板, not plain 上市, in this section.
    row = ["2237　合成-創", "TW0002237005", "2026/08/10", "上市臺灣創新板", "汽車工業", "ESVUFR", ""]
    instruments = parse_isin_master(synthetic_instruments(rows=[("創新板", row)]), "TWSE")
    assert instruments[0].security_type == "COMMON"
    assert instruments[0].raw_cells[3] == "上市臺灣創新板"


@pytest.mark.parametrize("target", ["instrument", "issuer", "status", "proof", "root", "market", "source"])
def test_compact_proof_rejects_modified_projection_even_if_selection_id_rehashed(target):
    selection = select_security_master(packet(), "2330", CUTOFF)
    changed = deepcopy(selection)
    if target == "instrument": changed["instrument"]["isin"] = "TW0008112A08"
    elif target == "issuer": changed["issuer"]["symbol"] = "8112"
    elif target == "status": changed["status"] = "OUT_OF_SCOPE"
    elif target == "proof": changed["instrument_inclusion"]["siblings"][0] = "0" * 64
    elif target == "root": changed["master"]["instruments_root"] = "0" * 64
    elif target == "market": changed["market"] = "TPEX"
    else: changed["master"]["instruments_document"]["url"] = "https://isin.twse.com.tw.evil.example/isin/C_public.jsp?strMode=2"
    changed["selection_id"] = content_id({key: value for key, value in changed.items() if key not in {"master", "selection_id"}})
    with pytest.raises(ValueError):
        verify_compact_security_selection(changed, cutoff=CUTOFF)


def test_compact_cutoff_cannot_be_changed_without_reselection():
    selection = select_security_master(packet(), "2330", CUTOFF)
    with pytest.raises(ValueError, match="CUTOFF_OR_SCHEMA"):
        verify_compact_security_selection(selection, cutoff="2026-10-15T14:01:00+08:00")


def test_integrity_inclusion_cannot_be_relabelled_authenticity_or_historical_certificate():
    for key, value in [("source_authenticity", "CERTIFIED"), ("historical_information_set_certified", True),
                       ("limitations", []), ("availability_basis", "ORIGINAL_PUBLICATION")]:
        selected = select_security_master(packet(), "2330", CUTOFF)
        selected["master"][key] = value
        with pytest.raises(ValueError, match="HEADER"):
            verify_compact_security_selection(selected, cutoff=CUTOFF)


def test_full_master_rejects_raw_body_hash_mismatch_and_forged_root():
    master = packet(); master["instruments_document"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_security_master(master)
    master = packet(); master["instruments_root"] = "0" * 64
    with pytest.raises(ValueError, match="CONTENT_ID"):
        validate_security_master(master)


def test_duplicate_rows_and_schema_changes_fail_closed():
    raw = synthetic_instruments()
    rows = parse_isin_master(raw, "TWSE")
    duplicate = synthetic_instruments(rows=[(rows[0].source_category, list(rows[0].raw_cells))] * 2)
    with pytest.raises(ValueError, match="DUPLICATE_INSTRUMENT"):
        parse_isin_master(duplicate, "TWSE")
    bad_header = raw.replace("CFICode".encode(), "FakeCode".encode())
    with pytest.raises(ValueError, match="COLUMN_SCHEMA"):
        parse_isin_master(bad_header, "TWSE")


def test_parsed_rows_are_frozen_and_public_proof_cannot_pollute_next_selection():
    rows = parse_isin_master(synthetic_instruments(), "TWSE")
    with pytest.raises(FrozenInstanceError):
        rows[0].name = "mutated"
    context = prepare_security_master(packet())
    with pytest.raises(TypeError):
        context.instrument_index["2330"] = 1
    selected = select_security_master(context, "2330", CUTOFF)
    selected["instrument"]["name"] = "mutated"
    assert select_security_master(context, "2330", CUTOFF)["instrument"]["name"] == "合成普通股"


def test_work_context_does_not_reparse_or_reread_shared_originals_per_stock(monkeypatch):
    context = prepare_security_master(packet())
    def forbidden(*args, **kwargs): raise AssertionError("full original traversal repeated")
    monkeypatch.setattr("infrastructure.crawler.security_master._document_content", forbidden)
    assert select_security_master(context, "2330", CUTOFF)["status"] == "VERIFIED"
    assert select_security_master(context, "8112A", CUTOFF)["status"] == "OUT_OF_SCOPE"


def test_context_source_guard_is_checked_for_each_selection():
    seen = []
    context = prepare_security_master(packet(), check_sources=lambda: seen.append("checked"))
    select_security_master(context, "2330", CUTOFF)
    select_security_master(context, "8112A", CUTOFF)
    assert seen == ["checked", "checked"]


def test_capture_deterministic_id_and_changed_source_new_identity():
    assert packet()["evidence_id"] == packet()["evidence_id"]
    assert packet()["evidence_id"] != packet(listing="19940906")["evidence_id"]


class SyntheticClient:
    def __init__(self):
        self.config = SimpleNamespace(allow_insecure_ssl_fallback=False, min_request_interval_seconds=1)
        self.calls = []
    def request_bytes(self, method, url, *, before_attempt, content_validator, context):
        before_attempt(); self.calls.append(url)
        content = synthetic_instruments() if "isin" in url else synthetic_issuers()
        content_validator(content)
        return SimpleNamespace(content=content)


def test_batch_collection_write_once_exact_budget_and_readback(tmp_path):
    client = SyntheticClient()
    path = collect_security_master(tmp_path, "TWSE", client=client, now=lambda: OBSERVED)
    first = path.read_bytes()
    assert len(client.calls) == 2
    assert collect_security_master(tmp_path, "TWSE", client=SyntheticClient(), now=lambda: OBSERVED) == path
    assert path.read_bytes() == first
    master, docs = load_security_master(path)
    proof = select_security_master(master, "2330", CUTOFF, document_reader=docs.read)
    assert verify_security_selection(proof, cutoff=CUTOFF, document_reader=docs.read) == proof
    assert json.loads(path.read_text())["instruments_document"]["body_ref"]["sha256"] == master["instruments_document"]["sha256"]


def test_unsafe_transport_and_extra_request_budget_refused_before_fetch(tmp_path):
    client = SyntheticClient(); client.config.allow_insecure_ssl_fallback = True
    with pytest.raises(ValueError, match="SECURE_RATE"):
        collect_security_master(tmp_path, "TWSE", client=client)
    with pytest.raises(ValueError, match="TWO_BATCH"):
        collect_security_master(tmp_path, "TWSE", client=SyntheticClient(), max_requests=3)
    assert client.calls == []


def test_timezone_and_wrong_market_rejected():
    with pytest.raises(ValueError, match="AWARE_TIME"):
        capture_security_master(market="TWSE", instrument_content=synthetic_instruments(),
            issuer_content=synthetic_issuers(), fetched_at="2026-10-14T14:00:00")
    with pytest.raises(ValueError, match="TITLE"):
        parse_isin_master(synthetic_instruments(), "TPEX")
    with pytest.raises(ValueError):
        parse_issuer_master(synthetic_issuers(), "TPEX")


def test_legacy_identity_calls_remain_unchanged():
    from domain.security_identity import resolve_security_identity
    assert resolve_security_identity("3036A", "文曄甲特").issuer_symbol == "3036"
    assert resolve_security_identity("8112A", "無法辨識").issuer_symbol == "8112A"

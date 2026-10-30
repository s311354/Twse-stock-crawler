"""SYNTHETIC ONLY: lossless decoding/provenance, no real issuer certification."""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json

import pytest

from application.les_v7 import prepare_quality_context
from domain.fundamental_derivation import FiledFact, FiledFinancials, _derive_audit_evidence
from domain.les_v7.financial_sources_r5 import audit_proof_r5
from domain.les_v7.quality_policy import r6_quality_policy
from domain.les_v7.quality_reported import evaluate_financials_reported, validate_reported_result
from domain.strategies.low_entry_v6_filings import ixbrl_record
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl, parse_ixbrl_evidence, IXBRL_PARSER_VERSION
from infrastructure.fundamentals.mops_repository import MopsRepository
from tests.fundamentals.test_filing_refresh_planner import _candidate
from tests.fundamentals.test_ttm_audit import document as filing_document
from tests.fundamentals.test_v7_audit_cash_r5 import original


CUTOFF = "2026-10-14T13:50:50.434430Z"
AVAILABLE = "2026-08-16T06:29:20.942116Z"
BODY = ("公司民國115年及114年6月30日之合併資產負債表，業經本會計師核閱。\n"
        "結論\n依本會計師核閱結果，並未發現上開合併財務報表未允當表達。")


def document(body, *, damage=b"", extra=b"", date="2026-08-11"):
    return (b'<html xmlns:xbrli="http://www.xbrl.org/2003/instance" '
            b'xmlns:currency="http://www.xbrl.org/2003/iso4217"><head>'
            b'<meta charset="big5"><meta charset="utf-8"></head><body>'
            b'<xbrli:context id="AUDIT"><xbrli:period><xbrli:instant>'
            + date.encode() + b'</xbrli:instant></xbrli:period></xbrli:context>'
            b'<xbrli:unit id="money"><xbrli:measure>currency:TWD</xbrli:measure></xbrli:unit>'
            b'<ix:nonFraction name="ifrs-full:Revenue" contextRef="AUDIT" unitRef="money" scale="3">1,234</ix:nonFraction>'
            b'<ix:nonNumeric name="tifrs-ar:UnqualifiedOpinionAbstract" contextRef="AUDIT">Y</ix:nonNumeric>'
            b'<ix:nonNumeric name="tifrs-ar:AccountantsReportBody" contextRef="AUDIT">'
            + body + b'</ix:nonNumeric>' + extra + b'<div>' + damage + b'</div></body></html>')


def filing(content, facts=None):
    facts = parse_ixbrl(content) if facts is None else facts
    return FiledFinancials(2026, 2, AVAILABLE, 1, tuple(FiledFact(**asdict(f)) for f in facts),
        content_sha256=sha256(content).hexdigest(),
        source_url="https://mopsov.twse.com.tw/SYNTHETIC_ONLY.html",
        period_end="2026-06-30", audit_status="reviewed")


@pytest.mark.parametrize("codec", ["utf-8", "big5", "cp950"])
def test_valid_document_preserves_existing_text_numbers_units_and_layout(codec):
    text = BODY.replace("\n", "<br/>")
    content = document(text.encode(codec))
    facts, receipt = parse_ixbrl_evidence(content)
    assert facts == parse_ixbrl(content)
    assert facts[-1].text_value == BODY
    assert facts[0].numeric_value == 1_234_000 and facts[0].unit == "iso4217:TWD"
    assert receipt["mode"] == "DOCUMENT_STRICT" and not receipt["facts"]
    assert receipt["content_sha256"] == sha256(content).hexdigest()


def test_unrelated_bad_bytes_do_not_force_entire_audit_body_to_latin1():
    content = document(BODY.encode("big5"), damage=b"\x84P")
    facts, receipt = parse_ixbrl_evidence(content)
    assert facts[-1].text_value == BODY
    assert facts[0].numeric_value == 1_234_000
    assert receipt["mode"] == "STRICT_FACT_RANGES" and len(receipt["document_failures"]) == 3
    entry = receipt["facts"][0]
    raw = content[entry["byte_start"]:entry["byte_end"]]
    assert raw == BODY.encode("big5")
    assert entry["raw_sha256"] == sha256(raw).hexdigest()
    assert entry["text_sha256"] == sha256(BODY.encode()).hexdigest()
    assert entry["status"] == "VERIFIED_DECODED" and entry["equivalent_encodings"] == ["big5", "cp950"]
    assert raw.decode(entry["encoding"]).encode(entry["encoding"]) == raw
    assert parse_ixbrl_evidence(content) == (facts, receipt)
    legacy = filing(content, parse_ixbrl(content.decode("latin-1")))
    old_proof = audit_proof_r5(legacy, symbol="SYNTHETIC_ONLY", original_verified=True)
    frozen = deepcopy(old_proof)
    assert old_proof["status"] == "PENDING_VERIFICATION"
    proof = audit_proof_r5(filing(content), symbol="SYNTHETIC_ONLY", original_verified=True)
    assert proof["status"] == "VERIFIED_REPORTED" and proof["audit"]["audit_opinion"] == "REVIEWED_UNMODIFIED"
    assert proof["available_at"] == AVAILABLE and proof["content_sha256"] == receipt["content_sha256"]
    assert proof["period_binding"]["auditor_report_date"] == "2026-08-11"
    assert audit_proof_r5(legacy, symbol="SYNTHETIC_ONLY", original_verified=True) == frozen


@pytest.mark.parametrize("date", ["2026-06-30", "2026-08-11"])
def test_invalid_bytes_inside_body_stay_pending_even_with_english_binding_and_unqualified_flag(date):
    body = (b'Reviewed consolidated financial statements as of June 30, 2026. '
            + BODY.encode("big5") + b"\x84P")
    content = document(body, date=date)
    facts, receipt = parse_ixbrl_evidence(content)
    assert facts[-1].text_value.startswith("UNVERIFIED_TEXT_ENCODING[")
    assert facts[0].numeric_value == 1_234_000
    assert receipt["facts"][0]["status"] == "PENDING_VERIFICATION"
    assert receipt["facts"][0]["reason"] == "UNDECODABLE_FACT_BYTES"
    proof = audit_proof_r5(filing(content), symbol="SYNTHETIC_ONLY", original_verified=True)
    assert proof["status"] == "PENDING_VERIFICATION"  # Never ignore bytes or missing body.


@pytest.mark.parametrize("concept", ["UnqualifiedOpinionAbstract", "QualifiedOpinionAbstract", "AdverseOpinionAbstract"])
@pytest.mark.parametrize("date", ["2026-06-30", "2026-08-11"])
def test_unreadable_opinion_flag_is_not_truthy_or_hidden_by_good_flag_or_body(concept, date):
    content = document(BODY.encode("big5"), date=date)
    broken_flag = (b'<ix:nonNumeric name="tifrs-ar:' + concept.encode()
                   + b'" contextRef="AUDIT">\x84P</ix:nonNumeric>')
    content = content.replace(b'</body>', broken_flag + b'</body>')
    facts, receipt = parse_ixbrl_evidence(content)
    assert any(r["concept"] == "tifrs-ar:" + concept and r["status"] == "PENDING_VERIFICATION" for r in receipt["facts"])
    proof = audit_proof_r5(filing(content), symbol="SYNTHETIC_ONLY", original_verified=True)
    assert proof["audit"]["audit_opinion"] == "UNKNOWN"
    assert proof["audit"]["audit_review_required"] is True
    assert proof["audit"]["going_concern"] is None
    # The generic v4/v5 derivation must not turn a nonempty diagnostic into Y.
    if date == "2026-06-30":
        assert _derive_audit_evidence(filing(content))["audit_opinion"] == "UNKNOWN"


@pytest.mark.parametrize("concept", ["AdverseOpinionAbstract", "DisclaimerOfOpinionAbstract"])
def test_confirmed_hard_audit_risk_survives_another_unreadable_abstract(concept):
    content = document(BODY.encode("big5"), date="2026-06-30").replace(
        b'>Y</ix:nonNumeric>', b'>\x84P</ix:nonNumeric>', 1)
    valid_flag = (b'<ix:nonNumeric name="tifrs-ar:' + concept.encode()
                  + b'" contextRef="AUDIT">Y</ix:nonNumeric>')
    content = content.replace(b'</body>', valid_flag + b'</body>')
    expected = "ADVERSE" if concept.startswith("Adverse") else "DISCLAIMER"
    assert _derive_audit_evidence(filing(content))["audit_opinion"] == expected
    assert audit_proof_r5(filing(content), symbol="SYNTHETIC_ONLY", original_verified=True)["audit"]["audit_opinion"] == expected


@pytest.mark.parametrize("damage", ["unclosed", "wrong_close", "nested"])
def test_damaged_document_incomplete_audit_payload_is_not_silently_dropped(damage):
    content = document(BODY.encode("big5"), date="2026-06-30", damage=b"\x84P")
    ending = BODY.encode("big5") + b'</ix:nonNumeric>'
    if damage == "unclosed":
        content = content.replace(ending, BODY.encode("big5"))
    elif damage == "wrong_close":
        content = content.replace(ending, BODY.encode("big5") + b'</ix:nonFraction>')
    else:
        content = content.replace(ending, BODY.encode("big5")
            + b'<ix:nonNumeric name="tifrs-notes:Nested" contextRef="AUDIT">x</ix:nonNumeric></ix:nonNumeric>')
    facts, receipt = parse_ixbrl_evidence(content)
    body = next(f for f in facts if f.concept == 'tifrs-ar:AccountantsReportBody')
    assert body.text_value.startswith("UNVERIFIED_TEXT_ENCODING[")
    assert receipt["facts"][0]["reason"] == "INCOMPLETE_FACT_STRUCTURE"
    assert audit_proof_r5(filing(content), symbol="SYNTHETIC_ONLY", original_verified=True)["status"] == "PENDING_VERIFICATION"
    assert _derive_audit_evidence(filing(content))["audit_opinion"] == "UNKNOWN"


def test_mixed_valid_encodings_are_selected_per_fact_and_html_entities_preserved():
    extra = ('<ix:nonNumeric name="tifrs-notes:Note" contextRef="AUDIT">'
             '附註&nbsp;\n甲<br/>乙<span>丙</span></ix:nonNumeric>').encode()
    content = document(BODY.encode("big5"), damage=b"\x84P", extra=extra)
    facts, receipt = parse_ixbrl_evidence(content)
    assert facts[-2].text_value == BODY
    assert facts[-1].text_value == "附註\u00a0\n甲\n乙丙"
    assert [r["encoding"] for r in receipt["facts"]] == ["big5", "utf-8"]


def test_strict_cp950_extension_keeps_currency_text_without_substitution():
    extra = (b'<ix:nonNumeric name="tifrs-notes:Currency" contextRef="AUDIT">'
             + "金額€".encode("cp950") + b'</ix:nonNumeric>')
    facts, receipt = parse_ixbrl_evidence(document(BODY.encode("big5"), damage=b"\x84P", extra=extra))
    assert facts[-1].text_value == "金額€"
    assert receipt["facts"][-1]["encoding"] == "cp950"
    assert receipt["facts"][-1]["equivalent_encodings"] == ["cp950"]


def test_conflicting_reversible_encodings_are_not_resolved_by_misleading_meta():
    content = document(b"\xc2\xa1", damage=b"\x84P")
    facts, receipt = parse_ixbrl_evidence(content)
    assert facts[-1].text_value.startswith("UNVERIFIED_TEXT_ENCODING[")
    assert receipt["facts"][0]["reason"] == "AMBIGUOUS_FACT_ENCODING"
    assert set(receipt["facts"][0]["candidate_encodings"]) == {"utf-8", "big5", "cp950"}


def test_self_closing_nil_payload_does_not_get_a_reversed_byte_range():
    extra = b'<ix:nonFraction name="ifrs-full:Nil" contextRef="AUDIT" xsi:nil="true"/>'
    facts, receipt = parse_ixbrl_evidence(document(BODY.encode("big5"), damage=b"\x84P", extra=extra))
    assert facts[-1].numeric_value is None and facts[-1].text_value is None
    assert all(0 <= r["byte_start"] <= r["byte_end"] for r in receipt["facts"])


def test_repaired_text_keeps_actual_hard_risks_and_cutoff_binding():
    content = document((BODY + "\n繼續經營重大不確定性\n本公司有重大繼續經營風險。").encode("big5"), damage=b"\x84P")
    proof = audit_proof_r5(filing(content), symbol="SYNTHETIC_ONLY", original_verified=True)
    assert proof["audit"]["going_concern"] is True
    assert proof["available_at"] == AVAILABLE
    # A signing date after acquisition cannot establish current-period safety.
    future = document(BODY.encode("big5"), damage=b"\x84P", date="2026-08-17")
    assert audit_proof_r5(filing(future), symbol="SYNTHETIC_ONLY", original_verified=True)["status"] == "PENDING_VERIFICATION"


@pytest.mark.parametrize("bad_flag", [False, True])
def test_native_source_overlay_repairs_legacy_facts_without_rewriting_sqlite_or_original(tmp_path, bad_flag):
    _, source = original(BODY, opinion="UnqualifiedOpinionAbstract")
    source = replace(source, audit_status="reviewed")
    document_text = filing_document(source).decode().replace('<body>',
        '<body><link:schemaRef xlink:href="https://mops.twse.com.tw/tifrs-ci-cr-2026-03-31.xsd"/>')
    content = document_text.encode("big5") + b'<div>\x84P</div>'
    if bad_flag:
        content = content.replace(b'>Y</ix:nonNumeric>', b'>\x84P</ix:nonNumeric>', 1)
    legacy = parse_ixbrl(content.decode("latin-1"))
    repo = MopsRepository(tmp_path / "SYNTHETIC-mops")
    candidate = replace(_candidate(2026, 2), source_url=source.source_url, audit_status="reviewed")
    stored = repo.save_revision(candidate, content, legacy, available_at=source.available_at,
                               fetched_at=source.available_at)
    original_hash, database_hash = sha256(repo.raw_path(stored).read_bytes()).hexdigest(), sha256(repo.database_path.read_bytes()).hexdigest()
    old = replace(source, facts=tuple(FiledFact(**asdict(f)) for f in legacy), content_sha256=stored.content_sha256,
                  source_url=stored.source_url, available_at=stored.available_at)
    row = ixbrl_record("2330", old)
    before = deepcopy(row)
    values = repo.load_v7_financial_sources("2330", CUTOFF, periods=((2026, 2),), source_repair=True)
    repair = values["financial_repairs_r5"][stored.content_sha256]
    assert repair["audit_evidence_r5"]["status"] == "VERIFIED_REPORTED"
    receipt = repair["source_repair_r5"]["source_decoding"]
    assert receipt["content_sha256"] == stored.content_sha256 and receipt["parser_version"] == IXBRL_PARSER_VERSION
    context = dict(symbol="2330", market="TWSE", as_of=CUTOFF, quality_policy=r6_quality_policy(), filings=(row,))
    prepare_quality_context(context, financial_audits=values["financial_audits"],
        financial_supplements=values["financial_supplements"], financial_repairs_r5=values["financial_repairs_r5"])
    result = evaluate_financials_reported(context["filings"], symbol="2330", market="TWSE",
                                         cutoff=CUTOFF, policy=r6_quality_policy())
    assert result["financial_safety"]["state"] == ("UNKNOWN" if bad_flag else "PASS"), result["financial_safety"]
    if bad_flag:
        assert result["status"] == "DATA_INSUFFICIENT"
        assert "FINANCIAL_AUDIT_REVIEW_REQUIRED" in result["financial_safety"]["reasons"]
    validate_reported_result(result)
    assert row == before
    assert sha256(repo.database_path.read_bytes()).hexdigest() == database_hash
    assert sha256(repo.raw_path(stored).read_bytes()).hexdigest() == original_hash == stored.content_sha256
    assert repo.load_facts(stored.filing_id)[-1].text_value == legacy[-1].text_value
    assert repo.load_v7_financial_sources("2330", "2026-08-01T13:30:00+08:00",
        periods=((2026, 2),), source_repair=True)["financial_repairs_r5"] == {}
    # Public receipts are detached; mutation cannot poison the parsed work memo.
    receipt["facts"].clear()
    fresh = repo.load_v7_financial_sources("2330", CUTOFF, periods=((2026, 2),), source_repair=True)
    assert fresh["financial_repairs_r5"][stored.content_sha256]["source_repair_r5"]["source_decoding"]["facts"]
    json.dumps(fresh, allow_nan=False)


def test_decoder_version_is_part_of_legacy_and_work_cache_identity(tmp_path, monkeypatch):
    from infrastructure.fundamentals import mops_queries
    from application.daily_performance import DailyPerformance
    from tests.fundamentals.test_raw_parse_lifetime import work_scope
    content = document(BODY.encode("big5"), damage=b"\x84P")
    path = tmp_path / "SYNTHETIC.html"
    path.write_bytes(content)
    stat = path.stat()
    args = (path, sha256(content).hexdigest(), (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size), "mapping-v1")
    mops_queries._raw_normalized_facts.cache_clear()
    first = mops_queries._raw_normalized_facts(*args)
    with DailyPerformance().activate() as probe, work_scope():
        assert mops_queries._raw_normalized_facts(*args) == first
        assert mops_queries._raw_normalized_facts(*args) == first
        monkeypatch.setattr(mops_queries, "IXBRL_PARSER_VERSION", IXBRL_PARSER_VERSION + "-test-changed")
        assert mops_queries._raw_normalized_facts(*args) == first
        assert probe.counters["raw_unit_parse_calls"] == 2
    mops_queries._raw_normalized_facts(*args)
    assert mops_queries._raw_normalized_facts.cache_info().misses == 2
    mops_queries._raw_normalized_facts.cache_clear()


def test_disk_memo_restart_and_cache_disabled_have_identical_source_facts_and_receipts(tmp_path, monkeypatch):
    from application.daily_performance import DailyPerformance
    from application.daily_work import DailyWorkContext
    from infrastructure.fundamentals import mops_queries
    from infrastructure.storage.research_cache import SQLiteRawParseCache
    content = document(BODY.encode("big5"), damage=b"\x84P")
    path = tmp_path / "SYNTHETIC.html"
    path.write_bytes(content)
    stat = path.stat()
    args = (path, sha256(content).hexdigest(), (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size), "mapping-v1")
    results, counts = [], []
    for enabled in (False, True, True):
        probe = DailyPerformance()
        with SQLiteRawParseCache(tmp_path / "SYNTHETIC-parse.sqlite3") as disk:
            work = DailyWorkContext("SYNTHETIC-same-source-code", enabled=enabled, parse_memo=disk)
            try:
                with work.activate(), probe.activate():
                    results.append(mops_queries._raw_normalized_facts(*args, include_decoding=True))
            finally:
                work.close()
        counts.append(probe.counters["raw_unit_parse_calls"])
    assert results[0] == results[1] == results[2]
    assert counts == [1, 1, 0]
    monkeypatch.setattr(mops_queries, "IXBRL_PARSER_VERSION", IXBRL_PARSER_VERSION + "-new-decoder")
    probe = DailyPerformance()
    with SQLiteRawParseCache(tmp_path / "SYNTHETIC-parse.sqlite3") as disk:
        work = DailyWorkContext("SYNTHETIC-same-source-code", parse_memo=disk)
        try:
            with work.activate(), probe.activate():
                assert mops_queries._raw_normalized_facts(*args, include_decoding=True) == results[0]
                assert probe.counters["raw_unit_parse_calls"] == 1
        finally:
            work.close()

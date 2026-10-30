"""SYNTHETIC ONLY: bounded read-only audit, never certified market evidence."""
from contextlib import closing
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from application.quality_audit_v7 import (QualityAuditStock, _cfo_comparability, _field_gaps, audit_quality_v7,
    _financial_source_diagnostics, _industry_classification, quality_audit_summary, select_audit_symbols)
from domain.fundamental_derivation import FiledFinancials
from domain.les_v7.quality_policy import default_quality_policy, r2_quality_policy, r3_quality_policy
from infrastructure.storage.report_evidence import ReportEvidenceWriter
from interface.quality_audit_v7 import LocalQualityAuditSource, load_audit_universe, main, quality_source_versions
from tests.fundamentals.test_ttm_audit import cache
from tests.fundamentals.test_filing_refresh_planner import _candidate

CUTOFF = "2026-10-12T13:30:00+08:00"


def test_source_guard_covers_native_classifier_units_and_version_selection():
    versions = quality_source_versions()
    assert {"domain.fundamental_derivation", "domain.text_normalization",
            "application.research_evidence", "application.replay_contracts",
            "infrastructure.fundamentals.ixbrl_units", "infrastructure.fundamentals.mops_sqlite"} <= versions["module_sha256"].keys()
    assert len(versions["content_sha256"]) == 64


def test_subset_keeps_fixed_order_and_reports_unprocessed_not_full_universe():
    calls, evaluated = [], []
    class Source:
        def load_stock(self, symbol, cutoff, *, period_count):
            calls.append((symbol, cutoff, period_count))
            return QualityAuditStock((), provenance=(dict(symbol=symbol, synthetic=True),))
    def evaluator(records, **kwargs):
        evaluated.append((records, kwargs))
        return dict(status="DATA_INSUFFICIENT", quality_score=None, reason_codes=["SYNTHETIC_NO_ORIGINAL"],
            field_checks=[dict(field="cfo", period="2026Q2", status="MISSING", category="MISSING_DATA", reason="NO_ORIGINAL")])
    report = audit_quality_v7(Source(), ("2330", "3014", "6834", "2357"), CUTOFF,
        symbols=("6834", "3014"), limit=1, data_kind="SYNTHETIC_ONLY", evaluator=evaluator)
    assert calls == [("3014", CUTOFF, 9)]
    assert len(evaluated) == 1
    assert report["fixed_universe"] == ["2330", "3014", "6834", "2357"]
    assert report["requested_symbols"] == ["3014", "6834"]
    assert report["processed_symbols"] == ["3014"]
    assert report["unprocessed_symbols"] == ["2330", "6834", "2357"]
    assert report["summary"]["fixed_universe_count"] == 4
    assert report["summary"]["requested_count"] == 2
    assert report["summary"]["processed_count"] == 1
    assert report["summary"]["full_universe_processed"] is False
    assert report["summary"]["field_gap_category_counts"] == {"MISSING_DATA": 1}
    assert "Fixed universe 4; requested 2; processed 1" in quality_audit_summary(report)


@pytest.mark.parametrize("universe,kwargs", [([], {}), (["2330", "2330"], {}),
    (["2330"], dict(symbols=["9999"])), (["2330"], dict(limit=0)), (["2330"], dict(limit=True)),
    ([" 2330"], {}), (["2330"], dict(symbols=[]))])
def test_invalid_universe_or_subset_fails_before_loading(universe, kwargs):
    with pytest.raises(ValueError): select_audit_symbols(universe, **kwargs)


def test_one_stock_read_failure_keeps_explicit_gap_and_does_not_skip_company():
    class Source:
        def load_stock(self, symbol, cutoff, *, period_count): raise ValueError("RAW_HASH_MISMATCH")
    report = audit_quality_v7(Source(), ("2330",), CUTOFF, data_kind="SYNTHETIC_ONLY")
    assert report["rows"][0]["after"]["status"] == "DATA_INSUFFICIENT"
    assert report["rows"][0]["after"]["quality_score"] is None
    assert report["summary"]["processed_count"] == 1
    assert "RAW_HASH_MISMATCH" in report["rows"][0]["diagnostics"][0]["reason"]
    assert report["rows"][0]["industry_classification"]["status"] == "NOT_EVALUATED"


def test_r1_uses_same_original_legacy_mapping_not_r2_only_fields(monkeypatch):
    calls = []
    legacy = dict(symbol="2330", content_sha256="a" * 64, source="SYNTHETIC_ONLY")
    mapped = dict(legacy, financial_mapping_version="SYNTHETIC_R2", parent_equity=123)
    class Source:
        def load_stock(self, *args, **kwargs):
            return QualityAuditStock((mapped,), baseline_records=(legacy,))
    def evaluate(records, **kwargs):
        calls.append((records, kwargs["policy"]["profile_id"]))
        return dict(status="DATA_INSUFFICIENT", reason_codes=["SYNTHETIC_ONLY"])
    monkeypatch.setattr("domain.les_v7.quality.evaluate_financials", evaluate)
    audit_quality_v7(Source(), ("2330",), CUTOFF, data_kind="SYNTHETIC_ONLY",
        policy_resolver=lambda *args, **kwargs: dict(profile_id="QG_V7_EVIDENCE_R2", latest_required_period="2026Q2"))
    assert calls == [((legacy,), "QG_V7_BOUNDED_GAPS_R1"), ((mapped,), "QG_V7_EVIDENCE_R2")]
    assert calls[0][0][0]["content_sha256"] == calls[1][0][0]["content_sha256"]


def test_default_r4_audit_compares_same_original_r3_and_labels_diagnostic_gaps():
    from tests.strategies.test_v7_reported_quality import source_records
    class Source:
        def load_stock(self, *args, **kwargs):
            return QualityAuditStock(tuple(source_records()))
    report = audit_quality_v7(Source(), ("2330",), CUTOFF, data_kind="SYNTHETIC_ONLY")
    row = report["rows"][0]
    assert report["schema"] == "local-v7-quality-audit/v3"
    assert row["before"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R3"
    assert row["after"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R4"
    assert row["after"]["status"] == "PASS"
    assert row["after"]["core_reported_pass"] is True
    assert "core_field_gap_category_counts" in report["summary"]
    assert all("affects_gate" in gap for gap in row["field_gaps"])
    assert report["field_evidence_schema"] == "quality-field-audit/v1"
    assert row["cfo_comparability"]["status"] in {"VERIFIED_SAME_PERIOD_BASIS", "UNKNOWN"}
    assert row["industry_classification"]["status"] == "UNKNOWN"
    assert row["industry_classification"]["reason"] == "OFFICIAL_INDUSTRY_MASTER_NOT_SUPPLIED"
    assert "R4:" in quality_audit_summary(report)


def test_quality_field_gaps_expose_source_period_scope_units_revision_and_availability():
    check = dict(field="cfo", period="2026Q2", status="PENDING_VERIFICATION",
        category="INCOMPARABLE", reason="REVISION_BASIS_UNVERIFIED", source="MOPS_IXBRL",
        source_url="https://example.invalid/filing", scope="consolidated", currency="TWD",
        unit="thousand", value_unit="TWD thousand", basis="YTD", duration="YTD",
        revision_id="revision-2", content_sha256="a" * 64,
        available_at="2026-08-01T08:00:00Z", published_at=None, fetched_at="2026-09-01T00:00:00Z",
        availability_source="FIRST_OBSERVED_DOWNLOAD", checks=dict(identity=True, version=False))
    gaps = _field_gaps(dict(profile_id="QG_V7_REPORTED_PERIOD_R6", field_checks={"2026Q2:cfo": check},
        reported_result={"metrics": {"quality_verification": {"dependencies": ["2026Q2:cfo"]}}}))
    gap = gaps[0]
    assert gap["affects_gate"] is True
    assert (gap["source"], gap["period"], gap["scope"], gap["unit"], gap["basis"]) == (
        "MOPS_IXBRL", "2026Q2", "consolidated", "thousand", "YTD")
    assert gap["revision_id"] == "revision-2" and gap["available_at"] == "2026-08-01T08:00:00Z"
    assert gap["verification_checks"] == dict(identity=True, version=False)


def test_cfo_comparability_requires_verified_same_period_scope_basis_and_units():
    def receipt(period, basis="YTD", scope="consolidated", unit="thousand"):
        return dict(field="cfo", symbol="2330", market="TWSE", period=period, status="VERIFIED_REPORTED", basis=basis,
            scope=scope, currency="TWD", unit=unit, source="MOPS_IXBRL", source_url="https://example.invalid",
            revision_id="rev-" + period, content_sha256="b" * 64, available_at="2026-08-01T08:00:00Z",
            checks=dict(identity=True, period=True, currency_unit=True, version=True, availability=True, original_context=True))
    quality = dict(field_checks={"2026Q2:cfo": receipt("2026Q2"),
        "2026Q2:comparative:cfo": receipt("2025Q2")}, reported_inputs=dict(period="2026Q2",
        current_keys={"cfo": "2026Q2:cfo"}, comparison_keys={"cfo": "2026Q2:comparative:cfo"}))
    result = _cfo_comparability(quality)
    assert result["status"] == "VERIFIED_SAME_PERIOD_BASIS"
    assert result["same_accounting_basis"] is result["same_period"] is True
    quality["field_checks"]["2026Q2:comparative:cfo"]["scope"] = "individual"
    changed = _cfo_comparability(quality)
    assert changed["status"] == "UNKNOWN" and changed["same_accounting_basis"] is False
    assert changed["reason"] == "CFO_PROOF_OR_PERIOD_BASIS_COMPARABILITY_UNVERIFIED"


def test_official_industry_classification_is_bound_to_symbol_market_and_observation_cutoff():
    from tests.test_security_master import packet as security_packet
    from infrastructure.crawler.security_master import select_security_master

    cutoff = "2026-10-15T13:30:00+08:00"
    selected = select_security_master(security_packet(), "2330", cutoff)
    result = _industry_classification(selected, symbol="2330", market="TWSE", cutoff=cutoff)
    assert result["status"] == "VERIFIED_SOURCE_LABEL"
    assert result["industry_name"] == "半導體業"
    assert result["source"] == "TWSE_TPEX_OFFICIAL_ISIN_MASTER"
    assert result["industry_code"] is None
    assert result["industry_code_status"] == "UNKNOWN_NO_VERIFIED_CODEBOOK_MAPPING"
    assert result["historical_information_set_certified"] is False

    old_cutoff = "2026-09-16T13:30:00+08:00"
    old = select_security_master(security_packet(), "2330", old_cutoff)
    assert _industry_classification(old, symbol="2330", market="TWSE", cutoff=old_cutoff)["status"] == "UNKNOWN"
    assert _industry_classification(selected, symbol="2317", market="TWSE", cutoff=cutoff)["status"] == "CONFLICT"
    assert _industry_classification(None, symbol="2330", market="TWSE", cutoff=cutoff)["status"] == "UNKNOWN"


def test_financial_source_diagnostics_expose_audit_period_opinion_and_cash_mapping_receipts():
    from domain.financial_taxonomy import IAS7_OTHER_CASH_CONCEPT, DISPOSAL_GROUP_CASH_CONCEPT

    origin = dict(source="MOPS_IXBRL", source_url="https://example.invalid/filing", symbol="2330",
        market="TWSE", period_end="2026-06-30", available_at="2026-08-01T00:00:00Z",
        fetched_at="2026-08-02T00:00:00Z", scope="consolidated", currency="TWD", unit="thousand",
        content_sha256="a" * 64, revision_id="rev-current", revision_number=2)
    fields = {}
    for field, concept in (("cfo", "ifrs-full:CashFlowsFromUsedInOperatingActivities"),
                           ("cash_ias7_adjustment", IAS7_OTHER_CASH_CONCEPT),
                           ("cash_disposal_group", DISPOSAL_GROUP_CASH_CONCEPT)):
        fields[field] = dict(origin, status="RESOLVED", reason="REPORTED_EXACT_CONTEXT", category=None,
            basis="YTD" if field == "cfo" else "INSTANT", start_date="2026-01-01" if field == "cfo" else None,
            end_date="2026-06-30", reported_instant=None if field == "cfo" else "2026-06-30",
            cash_flow_period_start=None if field != "cash_ias7_adjustment" else "2026-01-01",
            contexts=[dict(concept=concept, context_id="ctx", start_date="2026-01-01" if field == "cfo" else None,
                end_date="2026-06-30", instant=None if field == "cfo" else "2026-06-30",
                unit="iso4217:TWD", dimensions=[], numeric_value=1000.)])
    record = dict(origin, taxonomy=dict(status="RESOLVED", schema_name="tifrs-ci-cr-2026-03-31.xsd",
        schema_refs=["https://www.xbrl.org/2026/tifrs-ci-cr-2026-03-31.xsd"], mapping_version="mapping-r1"),
        field_evidence=fields, validation_checks=dict(cash_flow=[dict(check_id="CASH_BALANCE_TIE", status="PASS",
            method="SOURCE_BOUND_CASH_COMPONENTS", inputs=["cash", "cash_ias7_adjustment",
                "cash_disposal_group", "cash_ending"], variance=0., source_binding_errors=[])],
            comparative_cash_flow=[]), cfo=1000., cash_ias7_adjustment=0., cash_disposal_group=0.)
    record["audit_evidence_r5"] = dict(status="VERIFIED_REPORTED", reason="SOURCE_BOUND",
        policy_version="AUDIT_R5", source="MOPS_IXBRL", source_url=origin["source_url"],
        content_sha256=origin["content_sha256"], available_at=origin["available_at"], scope=origin["scope"],
        symbol=origin["symbol"], market=origin["market"], currency=origin["currency"], unit=origin["unit"],
        period_end=origin["period_end"], period_binding=dict(method="SOURCE_BOUND_ACCOUNTANT_REPORT_DATE",
            financial_period_end="2026-06-30", auditor_report_date="2026-08-01",
            body_bindings=[dict(body_sha256="b" * 64, financial_period_quotes=["115年6月30日"], scope="consolidated")]),
        audit=dict(audit_opinion="QUALIFIED_SCOPE_LIMITATION", current_audit_opinion="QUALIFIED_SCOPE_LIMITATION",
            comparative_audit_opinion="UNMODIFIED", audit_opinion_source="REPORT_BODY_FALLBACK",
            comparative_audit_opinion_source="ABSTRACT_COMPARATIVE", audit_evidence_available=True))

    result = _financial_source_diagnostics((record,))[0]
    audit = result["audit_report"]
    assert audit["binding_verification"]["status"] == "VERIFIED_BOUND_AND_PARSED"
    assert audit["period_binding"]["financial_period_end"] == "2026-06-30"
    assert audit["period_binding"]["auditor_report_date"] == "2026-08-01"
    assert audit["opinions"]["audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert audit["opinions"]["comparative_audit_opinion"] == "UNMODIFIED"
    mappings = {(item["group"], item["field"]): item for item in result["cash_flow_mapping"]["fields"]}
    assert IAS7_OTHER_CASH_CONCEPT in mappings[("CURRENT", "cash_ias7_adjustment")]["expected_concepts"]
    assert DISPOSAL_GROUP_CASH_CONCEPT in mappings[("CURRENT", "cash_disposal_group")]["expected_concepts"]
    assert mappings[("CURRENT", "cash_ias7_adjustment")]["observed_concepts"] == [IAS7_OTHER_CASH_CONCEPT]
    assert mappings[("COMPARATIVE_YTD", "cfo")]["group"] == "COMPARATIVE_YTD"
    assert result["cash_flow_mapping"]["current_reconciliation"][0]["status"] == "PASS"
    changed = dict(record, audit_evidence_r5=dict(record["audit_evidence_r5"], period_binding=dict(
        record["audit_evidence_r5"]["period_binding"], financial_period_end="2025-06-30")))
    assert _financial_source_diagnostics((changed,))[0]["audit_report"]["binding_verification"]["status"] == "CONFLICT"


def test_local_audit_source_attaches_verified_industry_master_selection(tmp_path):
    from infrastructure.crawler.security_master import capture_security_master
    from infrastructure.research_documents import EvidenceDocuments
    from tests.test_security_master import OBSERVED, synthetic_instruments, synthetic_issuers

    root = tmp_path / "mops"
    cache(root)
    master_root = tmp_path / "security-master"
    documents = EvidenceDocuments(master_root / "objects")
    packet = capture_security_master(market="TWSE", instrument_content=synthetic_instruments(),
        issuer_content=synthetic_issuers(), fetched_at=OBSERVED, document_store=documents.store)
    manifest = master_root / packet["evidence_id"] / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(packet), encoding="utf-8")
    database = root / "mops.sqlite3"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        source = LocalQualityAuditSource(root, connection, industry_master_manifest=manifest)
        stock = source.load_stock("2330", "2026-10-15T13:30:00+08:00", period_count=9)
    assert stock.industry_evidence["status"] == "VERIFIED"
    assert stock.industry_evidence["instrument"]["sector"] == "半導體業"
    assert source.industry_master_manifest == manifest.resolve()


def test_explicit_r3_audit_retains_original_r2_comparison_contract():
    from tests.strategies.test_v7_reported_quality import source_records
    class Source:
        def load_stock(self, *args, **kwargs): return QualityAuditStock(tuple(source_records()))
    report = audit_quality_v7(Source(), ("2330",), CUTOFF, data_kind="SYNTHETIC_ONLY", policy_resolver=r3_quality_policy)
    assert report["schema"] == "local-v7-quality-audit/v2"
    assert report["quality_profile"] == "QG_V7_REPORTED_PERIOD_R3"
    assert report["baseline_profile"] == "QG_V7_EVIDENCE_R2"
    assert report["rows"][0]["before"]["profile_id"] == "QG_V7_EVIDENCE_R2"
    assert report["rows"][0]["after"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R3"
    assert report["comparison"] == "SAME_LOCAL_ORIGINALS_R2_TTM_REQUIRED_VS_R3_REPORTED_PERIOD_FINANCIAL_QUALITY_ONLY"
    assert "R3:" in quality_audit_summary(report)


def test_empty_r3_native_result_keeps_identity_mismatch_gap_without_crashing():
    from tests.strategies.test_v7_reported_quality import source_records
    class Source:
        def load_stock(self, *args, **kwargs): return QualityAuditStock(tuple(source_records()))
    report = audit_quality_v7(Source(), ("9999",), CUTOFF, data_kind="SYNTHETIC_ONLY")
    assert report["rows"][0]["after"]["status"] == "DATA_INSUFFICIENT"
    assert report["rows"][0]["after"]["reported_result"] is None


def test_explicit_r2_audit_retains_version_one_contract():
    class Source:
        def load_stock(self, *args, **kwargs): return QualityAuditStock(())
    report = audit_quality_v7(Source(), ("2330",), CUTOFF,
        data_kind="SYNTHETIC_ONLY", policy_resolver=r2_quality_policy)
    assert report["schema"] == "local-v7-quality-audit/v1"
    assert "quality_profile" not in report
    assert "R2:" in quality_audit_summary(report)


def test_r3_read_failure_is_not_mislabelled_as_a_legacy_r2_audit():
    class Source:
        def load_stock(self, *args, **kwargs): raise ValueError("SYNTHETIC_MISSING_ORIGINAL")
    report = audit_quality_v7(Source(), ("2330",), CUTOFF, data_kind="SYNTHETIC_ONLY")
    assert report["schema"] == "local-v7-quality-audit/v3"
    assert report["quality_profile"] == "QG_V7_REPORTED_PERIOD_R4"
    assert report["rows"][0]["after"]["status"] == "DATA_INSUFFICIENT"


def test_explicit_r3_read_failure_keeps_r3_not_default_r4_audit_identity():
    class Source:
        def load_stock(self, *args, **kwargs): raise ValueError("SYNTHETIC_MISSING_ORIGINAL")
    report = audit_quality_v7(Source(), ("2330",), CUTOFF, data_kind="SYNTHETIC_ONLY", policy=r3_quality_policy())
    assert report["schema"] == "local-v7-quality-audit/v2"
    assert report["quality_profile"] == "QG_V7_REPORTED_PERIOD_R3"
    assert report["baseline_profile"] == "QG_V7_EVIDENCE_R2"


def reference_bundle(folder, symbols):
    folder.mkdir(parents=True, exist_ok=True)
    references = []
    with ReportEvidenceWriter(folder / "report_evidence.sqlite3") as writer:
        for symbol in symbols:
            references.append(writer.retain(dict(symbol=symbol, signal_date="2026-10-12",
                model_version="7.0.0", parameter_version="SYNTHETIC_ONLY", data_kind="SYNTHETIC_ONLY")))
    report = folder / "reports.json"
    report.write_text(json.dumps(dict(reports=references[:1], stale_reports=references[1:],
                                     signal_date="2026-10-12")))
    return report


def test_full_legacy_range_binds_all_verified_active_and_stale_symbols(tmp_path):
    stocklist = tmp_path / "indices"
    stocklist.write_text("0;1;2")
    bundle = reference_bundle(tmp_path / "report", ("2330", "3014", "6834"))
    symbols, metadata = load_audit_universe(stocklist, universe_report=bundle)
    assert symbols == ("2330", "3014", "6834")
    assert metadata["referenced_reports"] == 3
    assert metadata["report_envelope_metadata"]["signal_date"] == "2026-10-12"
    assert metadata["universe_time_policy"] == "CURRENT_REPORT_UNIVERSE_NOT_HISTORICAL_MEMBERSHIP_PROOF"
    assert metadata["stocklist_sha256"] == hashlib.sha256(stocklist.read_bytes()).hexdigest()
    stocklist.write_text("0;2")
    with pytest.raises(ValueError, match="COMPLETE_CONTIGUOUS"):
        load_audit_universe(stocklist, universe_report=bundle)
    with pytest.raises(ValueError, match="REQUIRES_VERIFIED"):
        load_audit_universe(stocklist)


def test_universe_stream_rejects_corrupt_later_reference_not_just_first_subset(tmp_path):
    stocklist = tmp_path / "indices"
    stocklist.write_text("0;1")
    bundle = reference_bundle(tmp_path / "report", ("2330", "3014"))
    body = json.loads(bundle.read_text())
    body["stale_reports"][0]["report_sha256"] = "0" * 64
    bundle.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        load_audit_universe(stocklist, universe_report=bundle)


def test_embedded_unverified_universe_report_is_not_accepted(tmp_path):
    stocklist, report = tmp_path / "symbols", tmp_path / "reports.json"
    stocklist.write_text("2330")
    report.write_text(json.dumps(dict(reports=[dict(symbol="2330")], stale_reports=[])))
    with pytest.raises(ValueError, match="REFERENCE_REQUIRED"):
        load_audit_universe(stocklist, universe_report=report)


def test_native_loader_uses_one_bounded_join_and_keeps_cache_bytes_unchanged(tmp_path):
    root = tmp_path / "mops"
    cache(root)
    db = root / "mops.sqlite3"
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    statements = []
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        connection.set_trace_callback(statements.append)
        loaded = LocalQualityAuditSource(root, connection).load_stock("2330", CUTOFF, period_count=9)
        missing = LocalQualityAuditSource(root, connection).load_stock("9999", CUTOFF, period_count=9)
    joins = [statement for statement in statements if "LEFT JOIN facts" in statement]
    assert len(joins) == 1
    assert loaded.records and loaded.provenance
    assert missing.records == ()
    assert loaded.preflight["derived_snapshot_json_reads"] == 0
    assert not any("snapshot_json" in statement for statement in statements)
    assert not any(statement.lstrip().upper().startswith(("UPDATE", "INSERT", "DELETE", "CREATE")) for statement in statements)
    assert before == hashlib.sha256(db.read_bytes()).hexdigest()
    assert all(record["audit_evidence"]["status"] == "PENDING_VERIFICATION" for record in loaded.records)
    assert all(r["financial_mapping_version"] for r in loaded.records)
    assert len(loaded.baseline_records) == len(loaded.records)
    assert all("financial_mapping_version" not in r for r in loaded.baseline_records)
    assert [r["content_sha256"] for r in loaded.baseline_records] == [r["content_sha256"] for r in loaded.records]


def test_missing_original_does_not_promote_stored_amounts_or_audit_status(tmp_path):
    root = tmp_path / "mops"
    repo = cache(root)
    filing = repo.load_filings("2330")[-1]
    (root / filing.raw_path).unlink()
    with closing(sqlite3.connect((root / "mops.sqlite3").as_uri() + "?mode=ro", uri=True)) as connection:
        loaded = LocalQualityAuditSource(root, connection).load_stock("2330", CUTOFF, period_count=9)
    assert any(item["category"] == "MISSING_ORIGINAL" for item in loaded.diagnostics)
    latest = next(record for record in loaded.records if record["content_sha256"] == filing.content_sha256)
    assert latest["audit_evidence"]["status"] == "PENDING_VERIFICATION"
    assert latest["revenue"] is None


def test_future_revision_header_is_diagnosed_but_cutoff_originals_are_unchanged(tmp_path):
    root = tmp_path / "mops"
    repo = cache(root)
    def load():
        with closing(sqlite3.connect((root / "mops.sqlite3").as_uri() + "?mode=ro", uri=True)) as connection:
            return LocalQualityAuditSource(root, connection).load_stock("2330", CUTOFF, period_count=9)
    before = load()
    latest = repo.load_filings("2330")[-1]
    candidate = replace(_candidate(latest.fiscal_year, latest.quarter), period_end=latest.period_end,
                        source_url="https://example.test/SYNTHETIC_FUTURE")
    repo.save_revision(candidate, b"<html>SYNTHETIC FUTURE VERSION</html>", [],
        available_at="2026-12-01T10:00:00+08:00", fetched_at="2026-12-01T10:00:00+08:00")
    after = load()
    assert after.records == before.records
    assert after.baseline_records == before.baseline_records
    assert after.provenance == before.provenance
    assert any(item["category"] == "EXCLUDED_FUTURE_VERSION" for item in after.diagnostics)


def test_latest_conflict_remains_unusable_policy_candidate_not_older_fallback(tmp_path):
    root = tmp_path / "mops"
    source = LocalQualityAuditSource(root, None)
    candidate = _candidate(2026, 2)
    available = "2026-08-15T10:00:00+08:00"
    headers = [SimpleNamespace(filing_id=i, fiscal_year=2026, quarter=2, available_at=available,
        fetched_at=available, statement_scope="consolidated", content_sha256=char * 64,
        revision_number=1, period_end="2026-06-30", source_url=candidate.source_url,
        raw_path=f"raw/{i}.html", availability_source="FIRST_OBSERVED_DOWNLOAD") for i, char in ((1, "a"), (2, "b"))]
    filings = [FiledFinancials(fiscal_year=2026, quarter=2, available_at=available, revision_number=1,
        facts=(), filing_id=h.filing_id, content_sha256=h.content_sha256, period_end="2026-06-30") for h in headers]
    source.queries = SimpleNamespace(load_filings=lambda *args, **kwargs: headers,
                                    load_filed_financials=lambda *args, **kwargs: filings)
    loaded = source.load_stock("2330", CUTOFF, period_count=9)
    assert len(loaded.records) == 2 and all(r["catalog_only"] for r in loaded.records)
    assert all(r["field_evidence"] == {} for r in loaded.records)
    policy = default_quality_policy(loaded.records, symbol="2330", market="TWSE", cutoff=CUTOFF)
    assert policy["latest_required_period"] is None
    assert policy["period_selection_status"] == "LATEST_PERIOD_ORIGINAL_UNVERIFIED"
    assert "2026Q2" in loaded.preflight["missing_periods"]


def test_cli_new_immutable_output_and_subset_manifest(tmp_path, capsys):
    root, output, stocklist = tmp_path / "mops", tmp_path / "audit.json", tmp_path / "symbols"
    cache(root)
    stocklist.write_text("2330;9999")
    before = hashlib.sha256((root / "mops.sqlite3").read_bytes()).hexdigest()
    args = ["--cache", str(root), "--cutoff", CUTOFF, "--stocklist", str(stocklist),
            "--limit", "1", "--data-kind", "SYNTHETIC_ONLY", "--output", str(output)]
    assert main(args) == 0
    report = json.loads(output.read_text())
    assert report["fixed_universe"] == ["2330", "9999"]
    assert report["processed_symbols"] == ["2330"]
    assert report["unprocessed_symbols"] == ["9999"]
    assert report["data_kind"] == "SYNTHETIC_ONLY"
    assert report["source_metadata"]["code_versions"]["content_sha256"]
    assert report["rows"][0]["after"]["status"] != "PASS"  # Fixture lacks required audit/extended proofs.
    assert before == hashlib.sha256((root / "mops.sqlite3").read_bytes()).hexdigest()
    first = output.read_bytes()
    with pytest.raises(ValueError, match="OUTPUT_EXISTS"): main(args)
    assert output.read_bytes() == first
    with pytest.raises(ValueError, match="MUST_NOT_MODIFY_SOURCE"):
        main([*args[:-1], str(root / "audit.json")])


def test_timezone_missing_cutoff_rejected():
    with pytest.raises(ValueError, match="TIMEZONE"):
        audit_quality_v7(None, ("2330",), "2026-10-12")

"""SYNTHETIC original-context regressions; never production evidence."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib

import pandas as pd
import pytest

from application.filing_publication import audit_filings, render_filing_audit
from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.strategies.low_entry_v6_filings import ixbrl_record
from domain.strategies.low_entry_v6_quality import (evaluate_quality, financial_field_checks,
    normalize_quarters, normalize_quarters_with_diagnostics, verification_diagnostics)
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.mops_repository import MopsRepository
from tests.fundamentals.test_ixbrl_units import document, filed
from tests.fundamentals.test_filing_refresh_planner import _candidate
from tests.test_filing_publication_repair import fixture, reviewed
from infrastructure.publication_repair import audit_difference
from tests.strategies.test_low_entry_score_v6 import financials


def source(*, single_quarter=True, predecessor=False):
    body = document().decode()
    contexts = ""
    if single_quarter:
        contexts += '<xbrli:context id="Q2"><xbrli:period><xbrli:startDate>2026-04-01</xbrli:startDate><xbrli:endDate>2026-06-30</xbrli:endDate></xbrli:period></xbrli:context>'
        contexts += '<ix:nonFraction name="ifrs-full:Revenue" contextRef="Q2" unitRef="money" scale="3">1,800</ix:nonFraction>'
        contexts += '<ix:nonFraction name="ifrs-full:BasicEarningsLossPerShare" contextRef="Q2" unitRef="EarningsPerShare">0.20</ix:nonFraction>'
    if predecessor:
        contexts += '<xbrli:context id="Q1"><xbrli:period><xbrli:startDate>2026-01-01</xbrli:startDate><xbrli:endDate>2026-03-31</xbrli:endDate></xbrli:period></xbrli:context>'
        contexts += '<ix:nonFraction name="ifrs-full:Revenue" contextRef="Q1" unitRef="money" scale="3">3,000</ix:nonFraction>'
    return body.replace('</body>', contexts + '</body>').encode()


def record(body):
    filing = replace(filed(parse_ixbrl(body)), content_sha256=hashlib.sha256(body).hexdigest(),
                     source_url="https://mops.twse.com.tw/SYNTHETIC.html")
    return ixbrl_record("3014", filing)


def test_original_quarter_value_is_used_not_an_unrelated_predecessor():
    row = record(source())
    periods = {pd.Period("2026Q1"): dict(flow_basis="YTD", revenue=4999, revision_id="other"),
               pd.Period("2026Q2"): row}
    result, trace = normalize_quarters_with_diagnostics(periods)
    assert row["revenue"] == 5000
    assert result[pd.Period("2026Q2")]["revenue"] == 1800
    assert result[pd.Period("2026Q2")]["basic_eps"] == .20
    assert trace[pd.Period("2026Q2")]["revenue"]["reason"] == "REPORTED_SINGLE_QUARTER"
    assert trace[pd.Period("2026Q2")]["revenue"]["inputs"][0]["content_sha256"] == row["content_sha256"]
    assert "total_assets" not in result[pd.Period("2026Q2")]
    assert periods[pd.Period("2026Q1")]["revenue"] == 4999  # No source mutation.


def test_only_same_original_compatible_ytd_can_be_subtracted():
    row = record(source(single_quarter=False, predecessor=True))
    q = pd.Period("2026Q2")
    values, trace = normalize_quarters_with_diagnostics({q: row})
    assert values[q]["revenue"] == 2000
    assert trace[q]["revenue"]["reason"] == "SAME_DOCUMENT_YTD_DIFFERENCE"
    assert values[q]["basic_eps"] is None  # EPS is not additive.
    assert not any(k in values[q] for k in ("total_assets", "total_equity"))
    row["prior_ytd_field_evidence"]["revenue"]["content_sha256"] = "other-revision"
    assert normalize_quarters({q: row})[q]["revenue"] is None


@pytest.mark.parametrize("damage", ["scope", "unit", "date", "revision", "currency"])
def test_same_document_derivation_requires_exact_basis(damage):
    row = record(source(single_quarter=False, predecessor=True))
    field = {"scope": "scope", "unit": "unit", "date": "end_date", "revision": "revision_id", "currency": "currency"}[damage]
    row["prior_ytd_field_evidence"]["revenue"][field] = "incompatible"
    assert normalize_quarters({pd.Period("2026Q2"): row})[pd.Period("2026Q2")]["revenue"] is None


def test_local_revision_number_one_is_not_a_shared_accounting_basis():
    second = record(source(single_quarter=False))
    first = dict(second, period_end="2026-03-31", revenue=3000, revision_id="a-different-original", revision_number=1)
    q = pd.Period("2026Q2")
    values, trace = normalize_quarters_with_diagnostics({q-1: first, q: second})
    assert values[q]["revenue"] is None
    assert trace[q]["revenue"]["reason"] == "REVISION_BASIS_UNVERIFIED"


@pytest.mark.parametrize("replacement", [b'scale="broken"', b'scale="999999"', b'scale="3" format="ixt:numcommadecimal"'])
def test_bad_scale_or_unsupported_transform_is_not_a_guessed_amount(replacement):
    body = document().replace(b'scale="3"', replacement)
    fact = parse_ixbrl(body)[0]
    assert fact.numeric_value is None and fact.text_value == "5,000"
    check = record(body)["field_evidence"]["revenue"]
    assert check["category"] == "PARSE_FAILED" and check["status"] == "UNKNOWN"


def test_typed_dimension_cannot_be_used_as_company_revenue():
    body = document().replace(b'<xbrli:period>', b'<xbrli:entity><xbrli:segment><xbrldi:typedMember dimension="segment:Customer"><segment:id>one</segment:id></xbrldi:typedMember></xbrli:segment></xbrli:entity><xbrli:period>', 1)
    fact = parse_ixbrl(body)[0]
    assert fact.dimensions == (("segment:Customer", "one"),)
    row = record(body)
    assert row["revenue"] is None
    assert row["field_evidence"]["revenue"]["category"] == "INCOMPARABLE"


def test_conflicting_alias_or_unit_never_silently_chooses_a_number():
    body = document().replace(b'</body>', b'<ix:nonFraction name="ifrs-full:RevenueFromContractsWithCustomers" contextRef="YTD" unitRef="money" scale="3">99</ix:nonFraction></body>')
    assert record(body)["field_evidence"]["revenue"]["reason"] == "CONFLICTING_AGGREGATE_VALUES"
    body = document().replace(b'currency:TWD', b'currency:USD')
    assert record(body)["field_evidence"]["revenue"]["reason"] == "UNIT_MISSING_OR_INCOMPATIBLE"


def test_a_conflicting_reported_quarter_cannot_be_hidden_by_ytd_subtraction():
    body = source(predecessor=True).replace(b'</body>', b'<ix:nonFraction name="ifrs-full:Revenue" contextRef="Q2" unitRef="money" scale="3">1,999</ix:nonFraction></body>')
    row = record(body)
    q = pd.Period("2026Q2")
    values, trace = normalize_quarters_with_diagnostics({q: row})
    assert values[q]["revenue"] is None
    assert trace[q]["revenue"]["reason"] == "CONFLICTING_AGGREGATE_VALUES"


def test_cost_is_derived_from_same_quarter_gross_profit_not_ytd_cost():
    body = source().replace(b'</body>', b'<ix:nonFraction name="ifrs-full:CostOfSales" contextRef="YTD" unitRef="money" scale="3">3,000</ix:nonFraction><ix:nonFraction name="ifrs-full:GrossProfit" contextRef="Q2" unitRef="money" scale="3">800</ix:nonFraction></body>')
    row = record(body)
    q = pd.Period("2026Q2")
    assert row["cost_of_revenue"] == 3000
    assert normalize_quarters({q: row})[q]["cost_of_revenue"] == 1000
    assert row["quarter_field_evidence"]["cost_of_revenue"]["reason"] == "REVENUE_MINUS_GROSS_PROFIT_SAME_DOCUMENT"


def test_missing_data_parse_failure_and_unverified_publication_have_distinct_attribution():
    data, item = fixture()
    cutoff = "2026-08-04T13:30:00+08:00"
    before = deepcopy(data)
    audit = audit_filings(data, ("2330",), cutoff, parse_filing=parse_ixbrl)
    assert audit["field_audit_schema"] == "financial-field-audit/v1"
    assert all(c["category"] == "PENDING_VERIFICATION" for c in audit["rows"][0]["field_checks"])
    assert audit["native_quality"]["2330"]["status"] == "DATA_INSUFFICIENT"
    financial = audit["financials"]["2330"]
    assert financial["missing_period_reasons"] == {}
    assert financial["diagnostic_missing_period_reasons"]["2026Q2"]["category"] == "PENDING_VERIFICATION"
    assert financial["diagnostic_missing_period_reasons"]["2025Q3"]["category"] == "MISSING_DATA"
    assert audit["status"] == "PARTIAL"
    assert data == before
    data["raw_documents"].clear()
    missing = audit_filings(data, ("2330",), cutoff, parse_filing=parse_ixbrl)
    assert all(c["category"] == "MISSING_DATA" for c in missing["rows"][0]["field_checks"])
    data, item = fixture()
    bad = audit_filings(data, ("2330",), cutoff, parse_filing=lambda _: [])
    assert all(c["category"] == "PARSE_FAILED" for c in bad["rows"][0]["field_checks"])
    assert "待核實" in render_filing_audit(audit)
    assert item["content_sha256"] in render_filing_audit(audit)


def test_verified_publication_repairs_are_traced_without_promising_native_pass():
    data, item = fixture()
    before = audit_filings(data, ("2330",), "2026-08-04T13:30:00+08:00", parse_filing=parse_ixbrl)
    original = deepcopy(data)
    item["publication_proof"] = reviewed(data, item)
    after = audit_filings(data, ("2330",), "2026-08-04T13:30:00+08:00", parse_filing=parse_ixbrl)
    difference = audit_difference(before, after)
    assert difference["improved_count"] == 1
    assert difference["changed_fields"]
    assert difference["native_quality_after"]["2330"]["status"] != "PASS"
    assert data["raw_documents"] == original["raw_documents"]
    assert data["filings"]["2330"][0]["receipt"] == original["filings"]["2330"][0]["receipt"]


def test_cli_summary_is_write_once_and_retries_are_identical(tmp_path):
    from infrastructure.research_archive import create_dataset
    from interface.research import main
    data, _ = fixture()
    evidence = create_dataset(data, tmp_path / "datasets")
    output, summary = tmp_path / "audit.json", tmp_path / "audit.md"
    args = ["filing-audit", "--evidence", str(evidence), "--as-of", "2026-08-04T13:30:00+08:00",
            "--output", str(output), "--summary-output", str(summary), "2330"]
    main(args)
    first = (output.read_bytes(), summary.read_bytes())
    main(args)
    assert first == (output.read_bytes(), summary.read_bytes())
    assert "UNKNOWN" in summary.read_text()


def test_resolved_quarter_is_not_mislabeled_missing_ytd_and_native_rules_remain():
    row = record(source())
    row["revenue"] = None  # Direct quarter still exists in original.
    diag = verification_diagnostics([row], symbol="3014", as_of="2026-09-01T13:30:00+08:00")
    check = next(c for c in diag["field_checks"] if c["period"] == "2026Q2" and c["field"] == "revenue")
    assert check["status"] == "RESOLVED" and check["value"] == 1800
    assert evaluate_quality([row], symbol="3014", market="TWSE", as_of="2026-09-01T13:30:00+08:00").status != "PASS"
    good = financials()
    expected = evaluate_quality(good, symbol="2330", market="TWSE", as_of="2026-09-18T13:30:00+08:00")
    assert expected.status == "PASS"
    future = dict(good[-1], available_at="2026-10-01T13:30:00+08:00", revenue=99999999, revision_id="future")
    assert asdict(evaluate_quality([*good, future], symbol="2330", market="TWSE", as_of="2026-09-18T13:30:00+08:00")) == asdict(expected)


@pytest.mark.parametrize("field,value", [("scope", "unknown"), ("currency", "USD"), ("unit", "million")])
def test_uniform_but_unverified_metadata_cannot_grant_native_pass(field, value):
    records = [dict(row, **{field: value}) for row in financials()]
    result = evaluate_quality(records, symbol="2330", market="TWSE", as_of="2026-09-18T13:30:00+08:00")
    assert result.status != "PASS"
    diagnostic = verification_diagnostics(records, symbol="2330", as_of="2026-09-18T13:30:00+08:00")
    assert diagnostic["local_review_required"]
    assert all(c["status"] == "UNKNOWN" for c in diagnostic["field_checks"])


def test_invalid_diagnostic_cutoff_remains_unknown_without_crashing():
    result = verification_diagnostics(financials(), symbol="2330", as_of="not-a-time")
    assert result["local_review_required"] and result["unresolved_fields"]


def test_reparse_repairs_old_cached_contexts_without_writing_receipts_or_facts(tmp_path):
    body = source()
    repo = MopsRepository(tmp_path / "mops")
    original = parse_ixbrl(body)
    legacy = [replace(f, numeric_value=None, text_value="old parser failed") if f.context_id == "Q2" else f for f in original]
    saved = repo.save_revision(_candidate(2026, 2, symbol="3014"), body, legacy,
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:00:00Z")
    loaded = repo.load_filed_financials("3014", "2026-08-11T05:00:00Z")[0]
    row = ixbrl_record("3014", loaded)
    assert normalize_quarters({pd.Period("2026Q2"): row})[pd.Period("2026Q2")]["revenue"] == 1800
    assert loaded.content_sha256 == saved.content_sha256 and loaded.available_at == saved.available_at
    assert repo.load_facts(saved.filing_id) == legacy
    assert repo.load_filed_financials("3014", "2026-08-09T05:00:00Z") == []


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_canonical_cached_numbers_cannot_hide_an_unverifiable_original(tmp_path, damage):
    body = source()
    repo = MopsRepository(tmp_path / "mops")
    saved = repo.save_revision(_candidate(2026, 2, symbol="3014"), body, parse_ixbrl(body),
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:00:00Z")
    before = repo.load_facts(saved.filing_id)
    if damage == "missing":
        repo.raw_path(saved).unlink()
    else:
        repo.raw_path(saved).write_bytes(source(single_quarter=False))
    loaded = repo.load_filed_financials("3014", "2026-08-11T05:00:00Z")[0]
    assert all(f.numeric_value is None for f in loaded.facts)
    assert ixbrl_record("3014", loaded)["revenue"] is None
    assert repo.load_facts(saved.filing_id) == before

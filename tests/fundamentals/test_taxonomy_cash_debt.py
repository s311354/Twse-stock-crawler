"""SYNTHETIC taxonomy/period/debt controls, not historical research evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib

import pandas as pd
import pytest

from application.research_evidence import materialize_filings
from domain.financial_taxonomy import (
    DEBT_COMPONENTS, TAXONOMY_PROFILES, ixbrl_schema_refs, taxonomy_mapping,
)
from domain.strategies.low_entry_v6_filings import ixbrl_record
from domain.strategies.low_entry_v6_quality import normalize_quarters, normalize_quarters_with_diagnostics
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.fundamentals.taxonomy_audit import audit_local_cache, main as audit_main
from tests.fundamentals.test_financial_field_repair import source
from tests.fundamentals.test_filing_refresh_planner import _candidate
from tests.fundamentals.test_ixbrl_units import filed
from tests.test_research_archive import raw_fixture, add_raw
from infrastructure.storage.research_cache import SQLiteFilingHistory
from application.replay_contracts import aware_timestamp, digest

SCHEMA = "tifrs-ci-cr-2026-03-31.xsd"


def document(*, schema=SCHEMA, amounts=None, flows=None, predecessor=False):
    body = source(single_quarter=False, predecessor=predecessor).decode()
    context = '<xbrli:context id="I"><xbrli:period><xbrli:instant>2026-06-30</xbrli:instant></xbrli:period></xbrli:context>'
    tags = f'<link:schemaRef xlink:href="{schema}"/>' if schema else ""
    for concept, value in (amounts or {}).items():
        context += f'<ix:nonFraction name="{concept}" contextRef="I" unitRef="money" scale="3">{value}</ix:nonFraction>'
    for (concept, ctx), value in (flows or {}).items():
        context += f'<ix:nonFraction name="{concept}" contextRef="{ctx}" unitRef="money" scale="3">{value}</ix:nonFraction>'
    return body.replace('<body>', '<body>' + tags).replace('</body>', context + '</body>').encode()


def adapter(body):
    return ixbrl_record("2330", replace(filed(parse_ixbrl(body)),
        content_sha256=hashlib.sha256(body).hexdigest(), period_end="2026-06-30",
        taxonomy_schema_refs=ixbrl_schema_refs(body)))


def cash_document(**kwargs):
    values = dict(cfo=4000, investing_cash_flow=-2000, financing_cash_flow=-800,
                  cash_fx_effect=100, cash_change=1300)
    values.update(kwargs)
    mapping, _ = taxonomy_mapping((SCHEMA,))
    mapping.update(cfo=("ifrs-full:CashFlowsFromUsedInOperatingActivities",),
                   capex=("ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",))
    flows = {(mapping[k][0], "YTD"): v for k, v in values.items() if v is not None}
    flows[mapping["capex"][0], "YTD"] = -2000
    return document(flows=flows, amounts={mapping["cash_beginning"][0]: 5000,
        mapping["cash_ending"][0]: 6300, "ifrs-full:CashAndCashEquivalents": 6300})


@pytest.mark.parametrize("schema", TAXONOMY_PROFILES)
def test_version_is_selected_from_original_schema_not_fiscal_year(schema):
    body = cash_document().replace(SCHEMA.encode(), schema.encode())
    row = adapter(body)
    assert row["taxonomy"]["schema_name"] == schema
    assert row["taxonomy"]["status"] == "RESOLVED"
    assert row["financing_cash_flow"] == -800 and row["cash_fx_effect"] == 100
    assert all(c["status"] == "PASS" for c in row["validation_checks"]["cash_flow"])
    # The actual beginning-cash context is the report end. It is NOT rewritten
    # as a Jan 1 balance; the separate flow-period binding explains its meaning.
    anchor = row["field_evidence"]["cash_beginning"]
    assert anchor["reported_instant"] == "2026-06-30"
    assert anchor["cash_flow_period_start"] == "2026-01-01"
    assert anchor["contexts"][0]["instant"] == "2026-06-30"


def test_schema_ref_comments_text_and_scripts_cannot_supply_a_version():
    fake = f'<link:schemaRef xlink:href="{SCHEMA}"/>'
    body = f'<!--{fake}--><p>&lt;link:schemaRef xlink:href="{SCHEMA}"/&gt;</p><script>{fake}</script>'
    assert ixbrl_schema_refs(body) == ()
    assert ixbrl_schema_refs(f'<link:schemaRef xlink:href="https://www.xbrl.org/{SCHEMA}"/>')
    assert taxonomy_mapping(("unknown-2030-01-01.xsd",))[1]["status"] == "UNKNOWN"


@pytest.mark.parametrize("schema,reason", [(None, "MISSING_SCHEMA_REF"),
    ("tifrs-ci-cr-2030-01-01.xsd", "UNSUPPORTED_TAXONOMY_VERSION")])
def test_unknown_version_does_not_use_a_latest_or_date_inferred_mapping(schema, reason):
    body = cash_document().replace(SCHEMA.encode(), (schema or "").encode())
    row = adapter(body)
    assert row["taxonomy"]["reason"] == reason
    assert row["investing_cash_flow"] is None
    assert row["field_evidence"]["investing_cash_flow"]["reason"] == reason
    assert all(c["status"] == "UNKNOWN" for c in row["validation_checks"]["cash_flow"])


def test_mixed_schema_versions_are_not_resolved_by_picking_latest():
    body = cash_document().replace(b'</body>', b'<link:schemaRef xlink:href="tifrs-ci-cr-2020-06-30.xsd"/></body>')
    row = adapter(body)
    assert row["taxonomy"]["reason"] == "CONFLICTING_SCHEMA_REFS"
    assert row["investing_cash_flow"] is None


def test_confirmed_cash_bridge_failure_blocks_cash_inputs_but_missing_fx_is_not_zero():
    row = adapter(cash_document(cash_change=1))
    assert row["cfo"] is row["capex"] is None
    assert row["field_evidence"]["cfo"]["parsed_value"] == 4000
    assert row["field_evidence"]["cfo"]["reason"] == "CASH_FLOW_RECONCILIATION_FAILED"
    assert row["validation_checks"]["cash_flow"][0]["status"] == "FAIL"
    missing = adapter(cash_document(cash_fx_effect=None))
    assert missing["cash_fx_effect"] is None and missing["cfo"] == 4000
    check = missing["validation_checks"]["cash_flow"][0]
    assert check["status"] == "UNKNOWN" and check["missing_fields"] == ["cash_fx_effect"]


def test_cash_bridge_does_not_turn_half_year_cash_flow_into_quarter_or_ttm():
    row = adapter(cash_document())
    q = pd.Period("2026Q2")
    values, trace = normalize_quarters_with_diagnostics({q: row})
    assert row["cfo"] == 4000  # Reported H1, not Q2.
    assert values[q]["cfo"] is None
    assert trace[q]["cfo"]["reason"] == "REVISION_BASIS_UNVERIFIED"
    assert row["quarter_fields"]["cfo"] is None


def test_same_original_cash_period_subtraction_preserves_sign_and_does_not_subtract_balances():
    flows = {("ifrs-full:CashFlowsFromUsedInOperatingActivities", "YTD"): 4000,
             ("ifrs-full:CashFlowsFromUsedInOperatingActivities", "Q1"): 1500,
             ("ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "YTD"): -2000,
             ("ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities", "Q1"): -700}
    row = adapter(document(predecessor=True, flows=flows, amounts={"ifrs-full:CashAndCashEquivalents": 6300}))
    q = pd.Period("2026Q2")
    values = normalize_quarters({q: row})[q]
    assert values["cfo"] == 2500 and values["capex"] == -1300
    assert "cash" not in values and row["cash"] == 6300
    row["prior_ytd_field_evidence"]["cfo"]["available_at"] = "2026-08-11T05:00:00Z"
    assert normalize_quarters({q: row})[q]["cfo"] is None


def test_report_period_end_mismatch_invalidates_values_not_just_audit_status():
    body = cash_document()
    filing = replace(filed(parse_ixbrl(body)), taxonomy_schema_refs=ixbrl_schema_refs(body), period_end="2026-09-30")
    row = ixbrl_record("2330", filing)
    assert row["cfo"] is row["cash"] is None
    assert row["validation_checks"]["period"]["status"] == "FAIL"
    assert row["field_evidence"]["cfo"]["reason"] == "FILING_PERIOD_END_MISMATCH"


def complete_debt():
    profile = TAXONOMY_PROFILES[SCHEMA]
    values = [300, 0, 20, 10, 500, 200]
    amounts = {profile[k][0]: v for k, v in zip(DEBT_COMPONENTS, values)}
    amounts.update({"ifrs-full:CurrentLeaseLiabilities": 5, "ifrs-full:NoncurrentLeaseLiabilities": 25})
    return amounts


def test_complete_classified_debt_does_not_sum_longterm_parent_and_alias_again():
    amounts = complete_debt()
    amounts["ifrs-full:NoncurrentPortionOfNoncurrentLoansReceived"] = 500
    row = adapter(document(amounts=amounts))
    assert row["interest_bearing_debt"] is None  # Leaves do not prove all debt categories.
    assert row["validation_checks"]["debt"]["reported_component_sum"] == 1030
    assert row["long_term_borrowings"] == 500  # Not 500 + the same 500 again.
    amounts["ifrs-full:Borrowings"] = 1030
    amounts["ifrs-full:CurrentBorrowings"] = 330
    amounts["ifrs-full:NoncurrentBorrowings"] = 700
    row = adapter(document(amounts=amounts))
    assert row["interest_bearing_debt"] == 1060
    assert row["validation_checks"]["debt"]["missing_fields"] == []
    assert row["validation_checks"]["debt"]["used_fields"] == [
        "borrowings", "current_lease_liabilities", "noncurrent_lease_liabilities"]


@pytest.mark.parametrize("damage", ["missing_bonds", "missing_notes", "missing_lease", "negative", "alias_conflict", "ambiguous_maturity"])
def test_incomplete_or_conflicting_debt_is_never_assumed_zero(damage):
    amounts = complete_debt()
    if damage == "missing_bonds": del amounts["ifrs-full:NoncurrentPortionOfNoncurrentBondsIssued"]
    elif damage == "missing_notes": del amounts["tifrs-bsci-ci:ShorttermNotesAndBillsPayable"]
    elif damage == "missing_lease": del amounts["ifrs-full:CurrentLeaseLiabilities"]
    elif damage == "negative": amounts["ifrs-full:LongtermBorrowings"] = -500
    elif damage == "alias_conflict": amounts["ifrs-full:NoncurrentPortionOfNoncurrentLoansReceived"] = 499
    else: amounts["tifrs-bsci-ci:LongtermLiabilitiesCurrentPortion"] = 1000
    row = adapter(document(amounts=amounts))
    assert row["interest_bearing_debt"] is None
    assert row["validation_checks"]["debt"]["status"] == "UNKNOWN"
    assert row["validation_checks"]["debt"]["missing_fields"] or row["validation_checks"]["debt"]["conflicts"]


def test_partial_components_exceeding_reported_total_are_not_ignored():
    amounts = {"ifrs-full:Borrowings": 100, "ifrs-full:ShorttermBorrowings": 200,
               "ifrs-full:CurrentLeaseLiabilities": 0, "ifrs-full:NoncurrentLeaseLiabilities": 0}
    row = adapter(document(amounts=amounts))
    assert row["interest_bearing_debt"] is None
    assert row["validation_checks"]["debt"]["conflicts"][0]["reason"] == "DEBT_PARENT_COMPONENT_CONFLICT"


def test_versioned_current_debt_aggregate_is_not_short_term_loans_only():
    row = adapter(document(amounts={"ifrs-full:CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings": 330,
        "ifrs-full:NoncurrentBorrowings": 700, "ifrs-full:CurrentLeaseLiabilities": 5,
        "ifrs-full:NoncurrentLeaseLiabilities": 25, "ifrs-full:ShorttermBorrowings": 300}))
    assert row["current_borrowings"] == 330
    assert row["short_term_borrowings"] == 300
    assert row["interest_bearing_debt"] == 1060


def test_an_untrusted_schema_url_cannot_enable_the_reviewed_profile():
    assert taxonomy_mapping(("https://example.test/" + SCHEMA,))[1]["reason"] == "UNTRUSTED_SCHEMA_REF"


def test_local_raw_migration_keeps_sqlite_original_and_does_not_request_network(tmp_path, monkeypatch):
    repo = MopsRepository(tmp_path / "mops")
    body = cash_document()
    stored = repo.save_revision(_candidate(2026, 2, symbol="2330"), body, parse_ixbrl(body),
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:00:00Z")
    before = repo.database_path.read_bytes() if hasattr(repo, 'database_path') else None
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("local migration attempted remote fetch"))
    records = repo.load_filed_financials("2330", "2026-09-11T13:30:00+08:00")
    assert records[0].taxonomy_schema_refs == (SCHEMA,)
    assert records[0].content_sha256 == stored.content_sha256
    assert records[0].available_at == stored.available_at
    assert ixbrl_record("2330", records[0])["cash_ending"] == 6300
    assert repo.raw_path(stored).read_bytes() == body
    if before is not None: assert repo.database_path.read_bytes() == before


def test_materializer_reestablishes_schema_from_bytes_and_cache_paths_match(tmp_path, monkeypatch):
    payload = raw_fixture()
    monkeypatch.setattr("tests.test_research_archive.document", cash_document)
    item = add_raw(payload)
    item["taxonomy_schema_refs"] = ["caller-asserted-unknown.xsd"]
    original = deepcopy(payload)
    fresh, errors = materialize_filings(payload, parse_filing=parse_ixbrl)
    assert errors == {"2330": []}
    assert fresh["2330"][0]["taxonomy_schema_refs"] == [SCHEMA]
    cache = SQLiteFilingHistory(tmp_path / "parse.sqlite3", evidence_id=digest(payload), version="synthetic-current", parse_filing=parse_ixbrl)
    at = aware_timestamp("2026-09-11T13:30:00+08:00")
    cached, errors = cache.load(payload, ("2330",), at)
    assert digest(fresh) == digest(cached)  # Full public JSON identity, including dimensions.
    assert payload == original


def test_bounded_local_audit_is_read_only_and_does_not_hide_missing_quarters(tmp_path, monkeypatch):
    repo = MopsRepository(tmp_path / "mops")
    body = cash_document()
    stored = repo.save_revision(_candidate(2026, 2, symbol="2330"), body, parse_ixbrl(body),
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:00:00Z")
    before = repo.database_path.read_bytes()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("offline audit attempted remote fetch"))
    args = (repo.root, ("2330",), "2026-09-11T13:30:00+08:00")
    audit = audit_local_cache(*args, period_count=3)
    assert audit == audit_local_cache(*args, period_count=3)
    company = audit["companies"][0]
    assert company["periods"] == ["2026Q2", "2026Q1", "2025Q4"]
    assert company["missing_periods"] == ["2026Q1", "2025Q4"]
    assert company["filings"][0]["original"]["content_sha256"] == stored.content_sha256
    assert company["filings"][0]["record"]["taxonomy"]["schema_name"] == SCHEMA
    assert company["filings"][0]["normalized_quarter"]["cfo"] is None
    assert repo.database_path.read_bytes() == before
    assert repo.raw_path(stored).read_bytes() == body
    assert repo.load_filed_financials("2330", args[2], periods=()) == []
    assert repo.load_filed_financials("2330", args[2], periods=((2026, 1),)) == []
    with pytest.raises(ValueError, match="INVALID_FILING_PERIOD_FILTER"):
        repo.load_filed_financials("2330", args[2], periods=((2026, 9),))
    with pytest.raises(ValueError, match="AUDIT_OUTPUT_MUST_NOT_MODIFY_MOPS_CACHE"):
        audit_main(["--cache", str(repo.root), "--as-of", args[2], "--output", str(repo.root / "invalid.json"), "2330"])
    assert not (repo.root / "invalid.json").exists()


def test_local_audit_cutoff_does_not_use_later_original(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    body = cash_document()
    repo.save_revision(_candidate(2026, 2, symbol="2330"), body, parse_ixbrl(body),
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:00:00Z")
    audit = audit_local_cache(repo.root, ("2330",), "2026-08-09T23:59:59+08:00")
    assert audit["companies"][0]["filings"] == []


def test_non_ixbrl_original_keeps_legacy_read_amount_but_not_v6_unit_evidence(tmp_path):
    repo = MopsRepository(tmp_path / "mops")
    persisted = parse_ixbrl(cash_document())
    repo.save_revision(_candidate(2026, 2, symbol="2330"), b"SYNTHETIC-NOT-IXBRL", persisted,
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:00:00Z")
    filing = repo.load_filed_financials("2330", "2026-09-11T13:30:00+08:00")[0]
    assert filing.facts[0].numeric_value == persisted[0].numeric_value
    assert all(f.unit is None for f in filing.facts)
    row = ixbrl_record("2330", filing)
    assert row["cfo"] is row["capex"] is row["cash"] is None

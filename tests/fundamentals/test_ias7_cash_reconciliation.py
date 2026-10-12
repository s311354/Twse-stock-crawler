"""SYNTHETIC source-bound IAS 7 controls; not historical evidence or returns."""
from copy import deepcopy
from dataclasses import replace
import hashlib

import pandas as pd
import pytest

import application.fundamental_deriver as deriver_module
from application.fundamental_deriver import FundamentalDeriver
from application.mops_refresh_control import MopsRefreshPolicy
from domain.financial_taxonomy import (
    IAS7_OTHER_CASH_CONCEPT, TAXONOMY_PROFILES, cash_flow_checks, ixbrl_schema_refs,
)
from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION, ixbrl_record
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.mops_repository import MopsRepository
from tests.fundamentals.test_filing_refresh_planner import _candidate, _FakeMopsClient
from tests.fundamentals.test_ixbrl_units import filed
from tests.fundamentals.test_mops_refresh_control import Clock, SIGNAL, service
from tests.fundamentals.test_taxonomy_cash_debt import SCHEMA, cash_document


def document(*, adjustment=1800, cash=4500, context="I", unit="money", schema=SCHEMA):
    body = cash_document().decode().replace(
        'name="ifrs-full:CashAndCashEquivalents" contextRef="I" unitRef="money" scale="3">6300',
        f'name="ifrs-full:CashAndCashEquivalents" contextRef="I" unitRef="money" scale="3">{cash}')
    if adjustment is not None:
        body = body.replace('</body>', f'<ix:nonFraction name="{IAS7_OTHER_CASH_CONCEPT}" '
            f'contextRef="{context}" unitRef="{unit}" scale="3">{adjustment}</ix:nonFraction></body>')
    return body.replace(SCHEMA, schema).encode()


def record(body):
    return ixbrl_record("2330", replace(filed(parse_ixbrl(body)),
        content_sha256=hashlib.sha256(body).hexdigest(), period_end="2026-06-30",
        source_url="https://mopsov.twse.com.tw/server-java/SYNTHETIC-2330-2026Q2.html",
        taxonomy_schema_refs=ixbrl_schema_refs(body)))


def tie(row):
    return row["validation_checks"]["cash_flow"][-1]


@pytest.mark.parametrize("schema", TAXONOMY_PROFILES)
@pytest.mark.parametrize("cash,adjustment", [(4500, 1800), (6300, 0), (8100, -1800)])
def test_reported_signed_adjustment_binds_three_operands_and_restores_cash_inputs(schema, cash, adjustment):
    row = record(document(schema=schema, cash=cash, adjustment=adjustment))
    check = tie(row)
    assert check["status"] == "PASS" and check["variance"] == 0
    assert check["method"] == "IAS7_SOURCE_BOUND_ADJUSTMENT"
    assert check["inputs"] == ["cash", "cash_ias7_adjustment", "cash_ending"]
    assert check["expected_value"] == check["observed_value"] == 6300
    assert check["source_binding_errors"] == [] and not check["blocks_cash_inputs"]
    assert row["cfo"] == 4000 and row["capex"] == -2000
    for operand in check["source_inputs"]:
        assert operand["content_sha256"] == row["content_sha256"]
        assert operand["revision_id"] == row["revision_id"]
        assert operand["available_at"] == row["available_at"]
        assert len(operand["contexts"]) == 1
        assert operand["contexts"][0]["instant"] == "2026-06-30"
    assert row["field_evidence"]["cash_ias7_adjustment"]["basis"] == "INSTANT"
    assert row["cash_ias7_adjustment"] == adjustment


def test_absent_adjustment_is_not_zero_or_an_inferred_plug():
    row = record(document(adjustment=None))
    assert row["cash_ias7_adjustment"] is None
    assert tie(row)["status"] == "FAIL" and tie(row)["method"] == "DIRECT"
    assert row["cfo"] is row["capex"] is None
    assert row["field_evidence"]["cfo"]["parsed_value"] == 4000
    ordinary = record(document(adjustment=None, cash=6300))
    assert tie(ordinary)["status"] == "PASS" and ordinary["cash_ias7_adjustment"] is None
    # An unrelated balance with precisely the difference is NOT cash evidence.
    body = document(adjustment=None).replace(b'</body>',
        b'<ix:nonFraction name="ifrs-full:HedgingInstrumentAssets" contextRef="I" '
        b'unitRef="money" scale="3">1800</ix:nonFraction></body>')
    assert tie(record(body))["status"] == "FAIL"


@pytest.mark.parametrize("adjustment", [1797, 1803, 0])
def test_actual_ias7_arithmetic_mismatch_is_still_quarantined(adjustment):
    row = record(document(adjustment=adjustment))
    assert tie(row)["status"] == "FAIL"
    assert row["cfo"] is row["capex"] is None
    assert row["field_evidence"]["cfo"]["reason"] == "CASH_FLOW_RECONCILIATION_FAILED"


@pytest.mark.parametrize("damage", ["wrong_instant", "duration", "wrong_unit", "dimensions", "nil", "conflict"])
def test_invalid_reported_adjustment_never_falls_back_to_direct_equality(damage):
    # Even when CF/BS cash directly tie, a known unresolved adjustment must not
    # disappear. It could change the cash definition and is not an absent item.
    body = document(cash=6300, adjustment=0).decode()
    if damage in {"wrong_instant", "duration", "dimensions"}:
        period = '<xbrli:instant>2025-06-30</xbrli:instant>' if damage == "wrong_instant" else (
            '<xbrli:startDate>2026-01-01</xbrli:startDate><xbrli:endDate>2026-06-30</xbrli:endDate>'
            if damage == "duration" else '<xbrli:instant>2026-06-30</xbrli:instant>')
        segment = '<xbrli:entity><xbrli:segment><xbrldi:explicitMember dimension="segment:Region">segment:A</xbrldi:explicitMember></xbrli:segment></xbrli:entity>' if damage == "dimensions" else ''
        body = body.replace('</body>', f'<xbrli:context id="BAD">{segment}<xbrli:period>{period}</xbrli:period></xbrli:context></body>')
        body = body.replace(f'name="{IAS7_OTHER_CASH_CONCEPT}" contextRef="I"',
                            f'name="{IAS7_OTHER_CASH_CONCEPT}" contextRef="BAD"')
    elif damage == "wrong_unit":
        body = body.replace(f'name="{IAS7_OTHER_CASH_CONCEPT}" contextRef="I" unitRef="money"',
                            f'name="{IAS7_OTHER_CASH_CONCEPT}" contextRef="I" unitRef="EarningsPerShare"')
    elif damage == "nil":
        body = body.replace(f'name="{IAS7_OTHER_CASH_CONCEPT}" contextRef="I" unitRef="money" scale="3">0',
                            f'name="{IAS7_OTHER_CASH_CONCEPT}" contextRef="I" unitRef="money" scale="3" xsi:nil="true">')
    else:
        body = body.replace('</body>', f'<ix:nonFraction name="{IAS7_OTHER_CASH_CONCEPT}" '
            'contextRef="I" unitRef="money" scale="3">99</ix:nonFraction></body>')
    row = record(body.encode())
    assert tie(row)["status"] == "UNKNOWN" and tie(row)["blocks_cash_inputs"]
    assert tie(row)["source_binding_errors"]
    assert row["cfo"] is row["capex"] is None
    assert row["field_evidence"]["cfo"]["reason"] == "CASH_FLOW_RECONCILIATION_UNVERIFIED"


@pytest.mark.parametrize("key,value", [("content_sha256", "0" * 64), ("revision_id", "different"),
    ("available_at", "2026-08-11T05:00:00Z"), ("scope", "individual"), ("currency", "USD"),
    ("unit", "million"), ("end_date", "2025-06-30"), ("symbol", "2344"),
    ("taxonomy_mapping_version", "old"), ("source_url", "https://mopsov.twse.com.tw/OTHER.html")])
def test_equal_arithmetic_cannot_hide_operand_source_mismatch(key, value):
    row = record(document())
    evidence = deepcopy(row["field_evidence"])
    before = deepcopy(row)
    evidence["cash_ias7_adjustment"][key] = value
    check = cash_flow_checks(row, evidence=evidence, start_date="2026-01-01", end_date="2026-06-30")[-1]
    assert check["variance"] == 0  # Not enough to establish a source-bound tie.
    assert check["status"] == "UNKNOWN" and check["blocks_cash_inputs"]
    assert row == before


def test_numeric_adjustment_without_original_proofs_is_not_verified():
    row = record(document())
    check = cash_flow_checks(row, start_date="2026-01-01", end_date="2026-06-30")[-1]
    assert check["status"] == "UNKNOWN" and check["source_binding_errors"]


def test_reconciled_cash_does_not_manufacture_or_change_operating_cash_flow():
    negative = document()
    negative = negative.replace(
        b'name="ifrs-full:CashFlowsFromUsedInOperatingActivities" contextRef="YTD" unitRef="money" scale="3">4000',
        b'name="ifrs-full:CashFlowsFromUsedInOperatingActivities" contextRef="YTD" unitRef="money" scale="3">-400')
    negative = negative.replace(
        b'name="tifrs-SCF:CashFlowsFromUsedInFinancingActivities" contextRef="YTD" unitRef="money" scale="3">-800',
        b'name="tifrs-SCF:CashFlowsFromUsedInFinancingActivities" contextRef="YTD" unitRef="money" scale="3">3600')
    row = record(negative)
    assert all(c["status"] == "PASS" for c in row["validation_checks"]["cash_flow"])
    assert row["cfo"] == -400  # A known negative CFO remains negative, not missing or positive.
    absent = document().replace(
        b'<ix:nonFraction name="ifrs-full:CashFlowsFromUsedInOperatingActivities" contextRef="YTD" unitRef="money" scale="3">4000</ix:nonFraction>', b'')
    missing = record(absent)
    assert tie(missing)["status"] == "PASS" and missing["cfo"] is None
    assert missing["validation_checks"]["cash_flow"][0]["status"] == "UNKNOWN"


def test_same_original_comparative_cash_uses_its_own_instant_and_flow_window():
    body = document().decode()
    context = '<xbrli:context id="PI"><xbrli:period><xbrli:instant>2025-06-30</xbrli:instant></xbrli:period></xbrli:context>'
    context += '<xbrli:context id="PY"><xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate><xbrli:endDate>2025-06-30</xbrli:endDate></xbrli:period></xbrli:context>'
    fields = {"ifrs-full:CashAndCashEquivalents": 6300, IAS7_OTHER_CASH_CONCEPT: 0,
        "tifrs-SCF:CashAndCashEquivalentsAtBeginningOfPeriod": 5000,
        "tifrs-SCF:CashAndCashEquivalentsAtEndOfPeriod": 6300}
    flows = {"ifrs-full:CashFlowsFromUsedInOperatingActivities": 4000,
        "tifrs-SCF:NetCashFlowsFromUsedInInvestingActivities": -2000,
        "tifrs-SCF:CashFlowsFromUsedInFinancingActivities": -800,
        "ifrs-full:EffectOfExchangeRateChangesOnCashAndCashEquivalents": 100,
        "ifrs-full:IncreaseDecreaseInCashAndCashEquivalents": 1300}
    for ctx, values in (("PI", fields), ("PY", flows)):
        context += ''.join(f'<ix:nonFraction name="{concept}" contextRef="{ctx}" unitRef="money" scale="3">{value}</ix:nonFraction>'
                           for concept, value in values.items())
    row = record(body.replace('</body>', context + '</body>').encode())
    assert all(c["status"] == "PASS" for c in row["validation_checks"]["comparative_cash_flow"])
    proof = row["comparative_ytd_field_evidence"]["cash_ending"]
    assert proof["cash_flow_period_start"] == "2025-01-01"
    assert proof["reported_instant"] == "2025-06-30"
    assert row["comparative_ytd_fields"]["cash_ias7_adjustment"] == 0
    assert row["cash_ias7_adjustment"] == 1800  # The comparative zero is not the current value.


def test_schema_upgrade_repairs_old_snapshot_locally_without_rewriting_originals(tmp_path, monkeypatch):
    repo = MopsRepository(tmp_path / "synthetic-mops")
    body = document()
    stored = repo.save_revision(_candidate(2026, 2), body, parse_ixbrl(body),
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:00:00Z")
    facts = repo.load_facts(stored.filing_id)
    originals = repo.load_filed_financials("2330", SIGNAL)
    native = deriver_module.ixbrl_record

    def old_adapter(symbol, filing):
        row = native(symbol, filing)
        row.update(cfo=None, capex=None, filing_schema="synthetic-old-direct-cash-tie")
        return row

    with monkeypatch.context() as patch:
        patch.setattr(deriver_module, "FUNDAMENTAL_DERIVATION_VERSION", "synthetic-old-derivation")
        patch.setattr(deriver_module, "V6_FILING_SCHEMA_VERSION", "synthetic-old-direct-cash-tie")
        patch.setattr(deriver_module, "ixbrl_record", old_adapter)
        with repo.unit_of_work() as uow:
            old, _ = FundamentalDeriver(repo).load_or_derive("2330", parsed_signal=pd.Timestamp(SIGNAL),
                resolved_sector="24", filed_financials=originals, uow=uow)
    assert old.snapshot["les_v6_mops_filings"][0]["cfo"] is None
    client = _FakeMopsClient(repo, [], [])
    svc = service(repo, client, Clock(), MopsRefreshPolicy(max_requests=0))
    assert svc.migrate_local_cache(["2330"], signal_date=SIGNAL, sector="24") == 1
    migrated = repo.load_cached_bundles({"2330": "24"}, as_of=SIGNAL,
        derivation_version=deriver_module.FUNDAMENTAL_DERIVATION_VERSION)["2330"].bundle.snapshot
    assert migrated["les_v6_filing_schema_version"] == V6_FILING_SCHEMA_VERSION
    assert migrated["les_v6_mops_filings"][0]["cfo"] == 4000
    assert migrated["les_v6_mops_filings"][0]["content_sha256"] == stored.content_sha256
    assert migrated["les_v6_mops_filings"][0]["available_at"] == stored.available_at
    assert repo.load_facts(stored.filing_id) == facts
    assert repo.raw_path(stored).read_bytes() == body
    assert client.discovery_calls == 0 and client.downloaded_periods == []
    assert svc.migrate_local_cache(["2330"], signal_date=SIGNAL, sector="24") == 0


def test_later_ias7_revision_cannot_change_earlier_cutoff_cash_basis(tmp_path):
    repo = MopsRepository(tmp_path / "synthetic-mops")
    body = document()
    saved = repo.save_revision(_candidate(2026, 2), body, parse_ixbrl(body),
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:00:00Z")
    cutoff = "2026-09-11T05:30:00Z"
    original = ixbrl_record("2330", repo.load_filed_financials("2330", cutoff)[0])
    future = document(adjustment=1790)
    repo.save_revision(_candidate(2026, 2), future, parse_ixbrl(future),
        available_at="2026-09-12T05:00:00Z", fetched_at="2026-09-12T05:00:00Z")
    earlier = repo.load_filed_financials("2330", cutoff)
    assert len(earlier) == 1 and earlier[0].content_sha256 == saved.content_sha256
    assert ixbrl_record("2330", earlier[0]) == original
    later = repo.load_filed_financials("2330", "2026-09-13T05:00:00Z")
    latest = max(later, key=lambda f: f.revision_number)
    changed = ixbrl_record("2330", latest)
    assert changed["content_sha256"] != saved.content_sha256
    assert tie(changed)["status"] == "FAIL" and changed["cfo"] is None

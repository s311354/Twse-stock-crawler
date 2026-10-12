"""SYNTHETIC disposal-group cash proofs; not historical evidence or returns."""
from copy import deepcopy

import pandas as pd
import pytest

import application.fundamental_deriver as deriver_module
from application.fundamental_deriver import FundamentalDeriver
from application.mops_refresh_control import MopsRefreshPolicy
from domain.financial_taxonomy import (
    DISPOSAL_GROUP_CASH_CONCEPT, IAS7_OTHER_CASH_CONCEPT, TAXONOMY_PROFILES, cash_flow_checks,
)
from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION, ixbrl_record
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.fundamentals.mops_repository import MopsRepository
from tests.fundamentals.test_filing_refresh_planner import _candidate, _FakeMopsClient
from tests.fundamentals.test_ias7_cash_reconciliation import document as ias7_document, record, tie
from tests.fundamentals.test_mops_refresh_control import Clock, SIGNAL, service


def document(**kwargs):
    return ias7_document(**kwargs).replace(IAS7_OTHER_CASH_CONCEPT.encode(), DISPOSAL_GROUP_CASH_CONCEPT.encode())


def fact(concept, value, context="I"):
    return (f'<ix:nonFraction name="{concept}" contextRef="{context}" '
            f'unitRef="money" scale="3">{value}</ix:nonFraction>').encode()


@pytest.mark.parametrize("schema", TAXONOMY_PROFILES)
@pytest.mark.parametrize("cash,adjustment", [(4500, 1800), (6300, 0)])
def test_reported_disposal_cash_binds_original_period_units_and_revision(schema, cash, adjustment):
    row = record(document(schema=schema, cash=cash, adjustment=adjustment))
    check = tie(row)
    assert check["status"] == "PASS" and check["variance"] == 0
    assert check["method"] == "DISPOSAL_GROUP_SOURCE_BOUND_ADJUSTMENT"
    assert check["inputs"] == ["cash", "cash_disposal_group", "cash_ending"]
    assert check["expected_value"] == check["observed_value"] == 6300
    assert not check["blocks_cash_inputs"] and check["source_binding_errors"] == []
    assert row["cfo"] == 4000 and row["capex"] == -2000
    assert row["cash_disposal_group"] == adjustment and row["cash_ias7_adjustment"] is None
    for operand in check["source_inputs"]:
        assert operand["content_sha256"] == row["content_sha256"]
        assert operand["revision_id"] == row["revision_id"]
        assert operand["available_at"] == row["available_at"]
        assert len(operand["contexts"]) == 1
        assert operand["contexts"][0]["instant"] == "2026-06-30"


def test_distinct_cash_components_are_added_once_not_aliases():
    body = document(cash=4300).replace(b'</body>', fact(IAS7_OTHER_CASH_CONCEPT, 200) + b'</body>')
    # Repeat the SAME resolved fact; a duplicate occurrence is not another cash asset.
    body = body.replace(b'</body>', fact(DISPOSAL_GROUP_CASH_CONCEPT, 1800) + b'</body>')
    row = record(body)
    assert tie(row)["method"] == "SOURCE_BOUND_CASH_COMPONENTS"
    assert tie(row)["inputs"] == ["cash", "cash_ias7_adjustment", "cash_disposal_group", "cash_ending"]
    assert tie(row)["expected_value"] == tie(row)["observed_value"] == 6300
    assert tie(row)["status"] == "PASS" and row["cfo"] == 4000
    # Equal-valued IAS 7 and disposal components are distinct, not substitutes.
    double = record(document().replace(b'</body>', fact(IAS7_OTHER_CASH_CONCEPT, 1800) + b'</body>'))
    assert tie(double)["status"] == "FAIL" and double["cfo"] is None


def test_missing_disposal_cash_is_not_zero_a_plug_or_total_held_for_sale_assets():
    ordinary = record(document(adjustment=None, cash=6300))
    assert tie(ordinary)["status"] == "PASS" and ordinary["cash_disposal_group"] is None
    missing = record(document(adjustment=None))
    assert tie(missing)["status"] == "FAIL" and missing["cfo"] is None
    unrelated = document(adjustment=None).replace(b'</body>',
        fact("ifrs-full:NoncurrentAssetsOrDisposalGroupsClassifiedAsHeldForSale", 1800) + b'</body>')
    assert tie(record(unrelated))["status"] == "FAIL"


@pytest.mark.parametrize("damage", ["wrong_instant", "duration", "wrong_unit", "dimensions", "nil", "conflict"])
def test_bad_disposal_context_never_disappears_even_when_direct_cash_matches(damage):
    body = document(cash=6300, adjustment=0).decode()
    if damage in {"wrong_instant", "duration", "dimensions"}:
        period = '<xbrli:instant>2025-06-30</xbrli:instant>' if damage == "wrong_instant" else (
            '<xbrli:startDate>2026-01-01</xbrli:startDate><xbrli:endDate>2026-06-30</xbrli:endDate>'
            if damage == "duration" else '<xbrli:instant>2026-06-30</xbrli:instant>')
        segment = ('<xbrli:entity><xbrli:segment><xbrldi:explicitMember dimension="segment:Region">'
                   'segment:A</xbrldi:explicitMember></xbrli:segment></xbrli:entity>') if damage == "dimensions" else ''
        body = body.replace('</body>', f'<xbrli:context id="BAD">{segment}<xbrli:period>{period}</xbrli:period></xbrli:context></body>')
        body = body.replace(f'name="{DISPOSAL_GROUP_CASH_CONCEPT}" contextRef="I"',
                            f'name="{DISPOSAL_GROUP_CASH_CONCEPT}" contextRef="BAD"')
    elif damage == "wrong_unit":
        body = body.replace(f'name="{DISPOSAL_GROUP_CASH_CONCEPT}" contextRef="I" unitRef="money"',
                            f'name="{DISPOSAL_GROUP_CASH_CONCEPT}" contextRef="I" unitRef="EarningsPerShare"')
    elif damage == "nil":
        body = body.replace(f'name="{DISPOSAL_GROUP_CASH_CONCEPT}" contextRef="I" unitRef="money" scale="3">0',
                            f'name="{DISPOSAL_GROUP_CASH_CONCEPT}" contextRef="I" unitRef="money" scale="3" xsi:nil="true">')
    else:
        body = body.replace('</body>', fact(DISPOSAL_GROUP_CASH_CONCEPT, 99).decode() + '</body>')
    row = record(body.encode())
    assert tie(row)["status"] == "UNKNOWN" and tie(row)["blocks_cash_inputs"]
    assert tie(row)["source_binding_errors"]
    assert row["cfo"] is row["capex"] is None
    assert row["field_evidence"]["cfo"]["reason"] == "CASH_FLOW_RECONCILIATION_UNVERIFIED"


@pytest.mark.parametrize("key,value", [("content_sha256", "0" * 64), ("revision_id", "different"),
    ("available_at", "2026-08-11T05:00:00Z"), ("scope", "individual"), ("currency", "USD"),
    ("unit", "million"), ("end_date", "2025-06-30"), ("symbol", "2392"),
    ("taxonomy_mapping_version", "old"), ("source_url", "https://mopsov.twse.com.tw/OTHER.html")])
def test_correct_arithmetic_requires_matching_disposal_source_identity(key, value):
    row = record(document())
    evidence = deepcopy(row["field_evidence"])
    original = deepcopy(row)
    evidence["cash_disposal_group"][key] = value
    check = cash_flow_checks(row, evidence=evidence, start_date="2026-01-01", end_date="2026-06-30")[-1]
    assert check["variance"] == 0 and check["status"] == "UNKNOWN" and check["blocks_cash_inputs"]
    assert row == original


def test_amount_without_source_proof_or_valid_schema_cannot_establish_bound_cash_tie():
    row = record(document())
    assert cash_flow_checks(row, start_date="2026-01-01", end_date="2026-06-30")[-1]["status"] == "UNKNOWN"
    unsupported = record(document(schema="tifrs-ci-cr-2099-01-01.xsd"))
    assert unsupported["cash_disposal_group"] is None
    # Common IFRS reported CFO remains a raw value; unknown extended schema
    # cannot resolve cash anchors or certify the adjustment as reconciled.
    assert unsupported["cfo"] == 4000
    assert tie(unsupported)["status"] == "UNKNOWN"


def test_one_valid_component_cannot_mask_another_reported_unresolved_component():
    body = document().replace(b'</body>',
        fact(IAS7_OTHER_CASH_CONCEPT, 0).replace(b'scale="3">0', b'scale="3" xsi:nil="true">') + b'</body>')
    row = record(body)
    assert tie(row)["method"] == "SOURCE_BOUND_CASH_COMPONENTS"
    assert tie(row)["status"] == "UNKNOWN" and row["cfo"] is row["capex"] is None
    assert tie(row)["missing_fields"] == ["cash_ias7_adjustment"]


@pytest.mark.parametrize("adjustment", [1797, 1803])
def test_disposal_amount_does_not_override_arithmetic_mismatch(adjustment):
    row = record(document(adjustment=adjustment))
    assert tie(row)["status"] == "FAIL" and row["cfo"] is row["capex"] is None


def test_matching_disposal_cash_does_not_manufacture_or_flip_cfo():
    body = document().replace(fact("ifrs-full:CashFlowsFromUsedInOperatingActivities", 4000, "YTD"),
                              fact("ifrs-full:CashFlowsFromUsedInOperatingActivities", -400, "YTD"))
    body = body.replace(fact("tifrs-SCF:CashFlowsFromUsedInFinancingActivities", -800, "YTD"),
                        fact("tifrs-SCF:CashFlowsFromUsedInFinancingActivities", 3600, "YTD"))
    row = record(body)
    assert all(c["status"] == "PASS" for c in row["validation_checks"]["cash_flow"])
    assert row["cfo"] == -400
    absent = record(document().replace(fact("ifrs-full:CashFlowsFromUsedInOperatingActivities", 4000, "YTD"), b''))
    assert tie(absent)["status"] == "PASS" and absent["cfo"] is None


def test_comparative_disposal_cash_is_bound_to_its_own_instant_not_current_value():
    contexts = (b'<xbrli:context id="PI"><xbrli:period><xbrli:instant>2025-06-30</xbrli:instant></xbrli:period></xbrli:context>'
        b'<xbrli:context id="PY"><xbrli:period><xbrli:startDate>2025-01-01</xbrli:startDate>'
        b'<xbrli:endDate>2025-06-30</xbrli:endDate></xbrli:period></xbrli:context>')
    amounts = {"ifrs-full:CashAndCashEquivalents": 6300, DISPOSAL_GROUP_CASH_CONCEPT: 0,
        "tifrs-SCF:CashAndCashEquivalentsAtBeginningOfPeriod": 5000,
        "tifrs-SCF:CashAndCashEquivalentsAtEndOfPeriod": 6300}
    flows = {"ifrs-full:CashFlowsFromUsedInOperatingActivities": 4000,
        "tifrs-SCF:NetCashFlowsFromUsedInInvestingActivities": -2000,
        "tifrs-SCF:CashFlowsFromUsedInFinancingActivities": -800,
        "ifrs-full:EffectOfExchangeRateChangesOnCashAndCashEquivalents": 100,
        "ifrs-full:IncreaseDecreaseInCashAndCashEquivalents": 1300}
    body = document().replace(b'</body>', contexts + b''.join(fact(c, v, ctx)
        for ctx, fields in (("PI", amounts), ("PY", flows)) for c, v in fields.items()) + b'</body>')
    row = record(body)
    check = row["validation_checks"]["comparative_cash_flow"][-1]
    assert check["status"] == "PASS" and check["method"] == "DISPOSAL_GROUP_SOURCE_BOUND_ADJUSTMENT"
    assert row["comparative_ytd_fields"]["cash_disposal_group"] == 0
    assert row["cash_disposal_group"] == 1800
    for proof in check["source_inputs"]:
        assert proof["reported_instant"] == "2025-06-30"
    assert check["source_inputs"][-1]["cash_flow_period_start"] == "2025-01-01"


def test_local_schema_migration_reuses_original_without_remote_fetch_or_time_rewrite(tmp_path, monkeypatch):
    repo = MopsRepository(tmp_path / "synthetic-mops")
    body = document()
    stored = repo.save_revision(_candidate(2026, 2), body, parse_ixbrl(body),
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:00:00Z")
    facts = repo.load_facts(stored.filing_id)
    originals = repo.load_filed_financials("2330", SIGNAL)
    native = deriver_module.ixbrl_record

    def old_adapter(symbol, filing):
        row = native(symbol, filing)
        row.update(cfo=None, capex=None, filing_schema="synthetic-before-disposal-mapping")
        return row

    with monkeypatch.context() as patch:
        patch.setattr(deriver_module, "FUNDAMENTAL_DERIVATION_VERSION", "synthetic-old-disposal-derivation")
        patch.setattr(deriver_module, "V6_FILING_SCHEMA_VERSION", "synthetic-before-disposal-mapping")
        patch.setattr(deriver_module, "ixbrl_record", old_adapter)
        with repo.unit_of_work() as uow:
            old, _ = FundamentalDeriver(repo).load_or_derive("2330", parsed_signal=pd.Timestamp(SIGNAL),
                resolved_sector="24", filed_financials=originals, uow=uow)
    assert old.snapshot["les_v6_mops_filings"][0]["cfo"] is None
    client = _FakeMopsClient(repo, [], [])
    svc = service(repo, client, Clock(), MopsRefreshPolicy(max_requests=0))
    assert svc.migrate_local_cache(["2330"], signal_date=SIGNAL, sector="24") == 1
    snapshot = repo.load_cached_bundles({"2330": "24"}, as_of=SIGNAL,
        derivation_version=deriver_module.FUNDAMENTAL_DERIVATION_VERSION)["2330"].bundle.snapshot
    assert snapshot["les_v6_filing_schema_version"] == V6_FILING_SCHEMA_VERSION
    migrated = snapshot["les_v6_mops_filings"][0]
    assert migrated["cfo"] == 4000 and migrated["cash_disposal_group"] == 1800
    assert migrated["content_sha256"] == stored.content_sha256
    assert migrated["available_at"] == stored.available_at
    assert repo.load_facts(stored.filing_id) == facts and repo.raw_path(stored).read_bytes() == body
    assert client.discovery_calls == 0 and client.downloaded_periods == []
    assert svc.migrate_local_cache(["2330"], signal_date=SIGNAL, sector="24") == 0


def test_daily_snapshot_assembly_retains_source_bound_disposal_cash():
    from application.fundamental_snapshot import BatchSnapshotResolution, SnapshotAssembler
    from domain.base_quality_features import BaseQualityFeatureProducer
    from domain.fundamental_derivation import FundamentalBundle
    from tests.fundamentals.test_fundamental_pipeline_components import _snapshot
    from domain.strategies.low_entry_v6_quality import _eligible

    cutoff = pd.Timestamp("2026-10-02T13:30:00+08:00")
    row = record(document())
    summary = _snapshot(fiscal_year=2026, quarter=2, observed_at="2026-10-01T00:00:00Z")
    bundle = FundamentalBundle(pd.DataFrame(), {"les_v6_mops_filings": [row]}, cutoff.isoformat())
    attached = SnapshotAssembler(None, BaseQualityFeatureProducer()).attach_base_evidence(
        bundle, BatchSnapshotResolution(summary, "TWSE_OPENAPI_CACHE_FALLBACK", True),
        symbol="2330", signal_time=cutoff, uow=None, history=[summary], persist_snapshot=False)
    selected = _eligible(attached.snapshot["les_v6_filings"], "2330", "TWSE", cutoff)[pd.Period("2026Q2")]
    assert selected["source"] == "MOPS_IXBRL"
    assert selected["cash_disposal_group"] == 1800 and selected["cfo"] == 4000
    assert selected["field_evidence"]["cash_disposal_group"] == row["field_evidence"]["cash_disposal_group"]
    assert tie(selected) == tie(row)


def test_future_disposal_revision_cannot_change_earlier_cutoff(tmp_path):
    repo = MopsRepository(tmp_path / "synthetic-mops")
    body = document()
    saved = repo.save_revision(_candidate(2026, 2), body, parse_ixbrl(body),
        available_at="2026-08-10T05:00:00Z", fetched_at="2026-08-10T05:00:00Z")
    cutoff = "2026-09-11T05:30:00Z"
    before = ixbrl_record("2330", repo.load_filed_financials("2330", cutoff)[0])
    changed = document(adjustment=1790)
    repo.save_revision(_candidate(2026, 2), changed, parse_ixbrl(changed),
        available_at="2026-09-12T05:00:00Z", fetched_at="2026-09-12T05:00:00Z")
    earlier = repo.load_filed_financials("2330", cutoff)
    assert len(earlier) == 1 and earlier[0].content_sha256 == saved.content_sha256
    assert ixbrl_record("2330", earlier[0]) == before
    later = repo.load_filed_financials("2330", "2026-09-13T05:00:00Z")
    latest = ixbrl_record("2330", max(later, key=lambda f: f.revision_number))
    assert tie(latest)["status"] == "FAIL" and latest["cfo"] is None

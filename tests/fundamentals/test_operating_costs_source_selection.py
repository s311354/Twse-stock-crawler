"""SYNTHETIC regressions: source repair must not manufacture financial PASS."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pandas as pd
import pytest

from domain.strategies.low_entry_v6_filings import ixbrl_record, V6_FILING_SCHEMA_VERSION
from domain.strategies.low_entry_v6_quality import (
    _eligible, combine_filing_sources, QUALITY_SOURCE_SELECTION_POLICY,
)
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from tests.fundamentals.test_ixbrl_units import document, filed

CUTOFF = pd.Timestamp("2026-10-02T13:30:00+08:00")


def original(*, cost=3000, gross=2000, net=1998, adjustment=2, extra=b""):
    body = document()
    additions = extra
    for concept, value in (
        ("tifrs-bsci-ci:OperatingCosts", cost),
        ("tifrs-bsci-ci:GrossProfitLossFromOperations", gross),
        ("ifrs-full:GrossProfit", net),
        ("tifrs-bsci-ci:UnrealizedProfitLossFromSales", adjustment),
        ("ifrs-full:CashFlowsFromUsedInOperatingActivities", -400),
    ):
        if value is not None:
            additions += (f'<ix:nonFraction name="{concept}" contextRef="YTD" '
                          f'unitRef="money" scale="3">{value}</ix:nonFraction>').encode()
    body = body.replace(b'</body>', additions + b'</body>')
    return ixbrl_record("2330", replace(filed(parse_ixbrl(body)),
        content_sha256=sha256(body).hexdigest(), source_url="https://mops.twse.com.tw/SYNTHETIC.html"))


def batch(row):
    return {**{key: row[key] for key in (
        "symbol", "market", "currency", "unit", "scope", "period_end", "flow_basis",
        "revenue", "operating_income", "net_income", "basic_eps")},
        "source": "TWSE_OPENAPI", "available_at": "2026-10-01T00:00:00Z",
        "revision_id": "synthetic-batch", "cost_of_revenue": 3002,
        "scope_basis": "LEGACY_DEFAULT_NOT_SOURCE_REPORTED"}


def selected(records):
    return _eligible(records, "2330", "TWSE", CUTOFF)[pd.Period("2026Q2")]


def test_operating_costs_and_two_gross_bases_are_distinct_exact_contexts():
    row = original()
    assert row["cost_of_revenue"] == 3000
    assert row["field_evidence"]["cost_of_revenue"]["reason"] == "REPORTED_EXACT_CONTEXT"
    fields = row["validation_checks"]["profitability"]["2026-01-01/2026-06-30"]["fields"]
    assert fields["gross_profit"]["value"] == 2000
    assert fields["gross_profit_net"]["value"] == 1998
    assert fields["unrealized_sales_profit"]["value"] == 2
    assert all(fields[key]["evidence"]["status"] == "RESOLVED" for key in
               ("cost_of_revenue", "gross_profit", "gross_profit_net"))
    assert row["filing_schema"] == V6_FILING_SCHEMA_VERSION


def test_missing_cost_uses_pre_adjustment_gross_not_net_gross():
    row = original(cost=None)
    assert row["cost_of_revenue"] == 3000  # NOT 3002.
    proof = row["field_evidence"]["cost_of_revenue"]
    assert proof["reason"] == "REVENUE_MINUS_GROSS_PROFIT_SAME_DOCUMENT"
    assert proof["inputs"][1]["contexts"][0]["concept"] == "tifrs-bsci-ci:GrossProfitLossFromOperations"
    assert selected([row])["cost_of_revenue"] == 3000


def test_ordinary_ifrs_gross_only_is_compatible_but_known_adjustment_is_not_ignored():
    row = original(cost=None, gross=None, net=2000, adjustment=None)
    assert row["cost_of_revenue"] == 3000
    assert selected([row])["cost_of_revenue"] == 3000
    assert original(cost=None, gross=None)["cost_of_revenue"] is None


@pytest.mark.parametrize("extra", [
    b'<ix:nonFraction name="tifrs-bsci-ci:OperatingCosts" contextRef="YTD" unitRef="money" scale="3">3001</ix:nonFraction>',
    b'<ix:nonFraction name="tifrs-bsci-ci:OperatingCosts" contextRef="YTD" unitRef="EarningsPerShare" scale="3">3000</ix:nonFraction>',
])
def test_conflicting_cost_or_wrong_unit_cannot_be_overridden_by_gross_fallback(extra):
    row = original(extra=extra)
    assert row["cost_of_revenue"] is None
    assert row["field_evidence"]["cost_of_revenue"]["reason"] in {
        "CONFLICTING_AGGREGATE_VALUES", "UNIT_MISSING_OR_INCOMPATIBLE"}


def test_conflicting_pre_adjustment_gross_cannot_fall_back_to_net_gross():
    row = original(cost=None, extra=b'<ix:nonFraction name="tifrs-bsci-ci:GrossProfitLossFromOperations" '
                   b'contextRef="YTD" unitRef="money" scale="3">2001</ix:nonFraction>')
    assert row["cost_of_revenue"] is None


def test_unverified_batch_uses_whole_original_preserves_negative_cfo_and_differences():
    row = original()
    summary = batch(row)
    before = deepcopy([row, summary])
    combined = combine_filing_sources([row], [summary], as_of=CUTOFF)
    actual = selected(combined)
    assert actual["source"] == "MOPS_IXBRL"
    assert actual["cfo"] == -400
    assert actual["cost_of_revenue"] == 3000
    assert actual["revision_id"] == row["revision_id"]
    assert actual["available_at"] == row["available_at"]
    assert actual["source_selection"]["policy"] == QUALITY_SOURCE_SELECTION_POLICY
    assert actual["source_selection"]["batch_differences"]["cost_of_revenue"] == {
        "batch": 3002, "original": 3000}
    assert [row, summary] == before


def test_future_original_and_explicit_batch_scope_keep_existing_fail_closed_contract():
    row = original()
    summary = batch(row)
    future = dict(row, available_at="2026-10-03T00:00:00Z")
    assert selected(combine_filing_sources([future], [summary], as_of=CUTOFF))["source"] == "TWSE_OPENAPI"
    summary.pop("scope_basis")
    assert selected(combine_filing_sources([row], [summary], as_of=CUTOFF)).get("cfo") is None


def test_latest_original_is_selected_even_if_cfo_missing_no_older_good_value_fallback():
    row = original()
    newer = dict(original(), available_at="2026-09-30T00:00:00Z", revision_id="synthetic-new", cfo=None)
    # Explicit missing evidence cannot inherit a prior original's healthy value.
    newer["field_evidence"] = dict(newer["field_evidence"], cfo={})
    actual = selected(combine_filing_sources([row, newer], [batch(row)], as_of=CUTOFF))
    assert actual["revision_id"] == "synthetic-new" and actual["cfo"] is None


def test_ambiguous_canonical_versions_are_not_hidden_by_batch_fallback():
    row = original()
    conflict = dict(row, revision_id="synthetic-conflict", revenue=1)
    with pytest.raises(ValueError, match="AMBIGUOUS_FILING_VERSION"):
        selected(combine_filing_sources([row, conflict], [batch(row)], as_of=CUTOFF))


def test_missing_original_hash_cannot_claim_canonical_source_precedence():
    row = dict(original(), content_sha256=None)
    assert selected(combine_filing_sources([row], [batch(row)], as_of=CUTOFF))["source"] == "TWSE_OPENAPI"


def test_production_snapshot_assembler_retains_original_cash_and_bound_comparisons():
    from application.fundamental_snapshot import BatchSnapshotResolution, SnapshotAssembler
    from domain.base_quality_features import BaseQualityFeatureProducer
    from domain.fundamental_derivation import FundamentalBundle
    from tests.fundamentals.test_fundamental_pipeline_components import _snapshot

    row = original()
    summary = _snapshot(fiscal_year=2026, quarter=2, observed_at="2026-10-01T00:00:00Z")
    bundle = FundamentalBundle(pd.DataFrame(), {"les_v6_mops_filings": [row]}, CUTOFF.isoformat())
    result = SnapshotAssembler(None, BaseQualityFeatureProducer()).attach_base_evidence(
        bundle, BatchSnapshotResolution(summary, "TWSE_OPENAPI_CACHE_FALLBACK", True),
        symbol="2330", signal_time=CUTOFF, uow=None, history=[summary], persist_snapshot=False)
    actual = selected(result.snapshot["les_v6_filings"])
    assert actual["cfo"] == -400 and actual["cost_of_revenue"] == 3000
    assert actual["comparative_ytd_fields"] == row["comparative_ytd_fields"]
    assert actual["comparative_year_end_fields"] == row["comparative_year_end_fields"]
    assert actual["source"] == "MOPS_IXBRL"

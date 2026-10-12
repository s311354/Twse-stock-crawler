"""SYNTHETIC hierarchy regressions, not historical or production evidence."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from domain.financial_taxonomy import (OPERATING_COST_CHILDREN, OPERATING_COST_CONCEPTS,
    OPERATING_COST_RECONCILIATION_POLICY, operating_cost_checks)
from domain.strategies.low_entry_v6_filings import ixbrl_record
from domain.strategies.low_entry_v6_quality_evidence import validated_proof_value
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from tests.fundamentals.test_ixbrl_units import document, filed

SCHEMA = "https://www.xbrl.org/tw/tifrs/2026-03-31/tifrs-ci-cr-2026-03-31.xsd"
PARTS = dict(cost_of_sales=354443, cost_of_construction_and_engineering=84494,
             cost_of_engineering=84494, cost_of_services=219050, other_operating_costs=6674)


def original(*, total=664661, gross=170177, net=None, parts=None, extra=b"", schema=SCHEMA):
    body = document().replace(b">5,000<", b">834838<")
    additions = f'<link:schemaRef xlink:href="{schema}"/>'.encode() if schema else b""
    for concept, value in (
        ("tifrs-bsci-ci:OperatingCosts", total),
        ("tifrs-bsci-ci:GrossProfitLossFromOperations", gross),
        ("ifrs-full:GrossProfit", gross if net is None else net),
        *((OPERATING_COST_CONCEPTS[k][0], v) for k, v in (PARTS if parts is None else parts).items()),
    ):
        if value is not None:
            additions += (f'<ix:nonFraction name="{concept}" contextRef="YTD" '
                          f'unitRef="money" scale="3">{value}</ix:nonFraction>').encode()
    body = body.replace(b"</body>", additions + extra + b"</body>")
    filing = replace(filed(parse_ixbrl(body)), content_sha256=sha256(body).hexdigest(),
        source_url="https://mops.twse.com.tw/SYNTHETIC-cost-hierarchy.html", taxonomy_schema_refs=(schema,) if schema else ())
    return ixbrl_record("8011", filing)


def window(row, key="2026-01-01/2026-06-30"):
    return row["validation_checks"]["profitability"][key]


def checks(row):
    return {c["check_id"]: c for c in window(row)["checks"]}


def operands(row):
    fields = window(row)["fields"]
    values = {k: v["value"] for k, v in fields.items()}
    evidence = {k: v["evidence"] for k, v in fields.items()}
    values["revenue"] = row["revenue"]
    evidence["revenue"] = row["field_evidence"]["revenue"]
    return values, evidence


@pytest.mark.parametrize("schema", [
    SCHEMA.replace("2026-03-31", "2020-06-30"),
    SCHEMA.replace("2026-03-31", "2025-06-30"), SCHEMA,
])
def test_reported_total_is_not_its_child_and_nested_subtotal_is_counted_once(schema):
    row = original(schema=schema)
    assert row["cost_of_revenue"] == 664661
    proof = row["field_evidence"]["cost_of_revenue"]
    assert {c["concept"] for c in proof["contexts"]} == {"tifrs-bsci-ci:OperatingCosts"}
    assert validated_proof_value(proof, row, "cost_of_revenue", start="2026-01-01", end="2026-06-30") == 664661
    detail = checks(row)["OPERATING_COST_DETAIL_TIE"]
    assert detail["expected_value"] == detail["observed_value"] == 664661
    assert detail["status"] == "PASS" and detail["complete"] is False
    assert "cost_of_engineering" not in detail["inputs"]  # Parent already contains it.
    assert "cost_of_construction" in checks(row)["CONSTRUCTION_COST_DETAIL_TIE"]["missing_fields"]
    assert detail["scope"] == "REPORTED_COMPONENTS_ONLY_NOT_PROOF_OF_EXHAUSTIVE_SCOPE"
    assert checks(row)["REVENUE_COST_GROSS_TIE"]["status"] == "PASS"
    assert window(row)["policy"] == OPERATING_COST_RECONCILIATION_POLICY


def test_complete_signed_hierarchy_is_checked_without_replacing_missing_with_zero():
    parts = dict.fromkeys(OPERATING_COST_CHILDREN, 0)
    parts.update(cost_of_sales=100, standard_cost_variances=-20)
    row = original(total=80, gross=834758, parts=parts)
    detail = checks(row)["OPERATING_COST_DETAIL_TIE"]
    assert row["cost_of_revenue"] == 80
    assert detail["status"] == "PASS" and detail["complete"] is True
    assert detail["missing_fields"] == []
    parts["standard_cost_variances"] = -10
    bad = original(total=80, gross=834758, parts=parts)
    assert checks(bad)["OPERATING_COST_DETAIL_TIE"]["status"] == "FAIL"
    assert bad["cost_of_revenue"] is None
    assert bad["field_evidence"]["cost_of_revenue"]["reason"] == "OPERATING_COST_RECONCILIATION_FAILED"
    assert window(bad)["fields"]["cost_of_revenue"]["value"] == 80  # Audit, not model input.


@pytest.mark.parametrize("sales", [50, 150])
def test_partial_detail_mismatch_is_not_proof_of_total_conflict_or_absent_zero(sales):
    row = original(total=100, gross=834738, parts={"cost_of_sales": sales})
    detail = checks(row)["OPERATING_COST_DETAIL_TIE"]
    assert detail["status"] == "UNKNOWN" and not detail["blocks_cost_inputs"]
    assert row["cost_of_revenue"] == 100  # Verified total + gross, not a partial sum.
    assert window(row)["fields"]["standard_cost_variances"]["value"] is None


def test_complete_nested_branch_conflict_blocks_even_if_overall_total_and_gross_tie():
    row = original(parts={**PARTS, "cost_of_construction": 10})
    assert checks(row)["CONSTRUCTION_COST_DETAIL_TIE"]["status"] == "FAIL"
    assert row["cost_of_revenue"] is None


def test_orphan_leaf_conflict_is_not_silently_discarded_when_subtotal_is_missing():
    parts = {k: v for k, v in PARTS.items() if k != "cost_of_construction_and_engineering"}
    extra = b'<ix:nonFraction name="tifrs-bsci-ci:CostOfEngineeringSales" contextRef="YTD" unitRef="money" scale="3">84495</ix:nonFraction>'
    row = original(parts=parts, extra=extra)
    assert row["cost_of_revenue"] is None
    assert any(c["check_id"] == "COST_COMPONENT_SOURCE_CHECK" and c["blocks_cost_inputs"] for c in window(row)["checks"])


@pytest.mark.parametrize("extra", [
    b'<ix:nonFraction name="tifrs-bsci-ci:OperatingCosts" contextRef="YTD" unitRef="money" scale="3">664662</ix:nonFraction>',
    b'<ix:nonFraction name="ifrs-full:CostOfRevenue" contextRef="YTD" unitRef="money" scale="3">664662</ix:nonFraction>',
])
def test_true_total_alias_conflicts_are_not_overridden_by_child_or_gross(extra):
    row = original(extra=extra)
    assert row["cost_of_revenue"] is None
    assert window(row)["fields"]["cost_of_revenue"]["evidence"]["reason"] == "CONFLICTING_AGGREGATE_VALUES"


@pytest.mark.parametrize("extra", [
    b'<ix:nonFraction name="ifrs-full:CostOfSales" contextRef="YTD" unitRef="money" scale="3">354444</ix:nonFraction>',
    b'<ix:nonFraction name="ifrs-full:CostOfSales" contextRef="YTD" unitRef="EarningsPerShare" scale="3">354443</ix:nonFraction>',
    b'<ix:nonFraction name="ifrs-full:CostOfSales" contextRef="YTD" unitRef="money" scale="broken">354443</ix:nonFraction>',
])
def test_invalid_reported_component_is_not_ignored_as_a_missing_category(extra):
    row = original(extra=extra)
    detail = checks(row)["OPERATING_COST_DETAIL_TIE"]
    assert detail["status"] == "UNKNOWN" and detail["blocks_cost_inputs"]
    assert row["cost_of_revenue"] is None


def test_genuine_gross_conflict_blocks_reported_cost_and_never_derives_another_total():
    row = original(gross=170000)
    assert checks(row)["REVENUE_COST_GROSS_TIE"]["status"] == "FAIL"
    assert row["cost_of_revenue"] is None


def test_adjusted_net_gross_is_not_a_pre_adjustment_alias_but_complete_bridge_conflicts_block():
    extra = b'<ix:nonFraction name="tifrs-bsci-ci:UnrealizedProfitLossFromSales" contextRef="YTD" unitRef="money" scale="3">10</ix:nonFraction>'
    extra += b'<ix:nonFraction name="tifrs-bsci-ci:RealizedProfitLossFromSales" contextRef="YTD" unitRef="money" scale="3">1</ix:nonFraction>'
    # This fixture net gross equals pre gross, so the explicit -10+1 bridge fails.
    row = original(extra=extra)
    assert checks(row)["GROSS_SALES_ADJUSTMENT_TIE"]["status"] == "FAIL"
    assert row["cost_of_revenue"] is None
    good = original(extra=extra, net=170168)
    assert good["cost_of_revenue"] == 664661
    assert checks(good)["GROSS_SALES_ADJUSTMENT_TIE"]["status"] == "PASS"


def test_missing_total_does_not_promote_sales_subtotal_or_sum_incomplete_details():
    row = original(total=None)
    assert row["cost_of_revenue"] == 664661
    assert row["field_evidence"]["cost_of_revenue"]["reason"] == "REVENUE_MINUS_GROSS_PROFIT_SAME_DOCUMENT"
    assert original(total=None, gross=None)["cost_of_revenue"] is None
    assert original(total=None, gross=480395, parts={"cost_of_sales": 354443})["cost_of_revenue"] == 354443


@pytest.mark.parametrize("key", ["content_sha256", "revision_id", "revision_number", "scope", "unit", "currency",
                                      "source_url", "available_at", "start_date", "end_date"])
def test_reconciliation_does_not_mix_original_period_unit_version_or_time(key):
    row = original()
    values, evidence = operands(row)
    before = deepcopy(evidence)
    evidence["cost_of_services"] = {**evidence["cost_of_services"], key: "foreign-source"}
    result = operating_cost_checks(values, evidence, start_date="2026-01-01", end_date="2026-06-30", taxonomy=row["taxonomy"])
    detail = next(c for c in result if c["check_id"] == "OPERATING_COST_DETAIL_TIE")
    assert detail["status"] == "UNKNOWN" and detail["blocks_cost_inputs"]
    assert detail["source_binding_errors"]
    assert evidence["cost_of_sales"] == before["cost_of_sales"]  # No shared mutation.


@pytest.mark.parametrize("schema", ["", SCHEMA.replace("2026-03-31", "2099-01-01"),
                                     SCHEMA.replace("www.xbrl.org", "untrusted.test")])
def test_unreviewed_hierarchy_cannot_claim_verified_cost_detail(schema):
    row = original(schema=schema)
    assert row["cost_of_revenue"] is None
    assert checks(row)["OPERATING_COST_DETAIL_TIE"]["status"] == "UNKNOWN"


def test_quarter_and_comparative_checks_use_their_own_exact_duration():
    extra = b""
    for context, start, end, revenue, total, gross in (
        ("Q2", "2026-04-01", "2026-06-30", 400492, 304430, 96062),
        ("LY", "2025-01-01", "2025-06-30", 1025863, 729605, 296258),
    ):
        extra += (f'<xbrli:context id="{context}"><xbrli:period><xbrli:startDate>{start}</xbrli:startDate>'
                  f'<xbrli:endDate>{end}</xbrli:endDate></xbrli:period></xbrli:context>').encode()
        for concept, value in (("ifrs-full:Revenue", revenue), ("tifrs-bsci-ci:OperatingCosts", total),
                               ("tifrs-bsci-ci:GrossProfitLossFromOperations", gross), ("ifrs-full:CostOfSales", total)):
            extra += (f'<ix:nonFraction name="{concept}" contextRef="{context}" unitRef="money" '
                      f'scale="3">{value}</ix:nonFraction>').encode()
    row = original(extra=extra)
    assert row["cost_of_revenue"] == 664661
    assert row["quarter_fields"]["cost_of_revenue"] == 304430
    assert row["comparative_ytd_fields"]["cost_of_revenue"] == 729605
    for key, expected in (("2026-04-01/2026-06-30", 304430), ("2025-01-01/2025-06-30", 729605)):
        check = next(c for c in window(row, key)["checks"] if c["check_id"] == "OPERATING_COST_DETAIL_TIE")
        assert check["expected_value"] == expected and check["status"] == "PASS"
        assert all(r["start_date"] == key.split("/")[0] for r in check["source_inputs"])

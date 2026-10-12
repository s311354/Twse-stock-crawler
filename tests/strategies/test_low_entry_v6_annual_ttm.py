"""Synthetic annual originals: TTM recognition, not cross-document gap filling."""
from copy import deepcopy

import pandas as pd
import pytest

from domain.strategies.low_entry_v6_quality import (
    FLOW_FIELDS, combine_filing_sources, evaluate_quality, reported_annual_flow,
)
from tests.strategies.test_low_entry_score_v6 import financials
from tests.strategies.quality_evidence_fixtures import bind_synthetic_contexts


def annual_records():
    rows = [r for r in financials() if r["period_end"] <= "2025-12-31"]
    for index, row in enumerate(rows):
        period = pd.Period(row["period_end"], freq="Q")
        if period.quarter != 4:
            continue
        year = rows[index - 3:index + 1]
        values = {field: sum(r[field] for r in year) for field in FLOW_FIELDS}
        row.update(values, source="MOPS_IXBRL", content_sha256=f"synthetic-{period}",
                   revision_id=f"synthetic-{period}", flow_basis="YTD",
                   flow_derivation_policy="SAME_DOCUMENT_ONLY")
        bind_synthetic_contexts(row)
    return rows


def evaluate(rows):
    return evaluate_quality(rows, symbol="2330", market="TWSE", as_of="2026-09-11T13:30+08:00")


def test_verified_full_year_is_ttm_without_inventing_fourth_quarter_values():
    rows = annual_records()
    before = deepcopy(rows)
    q = evaluate(rows)
    for field in FLOW_FIELDS:
        if field != "basic_eps":
            assert q.metrics[field + "_ttm"] == rows[-1][field]
            assert q.metrics["quality_ttm_inputs"][field + "_ttm"]["method"] == "REPORTED_FULL_YEAR"
    assert q.metrics["cfo_ttm"] == rows[-1]["cfo"]
    assert q.metrics["fcf_ttm"] == rows[-1]["cfo"] - abs(rows[-1]["capex"])
    # Canonical year comparisons need same-original comparative contexts; an
    # unrelated earlier original is not a fallback for an absent comparison.
    assert q.metrics["revenue_growth"] is None
    assert q.metrics["quarter_flow_methods"]["2025Q4"]["cfo"] == "REVISION_BASIS_UNVERIFIED"
    assert q.metrics["gross_margin"] == pytest.approx(.5)  # Uniform TTM, not Q4 invented by subtraction.
    assert q.metrics["positive_eps_quarters"] is None
    assert q.coverage["growth"] == 0
    assert rows == before


@pytest.mark.parametrize("key,value", [("content_sha256", "other"), ("revision_id", "other"),
    ("available_at", "2025-01-01"), ("scope", "individual"), ("unit", "yuan"),
    ("start_date", "2025-07-01"), ("end_date", "2025-09-30"), ("status", "UNKNOWN")])
def test_annual_amount_must_be_bound_to_exact_original_year_scope_unit_and_version(key, value):
    rows = annual_records()
    rows[-1]["field_evidence"]["cfo"][key] = value
    q = evaluate(rows)
    assert q.metrics["cfo_ttm"] is None
    assert "cfo_reported" in q.metrics["quality_required_missing"]
    assert q.status == "DATA_INSUFFICIENT"


def test_half_year_amount_cannot_be_promoted_to_ttm_and_future_revision_is_excluded():
    row = annual_records()[-1]
    assert reported_annual_flow(dict(row, period_end="2025-06-30"), pd.Period("2025Q2"), "cfo") is None
    rows = annual_records()
    expected = evaluate(rows)
    future = dict(rows[-1], available_at="2026-09-12T00:00Z", revision_id="future", cfo=-10**12)
    assert evaluate(rows + [future]) == expected


def test_audit_and_required_source_gaps_are_not_mutually_overwritten():
    rows = annual_records()
    rows[-1]["cfo"] = None
    rows[-1]["audit_review_required"] = True
    q = evaluate(rows)
    assert q.status == "DATA_INSUFFICIENT"
    assert q.reasons == ("REQUIRED_QUALITY_SOURCES_UNKNOWN", "AUDIT_REVIEW_REQUIRED")


def test_batch_annual_flow_needs_complete_original_tie_not_just_period_match():
    rows = annual_records()
    batch = {k: v for k, v in rows[-1].items() if k not in
             {"field_evidence", "cfo", "capex", "income_tax", "pretax_income", "cash", "interest_bearing_debt"}}
    batch.update(source="TWSE_OPENAPI", revision_id="synthetic-batch", content_sha256="synthetic-batch")
    combined = combine_filing_sources(rows, [batch], as_of="2026-09-11T13:30+08:00")
    assert evaluate(combined).metrics["cfo_ttm"] == rows[-1]["cfo"]
    batch["revenue"] += 1
    combined = combine_filing_sources(rows, [batch], as_of="2026-09-11T13:30+08:00")
    assert evaluate(combined).metrics["cfo_ttm"] is None

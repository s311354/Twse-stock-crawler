"""SYNTHETIC cash-flow/ROIC evidence. Not historical or investment performance."""
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from domain.fundamental_derivation import FiledFact
from domain.strategies.low_entry_v6_decision import common_safety_caps
from domain.strategies.low_entry_v6_filings import ixbrl_record
from domain.strategies.low_entry_v6_quality import (
    QUALITY_BUDGETS, QUALITY_POLICY_VERSION, combine_filing_sources,
    evaluate_quality, normalize_quarters,
)
from domain.strategies.low_entry_v6_report import entry_gate_checks
from tests.fundamentals.test_financial_field_repair import record, source
from tests.fundamentals.test_ixbrl_units import filed
from tests.strategies.test_low_entry_score_v6 import execution_report, features, financials


def evaluate(rows):
    return evaluate_quality(rows, symbol="2330", market="TWSE", as_of="2026-09-11T13:30:00+08:00")


def test_five_budgets_total_100_cash_flow_is_scored_and_roic_is_diagnostic():
    result = evaluate(financials())
    assert sum(QUALITY_BUDGETS.values()) == 100
    assert result.status == "PASS"
    assert result.metrics["quality_policy_version"] == QUALITY_POLICY_VERSION
    assert result.coverage["cash_flow"] == result.coverage["return_on_capital"] == 100
    assert result.components["cash_flow"] == 15
    assert result.metrics["cash_conversion"] == pytest.approx(1.2)
    revenue = sum(r["revenue"] for r in financials()[-4:])
    assert result.metrics["cfo_ttm"] == pytest.approx(revenue * .3)
    assert result.metrics["capex_ttm"] == pytest.approx(revenue * .1)
    assert result.metrics["fcf_ttm"] == pytest.approx(revenue * .2)
    assert result.metrics["normalized_tax_rate"] == pytest.approx(.2)
    assert result.metrics["nopat_ttm"] == pytest.approx(revenue * .3 * .8)
    assert result.metrics["average_invested_capital"] == 7000
    assert result.metrics["roic"] == pytest.approx(revenue * .3 * .8 / 7000)


def test_negative_cfo_and_fcf_earn_zero_not_positive_bonus():
    rows = financials()
    for row in rows[-4:]:
        row["cfo"] = -abs(row["cfo"])
    good, bad = evaluate(financials()), evaluate(rows)
    assert bad.components["cash_flow"] == 0
    assert bad.coverage["cash_flow"] == 100  # Known poor results, not missing evidence.
    assert bad.score < good.score and bad.metrics["fcf_ttm"] < 0


def test_missing_capex_cannot_use_forged_fcf_or_assume_zero():
    rows = financials()
    rows[-1].update(capex=None, fcf=10**12, roic=999)
    q = evaluate(rows)
    assert q.metrics["capex_ttm"] is q.metrics["fcf_ttm"] is None
    assert q.components["cash_flow"] == 10
    assert q.coverage["cash_flow"] == pytest.approx(200/3)
    assert q.status == "PASS"
    assert q.metrics["quality_penalty"] == 1.25
    assert not common_safety_caps(q, features(), industry_mode="ACTIVE")


@pytest.mark.parametrize("field", ["cfo", "pretax_income", "income_tax", "cash", "interest_bearing_debt"])
def test_missing_cfo_is_required_but_roic_only_sources_are_diagnostic(field):
    rows = financials()
    rows[-1][field] = None
    q = evaluate(rows)
    assert q.status == ("DATA_INSUFFICIENT" if field == "cfo" else "PASS")
    if field != "cfo":
        assert q.metrics["roic"] is None
        assert q.coverage["return_on_capital"] == 100
        assert q.score == evaluate(financials()).score
        assert q.confidence == evaluate(financials()).confidence
    else:
        assert q.metrics["cfo_ttm"] is q.metrics["fcf_ttm"] is None


def test_capex_outflow_sign_is_normalized_but_net_investing_flow_is_not_a_substitute():
    rows = financials()
    for row in rows:
        row["capex"] = -row["capex"]
    assert evaluate(rows).metrics["fcf_ttm"] == evaluate(financials()).metrics["fcf_ttm"]
    rows[-1]["capex"] = None
    rows[-1]["investing_cash_flow"] = -1
    assert evaluate(rows).metrics["fcf_ttm"] is None


@pytest.mark.parametrize("damage", ["missing_prior", "nonpositive_capital", "nonpositive_pretax"])
def test_roic_does_not_substitute_current_capital_or_fixed_tax(damage):
    rows = financials()
    if damage == "missing_prior":
        rows[-5]["cash"] = None
    elif damage == "nonpositive_capital":
        # A consistent cash-heavy balance sheet, not cash exceeding all assets.
        rows[-1].update(cash=8_000, current_assets=8_000, non_current_assets=2_000)
    else:
        for row in rows[-4:]:
            row["pretax_income"] = -1
    q = evaluate(rows)
    assert q.metrics["roic"] is None and q.status == "PASS"


def test_future_cash_restatement_does_not_change_earlier_cutoff():
    rows = financials()
    future = dict(rows[-1], revision_id="later", available_at="2026-09-12T01:00:00Z", cfo=-10**9)
    assert evaluate(rows + [future]) == evaluate(rows)


@pytest.mark.parametrize("effective,expected", [(-.1, 0.), (.2, .2), (.8, .35)])
def test_tax_rate_is_derived_then_clamped_not_filled(effective, expected):
    rows = financials()
    for row in rows[-4:]:
        row["income_tax"] = row["pretax_income"] * effective
    q = evaluate(rows)
    assert q.metrics["effective_tax_rate"] == pytest.approx(effective)
    assert q.metrics["normalized_tax_rate"] == pytest.approx(expected)
    assert q.metrics["nopat_ttm"] == pytest.approx(q.metrics["operating_income_ttm"] * (1 - expected))


def test_report_cannot_claim_pass_with_old_four_component_coverage():
    report = execution_report()
    del report["quality"]["coverage"]["cash_flow"]
    assert entry_gate_checks(report)["quality"] != "PASS"


def test_capital_flow_uses_exact_context_and_same_original_ytd_basis():
    body = source(predecessor=True)
    additions = b''
    for context, value in (("YTD", 4000), ("Q1", 1500)):
        additions += f'<ix:nonFraction name="ifrs-full:CashFlowsFromUsedInOperatingActivities" contextRef="{context}" unitRef="money" scale="3">{value}</ix:nonFraction>'.encode()
    row = record(body.replace(b'</body>', additions + b'</body>'))
    q = pd.Period("2026Q2")
    assert row["cfo"] == 4000
    assert normalize_quarters({q: row})[q]["cfo"] == 2500
    row["prior_ytd_field_evidence"]["cfo"]["revision_id"] = "other"
    assert normalize_quarters({q: row})[q]["cfo"] is None


def test_cash_and_debt_exact_instant_do_not_count_missing_lease_as_zero():
    def instant(concept, value):
        return FiledFact(concept, value * 1000, None, None, None, "2026-06-30", unit="iso4217:TWD")
    facts = [instant("ifrs-full:CurrentBorrowings", 300), instant("ifrs-full:NoncurrentBorrowings", 500),
             instant("ifrs-full:CurrentLeaseLiabilities", 0), instant("ifrs-full:NoncurrentLeaseLiabilities", 20),
             instant("ifrs-full:CashAndCashEquivalents", 400)]
    filing = replace(filed([]), facts=tuple(facts), content_sha256="synthetic-original")
    row = ixbrl_record("2330", filing)
    assert row["interest_bearing_debt"] == 820 and row["cash"] == 400
    assert row["field_evidence"]["interest_bearing_debt"]["basis"] == "INSTANT"
    missing = ixbrl_record("2330", replace(filing, facts=tuple(facts[:2] + facts[3:])))
    assert missing["interest_bearing_debt"] is None
    overlapping = ixbrl_record("2330", replace(filing, facts=tuple(facts + [instant("ifrs-full:Borrowings", 800)])))
    assert overlapping["interest_bearing_debt"] == 820  # No double counting.
    conflict = ixbrl_record("2330", replace(filing, facts=tuple(facts + [instant("ifrs-full:Borrowings", 900)])))
    assert conflict["interest_bearing_debt"] is None
    negative = [instant("ifrs-full:CurrentBorrowings", -300), instant("ifrs-full:NoncurrentBorrowings", 1100)] + facts[2:]
    assert ixbrl_record("2330", replace(filing, facts=tuple(negative)))["interest_bearing_debt"] is None


def test_batch_supplement_requires_full_core_tie_and_does_not_mutate_evidence():
    # Synthetic complete quarter Q1 ties to the official batch core. Extra cash
    # evidence must not destroy the established EPS/core matching contract.
    mops = dict(financials()[0], source="MOPS_IXBRL", flow_basis="YTD",
                flow_derivation_policy="SAME_DOCUMENT_ONLY", field_evidence={}, content_sha256="synthetic")
    batch = {k: v for k, v in mops.items() if k not in
             {"cfo", "capex", "income_tax", "pretax_income", "cash", "interest_bearing_debt"}}
    batch.update(source="TWSE_OPENAPI", revision_id="batch")
    originals = deepcopy([mops, batch])
    combined = combine_filing_sources([mops], [batch], as_of="2026-09-11")
    q = evaluate(combined)
    # Only one quarter: insufficient TTM, but the supplemental cash amount is
    # preserved and never fabricates four quarters or PASS.
    assert q.metrics["quarter_flow_methods"]["2023Q1"]["cfo"] == "REPORTED_QUARTER"
    assert q.metrics["cfo_ttm"] is None and q.status != "PASS"
    assert [mops, batch] == originals


def test_missing_diagnostic_eps_does_not_block_core_tied_cash_flow_enrichment():
    mops = dict(financials()[0], source="MOPS_IXBRL", flow_basis="YTD",
                flow_derivation_policy="SAME_DOCUMENT_ONLY", field_evidence={},
                content_sha256="synthetic", basic_eps=None, basic_eps_quarter=None)
    batch = {k: v for k, v in mops.items() if k not in
             {"cfo", "capex", "income_tax", "pretax_income", "cash", "interest_bearing_debt"}}
    batch.update(source="TWSE_OPENAPI", revision_id="batch", basic_eps=.8)
    before = deepcopy([mops, batch])
    combined = combine_filing_sources([mops], [batch], as_of="2026-09-11")
    q = evaluate(combined)
    assert q.metrics["quarter_flow_methods"]["2023Q1"]["cfo"] == "REPORTED_QUARTER"
    assert q.metrics["positive_eps_quarters"] is None
    assert q.status != "PASS"  # One quarter cannot become a complete TTM.
    assert before == [mops, batch]
    mismatched = dict(mops, revenue=1.)
    rejected = evaluate(combine_filing_sources([mismatched], [batch], as_of="2026-09-11"))
    assert rejected.metrics["quality_field_checks"]["2023Q1:cfo"]["value"] is None

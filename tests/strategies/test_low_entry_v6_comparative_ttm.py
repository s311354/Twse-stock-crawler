"""SYNTHETIC period/provenance regressions, not real or investable evidence."""
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.strategies.low_entry_v6_filings import FIELD_CONCEPTS, ixbrl_record
from domain.strategies.low_entry_v6_quality import evaluate_quality, normalize_quarters
from domain.strategies.low_entry_v6_ttm import TTM_BRIDGE_POLICY_VERSION, comparative_ttm
from tests.strategies.test_low_entry_score_v6 import financials


CUTOFF = "2026-10-02T13:30+08:00"
POLICY = "(一)初次適用準則對本公司並無重大影響。(二)未來IFRS18影響仍待評估。"


def originals():
    quarters = financials()
    rows, filings = [], []
    for row in quarters:
        period = pd.Period(row["period_end"], freq="Q")
        facts = []
        for p in (period, period - 4):
            previous = [r for r in quarters if pd.Period(r["period_end"], freq="Q").year == p.year
                        and pd.Period(r["period_end"], freq="Q") <= p]
            if len(previous) != p.quarter:
                continue
            for field in ("revenue", "cost_of_revenue", "operating_income", "net_income", "cfo", "capex", "income_tax", "pretax_income"):
                facts.append(FiledFact(FIELD_CONCEPTS[field][0], sum(r[field] for r in previous) * 1000, None,
                    f"{p.year}-01-01", p.end_time.date().isoformat(), None, unit="iso4217:TWD", context_id=f"{p}-YTD"))
        for end in (row["period_end"], f"{period.year-1}-12-31"):
            for field in ("current_assets", "total_assets", "current_liabilities", "total_liabilities", "total_equity", "cash"):
                facts.append(FiledFact(FIELD_CONCEPTS[field][0], row[field] * 1000, None, None, None, end,
                    unit="iso4217:TWD", context_id=f"{end}-BALANCE"))
        facts.append(FiledFact("tifrs-notes:ApplicationOfNewlyIssuedOrAmendedStandardsAndInterpretations", None,
            POLICY, f"{period.year}-01-01", row["period_end"], None, context_id="POLICY"))
        filing = FiledFinancials(period.year, period.quarter, row["available_at"], 1, tuple(facts),
            content_sha256=f"SYNTHETIC-{period}", source_url="https://mops.twse.com.tw/SYNTHETIC",
            period_end=row["period_end"])
        filings.append(filing)
        rows.append(ixbrl_record("2330", filing))
    return rows, filings


def evaluate(rows):
    return evaluate_quality(rows, symbol="2330", market="TWSE", as_of=CUTOFF)


def test_bridge_recovers_required_ttm_not_missing_quarters_or_eps():
    rows, _ = originals()
    before = deepcopy(rows)
    q = evaluate(rows)
    for field in ("revenue", "cost_of_revenue", "operating_income", "net_income", "cfo"):
        trace = q.metrics["quality_ttm_inputs"][field + "_ttm"]
        assert trace["status"] == "RESOLVED"
        assert trace["method"] == "COMPARATIVE_ORIGINAL_TTM"
        assert q.metrics[field + "_ttm"] == pytest.approx(
            rows[-3][field] + rows[-1][field] - rows[-1]["comparative_ytd_fields"][field])
        assert len(trace["sources"]) == 3 and trace["quarter_values_filled"] is False
        assert pd.Timestamp(trace["available_at"]) <= pd.Timestamp(CUTOFF)
    assert q.metrics["quality_required_missing"] == []
    assert q.metrics["basic_eps_ttm"] is None
    assert normalize_quarters({pd.Period(r["period_end"], freq="Q"): r for r in rows})[pd.Period("2025Q4")]["cfo"] is None
    # Qualification is governed by the current TTM scoring/coverage policy;
    # this source regression checks that no missing quarter or EPS is invented.
    assert rows == before
    assert q.metrics["quality_profitability_basis"] == "REPORTED_SAME_PERIOD_MARGIN"
    assert q.metrics["positive_eps_quarters"] is None
    assert q.coverage["growth"] == 100.  # Same-original YTD growth; quarter EPS is diagnostic only.
    assert q.metrics["fcf_ttm"] == pytest.approx(q.metrics["cfo_ttm"] - q.metrics["capex_ttm"])
    assert q.metrics["quality_ttm_inputs"]["revenue_previous_ttm"]["status"] == "RESOLVED"


@pytest.mark.parametrize("group,key,value", [
    ("comparative_ytd_field_evidence", "content_sha256", "another"),
    ("comparative_ytd_field_evidence", "unit", "yuan"),
    ("comparative_ytd_field_evidence", "scope", "individual"),
    ("comparative_ytd_field_evidence", "start_date", "2025-04-01"),
    ("comparative_ytd_field_evidence", "available_at", "2025-01-01"),
    ("field_evidence", "status", "UNKNOWN"),
])
def test_provenance_mismatch_never_supplements_a_required_flow(group, key, value):
    rows, _ = originals()
    rows[-1][group]["cfo"][key] = value
    q = evaluate(rows)
    assert q.metrics["cfo_ttm"] is None
    if group == "field_evidence":
        assert q.status == "DATA_INSUFFICIENT" and q.metrics["cfo_reported"] is None
    else:
        assert q.status == "PASS" and q.metrics["cfo_reported"] == rows[-1]["cfo"]
        assert q.metrics["quality_field_checks"]["2026Q2:comparative:cfo"]["value"] is None
        assert "2026Q2:comparative:cfo" not in q.metrics["quality_verification"]["dependencies"]


@pytest.mark.parametrize("damage", ["comp_amount", "scope", "correction", "missing_original",
                                     "future_annual", "later_annual"])
def test_changed_or_unavailable_basis_is_not_hidden_by_a_bridge(damage):
    rows, _ = originals()
    if damage == "comp_amount":
        rows[-1]["comparative_ytd_fields"]["cfo"] += 1
    elif damage == "scope":
        rows[-3]["scope"] = "individual"
    elif damage == "correction":
        rows[-3]["correction_description"] = "SYNTHETIC revision"
    elif damage == "missing_original":
        del rows[-5]
    elif damage == "future_annual":
        rows[-3]["available_at"] = "2026-10-03T00:00Z"
    else:
        rows[-3]["available_at"] = "2026-09-01T00:00Z"
        rows[-3]["availability_source"] = rows[-1]["availability_source"] = "MOPS_PUBLIC_ANNOUNCEMENT"
    assert evaluate(rows).metrics["cfo_ttm"] is None


def bridge(rows, field="cfo"):
    return comparative_ttm({pd.Period(row["period_end"], freq="Q"): row for row in rows},
                           pd.Period("2026Q2"), field, CUTOFF)


def attest(row, field, *, issues):
    """SYNTHETIC explicit source review; not an official publication proof."""
    dependencies = {context["concept"] for group in ("field_evidence", "comparative_ytd_field_evidence")
                    for context in row.get(group, {}).get(field, {}).get("contexts", [])
                    if context.get("unit") == "iso4217:TWD"}
    policy_context = row["accounting_basis_evidence"]["contexts"][0]
    return dict({key: row.get(key) for key in
                 ("symbol", "market", "source", "source_url", "content_sha256", "revision_id", "scope",
                  "currency", "unit", "period_end", "available_at")},
                status="RESOLVED", impact="UNAFFECTED", field=field, policy=TTM_BRIDGE_POLICY_VERSION,
                verification_method="SOURCE_BOUND_EXPLICIT_FIELD_IMPACT", issues=issues,
                dependency_concepts=sorted(dependencies), evidence_available_at="2026-09-30T00:00Z",
                verified_at="2026-09-30T00:00Z", correction_description=row.get("correction_description"),
                contexts=[dict(context_id=policy_context["context_id"],
                               statement=policy_context["current_adoption_text"])])


def test_unrelated_ap_reclassification_is_diagnostic_not_a_cfo_operand():
    rows, _ = originals()
    before = deepcopy(rows)
    prior = rows[-5]
    prior["balance_ledger"]["ifrs-full:TradePayables"] = 300
    rows[-1]["comparative_year_end_ledger"]["ifrs-full:TradePayables"] = 350
    rows[-3]["balance_ledger"]["ifrs-full:TradePayables"] = 300
    value, trace = bridge(rows)
    assert value is not None and trace["status"] == "RESOLVED"
    assert any(diagnostic["comparison"] == "annual_end" and
               diagnostic["fields"] == ["ifrs-full:TradePayables"]
               for diagnostic in trace["diagnostics"])
    assert all(diagnostic["affects_ttm_field"] is False for diagnostic in trace["diagnostics"])
    assert "ifrs-full:TradePayables" not in trace["dependency_concepts"]
    # Evaluation itself is pure; the fixture edits above are the only changes.
    expected = deepcopy(rows)
    bridge(rows)
    assert rows == expected and before != rows


@pytest.mark.parametrize("damage", ["anchor", "ledger", "missing_anchor"])
def test_annual_balance_comparatives_are_kept_for_separate_bs_diagnostics(damage):
    rows, _ = originals()
    if damage == "anchor":
        rows[-1]["comparative_year_end_fields"]["total_assets"] += 1
    elif damage == "ledger":
        rows[-1]["comparative_year_end_ledger"]["ifrs-full:Assets"] += 1
    else:
        del rows[-1]["comparative_year_end_fields"]["total_assets"]
    value, trace = bridge(rows)
    assert value is not None and trace["diagnostics"]


def test_changed_revenue_dependency_blocks_revenue_not_unrelated_cfo():
    rows, _ = originals()
    rows[-1]["comparative_statement_ledger"]["ifrs-full:Revenue"] += 1
    revenue, trace = bridge(rows, "revenue")
    assert revenue is None
    assert trace["gaps"][0]["reason"] == "COMPARATIVE_FIELD_DEPENDENCY_VALUES_CHANGED"
    cfo, cfo_trace = bridge(rows)
    assert cfo is not None
    assert cfo_trace["diagnostics"][0]["fields"] == ["ifrs-full:Revenue"]


def test_derived_cost_keeps_revenue_and_gross_profit_dependencies():
    rows, filings = originals()
    converted = []
    for filing in filings:
        revenues = {fact.context_id: fact.numeric_value for fact in filing.facts
                    if fact.concept == "ifrs-full:Revenue"}
        facts = tuple(replace(fact, concept="ifrs-full:GrossProfit",
                              numeric_value=revenues[fact.context_id] - fact.numeric_value)
                      if fact.concept == FIELD_CONCEPTS["cost_of_revenue"][0] else fact for fact in filing.facts)
        converted.append(ixbrl_record("2330", replace(filing, facts=facts)))
    expected = bridge(rows, "cost_of_revenue")[0]
    value, trace = bridge(converted, "cost_of_revenue")
    assert value == pytest.approx(expected)
    assert trace["dependency_concepts"] == ["ifrs-full:GrossProfit", "ifrs-full:Revenue"]
    converted[-1]["comparative_statement_ledger"]["ifrs-full:GrossProfit"] += 1
    assert bridge(converted, "cost_of_revenue")[0] is None
    assert bridge(converted, "cfo")[0] is not None


@pytest.mark.parametrize("group", ["field_evidence", "comparative_ytd_field_evidence"])
def test_cfo_operand_cannot_disagree_with_original_numeric_context(group):
    rows, _ = originals()
    proof = rows[-1][group]["cfo"]
    exact_context = next(context for context in proof["contexts"]
                         if context["start_date"] == proof["start_date"] and context["end_date"] == proof["end_date"])
    exact_context["numeric_value"] += 1000
    assert bridge(rows)[0] is None


def test_policy_exception_needs_explicit_field_review_and_never_blanket_resolves():
    rows, filings = originals()
    source = filings[-1]
    text = "(一)準則採用涉及應付帳款重分類；營業活動現金流量不受影響。(二)未來IFRS18。"
    facts = tuple(replace(f, text_value=text) if f.context_id == "POLICY" else f for f in source.facts)
    rows[-1] = ixbrl_record("2330", replace(source, facts=facts))
    assert bridge(rows)[0] is None
    proof = attest(rows[-1], "cfo", issues=["ACCOUNTING_POLICY"])
    rows[-1]["field_impact_evidence"] = {"cfo": proof}
    value, trace = bridge(rows)
    assert value is not None and trace["field_impact_attestations"] == [proof]
    assert pd.Timestamp(trace["available_at"]) == pd.Timestamp(proof["verified_at"])
    assert bridge(rows, "revenue")[0] is None


@pytest.mark.parametrize("key,value", [
    ("content_sha256", "other-original"), ("revision_id", "other-revision"),
    ("symbol", "3016"), ("market", "TPEX"), ("scope", "individual"),
    ("policy", "old-policy"), ("impact", "UNKNOWN"), ("issues", []),
    ("dependency_concepts", []), ("verified_at", "2026-10-03T00:00Z"),
    ("verified_at", "2020-01-01T00:00Z"), ("evidence_available_at", "2026-10-03T00:00Z"),
    ("contexts", [{"context_id": "POLICY", "statement": "not in the original"}]),
])
def test_field_review_is_source_bound_versioned_and_cutoff_isolated(key, value):
    rows, _ = originals()
    rows[-1]["accounting_basis_evidence"].update(status="UNKNOWN", reason="ACCOUNTING_POLICY_EXCEPTION_REQUIRES_REVIEW")
    proof = attest(rows[-1], "cfo", issues=["ACCOUNTING_POLICY"])
    proof[key] = value
    rows[-1]["field_impact_evidence"] = {"cfo": proof}
    assert bridge(rows)[0] is None


def test_correction_review_binds_each_affected_original_and_its_description():
    rows, _ = originals()
    annual = rows[-3]
    annual["correction_description"] = "SYNTHETIC AP presentation correction; CFO unchanged"
    assert bridge(rows)[0] is None
    proof = attest(annual, "cfo", issues=["FILING_CORRECTION"])
    annual["field_impact_evidence"] = {"cfo": proof}
    assert bridge(rows)[0] is not None
    annual["correction_description"] = "SYNTHETIC different correction"
    assert bridge(rows)[0] is None


def test_later_observation_of_original_is_allowed_only_after_that_time_not_backdated():
    rows, _ = originals()
    row = rows[-3]
    row.update(available_at="2026-09-01T00:00Z", availability_source="FIRST_OBSERVED_DOWNLOAD")
    for proof in row["field_evidence"].values():
        proof["available_at"] = row["available_at"]
    assert evaluate(rows).metrics["cfo_ttm"] is not None
    assert evaluate_quality(rows, symbol="2330", market="TWSE", as_of="2026-08-31T13:30+08:00").metrics["cfo_ttm"] is None


@pytest.mark.parametrize("text,reason", [
    ("(一)除下列說明外，無重大影響；選擇不重編比較資訊。(二)未來IFRS18。", "ACCOUNTING_POLICY_EXCEPTION_REQUIRES_REVIEW"),
    ("(一)首次適用影響待評估。(二)未來準則不致造成重大影響。", "CURRENT_ACCOUNTING_POLICY_ATTESTATION_MISSING"),
    ("缺少分節而只有未來準則不致造成重大影響。", "CURRENT_ACCOUNTING_POLICY_ATTESTATION_MISSING"),
])
def test_current_adoption_is_not_inferred_from_future_or_exception_notes(text, reason):
    rows, filings = originals()
    source = filings[-1]
    facts = tuple(replace(f, text_value=text) if f.context_id == "POLICY" else f for f in source.facts)
    rows[-1] = ixbrl_record("2330", replace(source, facts=facts))
    q = evaluate(rows)
    assert q.metrics["cfo_ttm"] is None
    assert q.metrics["quality_ttm_inputs"]["cfo_ttm"]["comparative_bridge_gaps"][0]["reason"] == reason


def test_arabic_headings_and_cjk_layout_preserve_current_policy_only():
    rows, filings = originals()
    source = filings[-1]
    facts = tuple(replace(f, text_value="1.首次適用對本集團並無重 大 影 響。 2.未來IFRS18重編比較數。")
                  if f.context_id == "POLICY" else f for f in source.facts)
    rows[-1] = ixbrl_record("2330", replace(source, facts=facts))
    assert evaluate(rows).metrics["cfo_ttm"] is not None


def test_future_revision_cannot_change_earlier_ttm_or_required_gaps():
    rows, _ = originals()
    expected = evaluate(rows)
    future = deepcopy(rows[-1])
    future.update(available_at="2026-10-03T00:00Z", revision_id="future", cfo=-10**12)
    assert evaluate(rows + [future]) == expected


def test_conflicting_reported_quarter_is_not_hidden_by_ttm_fallback():
    rows, _ = originals()
    rows[-1]["quarter_field_evidence"]["cfo"].update(reason="CONFLICTING_AGGREGATE_VALUES", category="INCOMPARABLE")
    assert evaluate(rows).metrics["cfo_ttm"] is None


def test_capex_sign_and_eps_policy():
    rows, _ = originals()
    rows[-1]["comparative_ytd_fields"]["capex"] *= -1
    rows[-5]["capex"] *= -1
    p = {pd.Period(r["period_end"], freq="Q"): r for r in rows}
    assert comparative_ttm(p, pd.Period("2026Q2"), "capex", CUTOFF)[0] is None
    assert comparative_ttm(p, pd.Period("2026Q2"), "basic_eps", CUTOFF)[0] is None

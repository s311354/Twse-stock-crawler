"""SYNTHETIC ONLY: source/cutoff regressions, never official repaired evidence."""
from copy import deepcopy

import pandas as pd
import pytest

from domain.les_v7.contracts import identity
from domain.les_v7.quality import evaluate_financials
from domain.les_v7.quality_r2 import FinancialInputs, derive_verified_ttm, FLOW_FIELDS, PROFILE
from domain.strategies.low_entry_v6_quality import select_quality_inputs
from tests.strategies.test_low_entry_score_v7 import filings, POLICY, CUTOFF


def _records():
    rows = filings()
    for index, row in enumerate(rows):
        row.pop("revision_basis_id")
        row["source_url"] = "https://mops.twse.com.tw/SYNTHETIC_ONLY"
        for field, proof in row["field_evidence"].items():
            proof["source_url"] = row["source_url"]
            proof["contexts"][0]["concept"] = "ifrs-full:SYNTHETIC_" + field
    return rows


def _selected(rows, cutoff=CUTOFF):
    return select_quality_inputs(rows, symbol=rows[0]["symbol"], market="TWSE", cutoff=pd.Timestamp(cutoff))[0]


def _complete_records(opinion="UnqualifiedOpinionAbstract"):
    from domain.fundamental_derivation import FiledFact, FiledFinancials
    from domain.les_v7.financial_sources import audit_proof, FINANCIAL_MAPPING, PAYMENT_POLICY, PAYMENT_CONCEPTS
    from domain.financial_taxonomy import TAXONOMY_MAPPING_VERSION, _TIFRS_FIELDS
    rows = _records()
    for row in rows:
        period = pd.Period(row["period_end"], freq="Q")
        row.update(financial_mapping_version=FINANCIAL_MAPPING, payment_policy=PAYMENT_POLICY,
            receivable_basis="REPORTED_ORDINARY_NET_TRADE_EXCLUDING_RELATED_PARTIES")
        for proof in row["field_evidence"].values():
            proof.update(revision_number=1, period_end=row["period_end"],
                taxonomy_schema="tifrs-ci-cr-2026-03-31.xsd", taxonomy_mapping_version=TAXONOMY_MAPPING_VERSION)
        row["field_evidence"]["cash"].update(reason="REPORTED_EXACT_CONTEXT", reported_instant=row["period_end"])
        row["field_evidence"]["cash"]["contexts"][0]["concept"] = "ifrs-full:CashAndCashEquivalents"
        for field, concepts in PAYMENT_CONCEPTS.items():
            row["field_evidence"][field]["contexts"][0]["concept"] = concepts[0]
        base = row["field_evidence"]["cash"]
        for field, value in (("borrowings", 5000.), ("current_lease_liabilities", 0.), ("noncurrent_lease_liabilities", 0.)):
            row[field] = value
            proof = deepcopy(base)
            proof["contexts"][0].update(numeric_value=value * 1000, concept="ifrs-full:SYNTHETIC_" + field)
            row["field_evidence"][field] = proof
        row["field_evidence"]["interest_bearing_debt"].update(reason="VERIFIED_NONOVERLAPPING_DEBT_COMPONENTS",
            used_fields=["borrowings", "current_lease_liabilities", "noncurrent_lease_liabilities"], instant=row["period_end"])
        for field, value in (("investing_cash_flow", -5000.), ("financing_cash_flow", 0.), ("cash_fx_effect", 0.), ("cash_change", 0.)):
            row[field] = value
            proof = deepcopy(row["field_evidence"]["cfo"])
            proof.update(reason="REPORTED_EXACT_CONTEXT")
            proof["contexts"][0].update(concept=_TIFRS_FIELDS[field][0], numeric_value=value * 1000)
            row["field_evidence"][field] = proof
        for field, instant in (("cash_beginning", row["period_end"]), ("cash_ending", row["period_end"])):
            row[field] = row["cash"]
            proof = deepcopy(base)
            proof.update(reason="REPORTED_EXACT_CONTEXT", basis="CASH_FLOW_ANCHOR", start_date=None,
                end_date=instant, reported_instant=instant, cash_flow_period_start=period.start_time.date().isoformat(),
                taxonomy_schema="tifrs-ci-cr-2026-03-31.xsd", taxonomy_mapping_version=TAXONOMY_MAPPING_VERSION)
            proof["contexts"][0].update(concept=_TIFRS_FIELDS[field][0], instant=instant)
            row["field_evidence"][field] = proof
        filing = FiledFinancials(period.year, period.quarter, row["available_at"], 1,
            (FiledFact("tifrs-ar:" + opinion, None, "Y", None, row["period_end"], None, context_id="SYNTHETIC_ONLY_CURRENT_AUDIT"),),
            content_sha256=row["content_sha256"], source_url=row["source_url"], statement_scope=row["scope"], period_end=row["period_end"])
        row["audit_evidence"] = audit_proof(filing, symbol=row["symbol"], original_verified=True)
    return rows


def _evaluate_r2(rows, cutoff=CUTOFF):
    from domain.les_v7.quality_policy import r2_quality_policy as default_quality_policy
    return evaluate_financials(rows, symbol=rows[0]["symbol"], market="TWSE", cutoff=cutoff,
        policy=default_quality_policy())


def test_missing_unrelated_note_is_not_a_known_impact_on_reported_quarters():
    rows = _complete_records()
    before = _evaluate_r2(rows)
    for row in rows:
        row["accounting_basis_evidence"] = dict(status="UNKNOWN",
            reason="CURRENT_ACCOUNTING_POLICY_ATTESTATION_MISSING")
    result = _evaluate_r2(rows)
    assert result["core_ttm"] == before["core_ttm"]
    assert result["quality_score"] == before["quality_score"] and result["status"] == before["status"]


def test_missing_ttm_does_not_receive_a_pass_comparison_validation_credit():
    rows = _records()
    rows[-1]["cfo"] = None
    inputs = FinancialInputs(_selected(rows), CUTOFF)
    assert inputs.comparison_allowed("cfo", pd.Period("2026Q2")) is False
    assert inputs.comparison_checks["2026Q2:cfo"]["state"] == "UNKNOWN"


def _annual(rows, index):
    row = rows[index]
    old = deepcopy(row)
    period = pd.Period(row["period_end"], freq="Q")
    assert period.quarter == 4
    row.update(flow_basis="YTD", flow_derivation_policy="SAME_DOCUMENT_ONLY")
    row["quarter_fields"], row["quarter_field_evidence"] = {}, {}
    for field in FLOW_FIELDS:
        row[field] = sum(r[field] for r in rows[index - 3:index + 1])
        proof = row["field_evidence"][field]
        proof.update(basis="YTD", start_date=f"{period.year}-01-01")
        proof["contexts"][0].update(start_date=f"{period.year}-01-01", numeric_value=row[field] * 1000)
        row["quarter_fields"][field] = old[field]
        row["quarter_field_evidence"][field] = old["field_evidence"][field]
    return row


def test_r2_dispatch_is_explicit_and_legacy_policy_keeps_exact_result():
    rows = filings()
    baseline = evaluate_financials(rows, symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=CUTOFF, policy=POLICY)
    assert evaluate_financials(rows, symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=CUTOFF,
                              policy=dict(POLICY, profile_id="OTHER")) == baseline


def test_four_verified_quarters_do_not_require_global_revision_basis_id():
    rows = _records()
    before = deepcopy(rows)
    period = pd.Period("2026Q2")
    value, trace = derive_verified_ttm(_selected(rows), period, "cfo", CUTOFF)
    assert value == 20000.
    assert trace["method"] == "FOUR_VERIFIED_QUARTERS"
    assert trace["status"] == "VERIFIED_DERIVED"
    assert pd.Timestamp(trace["available_at"]) <= pd.Timestamp(CUTOFF)
    assert rows == before


def test_unrelated_field_proof_conflict_does_not_disable_cfo_ttm():
    rows = _records()
    baseline = derive_verified_ttm(_selected(rows), pd.Period("2026Q2"), "cfo", CUTOFF)
    rows[-1]["field_evidence"]["cost_of_revenue"]["contexts"][0]["numeric_value"] += 1000
    assert derive_verified_ttm(_selected(rows), pd.Period("2026Q2"), "cfo", CUTOFF) == baseline
    cost, trace = derive_verified_ttm(_selected(rows), pd.Period("2026Q2"), "cost_of_revenue", CUTOFF)
    assert cost is None and trace["status"] == "UNKNOWN"


@pytest.mark.parametrize("damage", ["missing", "future", "scope", "unit", "proof", "correction"])
def test_required_flow_gap_is_not_replaced_by_revision_or_zero(damage):
    rows = _records()
    if damage == "missing": rows.pop(-2)
    elif damage == "future": rows[-2]["available_at"] = "2026-10-13T00:00:00Z"
    elif damage == "scope": rows[-2]["scope"] = "individual"
    elif damage == "unit": rows[-2]["field_evidence"]["cfo"]["unit"] = "yuan"
    elif damage == "proof": rows[-2]["field_evidence"]["cfo"]["revision_id"] = "other"
    else: rows[-2]["correction_description"] = "SYNTHETIC known restatement; not reviewed"
    value, trace = derive_verified_ttm(_selected(rows), pd.Period("2026Q2"), "cfo", CUTOFF)
    assert value is None and trace["status"] == "UNKNOWN"


def test_future_revision_does_not_change_earlier_ttm_or_identity():
    rows = _records()
    baseline = derive_verified_ttm(_selected(rows), pd.Period("2026Q2"), "cfo", CUTOFF)
    future = deepcopy(rows[-1])
    future.update(available_at="2026-10-13T00:00:00Z", revision_id="future", cfo=-1e12)
    assert derive_verified_ttm(_selected(rows + [future]), pd.Period("2026Q2"), "cfo", CUTOFF) == baseline


def test_verified_full_year_and_quarter_paths_agree_before_selecting_precedence():
    rows = _records()
    row = _annual(rows, 5)  # 2025Q4; no fabricated EPS or balances
    value, trace = derive_verified_ttm(_selected(rows), pd.Period("2025Q4"), "cfo", CUTOFF)
    assert value == row["cfo"] == 20000.
    assert trace["method"] == "REPORTED_FULL_YEAR"
    assert {p["method"] for p in trace["paths"]} == {"REPORTED_FULL_YEAR", "FOUR_VERIFIED_QUARTERS"}
    row["cfo"] += 1
    row["field_evidence"]["cfo"]["contexts"][0]["numeric_value"] += 1000
    value, trace = derive_verified_ttm(_selected(rows), pd.Period("2025Q4"), "cfo", CUTOFF)
    assert value is None and trace["status"] == "CONFLICT"


@pytest.mark.parametrize("field", ["interest_expense", "parent_net_income", "intangible_capex", "dividends", "share_repurchases"])
def test_full_year_path_supports_exact_additional_monetary_flows_not_only_legacy_core(field):
    rows = _records()
    row = _annual(rows, 5)
    value, trace = derive_verified_ttm(_selected(rows), pd.Period("2025Q4"), field, CUTOFF)
    assert value == row[field]
    assert trace["method"] == "REPORTED_FULL_YEAR" and trace["status"] == "VERIFIED_REPORTED"


def test_verified_bridge_reuses_existing_source_bound_comparative_core():
    from tests.strategies.test_low_entry_v6_comparative_ttm import originals
    rows, _ = originals()
    value, trace = derive_verified_ttm(_selected(rows, "2026-10-02T13:30+08:00"), pd.Period("2026Q2"), "cfo", "2026-10-02T13:30+08:00")
    assert value == rows[-3]["cfo"] + rows[-1]["cfo"] - rows[-1]["comparative_ytd_fields"]["cfo"]
    assert trace["method"] == "COMPARATIVE_ORIGINAL_TTM"
    assert trace["quarter_values_filled"] is False
    rows[-1]["comparative_ytd_field_evidence"]["cfo"]["available_at"] = "2026-10-03T00:00:00Z"
    assert derive_verified_ttm(_selected(rows, "2026-10-02T13:30+08:00"), pd.Period("2026Q2"), "cfo", "2026-10-02T13:30+08:00")[0] is None


def test_financial_inputs_memo_is_bound_to_one_cutoff_not_mutable_outputs():
    inputs = FinancialInputs(_selected(_records()), CUTOFF)
    period = pd.Period("2026Q2")
    assert inputs.ttm("cfo", period) == inputs.ttm("cfo", period) == 20000.
    assert len(inputs.ttm_traces) == 1
    assert identity(inputs.ttm_traces) == identity(deepcopy(inputs.ttm_traces))


@pytest.mark.parametrize("field", ["basic_eps", "total_assets", "cash", "total_equity", "invented"])
def test_public_ttm_never_sums_eps_instant_or_unknown_fields(field):
    with pytest.raises(ValueError, match="TTM_ADDITIVE_FLOW_FIELD_REQUIRED"):
        derive_verified_ttm(_selected(_records()), pd.Period("2026Q2"), field, CUTOFF)


def test_r2_quality_has_native_safety_and_real_measured_confidence_without_model_bundle():
    rows = _complete_records()
    before = deepcopy(rows)
    result = _evaluate_r2(rows)
    assert result["profile_id"] == PROFILE and result["status"] == "PASS"
    assert result["financial_safety"]["state"] == "PASS"
    assert result["core_ttm_pass"] is True and result["quality_score"] >= 70
    assert result["quality_policy_id"] == identity(result["resolved_policy"])
    assert result["verification"]["checked_fields"] > 0
    assert result["verification"]["verified_checks"] == result["verification"]["checked_fields"]
    assert result["data_confidence"] == 100
    assert rows == before


def test_old_pass_flags_or_market_safety_receipt_cannot_replace_raw_audit():
    rows = _complete_records()
    rows[-1].pop("audit_evidence")
    rows[-1].update(auditor_opinion="UNMODIFIED", going_concern=False, material_accounting_issue=False)
    result = _evaluate_r2(rows)
    assert result["status"] == "DATA_INSUFFICIENT" and result["financial_safety"]["state"] == "UNKNOWN"
    assert "AUDIT_ORIGINAL_EVIDENCE_MISSING" in result["reason_codes"]


def test_known_audit_hard_risk_is_reject_not_missing_data():
    result = _evaluate_r2(_complete_records("AdverseOpinionAbstract"))
    assert result["status"] == "REJECT" and result["financial_safety"]["state"] == "FAIL"
    assert "CONFIRMED_FINANCIAL_AUDIT_HARD_RISK" in result["reason_codes"]


def test_r2_stale_original_and_lower_verification_do_not_keep_fixed_dc_points():
    result = _evaluate_r2(_complete_records(), "2027-04-03T13:30:00+08:00")
    assert result["status"] == "DATA_INSUFFICIENT" and "STALE_FINANCIAL_REPORT" in result["reason_codes"]
    assert result["verification"]["freshness"] == 0 and result["quality_score"] is None
    rows = _complete_records()
    rows[-1]["field_evidence"]["intangible_capex"]["revision_id"] = "NOT_THE_ORIGINAL"
    changed = _evaluate_r2(rows)
    assert changed["verification"]["validation"] < 100
    assert changed["verification"]["source_reliability"] < 100
    assert changed["data_confidence"] < 100


def test_nonpositive_income_is_known_not_meaningful_not_missing_cfo_or_income():
    rows = _complete_records()
    for row in rows[-4:]:
        row["net_income"] = -100.
        row["field_evidence"]["net_income"]["contexts"][0]["numeric_value"] = -100000.
    result = _evaluate_r2(rows)
    item = next(item for item in result["metric_ledger"] if item["metric_id"] == "C2")
    assert result["core_ttm"]["net_income"] == -400.
    assert item["value_status"] == "NOT_MEANINGFUL" and item["mapped_score"] == 0.
    assert item["root_gap"] is None and item["earned_points"] == 0.
    assert result["core_ttm_pass"] is True


def test_missing_current_necessary_ttm_cannot_become_bounded_gap():
    rows = _complete_records()
    rows[-2]["field_evidence"]["cfo"]["revision_id"] = "UNVERIFIED_REVISION"
    result = _evaluate_r2(rows)
    assert result["status"] == "DATA_INSUFFICIENT" and result["quality_score"] is None
    assert result["core_ttm_pass"] is False and result["core_ttm"]["cfo"] is None
    assert "REQUIRED_TTM_UNAVAILABLE:cfo" in result["reason_codes"]


def test_negative_cfo_is_known_ratio_failure_even_when_one_capex_source_is_missing():
    rows = _complete_records()
    for row in rows[-4:]:
        row.update(cfo=-5000., investing_cash_flow=5000.)
        row["field_evidence"]["cfo"]["contexts"][0]["numeric_value"] = -5000000.
        row["field_evidence"]["investing_cash_flow"]["contexts"][0]["numeric_value"] = 5000000.
    rows[-2]["field_evidence"]["capex"]["revision_id"] = "MISSING_ONE_VERIFIED_QUARTER"
    result = _evaluate_r2(rows)
    item = next(item for item in result["metric_ledger"] if item["metric_id"] == "C3")
    assert result["core_ttm"]["cfo"] == -20000.
    assert item["value_status"] == "NOT_MEANINGFUL" and item["mapped_score"] == 0.
    assert item["root_gap"] is None and result["core_ttm_pass"] is True
    assert result["financial_safety"]["state"] == "PASS"
    assert result["capex_ttm"] is None


def test_zero_revenue_denominator_is_known_and_does_not_become_config_incomplete():
    rows = _complete_records()
    for row in rows[-4:]:
        row["revenue"] = 0.
        row["field_evidence"]["revenue"]["contexts"][0]["numeric_value"] = 0.
    result = _evaluate_r2(rows)
    assert result["core_ttm_pass"] is True and result["status"] != "CONFIG_INCOMPLETE"
    for key in ("R3", "R4", "G2"):
        item = next(item for item in result["metric_ledger"] if item["metric_id"] == key)
        assert item["mapped_score"] == 0. and item["value_status"] == "NOT_MEANINGFUL"


def test_verified_signed_payment_magnitude_keeps_original_ttm_and_provenance():
    rows = _complete_records()
    for row in rows[-4:]:
        for field in ("capex", "intangible_capex"):
            row[field] = -abs(row[field])
            row["field_evidence"][field]["contexts"][0]["numeric_value"] = row[field] * 1000.
    before = deepcopy(rows)
    result = _evaluate_r2(rows)
    assert result["status"] == "PASS"
    assert result["ttm_traces"]["2026Q2:capex"]["value"] == -4000.
    assert result["payment_normalization"]["capex"]["raw_value"] == -4000.
    assert result["payment_normalization"]["capex"]["value"] == 4000.
    assert result["capex_ttm"] == 4400. and result["fcf_ttm"] == 15600.
    assert rows == before


@pytest.mark.parametrize("damage", ["mapping", "concept", "sign"])
def test_payment_sign_or_source_mapping_cannot_be_blindly_repaired_with_abs(damage):
    rows = _complete_records()
    for row in rows[-4:]:
        row["capex"] = -1000.
        row["field_evidence"]["capex"]["contexts"][0]["numeric_value"] = -1000000.
    if damage == "mapping": rows[-2].pop("financial_mapping_version")
    elif damage == "concept": rows[-2]["field_evidence"]["capex"]["contexts"][0]["concept"] = "ifrs-full:PurchaseOfTreasuryShares"
    else:
        rows[-2]["capex"] = 1000.
        rows[-2]["field_evidence"]["capex"]["contexts"][0]["numeric_value"] = 1000000.
    result = _evaluate_r2(rows)
    assert result["status"] == "DATA_INSUFFICIENT" and result["quality_score"] is None
    assert result["capex_ttm"] is None
    assert any(reason.startswith("SOURCE_MAPPING_ERROR:") for reason in result["reason_codes"])
    assert result["verification"]["validation"] < 100


@pytest.mark.parametrize("annual", [False, True])
def test_hidden_future_flow_origin_fails_required_checks_not_only_dc(annual):
    rows = _records()
    if annual: _annual(rows, 5)
    periods = _selected(rows)
    period = pd.Period("2025Q4" if annual else "2026Q2")
    row = periods[period]
    source = deepcopy(row)
    source["available_at"] = "2026-10-13T00:00:00Z"
    for proof in source["field_evidence"].values():
        proof["available_at"] = source["available_at"]
    row["quarter_flow_source"] = {key: source[key] for key in
        ("symbol", "market", "source", "scope", "currency", "unit", "available_at", "content_sha256", "revision_id")}
    row["quarter_eps_evidence"] = source
    for field in ("cfo", "cash"):
        row["field_evidence"][field] = source["field_evidence"][field]
    inputs = FinancialInputs(periods, CUTOFF)
    raw_check = inputs.check("cfo", period)
    assert raw_check["value"] is not None and raw_check["checks"]["availability"] is False
    assert inputs.ttm("cfo", period) is None
    assert inputs.balance("cash", period) is None


def test_annual_original_remains_ttm_when_only_comparison_impact_is_unverified():
    rows = _records()
    row = _annual(rows, 5)
    row["correction_description"] = "SYNTHETIC change with unresolved field impact"
    inputs = FinancialInputs(_selected(rows), CUTOFF)
    assert inputs.ttm("cfo", pd.Period("2025Q4")) == row["cfo"]
    assert inputs.comparison_allowed("cfo", pd.Period("2025Q4")) is False
    assert inputs.comparison_checks["2025Q4:cfo"]["state"] == "UNKNOWN"


def test_comparative_conflict_is_scoped_to_actual_field_not_other_financials():
    rows = _records()
    current = _annual(rows, 5)
    periods = _selected(rows)
    prior = periods[pd.Period("2024Q4")]
    # Make an explicit previous annual source for this focused comparison.
    prior["flow_basis"] = "YTD"
    for field in ("cfo", "revenue"):
        proof = prior["field_evidence"][field]
        proof["basis"], proof["start_date"] = "YTD", "2024-01-01"
        proof["contexts"][0]["start_date"] = "2024-01-01"
        comparative = deepcopy(proof)
        for key in ("content_sha256", "revision_id", "available_at", "source_url"):
            comparative[key] = current[key]
        if field == "revenue": comparative["contexts"][0]["numeric_value"] += 1000.
        periods[pd.Period("2025Q4")].setdefault("comparative_ytd_field_evidence", {})[field] = comparative
    inputs = FinancialInputs(periods, CUTOFF)
    assert inputs.comparison_allowed("revenue", pd.Period("2025Q4")) is False
    assert inputs.comparison_checks["2025Q4:revenue"]["state"] == "CONFLICT"
    assert inputs.comparison_allowed("cfo", pd.Period("2025Q4")) is True

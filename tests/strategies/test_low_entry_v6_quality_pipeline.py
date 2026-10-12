"""SYNTHETIC unit evidence; not real filings or investment performance."""
from copy import deepcopy
from dataclasses import replace

import pytest
import pandas as pd

from domain.strategies.low_entry_v6_quality import (
    QUALITY_BUDGETS, QUALITY_DC_WEIGHTS, evaluate_quality, quality_evidence_complete,
)
from tests.strategies.quality_evidence_fixtures import bind_synthetic_contexts
from tests.strategies.test_low_entry_score_v6 import financials
from domain.fundamental_derivation import FiledFact
from domain.strategies.low_entry_v6_filings import ixbrl_record
from domain.strategies.low_entry_v6_quality_evidence import validate_field
from tests.fundamentals.test_ixbrl_units import filed


def evaluate(rows):
    return evaluate_quality(rows, symbol="2330", market="TWSE", as_of="2026-09-11T13:30:00+08:00")


def originals():
    return [bind_synthetic_contexts(row) for row in financials()]


def test_field_validation_and_actual_dc_not_fixed_official_source_bonus():
    rows = originals()
    before = deepcopy(rows)
    original, legacy = evaluate(rows), evaluate(financials())
    assert rows == before
    assert original.score == legacy.score
    assert original.metrics["quality_verification"]["score"] == 100
    assert legacy.metrics["quality_verification"]["score"] < 100
    assert original.confidence > legacy.confidence
    assert original.confidence == pytest.approx(sum(QUALITY_DC_WEIGHTS[k] * v
        for k, v in original.metrics["quality_dc_inputs"].items()))
    assert quality_evidence_complete(original)
    for node in original.metrics["quality_field_checks"].values():
        assert node["status"] == "VERIFIED_REPORTED"
        assert all(node["checks"].values())
        assert node["content_sha256"] and node["available_at"]


@pytest.mark.parametrize("damage", ["amount", "unit", "context_period", "proof_version", "no_context", "conflicting_alias"])
def test_official_source_label_does_not_hide_invalid_field_context(damage):
    rows = originals()
    row = rows[-1]
    proof = row["field_evidence"]["cfo"]
    if damage == "amount":
        row["cfo"] += 1
    elif damage == "unit":
        proof["contexts"][0]["unit"] = "iso4217:USD"
    elif damage == "context_period":
        proof["contexts"][0]["end_date"] = "2026-03-31"
    elif damage == "proof_version":
        proof["revision_id"] = "unverified-revision"
    elif damage == "no_context":
        proof.pop("contexts")
    else:
        proof["contexts"].append(dict(proof["contexts"][0], numeric_value=123.))
    before = deepcopy(rows)
    q = evaluate(rows)
    assert rows == before
    assert q.status == "DATA_INSUFFICIENT" and not quality_evidence_complete(q)
    assert q.metrics["cfo_ttm"] is None
    assert q.metrics["quality_field_checks"]["2026Q2:cfo"]["status"] == "PENDING_VERIFICATION"


def test_unrelated_historical_balance_does_not_poison_flow_or_prior_capital_inputs():
    rows = originals()
    rows[-3]["total_assets"] *= 2
    bind_synthetic_contexts(rows[-3])
    q = evaluate(rows)
    assert q.status == "PASS"
    assert q.metrics["net_income_ttm"] == evaluate(originals()).metrics["net_income_ttm"]
    # The actual prior capital endpoint is still checked, not bypassed.
    rows[-2]["total_assets"] *= 2
    bind_synthetic_contexts(rows[-2])
    assert evaluate(rows).reasons == ("HISTORICAL_BALANCE_SHEET_IDENTITY_FAIL",)


def test_reported_profitability_is_uniform_and_ttm_history_is_diagnostic():
    q = evaluate(financials())
    assert q.metrics["quality_profitability_basis"] == "REPORTED_SAME_PERIOD_MARGIN"
    assert q.metrics["quality_profitability_window"] == 2
    rows = financials()
    for name, field in (("gross_margin", "cost_of_revenue"), ("operating_margin", "operating_income"), ("net_margin", "net_income")):
        expected = []
        for offset in (3, 2, 1, 0):
            window = rows[len(rows)-offset-4:len(rows)-offset]
            r = sum(row["revenue"] for row in window)
            n = sum(row[field] for row in window)
            expected.append((r-n)/r if name == "gross_margin" else n/r)
        assert q.metrics[name + "_ttm_history"] == pytest.approx(expected)
        direct = [rows[-5], rows[-1]]
        assert q.metrics[name + "_reported_history"] == pytest.approx([
            (r["revenue"] - r[field]) / r["revenue"] if name == "gross_margin" else r[field] / r["revenue"]
            for r in direct])
    assert sum(node["weight"] for node in q.metrics["quality_metric_checks"].values()) == 100
    for group, budget in QUALITY_BUDGETS.items():
        assert sum(node["weight"] for node in q.metrics["quality_metric_checks"].values() if node["group"] == group) == budget


def test_quarterly_peer_rank_never_substitutes_for_ttm_peer_evidence():
    rows = financials()
    expected = evaluate(rows)
    rows[-1].update(gross_margin_subsector_percentile=1., operating_margin_subsector_percentile=1., net_margin_subsector_percentile=1.)
    assert evaluate(rows) == expected


def test_nonpositive_net_income_not_meaningful_conversion_is_not_missing_cfo():
    rows = financials()
    for row in rows[-4:]:
        row["net_income"] = -1.
    q = evaluate(rows)
    assert q.metrics["cash_conversion"] is None
    assert q.metrics["cash_conversion_state"] == "NOT_MEANINGFUL"
    check = q.metrics["quality_metric_checks"]["cash_conversion"]
    assert check["status"] == "NOT_MEANINGFUL" and check["points"] == 0 and check["available"] is True
    assert q.coverage["cash_flow"] == 100
    assert "net_income_ttm" not in q.metrics["quality_required_missing"]
    rows[-1]["cfo"] = None
    missing = evaluate(rows)
    assert missing.metrics["cash_conversion_state"] == "MISSING_OR_INCOMPARABLE"
    assert missing.status == "DATA_INSUFFICIENT"
    assert missing.coverage["cash_flow"] < 100


@pytest.mark.parametrize("damage", ["confidence", "verification", "checks", "weights"])
def test_shared_quality_gate_rechecks_measured_dc_ledger(damage):
    q = evaluate(originals())
    assert quality_evidence_complete(q)
    metrics = deepcopy(q.metrics)
    if damage == "confidence":
        bad = replace(q, confidence=q.confidence + 1)
    else:
        if damage == "verification":
            metrics["quality_dc_inputs"]["verification"] -= 1
        elif damage == "checks":
            metrics["quality_field_checks"]["2026Q2:cfo"]["checks"]["original_context"] = False
        else:
            metrics["quality_dc_weights"]["verification"] = 0
        bad = replace(q, metrics=metrics)
    assert not quality_evidence_complete(bad)


def test_future_version_cannot_change_earlier_validation_or_dc():
    rows = originals()
    original = evaluate(rows)
    later = deepcopy(rows[-1])
    later.update(available_at="2026-09-12T00:00Z", revision_id="future", cfo=10**12)
    bind_synthetic_contexts(later)
    assert evaluate(rows + [later]) == original


def test_debt_validation_depends_on_used_aggregates_not_unreported_aliases():
    facts = tuple(FiledFact(concept, value * 1000, None, None, None, "2026-06-30", unit="iso4217:TWD")
        for concept, value in (("ifrs-full:Borrowings", 700), ("ifrs-full:CurrentLeaseLiabilities", 0),
                              ("ifrs-full:NoncurrentLeaseLiabilities", 20)))
    row = ixbrl_record("2330", replace(filed([]), facts=facts, content_sha256="SYNTHETIC-DEBT-ORIGINAL"))
    assert row["interest_bearing_debt"] == 720
    node = validate_field(row, "interest_bearing_debt", period=pd.Period("2026Q2"),
                          cutoff=pd.Timestamp("2026-09-11T13:30+08:00"), instant=True)
    assert node["status"] == "VERIFIED_DERIVED" and node["checks"]["original_context"] is True
    assert len(node["evidence"]["inputs"]) == 3
    assert row["field_evidence"]["current_borrowings"]["status"] == "UNKNOWN"
    row["field_evidence"]["noncurrent_lease_liabilities"]["revision_id"] = "wrong"
    assert validate_field(row, "interest_bearing_debt", period=pd.Period("2026Q2"),
        cutoff=pd.Timestamp("2026-09-11T13:30+08:00"), instant=True)["value"] is None

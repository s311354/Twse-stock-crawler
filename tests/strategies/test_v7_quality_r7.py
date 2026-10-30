"""Synthetic only: additive source interpretation, never missing-data relief."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pandas as pd
import pytest

from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.les_v7.contracts import identity, timestamp, quality_extension_schema, envelope_schema_version
from domain.les_v7.financial_audit_r7 import audit_source_r7, AVAILABLE_AT
from domain.les_v7.financial_sources_r5 import audit_proof_r5
from domain.les_v7.quality import evaluate_financials
from domain.les_v7.quality_policy import (r6_quality_policy, r7_quality_policy,
    native_financial_safety, validate_quality_policy)
from domain.les_v7.quality_reported import validate_reported_result
from tests.fundamentals.test_v7_audit_cash_r5 import original, QUALIFIED_BODY, repair
from tests.strategies.test_v7_reported_quality import source_records

AT = "2026-10-28T13:00:00Z"


def complete():
    rows = source_records()
    for row in rows:
        period = pd.Period(row["period_end"], freq="Q")
        proof = row["audit_evidence"]
        facts = tuple(FiledFact(**dict(f, dimensions=tuple(map(tuple, f.get("dimensions", ())))))
                      for f in proof["contexts"])
        filing = FiledFinancials(period.year, period.quarter, row["available_at"], row["revision_number"], facts,
            content_sha256=row["content_sha256"], source_url=row["source_url"],
            statement_scope=row["scope"], period_end=row["period_end"], audit_status=proof.get("audit_status"))
        row["audit_evidence_r5"] = audit_proof_r5(filing, symbol="2330", original_verified=True)
        row["audit_evidence_r7"] = audit_source_r7(filing, symbol="2330", original_verified=True)
    return rows


def test_new_profile_keeps_native_numbers_and_old_result_identity():
    rows = complete(); frozen = deepcopy(rows)
    old = evaluate_financials(rows, symbol="2330", market="TWSE", cutoff=AT, policy=r6_quality_policy())
    new = evaluate_financials(rows, symbol="2330", market="TWSE", cutoff=AT, policy=r7_quality_policy())
    assert old["status"] == new["status"] == "PASS"
    for key in ("quality_score", "earned_points", "missing_penalty_points", "data_confidence",
                "coverage_total", "coverage_by_domain", "metric_ledger", "core_reported", "field_checks"):
        assert old[key] == new[key]
    assert new["financial_safety"]["gate_version"] == "NATIVE_FINANCIAL_SAFETY_R7"
    assert new["profile_id"] == "QG_V7_REPORTED_PERIOD_R7"
    assert quality_extension_schema(new["profile_id"]) == "v7-quality-evidence/v7"
    assert envelope_schema_version(new["profile_id"]) == "LES_V7_ENVELOPE_7"
    validate_reported_result(new)
    assert rows == frozen
    assert evaluate_financials(rows, symbol="2330", market="TWSE", cutoff=AT, policy=r6_quality_policy()) == old


def test_verified_parser_repairs_only_the_source_interpretation_not_raw_values_or_time():
    row, filing = original(QUALIFIED_BODY + "民國115年6月30日存貨成本如下。")
    old_row = repair(row, filing)
    before = deepcopy(old_row)
    old = native_financial_safety(old_row, symbol=row["symbol"], market="TWSE", cutoff=AT, policy=r6_quality_policy())
    fixed = dict(old_row, audit_evidence_r7=audit_source_r7(filing, symbol=row["symbol"], original_verified=True))
    after = native_financial_safety(fixed, symbol=row["symbol"], market="TWSE", cutoff=AT, policy=r7_quality_policy())
    assert old["state"] == "UNKNOWN" and after["state"] == "PASS"
    assert after["audit_classification"]["audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert fixed["available_at"] == fixed["audit_evidence_r7"]["available_at"] == filing.available_at
    assert old_row == before


@pytest.mark.parametrize("damage", ["missing", "access_failure", "individual_scope", "future", "unit", "tamper"])
def test_unknown_sources_still_block_financial_safety(damage):
    text = (QUALIFIED_BODY + "本會計師無法取得充分證據。" if damage == "access_failure" else
            "公司民國115年6月30日之資產負債表，業經核閱。" if damage == "individual_scope" else QUALIFIED_BODY)
    row, filing = original(text)
    if damage == "individual_scope": filing = replace(filing, statement_scope="individual")
    row = repair(row, filing) if damage != "individual_scope" else dict(row, scope="individual")
    row["audit_evidence_r7"] = audit_source_r7(filing, symbol=row["symbol"], original_verified=True)
    if damage == "missing": row.pop("audit_evidence_r7")
    elif damage == "future": row["audit_evidence_r7"]["available_at"] = "2030-01-01T00:00:00Z"
    elif damage == "unit": row["audit_evidence_r7"]["unit"] = "million"
    elif damage == "tamper": row["audit_evidence_r7"]["audit"]["audit_opinion"] = "UNMODIFIED"
    result = native_financial_safety(row, symbol=row["symbol"], market="TWSE", cutoff=AT, policy=r7_quality_policy())
    assert result["state"] == "UNKNOWN"


@pytest.mark.parametrize("opinion", ["AdverseOpinionAbstract", "DisclaimerOfOpinionAbstract"])
def test_confirmed_audit_risk_is_never_a_soft_score_penalty(opinion):
    row, filing = original(opinion=opinion)
    row = repair(row, filing)
    row["audit_evidence_r7"] = audit_source_r7(filing, symbol=row["symbol"], original_verified=True)
    result = native_financial_safety(row, symbol=row["symbol"], market="TWSE", cutoff=AT, policy=r7_quality_policy())
    assert result["state"] == "FAIL"
    assert "CONFIRMED_FINANCIAL_AUDIT_HARD_RISK" in result["reasons"]


def test_release_and_future_revision_are_separate_from_original_availability():
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        validate_quality_policy(r7_quality_policy(), timestamp(AVAILABLE_AT) - timedelta(microseconds=1))
    rows = complete()
    first = evaluate_financials(rows, symbol="2330", market="TWSE", cutoff=AT, policy=r7_quality_policy())
    future = deepcopy(rows[-1]); future.update(available_at="2030-01-01T00:00:00Z", revision_id="FUTURE_SYNTHETIC")
    assert evaluate_financials(rows + [future], symbol="2330", market="TWSE", cutoff=AT,
                              policy=r7_quality_policy()) == first

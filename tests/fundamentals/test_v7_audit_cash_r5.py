"""SYNTHETIC ONLY: source-bound R5 repair, never actual issuer certification."""
from copy import deepcopy
from dataclasses import replace

import pytest

from domain.financial_taxonomy import IAS7_OTHER_CASH_CONCEPT
from domain.fundamental_derivation import FiledFact
from domain.les_v7.contracts import identity, quality_extension_schema, envelope_schema_version
from domain.les_v7.financial_sources import audit_proof, ixbrl_record_r2
from domain.les_v7.financial_sources_r5 import audit_proof_r5, repair_source_record_r5
from domain.les_v7.quality_policy import (
    default_quality_policy, r2_quality_policy, r3_quality_policy, r5_quality_policy,
    validate_quality_policy, native_financial_safety, financial_safety_source,
)
from domain.les_v7.quality_reported import evaluate_financials_reported, validate_reported_result
from tests.strategies.test_v7_quality_policy import native_original

CUTOFF = "2026-10-14T07:00:00Z"
CURRENT_INTRO = "會計師核閱報告\n前言\n公司民國115年及114年6月30日之合併資產負債表，業經本會計師核閱。\n"
QUALIFIED_BODY = (CURRENT_INTRO + "範圍\n除保留結論之基礎段所述者外，本會計師執行核閱。\n"
    "保留結論之基礎\n部分非重要子公司之財務報表未經會計師核閱，民國115年及114年6月30日資產如下。\n"
    "保留結論\n除上述影響外，並未發現上開合併財務報表未允當表達。\n")


def original(body=QUALIFIED_BODY, *, opinion="QualifiedOpinionAbstract", english=None):
    _, source = native_original(opinion)
    date = "2026-08-10"
    facts = tuple(replace(f, end_date=None, instant=date)
                  if f.concept.startswith("tifrs-ar:") else f for f in source.facts)
    bodies = (body,) if english is None else (body, english)
    facts += tuple(FiledFact("tifrs-ar:AccountantsReportBody", None, text, None, None, date,
                            context_id="SYNTHETIC_ONLY_R5_AUDIT_" + str(i))
                   for i, text in enumerate(bodies))
    source = replace(source, facts=facts, content_sha256=identity([str(f) for f in facts]), audit_status="reviewed")
    return ixbrl_record_r2("SYNTHETIC_ONLY", source, original_verified=True), source


def repair(row, filing):
    return repair_source_record_r5(row, filing, symbol="SYNTHETIC_ONLY", original_verified=True)


def safety(row, policy=None):
    return native_financial_safety(row, symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=CUTOFF,
                                  policy=policy or r5_quality_policy())


def test_current_dual_year_intro_does_not_turn_current_qualified_basis_into_prior_opinion():
    row, filing = original()
    before = deepcopy(row)
    assert safety(row, default_quality_policy())["state"] == "UNKNOWN"
    fixed = repair(row, filing)
    result = safety(fixed)
    assert result["state"] == "PASS", result
    assert result["audit_classification"]["audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert result["audit_classification"]["audit_opinion"] != "UNMODIFIED"
    assert fixed["audit_evidence_r5"]["audit"]["audit_report_text_raw"] == QUALIFIED_BODY
    assert fixed["audit_evidence"] == before["audit_evidence"]
    assert row == before


@pytest.mark.parametrize("body,expected", [
    (CURRENT_INTRO + "保留結論之基礎\n部分非重要子公司未經核閱，且無法取得查核證據。\n保留結論\n保留結論。", "UNKNOWN"),
    (CURRENT_INTRO + "保留結論之基礎\n存貨有重大不實表達。\n保留結論\n保留結論。", "FAIL"),
    (CURRENT_INTRO + "保留結論之基礎\n公司財務報表未經核閱。\n保留結論\n保留結論。", "UNKNOWN"),
    (CURRENT_INTRO + "前期保留結論之基礎\n部分非重要子公司未經核閱。\n前期保留結論\n前期保留結論。", "UNKNOWN"),
    (CURRENT_INTRO + "前期\n保留結論之基礎\n部分非重要子公司未經核閱。\n保留結論\n前期保留結論。", "UNKNOWN"),
])
def test_qualified_component_exception_is_not_a_general_missing_audit_bypass(body, expected):
    row, filing = original(body)
    assert safety(repair(row, filing))["state"] == expected


@pytest.mark.parametrize("opinion", ["AdverseOpinionAbstract", "DisclaimerOfOpinionAbstract"])
def test_known_adverse_or_disclaimer_survives_even_a_nonmaterial_component_basis(opinion):
    row, filing = original(opinion=opinion)
    assert safety(repair(row, filing))["state"] == "FAIL"


def test_current_going_concern_heading_remains_hard_risk_after_comparative_values():
    row, filing = original(QUALIFIED_BODY + "繼續經營重大不確定性\n本公司有重大繼續經營風險。\n")
    result = safety(repair(row, filing))
    assert result["state"] == "FAIL"
    assert result["audit_classification"]["going_concern"] is True


ENGLISH = ("Introduction\nWe reviewed the consolidated balance sheets as at June 30, 2026 and 2025.\n"
           "Basis for Qualified Conclusion\nCertain insignificant consolidated subsidiaries were not reviewed.\n"
           "Qualified Conclusion\nExcept for those effects, our conclusion is qualified.\n")


def test_bilingual_as_at_binding_requires_the_same_financial_period_and_scope():
    row, filing = original(english=ENGLISH)
    assert row["audit_evidence"]["status"] == "PENDING_VERIFICATION"
    fixed = repair(row, filing)
    assert fixed["audit_evidence_r5"]["status"] == "VERIFIED_REPORTED"
    assert len(fixed["audit_evidence_r5"]["period_binding"]["body_bindings"]) == 2
    assert safety(fixed)["state"] == "PASS"


@pytest.mark.parametrize("damage", ["period", "scope", "classification"])
def test_conflicting_translation_is_not_silently_discarded(damage):
    english = (ENGLISH.replace("June 30", "March 31") if damage == "period" else
               ENGLISH.replace("consolidated balance", "separate balance") if damage == "scope" else
               ENGLISH.replace("Certain insignificant consolidated subsidiaries were not reviewed.",
                               "We could not obtain evidence for material inventories."))
    row, filing = original(english=english)
    result = safety(repair(row, filing))
    assert result["state"] == "UNKNOWN"


def test_unverified_original_or_missing_new_proof_cannot_certify_r5_audit():
    row, filing = original()
    assert audit_proof_r5(filing, symbol="SYNTHETIC_ONLY")["status"] == "PENDING_VERIFICATION"
    assert safety(row)["state"] == "UNKNOWN"  # Old proof is never silently upgraded.
    with pytest.raises(ValueError, match="SAME_ORIGINAL"):
        repair_source_record_r5(row, filing, symbol="SYNTHETIC_ONLY")
    fixed = repair(row, filing)
    fixed["audit_evidence_r5"]["contexts"][0]["instant"] = "2025-08-10"
    assert safety(fixed)["state"] == "UNKNOWN"


@pytest.mark.parametrize("key,value", [("content_sha256", "f" * 64), ("available_at", "2026-08-15T10:00:00Z"),
                                        ("scope", "individual"), ("unit", "million")])
def test_source_rebuild_requires_exact_original_identity_and_never_writes_the_old_row(key, value):
    row, filing = original()
    changed = dict(row, **{key: value})
    with pytest.raises(ValueError, match="SAME_ORIGINAL"):
        repair(changed, filing)


def cash_original(adjustment=4000.):
    row, filing = original(CURRENT_INTRO + "結論\n並未發現上開合併財務報表未允當表達。\n",
                           opinion="UnqualifiedOpinionAbstract")
    facts = tuple(replace(f, numeric_value=16000000.) if f.concept == "ifrs-full:CashAndCashEquivalents" else f
                  for f in filing.facts)
    if adjustment is not None:
        facts += (FiledFact(IAS7_OTHER_CASH_CONCEPT, adjustment * 1000., None, None, None,
                            "2026-06-30", unit="iso4217:TWD", context_id="SYNTHETIC_ONLY_R5_IAS7"),)
    filing = replace(filing, facts=facts, content_sha256=identity([str(f) for f in facts]))
    return ixbrl_record_r2("SYNTHETIC_ONLY", filing, original_verified=True), filing


def test_cash_rebuild_uses_actual_same_original_ias7_fact_not_an_arithmetic_plug():
    row, filing = cash_original()
    cached = deepcopy(row)
    cached["cash_ias7_adjustment"] = cached["cfo"] = None
    cached["field_evidence"].pop("cash_ias7_adjustment")
    before = deepcopy(cached)
    fixed = repair(cached, filing)
    assert fixed["cash_ias7_adjustment"] == 4000. and fixed["cfo"] == 5000.
    assert safety(fixed)["state"] == "PASS"
    assert cached == before
    assert fixed["source_repair_r5"]["old_cash_proof_id"] != fixed["source_repair_r5"]["new_cash_proof_id"]


@pytest.mark.parametrize("adjustment,expected", [(None, "FAIL"), (3000., "FAIL")])
def test_missing_or_conflicting_cash_bridge_stays_blocked(adjustment, expected):
    row, filing = cash_original(adjustment)
    fixed = repair(row, filing)
    assert fixed["cash_ias7_adjustment"] == adjustment
    assert safety(fixed)["state"] == expected


def test_unresolved_reported_cash_adjustment_or_absent_cfo_stays_unknown_not_zero():
    row, filing = cash_original()
    for damage in ("unit", "missing_cfo"):
        facts = tuple(replace(f, unit="iso4217:USD") if damage == "unit" and f.concept == IAS7_OTHER_CASH_CONCEPT else f
                      for f in filing.facts if damage != "missing_cfo" or f.concept != "ifrs-full:CashFlowsFromUsedInOperatingActivities")
        damaged = replace(filing, facts=facts, content_sha256=identity([str(f) for f in facts]))
        fresh = ixbrl_record_r2("SYNTHETIC_ONLY", damaged, original_verified=True)
        fixed = repair(fresh, damaged)
        assert safety(fixed)["state"] == "UNKNOWN"
        assert fixed["cash_ias7_adjustment" if damage == "unit" else "cfo"] is None


def test_frozen_old_policies_and_schema_are_unchanged_and_r5_is_cutoff_available():
    assert r2_quality_policy()["policy_sha256"] == "6526f90f7764fa92e2e6c2281c7438fb6b937c6f26dc870fcdf583817def906a"
    assert r3_quality_policy()["policy_sha256"] == "9083b33449f4860c4090c1525eb9a7735bdbbc3a1f7e9413f0f26e3371a51cfa"
    assert default_quality_policy()["policy_sha256"] == "a181db4b008f1478095e51f46711eb33af5bec78f7956d845e18346eff179de1"
    assert default_quality_policy()["profile_id"] == "QG_V7_REPORTED_PERIOD_R4"
    assert r3_quality_policy()["profile_id"] == "QG_V7_REPORTED_PERIOD_R3"
    assert quality_extension_schema("QG_V7_REPORTED_PERIOD_R4") == "v7-quality-evidence/v4"
    assert quality_extension_schema("QG_V7_REPORTED_PERIOD_R5") == "v7-quality-evidence/v5"
    assert envelope_schema_version("QG_V7_REPORTED_PERIOD_R5") == "LES_V7_ENVELOPE_5"
    with pytest.raises(ValueError, match="NOT_AVAILABLE"):
        validate_quality_policy(r5_quality_policy(), "2026-10-14T06:33:17Z")
    row, filing = original()
    fixed = repair(row, filing)
    early = native_financial_safety(fixed, symbol="SYNTHETIC_ONLY", market="TWSE",
                                    cutoff="2026-10-14T06:33:17Z", policy=r5_quality_policy())
    assert early["state"] == "UNKNOWN"
    assert early["reasons"] == ["QUALITY_POLICY_NOT_AVAILABLE_AT_CUTOFF"]


def test_r5_native_reported_projection_and_rebuild_validate_without_a_second_quality_engine():
    row, filing = original()
    fixed = repair(row, filing)
    result = evaluate_financials_reported([fixed], symbol="SYNTHETIC_ONLY", market="TWSE",
                                         cutoff=CUTOFF, policy=r5_quality_policy())
    assert result["profile_id"] == "QG_V7_REPORTED_PERIOD_R5"
    assert result["financial_safety"]["state"] == "PASS"
    assert result["resolved_policy"]["source_mapping_id"] == "MOPS_REPORTED_PERIOD_EVIDENCE_R5"
    validate_reported_result(result)
    packet = financial_safety_source(fixed)
    packet["audit_evidence_r5"]["contexts"][0]["text_value"] = "SYNTHETIC_TAMPER"
    assert fixed["audit_evidence_r5"]["contexts"][0]["text_value"] != "SYNTHETIC_TAMPER"


def test_r5_audit_compares_original_r4_against_source_repaired_r5_not_a_v6_or_ttm_policy():
    from application.quality_audit_v7 import QualityAuditStock, audit_quality_v7, quality_audit_summary
    row, filing = original()
    fixed = repair(row, filing)
    class Source:
        def load_stock(self, *args, **kwargs):
            return QualityAuditStock((fixed,), source_baseline_records=(row,))
    result = audit_quality_v7(Source(), ("SYNTHETIC_ONLY",), CUTOFF,
                              data_kind="SYNTHETIC_ONLY", policy_resolver=r5_quality_policy)
    assert result["schema"] == "local-v7-quality-audit/v4"
    assert result["baseline_profile"] == "QG_V7_REPORTED_PERIOD_R4"
    assert result["quality_profile"] == "QG_V7_REPORTED_PERIOD_R5"
    assert result["summary"]["before_financial_safety_counts"] == {"UNKNOWN": 1}
    assert result["summary"]["after_financial_safety_counts"] == {"PASS": 1}
    assert result["rows"][0]["before"]["status"] == "DATA_INSUFFICIENT"
    assert result["account"] is None
    assert result["rows"][0]["source_repairs"][0]["content_sha256"] == filing.content_sha256
    assert "R5:" in quality_audit_summary(result)


def test_r5_local_source_does_not_repair_missing_original_or_write_source_cache(tmp_path):
    from contextlib import closing
    import hashlib
    import sqlite3
    from interface.quality_audit_v7 import LocalQualityAuditSource
    from tests.fundamentals.test_ttm_audit import cache
    root = tmp_path / "mops"
    cache(root)
    db = root / "mops.sqlite3"
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        loaded = LocalQualityAuditSource(root, connection, source_repair_r5=True).load_stock("2330", CUTOFF, period_count=9)
    assert loaded.source_baseline_records is not None
    assert len(loaded.source_baseline_records) == len(loaded.records)
    assert before == hashlib.sha256(db.read_bytes()).hexdigest()
    assert all(record.get("audit_evidence_r5", {}).get("status") != "VERIFIED_REPORTED" for record in loaded.records)


def test_r5_cli_is_explicit_immutable_and_preserves_fail_closed_readiness(tmp_path):
    import hashlib
    import json
    from interface.quality_audit_v7 import main
    from tests.fundamentals.test_ttm_audit import cache
    root, output, stocklist = tmp_path / "mops", tmp_path / "r5.json", tmp_path / "symbols"
    cache(root)
    stocklist.write_text("2330;9999")
    before = hashlib.sha256((root / "mops.sqlite3").read_bytes()).hexdigest()
    args = ["--cache", str(root), "--cutoff", CUTOFF, "--stocklist", str(stocklist), "--profile", "r5",
            "--data-kind", "SYNTHETIC_ONLY", "--output", str(output)]
    assert main(args) == 0
    result = json.loads(output.read_text())
    assert result["schema"] == "local-v7-quality-audit/v4"
    assert result["quality_profile"] == "QG_V7_REPORTED_PERIOD_R5"
    assert all(row["after"]["status"] != "PASS" for row in result["rows"])
    assert before == hashlib.sha256((root / "mops.sqlite3").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="OUTPUT_EXISTS"):
        main(args)

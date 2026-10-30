"""Synthetic parser candidates; no issuer-specific exceptions or entry grants."""
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from domain.fundamental_derivation import FiledFact
from domain.les_v7.financial_audit_r7 import audit_proof_r7, statement_bindings
from domain.les_v7.financial_sources_r5 import audit_proof_r5
from tests.fundamentals.test_v7_audit_cash_r5 import original, ENGLISH, CURRENT_INTRO, QUALIFIED_BODY

CUTOFF = "2026-10-28T12:30:00Z"


def proof(filing, **kwargs):
    return audit_proof_r7(filing, symbol="SYNTHETIC_ONLY", cutoff=CUTOFF,
        original_verified=True, **kwargs)


@pytest.mark.parametrize("date_text", ["一一五及一一四年六月三十日", "115年及民國114年6月30日",
    "115年6月30日及民國114年6月30日", "2026年及2025年6月30日", "115年6月30日"])
def test_explicit_coordinated_period_grammar(date_text):
    text = "本公司民國" + date_text + "之合併資產負債表，業經核閱。"
    assertions, ignored = statement_bindings(text, pd.Period("2026Q2"))
    assert assertions and {a["scope"] for a in assertions} == {"consolidated"}
    assert not ignored


@pytest.mark.parametrize("body", [
    "We reviewed the consolidated balance sheets of Example Co., Ltd. as at June 30, 2026 and 2025.",
    "We reviewed the separate financial statements as of June 30, 2026.",
])
def test_english_identity_and_description_are_bound_in_same_sentence(body):
    assertions, _ = statement_bindings(body, pd.Period("2026Q2"))
    assert assertions
    assert assertions[0]["scope"] == ("individual" if "separate" in body else "consolidated")


@pytest.mark.parametrize("note", ["民國115年6月30日存貨成本及備抵存貨評價損失如下。",
    "民國115年6月30日採用權益法之投資金額如下。"])
def test_incidental_note_date_does_not_taint_an_explicit_report_description(note):
    _, filing = original(QUALIFIED_BODY + note)
    old = audit_proof_r5(filing, symbol="SYNTHETIC_ONLY", original_verified=True)
    assert old["status"] == "PENDING_VERIFICATION"
    new = proof(filing)
    assert new["status"] == "VERIFIED_REPORTED"
    assert new["audit"]["audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert new["period_binding"]["body_bindings"][0]["incidental_date_mentions"] > 0
    assert audit_proof_r5(filing, symbol="SYNTHETIC_ONLY", original_verified=True) == old


def test_same_endpoint_conflicting_statement_scope_is_never_treated_as_incidental():
    _, filing = original(QUALIFIED_BODY + "民國115年6月30日之個體財務報表業經核閱。")
    new = proof(filing)
    assert new["status"] == "PENDING_VERIFICATION"
    assert "CURRENT_PERIOD_STATEMENT_SCOPE_CONFLICT" in new["period_binding"]["reasons"]


@pytest.mark.parametrize("english", ["NA", "N/A", "  na  "])
def test_exact_placeholder_is_retained_as_missing_translation_not_a_favorable_opinion(english):
    _, filing = original(english=english)
    before = deepcopy(filing)
    result = proof(filing)
    assert result["status"] == "VERIFIED_REPORTED"
    assert result["audit"]["audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert result["period_binding"]["excluded_bodies"][0]["body_sha256"]
    assert filing == before


def test_placeholder_only_is_not_a_current_report_body():
    _, filing = original("NA", english="N/A")
    assert proof(filing)["status"] == "PENDING_VERIFICATION"


@pytest.mark.parametrize("damage", ["wrong_period", "wrong_scope", "unrecognized_body"])
def test_actual_translation_content_cannot_be_silently_discarded(damage):
    text = ENGLISH.replace("June 30", "March 31") if damage == "wrong_period" else (
        ENGLISH.replace("consolidated balance", "individual balance") if damage == "wrong_scope" else
        "No independently interpretable opinion is supplied.")
    _, filing = original(english=text)
    assert proof(filing)["status"] == "PENDING_VERIFICATION"


def test_explicit_individual_scope_is_not_inferred_from_filing_label_or_other_body_keywords():
    text = "本公司115年6月30日之資產負債表業經核閱。附註另說明合併公司。"
    _, filing = original(text)
    filing = replace(filing, statement_scope="individual")
    assert proof(filing)["status"] == "PENDING_VERIFICATION"


ACTIVE = ENGLISH.replace("Certain insignificant consolidated subsidiaries were not reviewed.",
    "We did not review the financial statements of certain insignificant consolidated subsidiaries.")


def test_current_active_voice_equivalence_repairs_a_proven_bilingual_lexer_gap_not_the_raw_text():
    _, filing = original(english=ACTIVE)
    old = audit_proof_r5(filing, symbol="SYNTHETIC_ONLY", original_verified=True)
    assert old["audit"]["audit_opinion"] == "UNKNOWN"
    new = proof(filing)
    assert new["audit"]["audit_opinion"] == "QUALIFIED_SCOPE_LIMITATION"
    assert new["audit"]["audit_review_required"] is False
    assert new["audit"]["body_classifications"][1]["active_voice_basis_equivalence"]
    assert "We did not review" in new["audit"]["audit_report_text_raw"]
    assert audit_proof_r5(filing, symbol="SYNTHETIC_ONLY", original_verified=True) == old
    assert not new["entry_authority"] and not new["quality_gate_recomputed"]


@pytest.mark.parametrize("suffix,opinion", [
    (" We were unable to access the financial information or management.", "UNKNOWN"),
    (" We could not obtain sufficient evidence.", "UNKNOWN"),
    (" We could not access the financial information or management.", "UNKNOWN"),
    (" We cannot obtain audit evidence.", "UNKNOWN"),
    (" We were denied access to management.", "UNKNOWN"),
    (" Inventory is materially misstated.", "QUALIFIED_MATERIAL_MISSTATEMENT"),
])
def test_active_voice_does_not_remove_real_evidence_or_material_misstatement_risks(suffix, opinion):
    text = ACTIVE.replace("\nQualified Conclusion", suffix + "\nQualified Conclusion")
    _, filing = original(english=text)
    new = proof(filing)
    assert new["audit"]["audit_opinion"] == opinion
    if opinion == "UNKNOWN":
        assert new["audit"]["audit_review_required"]


@pytest.mark.parametrize("opinion", ["AdverseOpinionAbstract", "DisclaimerOfOpinionAbstract"])
def test_adverse_or_disclaimer_survives_the_new_binding(opinion):
    _, filing = original(english=ACTIVE, opinion=opinion)
    assert proof(filing)["audit"]["audit_opinion"] in {"ADVERSE", "DISCLAIMER"}


def test_going_concern_hard_risk_and_prior_only_basis_are_not_cleared():
    _, filing = original(QUALIFIED_BODY + "\n繼續經營重大不確定性\n公司存續有重大疑慮。")
    assert proof(filing)["audit"]["going_concern"] is True
    _, prior = original(CURRENT_INTRO + "\n前期\n保留結論之基礎\n部分非重要子公司未經核閱。")
    assert proof(prior)["audit"]["audit_opinion"] == "UNKNOWN"


def test_cutoff_future_contexts_dimensions_and_unverified_original_still_block():
    _, filing = original(english=ACTIVE)
    assert audit_proof_r7(filing, symbol="SYNTHETIC_ONLY", cutoff="2026-01-01T00:00:00Z",
        original_verified=True)["status"] == "PENDING_VERIFICATION"
    assert audit_proof_r7(filing, symbol="SYNTHETIC_ONLY", cutoff=CUTOFF)["status"] == "PENDING_VERIFICATION"
    two_dates = replace(filing, facts=filing.facts + (FiledFact("tifrs-ar:AccountantsReportBody", None,
        QUALIFIED_BODY, None, None, "2026-08-11", context_id="OTHER_PRESENT_DATE"),))
    assert proof(two_dates)["status"] == "PENDING_VERIFICATION"
    dimensional = replace(filing, facts=tuple(replace(f, dimensions=(("entity", "OTHER"),))
        if f.concept.startswith("tifrs-ar:") else f for f in filing.facts))
    assert proof(dimensional)["status"] == "PENDING_VERIFICATION"

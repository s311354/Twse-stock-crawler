"""SYNTHETIC ONLY: comparison integrity, not forecast or trading efficacy."""
from copy import deepcopy

import pytest

from domain.les_v7.contracts import canonical, identity
from examples.compare_fr2_eps import differences, restore_delta, summarize, compare_report


@pytest.mark.parametrize("before,after", [
    ({}, {"optional": None}),
    ({"optional": None}, {}),
    ({"a/b~c": {"x": 1}}, {"a/b~c": {"x": 2, "y": None}}),
    ({"array": [1, 2]}, {"array": [1, 3, None]}),
    ({"n": 1}, {"n": 1.}),
    ({"keep": {"x": 1}, "drop": None}, {"keep": {"x": 1}, "new": False}),
])
def test_lossless_delta_reconstructs_exact_type_presence_and_escaped_keys(before, after):
    before_id = identity(before)
    changes = list(differences(before, after))
    assert canonical(restore_delta(before, changes)) == canonical(after)
    assert identity(before) == before_id


def test_delta_rejects_a_different_original_instead_of_overwriting_it():
    changes = list(differences({"old": None}, {"old": 3}))
    with pytest.raises(ValueError, match="DELTA_SOURCE_MISMATCH"):
        restore_delta({}, changes)


def test_complete_census_keeps_gaps_separate_from_recomputed_verdicts():
    complete = dict(symbol="SYNTHETIC", state="FULL_POLICY_RECOMPUTED", old_decision="DATA_INSUFFICIENT",
        new_decision="DATA_INSUFFICIENT", new_score={"final_score": None}, frozen_operands_equal=True,
        admission=dict(valuation=dict(native_state="UNKNOWN", warning_eligible=False,
                                     reason_codes=["SOURCE_MISSING"]), hard_vetoes=[]))
    gap = dict(symbol="SYNTHETIC_GAP", state="SOURCE_OR_RECOMPUTATION_GAP", reason="UNVERIFIED")
    summary = summarize([complete, gap])
    assert summary["total"] == 2
    assert summary["states"] == {"FULL_POLICY_RECOMPUTED": 1, "SOURCE_OR_RECOMPUTATION_GAP": 1}
    assert summary["new_decisions"] == {"DATA_INSUFFICIENT": 1}
    assert summary["valuation_states"] == {"UNKNOWN": 1}
    assert summary["signals"] == []
    assert summary["validated_count"] == 1
    assert summary["full_frozen_operand_consistency"] is False
    assert summary["account_execution_or_profitability_assessed"] is False


def test_real_comparison_boundary_validates_frozen_fields_and_reversible_policy_only():
    from application.low_entry_v7_daily import daily_report
    from domain.les_v7.fr2_policy import restore_original
    from tests.test_v7_fr2_eps_daily import research_history

    research = daily_report(research_history())
    original = restore_original(research)
    model = research["diagnostics"]["fr2"]["predictor_bundle"]
    inputs = research["diagnostics"]["fr2"]["valuation_inputs"]
    before = canonical(dict(original=original, model=model, inputs=inputs))
    output, row, retained = compare_report(original, model, inputs)
    assert row["state"] == "FULL_POLICY_RECOMPUTED"
    assert row["result_sha256"] == identity(output)
    assert retained["original_report_id"] == identity(original)
    assert row["old_gates"] != row["new_gates"]
    assert row["frozen_operands_equal"] is True
    assert row["original_report_restored_exactly"] is True
    assert row["new_account_authorization"] is None
    assert row["original_authorization"] is None
    assert restore_delta(original, row["full_report_differences"]) == output
    assert canonical(dict(original=original, model=model, inputs=inputs)) == before


def test_comparison_never_accepts_a_live_authorized_source():
    saved = dict(pae_authorization={"live_order_authorized": True})
    with pytest.raises(ValueError, match="LIVE_AUTHORIZATION_FORBIDDEN"):
        compare_report(saved, {}, {}, validate=lambda _: None)


def test_internal_admission_cannot_run_before_the_original_validation_boundary():
    events = []
    def validate(value):
        events.append("VALIDATE_ORIGINAL")
        raise ValueError("SYNTHETIC_INVALID_ORIGINAL")
    def apply(*args):
        events.append("INTERNAL_ADMISSION")
        raise AssertionError("MUST_NOT_RUN")
    with pytest.raises(ValueError, match="SYNTHETIC_INVALID_ORIGINAL"):
        compare_report({}, {}, {}, validate=validate, apply=apply)
    assert events == ["VALIDATE_ORIGINAL"]

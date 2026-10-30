"""Fixed-input ablations are not formal reports, fills, or OOS evidence."""
from copy import deepcopy

import pytest

from domain.les_v7.rule_baseline import baseline_policy
from examples.compare_les_r25 import compare_report
from tests.test_v7_r19_pipeline import native
from tests.test_v7_r25_pipeline import bound_market


def test_control_is_reproduced_and_all_four_cells_keep_identical_safety_inputs():
    report = native(bound_market(), "V7_RULE_BASED_BASELINE_R24")
    before = deepcopy(report)
    result = compare_report(report)
    assert report == before
    assert result["full_native_r24_revalidated"]
    assert result["account_sha256"] == report["pae_authorization"]["account_sha256"]
    assert not result["new_formal_report"] and not result["new_pae_authorization"]
    assert not result["account_or_pae_writes"] and result["simulated_fills"] == 0
    assert result["outcomes"]["r24"]["decision"] == report["envelope"]["strategy"]["strategy_decision"]
    assert result["outcomes"]["r24"]["score"]["final_score"] == report["envelope"]["score"]["final_score"]
    assert result["unchanged_plan"] == report["diagnostics"]["reference_plan"]
    for outcome in result["outcomes"].values():
        assert outcome["penalties"] == result["outcomes"]["r24"]["penalties"]
        assert outcome["admission"] == result["outcomes"]["r24"]["admission"]
    assert result["outcomes"]["r24"]["score"] == result["outcomes"]["threshold_only"]["score"]
    assert result["outcomes"]["r25"]["score"] == result["outcomes"]["score_only"]["score"]


def test_tampered_parent_score_cannot_be_reused_as_a_validated_control():
    report = native(bound_market(), "V7_RULE_BASED_BASELINE_R24")
    report["envelope"]["score"]["final_score"] += 1
    with pytest.raises(ValueError): compare_report(report)


def test_new_policy_does_not_rewrite_old_curve_or_threshold_contract():
    old = baseline_policy(version="V7_RULE_BASED_BASELINE_R24")
    assert old["model_spec"]["curves"]["amount_ratio"] == [1., 1.5]
    assert old["unified_technical_policy"]["small_minimum"] == 60


def test_daily_input_errors_remain_missing_final_and_block_even_with_positive_diagnostic_score():
    from application.low_entry_v7_daily import finalize_daily_report
    report = native(bound_market(), "V7_RULE_BASED_BASELINE_R24")
    account = report.pop("pae_authorization")["account"]  # Fresh engine output has no prior quota.
    report = finalize_daily_report(report, account=account,
        daily_input_errors=["SYNTHETIC_MISSING_PRICE_INPUT"])
    result = compare_report(report)
    for value in result["outcomes"].values():
        assert value["score"]["final_score"] is None
        assert value["decision"] not in {"BUY", "BUY_SMALL"}
        assert value["pre_input_error_score"]["raw_score"] is not None

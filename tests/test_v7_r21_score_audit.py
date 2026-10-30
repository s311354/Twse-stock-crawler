"""Synthetic cached-ledger audit; original engine validation remains separate."""
from copy import deepcopy

import pytest

from examples.audit_les_r21 import rescore_saved
from examples.compare_les_r14 import business
from examples.compare_les_r20 import compare_report
from tests.test_v7_r14_comparison import fixture as source_fixture
from tests.test_v7_r19_pipeline import native
from tests.test_v7_r21_pipeline import fixture


def test_saved_technical_ledger_must_reproduce_before_new_score_comparison():
    old = native(fixture(), "V7_RULE_BASED_BASELINE_R20")
    body = business(old)
    before = deepcopy(body)
    result = rescore_saved(body)
    assert body == before
    assert result["old_raw"] == old["envelope"]["score"]["raw_score"]
    assert result["old_after_same_deduction"] == old["envelope"]["score"]["final_score"]
    assert result["new_entry_decision"] == "NOT_RECOMPUTED_IN_SCORE_ONLY_AUDIT"
    assert "strategy_decision" not in result
    forged = deepcopy(body)
    forged["baseline_calculation"]["score_evidence"]["conditions"][0]["earned_points"] += 1.
    with pytest.raises(ValueError, match="NOT_REPRODUCED"):
        rescore_saved(forged)


def test_full_native_pilot_uses_public_engine_and_distinct_policy_pair():
    saved, filings, _ = source_fixture()
    old, new, row = compare_report(saved, filings, evaluation_cutoff="2026-10-28T06:10:00Z",
        versions=("V7_RULE_BASED_BASELINE_R20", "V7_RULE_BASED_BASELINE_R21"))
    assert "r20" in row and "r21" in row and "r19" not in row
    assert row["state"] == "FULL_NATIVE_RECOMPUTED"
    assert old["pae_authorization"]["account_sha256"] == new["pae_authorization"]["account_sha256"]
    assert not new["pae_authorization"]["live_order_authorized"]
    assert row["full_report_differences"]

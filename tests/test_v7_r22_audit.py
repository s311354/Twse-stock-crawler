"""Comparison safeguards on explicit synthetic reports, not real source repair."""
from copy import deepcopy

import pytest

from examples.audit_les_r22 import score_pair, compare_native
from examples.compare_les_r14 import business
from tests.test_v7_r14_comparison import fixture as source_fixture
from tests.test_v7_r19_pipeline import native
from tests.test_v7_r22_pipeline import fixture


def test_score_sensitivity_first_reproduces_ledger_and_cannot_claim_decision():
    old=native(fixture(), "V7_RULE_BASED_BASELINE_R20")
    body=business(old);before=deepcopy(body)
    row=score_pair(body)
    assert body==before
    assert row["r22"]["raw_score"]>=row["r21"]["raw_score"]
    assert row["r22"]["total_penalty"]==row["r21"]["total_penalty"]
    assert row["native_decision"].startswith("NOT_RECOMPUTED")
    body["baseline_calculation"]["score_evidence"]["conditions"][0]["earned_points"]+=1
    with pytest.raises(ValueError,match="NOT_REPRODUCED"):score_pair(body)


def test_native_pair_preserves_account_and_source_safety_with_full_differences():
    saved,filings,_=source_fixture()
    old,new,diff=compare_native(saved,filings,cutoff="2026-10-28T06:35:00Z")
    assert old["pae_authorization"]["account_sha256"]==new["pae_authorization"]["account_sha256"]
    assert new["envelope"]["strategy"]["decision_policy_version"]=="RULE_BASED_PROGRESSIVE_CONFIRMATION_R22"
    assert diff["full_report_differences"]
    assert not new["pae_authorization"]["live_order_authorized"]

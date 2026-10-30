"""Synthetic policy/invariance tests, not evidence of investment efficacy."""
from dataclasses import replace

import pandas as pd
import pytest

from domain.features import SharedFeaturePipeline
from domain.strategies.erem_soft_reduction import SoftReductionResearchPolicy, apply_soft_reduction_policy
from domain.strategies.expected_return_exit_v3 import ExpectedReturnExitV3Strategy
from tests.strategies.test_expected_return_exit_v3 import make_history


def apply(commands, *, les=None, days=None, grace=None, kinds=None):
    n = len(commands)
    return apply_soft_reduction_policy(pd.Series(commands), entry_les=pd.Series(les or [70.] * n),
        holding_days=pd.Series(days or list(map(float, range(n)))),
        entry_decision=pd.Series(kinds or ["BUY_SMALL"] * n),
        entry_grace=pd.Series(grace if grace is not None else [False] * n),
        policy=SoftReductionResearchPolicy())


def test_two_observations_confirm_and_two_observations_cool_down():
    result = apply(["REDUCE 25%"] * 6)
    assert result.decision.tolist() == ["HOLD_REVIEW", "REDUCE 25%", "HOLD_REVIEW", "HOLD_REVIEW", "HOLD_REVIEW", "REDUCE 25%"]
    assert result.cooldown_remaining.tolist() == [0, 2, 1, 0, 0, 2]


def test_exit_is_never_delayed_even_with_grace_cooldown_or_missing_position():
    result = apply(["REDUCE 25%", "REDUCE 25%", "EXIT", "EXIT"],
                   les=[70., 70., 70., float("nan")], grace=[False, False, True, True])
    assert result.decision.iloc[-2:].tolist() == ["EXIT", "EXIT"]
    assert result.status.iloc[-2:].tolist() == ["NATIVE_EXIT_IMMEDIATE"] * 2


def test_grace_and_recovery_reset_warning_without_reclassifying_healthy_hold():
    result = apply(["HOLD", "REDUCE 25%", "HOLD", "REDUCE 25%", "REDUCE 25%"],
                   grace=[True, False, False, False, False])
    assert result.decision.tolist() == ["HOLD", "HOLD_REVIEW", "HOLD", "HOLD_REVIEW", "REDUCE 25%"]


@pytest.mark.parametrize("kwargs", [
    {"days": [10., 0.]}, {"les": [70., 80.]}, {"les": [float("nan"), 70.]},
    {"kinds": [None, "BUY_SMALL"]}, {"grace": [None, False]},
])
def test_position_change_or_unknown_input_cannot_carry_confirmation(kwargs):
    assert apply(["REDUCE 25%", "REDUCE 25%"], **kwargs).decision.tolist() == ["HOLD_REVIEW"] * 2


def test_causal_prefix_invariance_and_no_shared_account_state():
    commands = ["REDUCE 25%"] * 8 + ["EXIT", "HOLD"]
    whole = apply(commands)
    for n in range(1, len(commands)+1):
        pd.testing.assert_frame_equal(apply(commands[:n]), whole.iloc[:n])
    assert apply(["REDUCE 25%"]).decision.iloc[0] == "HOLD_REVIEW"


def test_policy_is_explicit_frozen_and_not_a_claim_of_better_returns():
    policy = SoftReductionResearchPolicy()
    assert policy.record()["performance_validated"] is False
    with pytest.raises(ValueError):
        replace(policy, confirmation_observations=1)
    with pytest.raises(ValueError):
        ExpectedReturnExitV3Strategy(soft_reduction_policy={"confirmation_observations": 2})


def test_native_default_and_research_share_identical_features_scores_and_latest_prefix(monkeypatch):
    history = make_history()
    def commands(self, score, *args, **kwargs):
        result = pd.Series("HOLD", index=score.index)
        result.iloc[-6:] = "REDUCE 25%"
        return result
    monkeypatch.setattr(ExpectedReturnExitV3Strategy, "_decision", commands)
    native = ExpectedReturnExitV3Strategy()
    research = ExpectedReturnExitV3Strategy(soft_reduction_policy=SoftReductionResearchPolicy())
    features = SharedFeaturePipeline(native._shared_feature_config()).build(history)
    baseline = native.run_with_features(history, features)
    result = research.run_with_features(history, features)
    assert "EREM Research Policy" not in baseline
    mutable = {"EREM 決策訊號", "EREM 建議減倉比例", "EREM 理由"}
    stable = [column for column in baseline if column not in mutable]
    pd.testing.assert_frame_equal(baseline[stable], result[stable])
    assert result["EREM Native Decision"].equals(baseline["EREM 決策訊號"])
    assert result["EREM 決策訊號"].tail(6).str.startswith("REDUCE").sum() == 2
    assert result.loc[result["EREM 決策訊號"] == "HOLD_REVIEW", "EREM 建議減倉比例"].eq(0).all()
    pd.testing.assert_frame_equal(result.tail(1), research.run_latest_with_features(history, features))
    assert native.version == "3.2.4" and research.version != native.version
    assert native.name == "expected_return_exit_v3" and research.name != native.name


def test_empty_research_result_cannot_look_like_a_native_complete_output():
    result = ExpectedReturnExitV3Strategy(soft_reduction_policy=SoftReductionResearchPolicy()).run(pd.DataFrame())
    assert result["EREM Reduction Status"].iloc[0] == "UNKNOWN_EMPTY_INPUT"

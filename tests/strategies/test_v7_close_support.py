"""SYNTHETIC ONLY: explicit close-profile timing, never a real BUY assertion."""
from copy import deepcopy

import pandas as pd
import pytest

from domain.les_v7.contracts import FEATURE_VERSION
from domain.les_v7.decision import vetoes
from domain.les_v7.features import (CLOSE_SUPPORT_FEATURE_VERSION, compute_features, select_bars)


def structural_bars():
    """Old floor100, new strict floor90 at117, known only at119 close."""
    n = 121
    bars = pd.DataFrame(dict(Date=pd.bdate_range(end="2026-10-19", periods=n).strftime("%Y-%m-%d"),
        Open=[105.] * n, High=[110.] * n, Low=[104.] * n,
        Close=[105.] * n, Volume=[10000.] * n, **{"TAIEX Close": [10000.] * n}))
    bars.loc[60, "Low"] = 100.
    bars.loc[100:, ["Open", "Close", "High", "Low"]] = [94., 94., 104., 93.]
    bars.loc[110, "High"] = 130.
    for i, close, low in ((117, 91., 90.), (118, 94., 93.), (119, 96., 93.), (120, 98., 94.)):
        bars.loc[i, ["Open", "Close", "Low"]] = [close, close, low]
    bars["Amount"] = bars["Close"] * bars["Volume"]
    bars["Bid"], bars["Ask"] = bars["Close"] - .1, bars["Close"]
    return bars


@pytest.mark.parametrize("right_count,last_index,old_support,new_support", [
    (1, 118, 100., 100.), (2, 119, 100., 90.), (3, 120, 90., 90.),
])
def test_exact_one_two_three_right_bar_timing_preserves_legacy_delay(
        right_count, last_index, old_support, new_support):
    bars = structural_bars().iloc[:last_index + 1]
    legacy, old_audit = compute_features(bars)
    current, audit = compute_features(bars, feature_version=CLOSE_SUPPORT_FEATURE_VERSION)
    assert legacy["support"] == old_support
    assert current["support"] == new_support
    assert old_audit["profile"] == FEATURE_VERSION and "support_policy" not in old_audit
    assert audit["profile"] == CLOSE_SUPPORT_FEATURE_VERSION
    assert audit["support_policy"] == dict(version="STRICT_PIVOT_CONFIRMED_AT_SECOND_RIGHT_CLOSE_R2",
        confirmation_lag_sessions=2, additional_delay_sessions=0, cutoff_mode="COMPLETED_DAILY_CLOSE")
    # Only the support operands change. Higher-low, all technical indicators
    # and trailing estimators retain the same calculation and warmup rules.
    dependent = {"support", "support_intact", "reclaim", "no_low2"}
    assert {k: v for k, v in current.items() if k not in dependent} == {
        k: v for k, v in legacy.items() if k not in dependent}
    assert {k: v for k, v in audit.items() if k not in {"profile", "support_policy"}} == {
        k: v for k, v in old_audit.items() if k != "profile"}


def test_default_is_exact_explicit_legacy_profile_and_does_not_mutate_input():
    bars = structural_bars().iloc[:120].copy()
    before = deepcopy(bars)
    assert compute_features(bars) == compute_features(bars, feature_version=FEATURE_VERSION)
    pd.testing.assert_frame_equal(bars, before)
    assert FEATURE_VERSION == "V7_BASE_FRACTAL_R1"


def test_equal_low_tie_never_becomes_a_strict_close_profile_pivot():
    bars = structural_bars().iloc[:120].copy()
    bars.loc[118, "Low"] = 90.
    for version in (FEATURE_VERSION, CLOSE_SUPPORT_FEATURE_VERSION):
        features, _ = compute_features(bars, feature_version=version)
        assert features["support"] == 100.


def test_cutoff_excludes_second_right_bar_before_close_and_ignores_future_extremes():
    bars = structural_bars()
    at_close = "2026-10-16T13:30:00+08:00"
    selected = select_bars(bars, at_close)
    original = compute_features(selected, feature_version=CLOSE_SUPPORT_FEATURE_VERSION)
    assert original[0]["support"] == 90.
    before = compute_features(select_bars(bars, "2026-10-16T13:29:59+08:00"),
                              feature_version=CLOSE_SUPPORT_FEATURE_VERSION)
    assert before[0]["support"] == 100.
    future = bars.copy()
    future.loc[120, ["Open", "High", "Low", "Close"]] = [200., 500., 1., 200.]
    assert compute_features(select_bars(future, at_close),
                            feature_version=CLOSE_SUPPORT_FEATURE_VERSION) == original
    future["Available At"] = future["Date"] + "T13:30:00+08:00"
    future.loc[119, "Available At"] = "2026-10-19T09:00:00+08:00"
    delayed = compute_features(select_bars(future, at_close),
                              feature_version=CLOSE_SUPPORT_FEATURE_VERSION)
    assert delayed[0]["support"] == 100.


def test_reconfirmed_floor_avoids_only_old_anchor_veto_not_a_genuine_new_floor_break():
    bars = structural_bars()
    old, _ = compute_features(bars.iloc[:120], feature_version=FEATURE_VERSION)
    current, _ = compute_features(bars.iloc[:120], feature_version=CLOSE_SUPPORT_FEATURE_VERSION)
    assert {"SUPPORT_FAILURE", "GAP_FAILURE"} <= set(vetoes(old))
    assert not ({"SUPPORT_FAILURE", "GAP_FAILURE"} & set(vetoes(current)))
    bars.loc[120, ["Open", "Close", "Low"]] = [88., 88., 87.]
    broken, _ = compute_features(bars, feature_version=CLOSE_SUPPORT_FEATURE_VERSION)
    assert broken["support"] == 90.
    assert {"SUPPORT_FAILURE", "GAP_FAILURE"} <= set(vetoes(broken))


@pytest.mark.parametrize("version", [None, "UNKNOWN_PROFILE", {}, True])
def test_unknown_feature_versions_cannot_silently_choose_close_semantics(version):
    with pytest.raises(ValueError, match="V7_UNSUPPORTED_FEATURE_VERSION"):
        compute_features(structural_bars(), feature_version=version)


def test_empty_close_profile_has_versioned_audit_without_a_fabricated_support():
    features, audit = compute_features(pd.DataFrame(), feature_version=CLOSE_SUPPORT_FEATURE_VERSION)
    assert features == {} and audit["status"] == "NO_BARS"
    assert audit["profile"] == CLOSE_SUPPORT_FEATURE_VERSION
    assert audit["support_policy"]["additional_delay_sessions"] == 0

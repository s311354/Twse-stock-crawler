"""SYNTHETIC ONLY: causal short-base hypotheses, not a market BUY validation."""
from copy import deepcopy

import pandas as pd
import pytest

from domain.les_v7.context_r14 import VARIANT, assess_short_base, select_context
from domain.les_v7.rule_baseline import prepare_baseline, select_baseline_context, RELAXED_BASELINE_POLICY
from domain.strategies.low_entry_v6_hybrid import ContextSelection


CUTOFF = "2026-10-20T13:30:00+08:00"


def short_base_bars():
    n = 100
    data = pd.DataFrame(dict(Date=pd.bdate_range(end="2026-10-20", periods=n).strftime("%Y-%m-%d"),
        Open=[80.] * n, High=[81.] * n, Low=[79.] * n, Close=[80.] * n,
        Volume=[10000.] * n, Amount=[800000.] * n, Trades=[100] * n,
        Bid=[79.9] * n, Ask=[80.] * n, **{"TAIEX Close": [1000.] * n}, Complete=[True] * n))
    # A genuine older overhead pivot, outside the latest sixty-bar position.
    data.loc[10, "High"] = 90.
    prior = [(80., 81., 80.5), (81.2, 81.6, 81.4), (80.2, 81., 80.6),
             (81.2, 81.7, 81.5), (81.1, 81.8, 81.6)]
    for index, (low, high, close) in zip(range(n - 6, n - 1), prior):
        data.loc[index, ["Open", "Low", "High", "Close", "Bid", "Ask"]] = [close, low, high, close, close - .1, close]
    data.loc[n - 1, ["Open", "Low", "High", "Close", "Volume", "Amount", "Bid", "Ask"]] = [82., 81.8, 82.2, 82.2, 20000., 1644000., 82.1, 82.2]
    return data


def prepared(data=None):
    data = short_base_bars() if data is None else data
    p = prepare_baseline(data, market="TWSE", cutoff=CUTOFF, policy_version=RELAXED_BASELINE_POLICY)
    original = select_baseline_context(p["features"], policy_version=RELAXED_BASELINE_POLICY)
    return data, p["features"], original


source_bars = short_base_bars


def valid_features(data=None):
    return prepared(data)[1]


def test_short_variant_is_a_causal_addition_not_a_relabeling_of_the_failed_original_base():
    data, features, original = prepared()
    assert original.context == "NO_CLEAR_CONTEXT"
    assert original.record()["evidence"]["LEADERSHIP_CONTINUATION"]["PRIOR_BASE"] is not True
    selection, audit = select_context(original, features, bars=data, cutoff=CUTOFF)
    assert audit["state"] == "PASS"
    assert audit["setup_variant"] == VARIANT
    assert selection.context == "LEADERSHIP_CONTINUATION"
    assert selection.record()["matches"] == ["LEADERSHIP_CONTINUATION"]
    assert audit["original_selection"] == original.record()
    assert "PRIOR_BASE" not in selection.record()["evidence"]["LEADERSHIP_CONTINUATION"]
    assert audit["support"]["value"] == 80.
    assert audit["support"]["touch_episode_count"] == 2
    assert audit["support"]["touch_episodes"][0]["first_session"] != audit["support"]["touch_episodes"][1]["first_session"]
    assert features["base_volume_ratio5_20"] == 1.  # no forced volume contraction


def test_original_recognized_setup_never_uses_the_short_branch_even_if_its_risk_will_fail():
    data, features, _ = prepared()
    original = ContextSelection("TREND_PULLBACK", (("TREND_PULLBACK", (("KNOWN", True),)),))
    selection, audit = select_context(original, features, bars=data, cutoff=CUTOFF)
    assert selection is original and audit["state"] == "NOT_APPLICABLE"
    assert audit["setup_variant"] is None


def test_unrelated_missing_original_operand_does_not_block_verified_short_branch_dependencies():
    data, features, original = prepared()
    incomplete = ContextSelection("NO_CLEAR_CONTEXT", tuple((name,
        tuple((key, None if name == "REVERSAL" and key == "LOW_ZONE" else value) for key, value in checks))
        for name, checks in original.evidence))
    selection, audit = select_context(incomplete, features, bars=data, cutoff=CUTOFF)
    assert selection.context == "LEADERSHIP_CONTINUATION" and audit["state"] == "PASS"
    features["rs20"] = None
    selection, audit = select_context(incomplete, features, bars=data, cutoff=CUTOFF)
    assert selection.context == "NO_CLEAR_CONTEXT" and audit["state"] == "UNKNOWN"
    assert audit["checks"]["POSITIVE_RS20"] is None


@pytest.mark.parametrize("field", ["ema10", "ema20", "ema20_slope", "rs20", "rs_slope5"])
def test_required_absence_cannot_reuse_a_stale_technical_value(field):
    data, features, _ = prepared()
    features["score_availability"] = dict(features["score_availability"], **{field: False})
    audit = assess_short_base(data, features, cutoff=CUTOFF)
    assert audit["state"] == "UNKNOWN" and audit["setup_variant"] is None


def test_missing_amount_is_unknown_not_zero_or_a_volume_substitute():
    data, features, _ = prepared()
    del data["Amount"]
    audit = assess_short_base(data, features, cutoff=CUTOFF)
    assert audit["state"] == "UNKNOWN" and audit["checks"]["BREAKOUT_AMOUNT"] is None


def test_touch_days_in_one_episode_are_not_two_independent_support_tests():
    data, features, _ = prepared()
    data.loc[len(data) - 5, "Low"] = 80.2  # remove the separating excursion
    audit = assess_short_base(data, features, cutoff=CUTOFF)
    assert audit["support"]["touch_episode_count"] == 1
    assert audit["checks"]["TWO_SEPARATED_SUPPORT_TESTS"] is False
    assert audit["state"] == "FAIL"


def test_wide_prior_base_cannot_be_repaired_by_inflating_signal_bar_range():
    data, features, _ = prepared()
    data.loc[len(data) - 4, "High"] = 86.
    before = assess_short_base(data, features, cutoff=CUTOFF)
    data.loc[len(data) - 1, "High"] = 120.
    after = assess_short_base(data, features, cutoff=CUTOFF)
    assert before["checks"]["PRIOR_5_WIDTH"] is False
    assert after["checks"]["PRIOR_5_WIDTH"] is False
    assert before["metrics"]["prior_atr"] == after["metrics"]["prior_atr"]
    assert before["metrics"]["prior5_width_atr"] == after["metrics"]["prior5_width_atr"]


def test_future_bars_and_mutating_returned_audit_do_not_pollute_cutoff_or_future_calls():
    data, features, _ = prepared()
    first = assess_short_base(data, features, cutoff=CUTOFF)
    future = data.tail(1).copy(); future["Date"] = "2026-10-21"
    future[["Open", "High", "Low", "Close"]] = [999., 1000., 1., 900.]
    more = pd.concat([data, future], ignore_index=True)
    second = assess_short_base(more, features, cutoff=CUTOFF)
    assert second == first
    saved = deepcopy(first)
    first["support"]["value"] = 9999.
    assert assess_short_base(data, features, cutoff=CUTOFF) == saved


def test_clv_is_native_minus_one_to_one_not_close_percentile():
    data, features, _ = prepared()
    # (2*82.2 - 82.4 - 81.8)/(.6) == 1/3, not a .6667 percentile.
    data.loc[len(data) - 1, "High"] = 82.4
    audit = assess_short_base(data, features, cutoff=CUTOFF)
    assert audit["metrics"]["clv"] == pytest.approx(1 / 3)
    assert audit["checks"]["CLOSE_LOCATION"] is False

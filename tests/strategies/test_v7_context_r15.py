"""SYNTHETIC ONLY: R15 support timing, not a real market signal or fill."""
from copy import deepcopy

import pytest

from domain.les_v7.context_r14 import assess_short_base as old_assess
from domain.les_v7.context_r15 import VARIANT, SUPPORT_METHOD, assess_short_base, select_context
from domain.strategies.low_entry_v6_hybrid import ContextSelection
from tests.strategies.test_v7_context_r14 import CUTOFF, prepared, short_base_bars


def source_bars():
    data = short_base_bars()
    # Only one floor formation; later closes, not an invented second retest.
    data.loc[len(data) - 4, ["Open", "Low", "High", "Close"]] = [81.4, 81.2, 81.5, 81.4]
    return data


def valid_features(data=None):
    return prepared(source_bars() if data is None else data)[1]


def test_r15_single_floor_requires_two_later_completed_prior_closes_and_is_not_a_pivot():
    data = source_bars(); _, f, original = prepared(data)
    old = old_assess(data, f, cutoff=CUTOFF)
    assert old["checks"]["TWO_SEPARATED_SUPPORT_TESTS"] is False
    selected, audit = select_context(original, f, bars=data, cutoff=CUTOFF)
    assert selected.context == "LEADERSHIP_CONTINUATION"
    assert audit["setup_variant"] == VARIANT and audit["state"] == "PASS"
    support = audit["support"]
    assert support["value"] == 80. and support["method"] == SUPPORT_METHOD
    assert support["formation_session"] == data.iloc[-6]["Date"]
    assert support["subsequent_close_count"] >= 2
    assert support["confirmed_at_session"] == support["subsequent_close_sessions"][1]
    assert all(session < data.iloc[-1]["Date"] for session in support["subsequent_close_sessions"])
    assert "touch_episode_count" not in support
    assert "TWO_SEPARATED_SUPPORT_TESTS" not in audit["checks"]


def test_signal_close_never_counts_as_one_of_two_prior_confirmations():
    data = source_bars()
    data.loc[len(data) - 3, "Low"] = 79.5
    _, f, _ = prepared(data)
    audit = assess_short_base(data, f, cutoff=CUTOFF)
    assert audit["support"]["subsequent_close_count"] == 1
    assert audit["checks"]["TWO_SUBSEQUENT_COMPLETED_PRIOR_CLOSES"] is False
    assert audit["state"] == "FAIL"


def test_last_prior_bar_floor_has_no_future_confirmation_or_nearer_anchor_substitution():
    data = source_bars(); data.loc[len(data) - 2, "Low"] = 79.
    _, f, _ = prepared(data)
    audit = assess_short_base(data, f, cutoff=CUTOFF)
    assert audit["support"]["value"] == 79.
    assert audit["support"]["subsequent_close_count"] == 0
    assert audit["support"]["confirmed_at_session"] is None
    assert audit["state"] == "FAIL"


def test_r15_stronger_participation_is_not_backported_to_the_old_rule():
    data = short_base_bars(); data.loc[len(data) - 1, ["Volume", "Amount"]] = [14000., 1120000.]
    _, f, _ = prepared(data)
    old, new = old_assess(data, f, cutoff=CUTOFF), assess_short_base(data, f, cutoff=CUTOFF)
    assert old["checks"]["BREAKOUT_VOLUME"] is True and old["checks"]["BREAKOUT_AMOUNT"] is True
    assert new["checks"]["BREAKOUT_VOLUME"] is False and new["checks"]["BREAKOUT_AMOUNT"] is False


@pytest.mark.parametrize("key", ["ema10", "ema20", "ema20_slope", "rs20", "rs_slope5"])
def test_missing_required_operands_stay_unknown(key):
    data = source_bars(); f = valid_features(data)
    f["score_availability"] = dict(f["score_availability"], **{key: False})
    audit = assess_short_base(data, f, cutoff=CUTOFF)
    assert audit["state"] == "UNKNOWN" and audit["setup_variant"] is None


def test_old_recognized_context_is_not_replaced_after_a_possible_risk_failure():
    data = source_bars(); f = valid_features(data)
    original = ContextSelection("REVERSAL", (("REVERSAL", (("STRUCTURE", True),)),))
    selected, audit = select_context(original, f, bars=data, cutoff=CUTOFF)
    assert selected is original and audit["state"] == "NOT_APPLICABLE"


def test_future_bars_and_mutating_results_do_not_change_the_cutoff_assessment():
    import pandas as pd
    data = source_bars(); f = valid_features(data)
    original = assess_short_base(data, f, cutoff=CUTOFF)
    saved = deepcopy(original)
    future = data.tail(1).copy(); future["Date"] = "2026-10-21"
    future[["Open", "High", "Low", "Close"]] = [999., 1000., 1., 900.]
    assert assess_short_base(pd.concat([data, future], ignore_index=True), f, cutoff=CUTOFF) == saved
    original["support"]["value"] = 999.
    assert assess_short_base(data, f, cutoff=CUTOFF) == saved

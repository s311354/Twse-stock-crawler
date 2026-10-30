"""SYNTHETIC structural/source semantics, not official evidence or profits."""
from copy import deepcopy
from datetime import timedelta

import pandas as pd
import pytest

from domain.les_v7 import plan_r17, plan_r18, plan_r19
from domain.les_v7.contracts import timestamp
from domain.les_v7.context_r17 import select_context
from domain.les_v7.policy_r22 import AVAILABLE_AT, SOURCE_SELECTION_VERSION
from domain.les_v7.rule_baseline import RELAXED_BASELINE_POLICY, baseline_policy
from tests.test_v7_plan_r16 import pullback_bars, synthetic_calendar
from tests.test_v7_plan_r17 import prepared as prepared_pullback
from tests.test_v7_plan_r18 import features, neutral_bars

CUTOFF = AVAILABLE_AT


def reference(*, marker=True):
    value = plan_r19.policy(baseline_policy(version=RELAXED_BASELINE_POLICY)["reference_policy"],
        available_at=AVAILABLE_AT)
    if marker:
        value["source_selection_policy"] = SOURCE_SELECTION_VERSION
    return value


def selected(data, *, marker=True, calendar=True):
    f = features(data)
    return plan_r19.prepare_selection(select_context(f), f, bars=data, cutoff=CUTOFF,
        market="TWSE", session_evidence=synthetic_calendar(data) if calendar else None,
        selection_policy=SOURCE_SELECTION_VERSION if marker else None)


def native(data, *, marker=True, calendar=True):
    choice, audit = selected(data, marker=marker, calendar=calendar)
    selected_bars = plan_r17.r16.select_bars(data, CUTOFF)
    _, atr = plan_r17.r16._atr(selected_bars)
    return plan_r19.make_plan(choice.context, dict(close=float(selected_bars.iloc[-1]["Close"]), atr=atr),
        reference(marker=marker), "a" * 64, bars=data, cutoff=CUTOFF,
        setup_variant=audit.get("setup_variant"),
        session_evidence=synthetic_calendar(data) if calendar else None)


def test_legacy_none_retains_the_exact_old_selection_and_plan():
    data = neutral_bars()
    f = features(data); initial = select_context(f)
    kwargs = dict(bars=data, cutoff=CUTOFF, market="TWSE", session_evidence=synthetic_calendar(data))
    assert plan_r19.prepare_selection(initial, f, **kwargs) == plan_r19.prepare_selection(
        initial, f, selection_policy=None, **kwargs)
    old, old_audit = native(data, marker=False)
    new, new_audit = native(data)
    assert (old.entry, old.band, old.stop, old.target) == (new.entry, new.band, new.stop, new.target)
    assert old_audit["conditional_entry"] == new_audit["conditional_entry"]
    assert "source_selection_policy" not in reference(marker=False)


@pytest.mark.parametrize("field", ["ema5_slope", "ema5", "clv", "obv_slope5"])
def test_soft_missing_has_no_second_structural_veto_or_default_points(field):
    data = pullback_bars(); f, _, _, old = prepared_pullback(data)
    f = deepcopy(f); f[field] = None
    # Making every other demand input unknown avoids accidentally satisfying
    # the retained two-condition summary with unrelated known strengths.
    if field in {"clv", "obv_slope5"}:
        f.update(cmf_improving=None, up_volume_ratio=None, vwap20=None)
    kwargs = dict(cutoff=CUTOFF, impulse_high=old["support"]["impulse_high"],
        market="TWSE", session_evidence=synthetic_calendar(data))
    legacy = plan_r17.assess_early_pullback(data, f, **kwargs)
    audit = plan_r17.assess_early_pullback(data, f, selection_policy=SOURCE_SELECTION_VERSION, **kwargs)
    assert legacy["state"] == "UNKNOWN"
    assert audit["state"] == "PASS" and audit["setup_variant"] == plan_r17.A_VARIANT
    assert all(value is True for value in audit["hard_checks"].values())
    assert audit["soft_unknowns"] and audit["source_selection_policy"] == SOURCE_SELECTION_VERSION
    assert audit["soft_missing_policy"] == "RETAIN_DIAGNOSTICS_NO_DEFAULT_POINTS_NO_STRUCTURAL_VETO"
    assert not {"score", "bonus", "earned_points"} & set(audit)


@pytest.mark.parametrize("field", ["history_ready", "market_ready"])
def test_missing_hard_operands_are_not_covered_by_soft_scores(field):
    data = pullback_bars(); f, _, _, old = prepared_pullback(data)
    f = deepcopy(f); f[field] = None
    audit = plan_r17.assess_early_pullback(data, f, cutoff=CUTOFF,
        impulse_high=old["support"]["impulse_high"], market="TWSE",
        session_evidence=synthetic_calendar(data), selection_policy=SOURCE_SELECTION_VERSION)
    assert audit["state"] == "UNKNOWN" and audit["setup_variant"] is None
    assert any(value is None for value in audit["hard_checks"].values())


def test_missing_calendar_remains_unknown_and_missing_open_bar_is_not_no_match():
    data = neutral_bars()
    _, audit = selected(data, calendar=False)
    assert audit["state"] == "UNKNOWN"
    assert audit["session_coverage"]["state"] == "UNKNOWN"
    plan, audit = native(data, calendar=False)
    assert plan is None and audit["state"] == "UNKNOWN"
    calendar = synthetic_calendar(data)
    incomplete = data.drop(index=len(data) - 10).reset_index(drop=True)
    f = features(incomplete)
    _, audit = plan_r19.prepare_selection(select_context(f), f, bars=incomplete, cutoff=CUTOFF,
        market="TWSE", session_evidence=calendar, selection_policy=SOURCE_SELECTION_VERSION)
    assert audit["session_coverage"]["state"] == "FAIL" and audit["state"] == "FAIL"


@pytest.mark.parametrize("anchor", ["support", "resistance"])
def test_complete_verified_no_anchor_is_known_no_plan_not_missing_data(anchor):
    data = neutral_bars()
    if anchor == "support":
        data.loc[len(data) - 12, "Low"] = 99.
    else:
        data.loc[len(data) - 70, "High"] = 101.
    _, legacy = selected(data, marker=False)
    _, audit = selected(data)
    assert legacy["state"] == "UNKNOWN"
    assert audit["state"] == "NO_MATCH" and audit["setup_variant"] is None
    assert audit["session_coverage"]["state"] == "PASS"
    assert audit[anchor] is None
    plan, sidecar = native(data)
    assert plan is None and sidecar["state"] == "NO_MATCH"
    assert sidecar["conditional_entry"] is None and sidecar["anchors"] is None
    assert sidecar["short_base_audit"]["source_selection_policy"] == SOURCE_SELECTION_VERSION


def test_a_broken_support_cannot_be_reclassified_as_no_match_or_reanchored():
    data = neutral_bars(); data.loc[len(data) - 1, "Low"] = 97.9
    _, audit = selected(data)
    assert audit["state"] == "FAIL" and audit["support"]["value"] == 98.
    assert audit["reasons"] == ["R18_CONFIRMED_SUPPORT_BROKEN"]
    plan, sidecar = native(data)
    assert plan is None and sidecar["state"] == "FAIL"


def test_no_anchor_does_not_hide_a_missing_required_close_or_atr():
    data = neutral_bars(); data.loc[len(data) - 12, "Low"] = 99.
    f = features(data); initial = select_context(f)
    f["atr14"] = None
    _, audit = plan_r19.prepare_selection(initial, f, bars=data, cutoff=CUTOFF,
        market="TWSE", session_evidence=synthetic_calendar(data),
        selection_policy=SOURCE_SELECTION_VERSION)
    assert audit["state"] == "UNKNOWN"
    assert audit["reasons"] == ["SOURCE_CLOSE_OR_ATR_MISSING_OR_CONFLICTING"]


def test_no_anchor_cannot_hide_missing_signal_trading_evidence():
    data = neutral_bars(); data.loc[len(data) - 12, "Low"] = 99.
    data.loc[len(data) - 1, "Volume"] = float("nan")
    # Native bar validation is stricter than selection: a missing volume
    # rejects the raw bar before any known-no-anchor classification is made.
    with pytest.raises(ValueError, match="INVALID_OHLCV"):
        selected(data)


def test_reference_marker_reaches_the_native_reconstructed_early_plan():
    data = pullback_bars()
    plan, sidecar = native(data)
    old_plan, old_sidecar = native(data, marker=False)
    assert plan is not None and plan.setup_variant == plan_r17.A_VARIANT
    assert sidecar["short_base_audit"]["source_selection_policy"] == SOURCE_SELECTION_VERSION
    assert "source_selection_policy" not in old_sidecar["short_base_audit"]
    assert (plan.stop, plan.target, plan.entry, plan.band) == (
        old_plan.stop, old_plan.target, old_plan.entry, old_plan.band)
    assert sidecar["policy_id"] != old_sidecar["policy_id"]


def test_marker_is_bound_to_reference_identity_and_cannot_precede_release():
    data = neutral_bars(); f = features(data); initial = select_context(f)
    before = (timestamp(AVAILABLE_AT) - timedelta(microseconds=1)).isoformat()
    with pytest.raises(ValueError, match="BEFORE_RELEASE"):
        plan_r19.prepare_selection(initial, f, bars=data, cutoff=before,
            market="TWSE", session_evidence=synthetic_calendar(data),
            selection_policy=SOURCE_SELECTION_VERSION)
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        plan_r19.prepare_selection(initial, f, bars=data, cutoff=CUTOFF, selection_policy="UNKNOWN_POLICY")
    bad = reference(); bad["source_selection_policy"] = "UNKNOWN_POLICY"
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        plan_r19.require_policy(bad, CUTOFF)
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        plan_r19.make_plan(initial.context, {}, reference(marker=False), "a" * 64,
            bars=data, cutoff=CUTOFF, selection_policy=SOURCE_SELECTION_VERSION)


def test_future_bars_cannot_change_selection_or_native_plan():
    data = neutral_bars()
    choice, audit = selected(data); plan, sidecar = native(data)
    future = data.tail(1).copy(); future["Date"] = "2026-10-29"
    future[["Open", "High", "Low", "Close"]] = [999., 1000., 1., 999.]
    more = pd.concat((data, future), ignore_index=True)
    assert selected(more) == (choice, audit)
    assert native(more) == (plan, sidecar)

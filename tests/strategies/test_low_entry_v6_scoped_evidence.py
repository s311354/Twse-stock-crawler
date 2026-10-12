"""Synthetic, ticker-independent tests of cutoff, comparability and risk policies."""
from copy import deepcopy

import pandas as pd
import pytest

from domain.strategies.low_entry_v6_features import _consolidation_features
from domain.strategies.low_entry_v6_payoff import resistance_target_research
from domain.strategies.low_entry_v6_quality import (
    _eligible, combine_filing_sources, evaluate_quality, verification_diagnostics,
)
from domain.strategies.low_entry_v6_signals import decide, net_reward_risk, risk_pass, trade_plan
from tests.strategies.test_low_entry_score_v6 import CONFIG, financials, quality, setup_features
from tests.strategies.test_low_entry_v6_base_recovery import base_bars


CUTOFF = pd.Timestamp("2026-09-16T13:30:00+08:00")


def evaluate(records, cutoff=CUTOFF):
    return evaluate_quality(records, symbol="2330", market="TWSE", as_of=cutoff)


@pytest.mark.parametrize("field,value", [("scope", "individual"), ("unit", "dollar"), ("currency", "USD")])
def test_unrelated_old_reports_do_not_poison_eight_quarter_comparability(field, value):
    records = financials()
    reference = evaluate(records)
    records[0][field] = value
    assert evaluate(records) == reference
    assert evaluate(records).metrics["comparison_periods"] == [
        str(p) for p in pd.period_range("2024Q3", "2026Q2", freq="Q")]


@pytest.mark.parametrize("field,value", [("scope", "individual"), ("unit", "dollar"), ("currency", "USD")])
def test_actual_reported_comparison_inputs_still_must_be_comparable(field, value):
    records = financials()
    records[-5][field] = value
    result = evaluate(records)
    assert result.status == "PROVISIONAL"
    assert result.metrics["excluded_quality_inputs"][0]["reason"] == "INCOMPARABLE_FINANCIAL_STATEMENTS"
    assert result.metrics["revenue_growth"] is None  # No cross-scope/unit arithmetic.


def test_ytd_dependency_before_eight_quarters_is_included_not_spliced():
    records = financials()
    records[-8]["flow_basis"] = "YTD"  # 2024Q3 needs 2024Q2.
    selected = _eligible(records, "2330", "TWSE", CUTOFF)
    assert pd.Period("2024Q2") in selected
    records[-9]["scope"] = "individual"
    assert evaluate(records).metrics["excluded_quality_inputs"][0]["reason"] == "INCOMPARABLE_FINANCIAL_STATEMENTS"
    # Removing the source doesn't manufacture Q3 standalone flows.
    del records[-9]
    result = evaluate(records)
    # This gap affects diagnostic TTM growth/EPS, not the selected reported
    # comparison. Do not transfer it to an unrelated scoring metric.
    assert result.coverage["profitability"] == 100
    assert result.metrics["quality_profitability_basis"] == "REPORTED_SAME_PERIOD_MARGIN"
    assert result.metrics["revenue_growth"] is not None
    assert result.metrics["positive_eps_quarters"] is None
    audit = verification_diagnostics(records, symbol="2330", as_of=CUTOFF)
    assert "2024Q2" in audit["diagnostic_missing_periods"] and audit["missing_periods"] == []


def test_future_revision_scope_does_not_change_earlier_window():
    records = financials()
    future = dict(records[-8], scope="individual", available_at="2026-09-17T00:00:00Z", revision_id="future")
    assert evaluate(records + [future]) == evaluate(records)
    assert evaluate(records + [future], pd.Timestamp("2026-09-18T13:30:00+08:00")).status == "PASS"
    assert evaluate(records + [future], pd.Timestamp("2026-09-18T13:30:00+08:00")).score == evaluate(records).score


def test_future_batch_cannot_hide_the_eligible_original_mops_period():
    records = [dict(r, source="MOPS_IXBRL") for r in financials()]
    future = dict(records[-1], source="TWSE_OPENAPI", available_at="2026-09-23T00:00:00Z",
                  revision_id="future-batch", total_equity=0.)
    original = deepcopy((records, future))
    combined = combine_filing_sources(records, [future], as_of=CUTOFF)
    assert evaluate(combined) == evaluate(records)
    assert (records, future) == original
    assert evaluate(combine_filing_sources(records, [future], as_of="2026-09-24T00:00:00Z"),
                    pd.Timestamp("2026-09-24T00:00:00Z")).status == "REJECT"


def test_unverified_batch_scope_never_relabels_an_individual_original():
    originals = [dict(r, scope="individual", source="MOPS_IXBRL") for r in financials()]
    batches = [dict(r, scope="consolidated", source="TWSE_OPENAPI", total_equity=0.,
                    scope_basis="LEGACY_DEFAULT_NOT_SOURCE_REPORTED", revision_id="batch-" + r["revision_id"])
               for r in originals]
    before = deepcopy((originals, batches))
    result = combine_filing_sources(originals, batches, as_of=CUTOFF)
    for row, original in zip(result, originals):
        assert {key: row[key] for key in original} == original
        assert row["source_selection"]["excluded_batch_revision_id"].startswith("batch-")
    assert evaluate(result) == evaluate(originals)
    assert (originals, batches) == before
    # Explicit consolidated metadata is not changed to an individual label.
    for batch in batches:
        batch.pop("scope_basis")
    assert all(r["source"] == "TWSE_OPENAPI" for r in combine_filing_sources(originals, batches, as_of=CUTOFF))


def extended_base(size):
    data = base_bars()
    extra = data.iloc[[-1]].copy()
    extra["low"] = 98.1
    data = pd.concat([data, extra], ignore_index=True)
    if size == 5:
        extra = extra.copy()
        extra["low"] = 98.05
        data = pd.concat([data, extra], ignore_index=True)
    return data


@pytest.mark.parametrize("size", [4, 5])
def test_extended_held_shelf_has_causal_support_not_latest_low(size):
    data = extended_base(size)
    complete = _consolidation_features(data)
    assert complete.iloc[-1].consolidation_confirmed
    assert complete.iloc[-1].consolidation_support == 98.
    assert complete.iloc[-1].consolidation_sessions == size
    for end in range(1, size + 1):
        pd.testing.assert_frame_equal(complete.iloc[:end], _consolidation_features(data.iloc[:end]))
    f = setup_features("REVERSAL", support=90., low=float(data.low.iloc[-1]),
                       consolidation_confirmed=True, consolidation_support=98., consolidation_sessions=size)
    plan = trade_plan(f, config=CONFIG, setup="REVERSAL")
    assert plan["stop_source"] == "CONFIRMED_CONSOLIDATION_LOW"
    assert plan["stop_anchor_age_sessions"] == size - 1
    assert plan["stop_price"] == pytest.approx(97.4)
    gap = trade_plan(f, entry_price=120., config=CONFIG, setup="REVERSAL")
    assert gap["stop_price"] == plan["stop_price"]
    assert not risk_pass("REVERSAL", "BUY_SMALL", f, gap, "NEUTRAL", CONFIG)[0]


@pytest.mark.parametrize("field,value", [("low", 97.9), ("volume", 0), ("market_ready", False),
                                        ("support_intact", False), ("close", float("nan"))])
def test_extended_shelf_does_not_carry_broken_or_missing_confirmation(field, value):
    data = extended_base(5)
    data.loc[4 if field == "low" else 2, field] = value
    assert not _consolidation_features(data).iloc[-1].consolidation_confirmed


def test_a_broken_shelf_can_only_be_replaced_after_two_new_held_sessions():
    data = extended_base(5)
    data.loc[2, "low"] = 97.9
    result = _consolidation_features(data)
    assert not result.iloc[2].consolidation_confirmed
    assert not result.iloc[3].consolidation_confirmed
    assert result.iloc[4].consolidation_support == 97.9
    assert result.iloc[4].consolidation_sessions == 3


def test_extended_shelf_cannot_widen_tolerance_with_later_atr_spike():
    data = extended_base(5)
    data.loc[0, "atr14"] = 1.
    data.loc[1:, "atr14"] = 20.
    assert not _consolidation_features(data).iloc[-1].consolidation_confirmed


def test_target_policy_study_preserves_formal_nearest_target_and_net_rr():
    f = setup_features("REVERSAL", support=92., swing_high=103., hh20=120., hh60=130.)
    before = deepcopy(f)
    decision = decide(quality(), f, config=CONFIG)
    plan = decision["plan"]
    study = resistance_target_research(plan, f, buy_cost_rate=CONFIG.buy_fee_rate,
                                       sell_cost_rate=CONFIG.sell_cost_rate)
    assert study["research_only"] and not study["affects_entry_gate"]
    assert study["formal_target_price"] == plan["target_price"] == 103.
    assert [s["target_price"] for s in study["scenarios"]] == [103., 120., 130.]
    split = study["staged_exit"]["conditional_both_targets_payoff"]
    assert split["net_reward_risk"] == pytest.approx(net_reward_risk(dict(plan, target_price=111.5), CONFIG))
    partial_stop = ((.5 * 103. + .5 * plan["stop_price"]) * (1 - CONFIG.sell_cost_rate)
                    / (plan["entry_price"] * (1 + CONFIG.buy_fee_rate)) - 1)
    assert study["staged_exit"]["conditional_first_target_then_stop_net_return"] == pytest.approx(partial_stop)
    assert plan["net_reward_risk"] == pytest.approx(net_reward_risk(plan, CONFIG))
    assert study["expected_return"] is study["estimated_win_rate"] is None
    assert decision == decide(quality(), f, config=CONFIG) and f == before
    assert decision["signal"] not in {"BUY", "BUY_SMALL"}


def test_unknown_target_stays_unknown_and_duplicate_resistances_are_not_new_reward():
    f = setup_features("LEADERSHIP_CONTINUATION", swing_high=99., hh20=99., hh60=99.)
    plan = trade_plan(f, config=CONFIG)
    study = resistance_target_research(plan, f, buy_cost_rate=.001425, sell_cost_rate=.004425)
    assert plan["target_price"] is None and study["scenarios"] == [] and study["staged_exit"] is None
    f.update(swing_high=110., hh20=110., hh60=110.)
    study = resistance_target_research(trade_plan(f, config=CONFIG), f,
                                       buy_cost_rate=.001425, sell_cost_rate=.004425)
    assert len(study["scenarios"]) == 1 and study["staged_exit"] is None
    assert study["scenarios"][0]["sources"] == ["swing_high", "hh20", "hh60"]

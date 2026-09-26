"""PIT source tie-outs and structural risk consistency; no ticker overrides."""
from copy import deepcopy

import pandas as pd
import pytest

from domain.strategies.low_entry_v6_quality import (
    _eligible, combine_filing_sources, evaluate_quality, normalize_quarters,
)
from domain.strategies.low_entry_v6_signals import decide, risk_pass, score_setups, trade_plan
from tests.strategies.test_low_entry_score_v6 import CONFIG, financials, quality, setup_features
from tests.strategies.test_low_entry_v6_confirmed_retest import retest_features


CUTOFF = pd.Timestamp("2026-09-16T13:30:00+08:00")


def eps_sources():
    batch = financials()[-1]
    batch.update(flow_basis="YTD", basic_eps=2., available_at="2026-09-13T00:00:00Z")
    mops = dict(batch, source="MOPS_IXBRL", revision_id="mops-v1",
                available_at="2026-08-14T00:00:00Z", basic_eps_quarter=.8)
    return batch, mops


def eligible(records, as_of=CUTOFF):
    return _eligible(records, "2330", "TWSE", pd.Timestamp(as_of))


def test_compatible_reported_quarter_eps_retains_primary_core_and_own_provenance():
    batch, mops = eps_sources()
    original = deepcopy((batch, mops))
    records = combine_filing_sources([mops], [batch], as_of=CUTOFF)
    row = eligible(records)[pd.Period("2026Q2")]
    assert row["source"] == "TWSE_OPENAPI"
    for key, value in batch.items():
        assert row[key] == value
    assert row["basic_eps_quarter"] == .8
    assert row["quarter_eps_provenance"]["revision_id"] == "mops-v1"
    assert row["quarter_eps_provenance"]["available_at"] == mops["available_at"]
    assert normalize_quarters({pd.Period("2026Q2"): row})[pd.Period("2026Q2")]["basic_eps"] == .8
    assert (batch, mops) == original


@pytest.mark.parametrize("key,value", [
    ("symbol", "9999"), ("market", "TPEX"), ("currency", "USD"),
    ("unit", "dollar"), ("scope", "individual"), ("flow_basis", "QUARTER"),
    ("period_end", "2026-03-31"), ("source", "THIRD_PARTY"),
    ("revenue", None), ("net_income", -1.), ("basic_eps", 2.01),
    ("total_assets", 10001.), ("basic_eps_quarter", None),
    ("basic_eps_quarter", float("nan")), ("revision_id", ""),
    ("available_at", "2026-09-23T00:00:00Z"), ("available_at", None),
    ("available_at", "2026-06-29T00:00:00Z"),
])
def test_conflicting_missing_or_future_supplement_cannot_fill_quarter_eps(key, value):
    batch, mops = eps_sources()
    mops[key] = value
    records = combine_filing_sources([mops], [batch], as_of=CUTOFF)
    # Inspect only the batch period; unrelated MOPS records retain normal gates.
    records = [r for r in records if r["source"] == "TWSE_OPENAPI"]
    row = eligible(records)[pd.Period("2026Q2")]
    assert row.get("basic_eps_quarter") is None


def test_future_enrichment_cannot_leak_when_a_later_bundle_is_replayed_earlier():
    batch, mops = eps_sources()
    mops["available_at"] = "2026-09-23T00:00:00Z"
    records = combine_filing_sources([mops], [batch], as_of="2026-09-24T00:00:00Z")
    assert eligible(records)[pd.Period("2026Q2")].get("basic_eps_quarter") is None
    later = eligible(records, "2026-09-24T00:00:00Z")[pd.Period("2026Q2")]
    assert later["basic_eps_quarter"] == .8


def test_latest_revision_must_tie_no_cherry_picking_an_older_matching_eps():
    batch, mops = eps_sources()
    changed = dict(mops, available_at="2026-09-15T00:00:00Z", revision_id="mops-v2", revenue=1.)
    for versions in ([mops, changed], [changed, mops]):
        row = eligible(combine_filing_sources(versions, [batch], as_of=CUTOFF))[pd.Period("2026Q2")]
        assert row.get("basic_eps_quarter") is None
    ambiguous = dict(mops, revision_id="other", basic_eps_quarter=10.)
    row = eligible(combine_filing_sources([mops, ambiguous], [batch], as_of=CUTOFF))[pd.Period("2026Q2")]
    assert row.get("basic_eps_quarter") is None


def test_enrichment_can_restore_evidence_but_does_not_guarantee_higher_quality():
    batch, mops = eps_sources()
    rows = financials()[:-1] + [batch]
    before = evaluate_quality(rows, symbol="2330", market="TWSE", as_of=CUTOFF)
    filled = combine_filing_sources([mops], rows, as_of=CUTOFF)
    after = evaluate_quality(filled, symbol="2330", market="TWSE", as_of=CUTOFF)
    assert before.metrics["positive_eps_quarters"] is None
    assert after.metrics["positive_eps_quarters"] == 8
    assert after.coverage["growth"] > before.coverage["growth"]
    negative = combine_filing_sources([dict(mops, basic_eps_quarter=-.8)], rows, as_of=CUTOFF)
    assert evaluate_quality(negative, symbol="2330", market="TWSE", as_of=CUTOFF).metrics["positive_eps_quarters"] == 7
    filled[-1]["total_equity"] = 0
    assert evaluate_quality(filled, symbol="2330", market="TWSE", as_of=CUTOFF).status == "REJECT"


def test_production_snapshot_assembler_preserves_dated_quarter_eps_without_writes():
    from application.fundamental_snapshot import BatchSnapshotResolution, SnapshotAssembler
    from domain.base_quality_features import BaseQualityFeatureProducer
    from domain.fundamental_derivation import FundamentalBundle
    from domain.strategies.low_entry_v6_quality import snapshot_record
    from tests.fundamentals.test_fundamental_pipeline_components import _snapshot

    batch = _snapshot(fiscal_year=2026, quarter=2, observed_at="2026-09-13T00:00:00Z")
    mops = dict(snapshot_record(batch), source="MOPS_IXBRL", revision_id="local-mops",
                available_at="2026-08-14T00:00:00Z", basic_eps_quarter=.8)
    bundle = FundamentalBundle(pd.DataFrame(), {"les_v6_mops_filings": [mops]}, CUTOFF.isoformat())
    # No repository methods are available: a replay assembly must not write.
    assembler = SnapshotAssembler(object(), BaseQualityFeatureProducer())
    result = assembler.attach_base_evidence(
        bundle, BatchSnapshotResolution(batch, "TWSE_OPENAPI_CACHE_FALLBACK", True),
        symbol="2330", signal_time=CUTOFF, uow=object(), history=[batch], persist_snapshot=False,
    )
    row = eligible(result.snapshot["les_v6_filings"])[pd.Period("2026Q2")]
    assert row["basic_eps_quarter"] == .8
    assert row["source"] == "TWSE_OPENAPI"
    assert row["quarter_eps_provenance"]["revision_id"] == "local-mops"


@pytest.mark.parametrize("setup", ["REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION"])
def test_percentage_wide_stop_can_use_confirmed_structure_even_below_atr_limit(setup):
    if setup == "LEADERSHIP_CONTINUATION":
        f = retest_features(support=93., stop_support=93., atr14=6.)
        expected = "CONFIRMED_BREAKOUT_RETEST"
    else:
        f = setup_features(setup, support=92., atr14=6., low=98.2,
                           consolidation_confirmed=True, consolidation_support=98.,
                           swing_high=112., hh20=120., hh60=125.)
        expected = "CONFIRMED_CONSOLIDATION_LOW"
    old = trade_plan(f, config=CONFIG)
    assert old["stop_distance_atr"] < 1.8 and old["stop_distance_fraction"] > .08
    plan = trade_plan(f, config=CONFIG, setup=setup)
    assert plan["stop_source"] == expected
    assert plan["target_price"] == old["target_price"]  # No target inflation.
    assert plan["stop_price"] == pytest.approx(plan["stop_support"] - .3 * f["atr14"])
    assert plan["stop_distance_fraction"] < .08
    assert score_setups(f, old, config=CONFIG)[setup]["entry_tier"] == "STARTER_ONLY"
    assert decide(quality(), f, config=CONFIG)["signal"] == "BUY_SMALL"
    gap_plan = trade_plan(f, entry_price=f["high"] * 1.15, config=CONFIG, setup=setup)
    assert gap_plan["stop_price"] == plan["stop_price"]  # Gap cannot pull up stop.
    assert not risk_pass(setup, "BUY_SMALL", f, gap_plan, "NEUTRAL", CONFIG)[0]


@pytest.mark.parametrize("setup,minimum", [("REVERSAL", 2.), ("TREND_PULLBACK", 1.8)])
def test_execution_score_uses_net_rr_not_a_pre_cost_pass(setup, minimum):
    f = setup_features(setup, high=100., support=97., atr14=2.,
                       swing_high=100. + minimum * 3.6, hh20=120., hh60=130.)
    plan = trade_plan(f, config=CONFIG)
    assert plan["reward_risk"] == pytest.approx(minimum)
    assert plan["net_reward_risk"] < minimum
    scored = score_setups(f, plan, config=CONFIG)[setup]
    # Stop and liquidity earn 4+1+2; net RR does not earn the extra 3 points.
    assert scored["components"]["execution"] == 7
    f["swing_high"] += 2.
    assert score_setups(f, trade_plan(f, config=CONFIG), config=CONFIG)[setup]["components"]["execution"] == 10


def test_sept16_3016_still_requires_restriction_and_realistic_risk_even_with_quality_pass():
    from tests.strategies.test_low_entry_v6_3016_causal import compute, raw_bars

    raw = raw_bars(include_future=True)
    raw = raw.loc[raw["Date"].le("2026-09-16")].copy()
    raw.loc[raw.index[-1], ["Bid", "Ask"]] = [115., 115.5]
    raw["Restricted"] = None
    f = compute(raw).iloc[-1].to_dict()
    result = decide(quality(score=74.76056338028168), f, config=CONFIG)
    assert result["signal"] == "SKIP"
    assert result["plan"]["entry_price"] == 116.
    assert result["plan"]["target_price"] == 119.
    assert result["plan"]["stop_distance_fraction"] > .16
    assert result["plan"]["net_reward_risk"] < .12
    # Diagnostic counterfactual only: clearing restrictions still cannot waive
    # the stop, nearby resistance or trigger. No production data is fabricated.
    cleared = decide(quality(score=100), dict(f, restricted=False), config=CONFIG)
    assert cleared["signal"] not in {"BUY", "BUY_SMALL"}
    assert "STOP_PERCENTAGE_EXCEEDS_LIMIT" in cleared["technical_observation"]["blocking_reasons"]

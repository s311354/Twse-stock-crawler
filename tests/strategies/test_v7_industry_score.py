"""SYNTHETIC ONLY: industry alignment, no real source or BUY claim."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from application.evidence_context import ReadOnlyEvidence
from application.industry_benchmark import prepare_industry_benchmark
from domain.les_v7.industry_score import apply_industry_features, industry_score_policy
from tests.test_industry_benchmark import add_row, packet

CUTOFF = "2026-09-11T13:30:00+08:00"


def inputs(count=61):
    dates = pd.bdate_range(end="2026-09-11", periods=count)
    close = np.linspace(100., 120., count)
    bars = pd.DataFrame(dict(Date=dates.strftime("%Y-%m-%d"), Open=close,
        High=close + 1, Low=close - 1, Close=close, Volume=10000.))
    raw = packet(dates=dates, closes=np.linspace(100., 101., count))
    return dict(close=120., score_availability={}), bars, raw


def apply(features, bars, raw, **options):
    return apply_industry_features(features, bars, prepare_industry_benchmark(raw) if raw else None,
        **{**dict(symbol="2330", market="TWSE", sector="SEMI", cutoff=CUTOFF), **options})


def test_verified_exact_dates_reconnect_existing_six_points_without_mutation():
    features, bars, raw = inputs()
    before = deepcopy((features, raw)); before_bars = bars.copy(deep=True)
    result, audit = apply(features, bars, raw)
    for n in (20, 60):
        expected = (bars["Close"].iloc[-1] / bars["Close"].iloc[-n - 1]
                    - raw["bars"][-1]["close"] / raw["bars"][-n - 1]["close"])
        assert result[f"industry_rs{n}"] == pytest.approx(expected)
        assert result["score_availability"][f"industry_rs{n}"] is True
    assert result["industry_status"] == audit["status"] == "PASS"
    evidence = audit["evidence"]
    assert len(evidence["observations"]) == 61
    assert evidence["selected_sha256"]
    assert evidence["observations"][-1]["index_receipt"]["sha256"]
    assert audit["window"] == dict(start="2026-06-19", end="2026-09-11", observations=61)
    result["score_availability"]["other"] = False
    assert (features, raw) == before
    pd.testing.assert_frame_equal(bars, before_bars)
    policy = industry_score_policy()
    assert policy["horizons"] == {"20": {"weight": 4}, "60": {"weight": 2}}
    policy["horizons"]["20"]["weight"] = 99
    assert industry_score_policy()["horizons"]["20"]["weight"] == 4


def test_missing_packet_clears_stale_operands_and_never_grants_neutral_points():
    features, bars, _ = inputs()
    features.update(industry_rs20=99., industry_rs60=99., industry_status="PASS")
    features["score_availability"].update(industry_rs20=True, industry_rs60=True)
    result, audit = apply(features, bars, None)
    assert result["industry_rs20"] is result["industry_rs60"] is None
    assert not result["score_availability"]["industry_rs20"]
    assert result["industry_status"] == audit["status"] == "UNKNOWN"
    assert audit["reason"] == "INDUSTRY_BENCHMARK_MISSING"


def test_verified_inputs_only_restore_existing_four_and_two_point_slots():
    from domain.les_v7.rule_baseline import CONFIG
    from domain.strategies.low_entry_v6_hybrid import score_factors, select_context
    from tests.strategies.test_v7_rule_baseline import unit_features
    _, bars, raw = inputs()
    features = unit_features()
    features.update(close=120., open=120., high=121., low=119.)
    baseline, _ = apply(features, bars, None)
    verified, _ = apply(features, bars, raw)
    plan = dict(entry_price=120., stop_price=118., target_price=128., stop_source="SYNTHETIC_ONLY",
        stop_distance_atr=1., stop_distance_fraction=2. / 120.)
    scores = [score_factors(f, select_context(f), plan, config=CONFIG)["score_evidence"]
              for f in (baseline, verified)]
    assert scores[1]["formal_score"] - scores[0]["formal_score"] == pytest.approx(6.)
    assert scores[1]["total_weight"] == scores[0]["total_weight"] == 100
    conditions = [{row["id"]: row for row in score["conditions"]} for score in scores]
    changed = {name for name in conditions[0] if conditions[0][name] != conditions[1][name]}
    assert changed == {"relative_strength:industry_rs20", "relative_strength:industry_rs60"}
    for name in changed:
        assert conditions[0][name]["earned_points"] == conditions[0][name]["available_weight"] == 0
        assert conditions[1][name]["earned_points"] == conditions[1][name]["weight"]
        assert conditions[0][name]["weight"] == conditions[1][name]["weight"]
    for group in scores[0]["components"]:
        if group != "relative_strength":
            assert scores[0]["components"][group] == scores[1]["components"][group]


def test_future_index_revision_and_future_stock_bar_do_not_change_cutoff_operands():
    features, bars, raw = inputs()
    expected = apply(features, bars, raw)
    row = raw["bars"][-1]
    add_row(raw, "bar", dict(trade_date=row["trade_date"], close=999.,
        membership_version=row["membership_version"], revision=1,
        series_version=row["series_version"], published_at="2026-09-12T09:00:00+08:00"))
    future = bars.iloc[[-1]].copy(); future.loc[:, "Date"] = "2026-09-14"
    future.loc[:, ["Open", "High", "Low", "Close"]] = [999., 1000., 998., 999.]
    assert apply(features, pd.concat((bars, future), ignore_index=True), raw) == expected


@pytest.mark.parametrize("changes", [dict(market="TPEX"), dict(sector="28"), dict(symbol="OTHER")])
def test_market_sector_and_historical_membership_mismatch_stay_unknown(changes):
    features, bars, raw = inputs()
    result, audit = apply(features, bars, raw, **changes)
    assert result["industry_status"] == audit["status"] == "UNKNOWN"
    assert result["industry_rs20"] is result["industry_rs60"] is None


def test_conflicting_current_versions_or_unproved_observations_remain_unknown():
    features, bars, raw = inputs()
    row = raw["bars"][-1]
    add_row(raw, "bar", dict(trade_date=row["trade_date"], close=200.,
        membership_version=row["membership_version"], revision=0,
        series_version=row["series_version"], published_at=row["trade_date"] + "T13:30:00+08:00"))
    result, audit = apply(features, bars, raw)
    assert audit["evidence"]["gaps"][-1]["reason"] == "INDEX_VERSION_CONFLICT"
    assert result["industry_rs20"] is result["industry_rs60"] is None
    raw = packet(dates=pd.to_datetime(bars["Date"]), proof=False)
    result, audit = apply(features, bars, raw)
    assert audit["status"] == "UNKNOWN"
    assert result["industry_rs20"] is result["industry_rs60"] is None


def test_short_window_does_not_invent_sixty_session_return():
    features, bars, raw = inputs(21)
    result, audit = apply(features, bars, raw)
    assert result["industry_rs20"] is not None and result["industry_rs60"] is None
    assert result["score_availability"]["industry_rs20"] is True
    assert result["score_availability"]["industry_rs60"] is False
    assert audit["periods"]["60"] == dict(available=False, value=None)


def test_empty_cutoff_window_has_no_evaluable_periods_or_stale_values():
    features, bars, raw = inputs()
    features.update(industry_rs20=99., industry_rs60=99., industry_status="PASS")
    result, audit = apply(features, bars.iloc[:0], raw)
    assert audit["reason"] == "INDUSTRY_STOCK_WINDOW_MISSING"
    assert audit["window"] is None and audit["evidence"] is None
    assert all(period == dict(available=False, value=None) for period in audit["periods"].values())
    assert result["industry_rs20"] is result["industry_rs60"] is None
    assert result["industry_status"] == "UNKNOWN"


def test_read_only_packet_is_supported_and_bad_stock_binding_stays_unscored():
    features, bars, raw = inputs()
    normalized = prepare_industry_benchmark(raw)
    expected = apply(features, bars, raw)
    actual = apply_industry_features(ReadOnlyEvidence(features), bars, ReadOnlyEvidence(normalized),
        symbol="2330", market="TWSE", sector="SEMI", cutoff=CUTOFF)
    assert actual == expected
    result, audit = apply(dict(features, close=119.), bars, raw)
    assert audit["reason"] == "INDUSTRY_STOCK_CLOSE_BINDING_MISMATCH"
    assert result["industry_rs20"] is result["industry_rs60"] is None


def test_invalid_source_binding_is_unknown_and_does_not_inherit_pass():
    features, bars, raw = inputs()
    normalized = prepare_industry_benchmark(raw)
    normalized["bars"][-1]["close"] = 999.
    result, audit = apply_industry_features(features, bars, normalized, symbol="2330",
        market="TWSE", sector="SEMI", cutoff=CUTOFF)
    assert audit["reason"] == "INDUSTRY_SOURCE_BINDING_MISMATCH"
    assert result["industry_status"] == "UNKNOWN"
    assert result["industry_rs20"] is result["industry_rs60"] is None

"""A subsequent rally cannot retrospectively authorize a risky entry plan.

Frozen TWSE ALLBUT0999 OHLCV/amount and TAIEX cache extract for 3016,
2026-05-27 through 2026-09-24. The decision cutoff is 2026-09-18; later
bars are used only to check prefix invariance. This bounded warm-up fixture
does not claim exact parity with the full production history or actual fills.
No test accesses a production database, network, broker or report artifact.
"""
from io import StringIO

import pandas as pd
import pytest

from domain.features import SharedFeaturePipeline
from domain.strategies.low_entry_v6_features import build_features
from domain.strategies.low_entry_v6_quality import QualityResult
from domain.strategies.low_entry_v6_signals import (
    EXECUTABLE_SIGNALS, LowEntryV6Config, decide, risk_pass, size_position,
    trade_plan,
)


CONFIG = LowEntryV6Config()
CUTOFF = pd.Timestamp("2026-09-18")
# Date, Open, High, Low, Close, Volume (shares), Amount (TWD), TAIEX Close.
RAW_BARS = """20260527,131,138,122,126,15299263,1974035359,44256.8
20260528,126,138.5,125,134.5,17126606,2324702302,43636.44
20260529,146.5,147.5,133.5,138,23100390,3234756238,44732.94
20260601,138.5,141,125,126,15785863,2041402164,45337.91
20260602,124.5,126.5,118.5,118.5,7910814,964398928,45557.31
20260603,120.5,122,112.5,115.5,9013879,1044596376,46459.16
20260604,113.5,114.5,109,110.5,4569912,508437884,45677.46
20260605,108.5,110.5,103,108,4538327,489138577,45070.94
20260608,97.2,106.5,97.2,104,4965303,498564234,43502.78
20260609,107,113,102,112,6130078,668567922,44704.44
20260610,107,113.5,103,103,5033055,542200004,43225.54
20260611,101,106,98.3,104,5035912,515761819,43149.46
20260612,114,114,110.5,114,4297699,489396825,44169.04
20260615,122,125,118.5,125,10244720,1265529184,45396.99
20260616,125.5,134.5,119,119.5,29035493,3683858511,45809.19
20260617,117.5,127,117,122.5,14167878,1750957133,45877.39
20260618,123,128,120.5,126.5,11958970,1490945923,46465.2
20260622,129.5,139,129,139,8906871,1221210569,47741.51
20260623,150,152.5,149,152.5,8287530,1260177127,47100.65
20260624,151.5,153,137.5,137.5,18860056,2693555598,46043.6
20260625,141.5,141.5,129.5,133.5,12052983,1621932476,46255.26
20260626,132.5,139,124.5,124.5,8914307,1172557766,44571.76
20260629,129.5,129.5,119,121.5,19713887,2318106477,44999.9
20260630,128.5,133.5,127,133.5,12505354,1664043927,46125.91
20260701,136.5,142,120.5,122,38593911,5081601220,47018.99
20260702,121,134,121,134,28157674,3674554856,46744.16
20260703,131,137.5,127,135.5,24047267,3186216497,46780.62
20260706,139.5,143,127,128.5,19909842,2665749588,46556.39
20260707,129.5,129.5,116,118,10142843,1232939324,45479.11
20260708,118,121.5,115,120,8576271,1014811147,45734.41
20260709,120,127,118.5,121.5,9446406,1168097955,45354.61
20260713,133,133.5,130.5,133.5,20361863,2713989553,45380.52
20260714,131,134.5,120.5,124,20534741,2576304092,44737.95
20260715,126.5,127.5,120.5,123,10413406,1284812467,45631.59
20260716,120.5,122,115.5,116,6596435,776948054,45624.98
20260717,108,111,104.5,104.5,5936314,631526003,42671.27
20260720,101,105.5,95.5,101,6792121,680175409,42449.7
20260721,102.5,103.5,97.6,101.5,6126063,615525388,44232.87
20260722,102.5,111.5,102.5,105,11278484,1202953329,44825.78
20260723,106.5,111.5,100,103,9425265,987382557,44850.81
20260724,100.5,101,95.1,95.3,5430956,531223647,43654.84
20260727,94.5,95.8,90.5,94.2,3551809,330309314,43634.19
20260728,87.8,89,84.8,84.8,3812956,327656756,41603.36
20260729,83.6,83.7,76.4,77.3,5721070,446670192,40039.18
20260730,74.9,80.5,72.7,74.4,5621832,426352806,39933.3
20260731,81.8,81.8,76.9,79.9,5874515,465920811,43119.75
20260803,77.5,87.8,77.5,87.8,3211226,270472243,43386.41
20260804,87,95.4,86.1,94.8,10624004,968608692,43360.66
20260805,99.4,104,96.2,104,11106435,1116226365,44611.6
20260806,104,104.5,98,103.5,7215937,734840614,44396.7
20260807,103.5,104.5,97.5,98,4023316,403619417,44225.91
20260810,100.5,107.5,99,107,5797696,605362253,44928.76
20260811,106,107.5,101.5,101.5,3761882,389829897,45120.72
20260812,102.5,109,102.5,104,8373062,888803455,45518.07
20260813,106,111,103.5,104,10040470,1074994565,46021.48
20260814,106,106,99.9,102.5,6007483,617535241,45811.01
20260817,100.5,108.5,99.5,106.5,8477223,888103811,45857.27
20260818,106,109,103,105,8729366,925811126,45308.68
20260819,101,105,100,102,2964464,303141994,44719.35
20260820,103,105,100.5,103,2713183,277963129,44933.74
20260821,103,103.5,97.9,98.4,3524488,350965752,45224.29
20260824,100,106,100,100,5558840,572884002,44762.32
20260825,99.8,99.8,93.9,96.8,4122205,397439945,45169.46
20260826,97.6,105,97.1,102,5178243,531278114,45832.62
20260827,102.5,104,99.1,99.9,3231808,325754136,45975.22
20260828,101.5,104,99.6,99.6,3253682,329684180,46331.45
20260831,98.7,104,97.6,103,4664276,471632242,46128.47
20260901,102,108,102,104,7198205,760000667,46948.72
20260902,103.5,104.5,99.7,100,3261035,330175817,46164.72
20260903,100.5,103,98.2,98.3,3531323,354240412,45857.66
20260904,100,101.5,97.5,101,3591167,357764025,46551.13
20260907,103,103.5,100,100.5,2695630,273503469,47326.27
20260908,102,108.5,101,101.5,11481206,1202736587,47105.78
20260909,102,102.5,99.4,100,3365101,338373233,47183.36
20260910,99.3,110,99.3,110,13980714,1499086316,46940.49
20260911,106.5,109.5,101,109.5,19649188,2064218617,46184.85
20260914,104,117,102,112.5,21536063,2407309412,45862.52
20260915,113,119,107.5,110,30912478,3519353372,45511.49
20260916,110.5,116,107.5,115.5,19161129,2154436181,45848.9
20260917,116,122,113.5,115.5,24388426,2860470729,46288
20260918,116.5,119,115,116.5,11658641,1356009457,47180.75
20260921,118,123,115.5,120.5,12523363,1493192954,47718.84
20260922,125,132.5,124.5,132.5,30369344,3962592045,47800.17
20260923,145,145.5,141.5,145.5,31085717,4515973276,48157.29
20260924,147.5,160,147,160,49083815,7567529613,48024.6
"""


def raw_bars(*, include_future=False):
    data = pd.read_csv(StringIO(RAW_BARS), header=None, names=(
        "Date", "Open", "High", "Low", "Close", "Volume", "Amount", "TAIEX Close"))
    data["Date"] = pd.to_datetime(data["Date"].astype(str), format="%Y%m%d")
    # Only the cutoff quote is needed; do not fabricate historical spreads.
    data["Bid"] = data["Ask"] = float("nan")
    data.loc[data["Date"].eq(CUTOFF), ["Bid", "Ask"]] = [116., 116.5]
    # Controlled unrestricted assumption isolates technical risk from the
    # independently verified production restriction and fundamental gates.
    data["Restricted"] = False
    return data if include_future else data.loc[data["Date"].le(CUTOFF)].copy()


def compute(data):
    return build_features(data, SharedFeaturePipeline().build(data))


def strong_quality():
    # Deliberately stronger than the historical quality classification: fixing
    # financial data or a high quality score must not waive technical risk.
    return QualityResult(status="PASS", score=100., confidence=100.,
                         integrity="PASS", components={}, coverage={}, metrics={})


def test_cutoff_has_no_confirmed_tighter_swing_or_compact_base():
    data = raw_bars()
    features = compute(data)
    f = features.iloc[-1]
    assert f["history_ready"] and f["market_ready"]
    assert f["support"] == f["stop_support"] == 99.3
    assert features.loc[data["Date"].ge("2026-09-15"), "swing_low"].eq(99.3).all()
    # Equal 107.5 lows are above the earlier 102 low, not a two-sided trough.
    assert data.loc[data["Date"].eq("2026-09-14"), "Low"].item() == 102.
    assert data.loc[data["Date"].isin(pd.to_datetime(["2026-09-15", "2026-09-16"])), "Low"].eq(107.5).all()
    assert not f["consolidation_confirmed"]
    assert pd.isna(f["consolidation_support"])
    width = data["High"].tail(3).max() - data["Low"].iloc[-3]
    assert width == 14.5 and width > 1.5 * features["atr14"].iloc[-3]


@pytest.mark.parametrize("setup", ["REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION"])
@pytest.mark.parametrize("signal", ["BUY", "BUY_SMALL"])
def test_known_resistance_and_structural_risk_block_every_entry_tier(setup, signal):
    f = compute(raw_bars()).iloc[-1].to_dict()
    plan = trade_plan(f, config=CONFIG, setup=setup)
    assert plan["stop_source"] == "CONFIRMED_SWING_LOW"
    assert plan["entry_price"] == 119. and plan["target_price"] == 122.
    assert plan["stop_price"] == pytest.approx(99.3 - .3 * f["atr14"])
    assert plan["liquidity_pass"]  # Not an accidental missing-quote rejection.
    assert plan["stop_distance_fraction"] > .18
    assert plan["reward_risk"] < .15
    assert plan["net_reward_risk"] < .11
    assert not risk_pass(setup, signal, f, plan, "NEUTRAL", CONFIG)[0]


def test_quality_pass_does_not_turn_the_cutoff_into_an_executable_entry():
    f = compute(raw_bars()).iloc[-1].to_dict()
    decision = decide(strong_quality(), f, valuation="NEUTRAL", config=CONFIG)
    assert decision["setup"] is not None and decision["score"] > 0
    assert decision["signal"] not in EXECUTABLE_SIGNALS
    assert decision["risk_fraction"] == 0
    assert decision["plan"]["target_price"] == 122.
    assert size_position(decision, portfolio_value=1_000_000., available_cash=1_000_000.,
                         intended_weight=.25, portfolio_capacity_weight=.25,
                         amount_ma20=f["amount_ma20"], config=CONFIG) == 0


def test_later_rally_cannot_rewrite_cutoff_features_score_or_decision():
    past, complete = raw_bars(), raw_bars(include_future=True)
    assert complete["Date"].iloc[-1] == pd.Timestamp("2026-09-24")
    assert complete["Close"].iloc[-1] == 160.
    before, after = compute(past), compute(complete)
    pd.testing.assert_frame_equal(before, after.iloc[:len(past)])
    original = decide(strong_quality(), before.iloc[-1].to_dict(), valuation="NEUTRAL", config=CONFIG)
    appended = decide(strong_quality(), after.iloc[len(past) - 1].to_dict(), valuation="NEUTRAL", config=CONFIG)
    assert appended == original
    assert appended["plan"]["target_price"] == 122.  # Never retrospective 160.

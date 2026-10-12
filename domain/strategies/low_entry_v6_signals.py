"""Pure setup scoring, vetoes, risk and sizing for LES v6."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isclose

from domain.strategies.low_entry_v6_quality import MINIMUM_ENTRY_QUALITY, QualityResult, number, ratio
from domain.strategies.low_entry_v6_payoff import payoff_scenarios
from domain.strategies.low_entry_v6_score_evidence import score_evidence, valid_price_anchor
from domain.strategies.low_entry_v6_execution_score import (
    DEPENDENCIES as EXECUTION_DEPENDENCIES, execution_conditions, validate_score_budget,
)


SETUPS = ("REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION")
EXECUTABLE_SIGNALS = frozenset({"BUY", "BUY_SMALL"})
VETO_NUMERIC_FIELDS = ("new_lows3", "clv", "ema5_slope", "ema20_slope", "atr_expansion",
                       "r5", "atr_percent", "close", "open", "volume_ratio", "gap",
                       "support", "rs20", "rs_slope5", "resilience")
VETO_BOOLEAN_FIELDS = ("support_break", "new_low", "undercut_reclaim")


@dataclass(frozen=True)
class LowEntryV6Config:
    """Uncalibrated choices are versioned and described in the v6 implementation notes."""

    decision_policy_version: int = 2
    correlation_policy_version: int = 2
    minimum_amount_ma20: float = 30_000_000.
    maximum_spread: float = .01
    maximum_order_adv: float = .01
    maximum_position_weight: float = .25
    starter_fraction: float = .30
    small_risk_fraction: float = .0025
    buy_risk_fraction: float = .0075
    stop_buffer_atr: float = .30
    risk_off_score_increment: float = 5.
    risk_off_position_multiplier: float = .5
    risk_off_stop_multiplier: float = .85
    expensive_rr_increment: float = .2
    expensive_position_multiplier: float = .75
    quote_max_age_seconds: float = 60.
    buy_fee_rate: float = .001425
    # Configurable cash-equity modelling assumption: fee plus sale tax, not a quote.
    sell_cost_rate: float = .004425
    buy_score_minimum: float = 75.
    reversal_starter_score_minimum: float = 65.
    pullback_starter_score_minimum: float = 65.
    leadership_starter_score_minimum: float = 65.
    buy_maximum_stop_atr: float = 1.8
    buy_maximum_stop_fraction: float = .08
    starter_maximum_stop_atr: float = 3.
    starter_maximum_stop_fraction: float = .08
    starter_minimum_net_rr: float = 1.7
    reversal_minimum_drawdown: float = .06
    reversal_maximum_position: float = .40
    recovery_maximum_atr_expansion: float = 1.25
    recovery_ema20_slope_tolerance: float = .05
    pullback_minimum_drawdown: float = .01
    pullback_maximum_drawdown: float = .15
    leadership_rs20_tolerance: float = .01
    leadership_policy_version: int = 2

    def __post_init__(self):
        if type(self.decision_policy_version) is not int or self.decision_policy_version not in {1, 2, 3, 4}:
            raise ValueError("Unsupported LES decision policy version")
        if type(self.correlation_policy_version) is not int or self.correlation_policy_version not in {1, 2}:
            raise ValueError("Unsupported LES correlation policy version")
        if type(self.leadership_policy_version) is not int or self.leadership_policy_version not in {1, 2}:
            raise ValueError("Unsupported LES leadership policy version")
        for name, value in vars(self).items():
            if number(value) is None or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("maximum_spread", "maximum_order_adv", "maximum_position_weight",
                     "starter_fraction", "small_risk_fraction", "buy_risk_fraction",
                     "risk_off_position_multiplier", "risk_off_stop_multiplier", "expensive_position_multiplier"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must not exceed one")
        if not .25 <= self.starter_fraction <= .4 or not .25 <= self.stop_buffer_atr <= .3:
            raise ValueError("Starter fraction / stop buffer outside v6 specification")
        if not .0025 <= self.small_risk_fraction <= .005 or not .005 <= self.buy_risk_fraction <= .01:
            raise ValueError("Risk fractions outside v6 specification")
        if self.maximum_order_adv > .03 or self.buy_fee_rate >= .1 or self.sell_cost_rate >= .1:
            raise ValueError("Invalid liquidity or fee cap")
        starters = (self.reversal_starter_score_minimum, self.pullback_starter_score_minimum,
                    self.leadership_starter_score_minimum)
        buy_floor, small_floor = (72, 62) if self.decision_policy_version == 4 else (75, 65)
        if not buy_floor <= self.buy_score_minimum <= 100 or any(not small_floor <= v <= self.buy_score_minimum for v in starters):
            raise ValueError("Invalid entry score thresholds")
        if not 1.2 <= self.starter_maximum_stop_atr <= 3.:
            raise ValueError("Invalid starter structural stop bound")
        if not 0 < self.starter_maximum_stop_fraction <= .08:
            raise ValueError("Invalid starter percentage stop bound")
        if not 1.2 <= self.buy_maximum_stop_atr <= 1.8 or not 0 < self.buy_maximum_stop_fraction <= .08:
            raise ValueError("Invalid confirmed entry structural stop bound")
        if not 1.7 <= self.starter_minimum_net_rr <= 3.:
            raise ValueError("Invalid starter net reward/risk bound")
        if not (0 < self.reversal_minimum_drawdown <= .08 and 0 < self.reversal_maximum_position <= .40
                and 1 <= self.recovery_maximum_atr_expansion <= 1.25
                and self.recovery_ema20_slope_tolerance <= .05
                and 0 < self.pullback_minimum_drawdown < self.pullback_maximum_drawdown <= .15
                and self.leadership_rs20_tolerance <= .01):
            raise ValueError("Invalid balanced setup bounds")


def relaxed_hybrid_config(**overrides) -> LowEntryV6Config:
    """Explicit versioned deployment; legacy constructor defaults stay intact."""
    parameters = dict(decision_policy_version=4, buy_score_minimum=72.,
                      reversal_starter_score_minimum=62., pullback_starter_score_minimum=62.,
                      leadership_starter_score_minimum=62.)
    parameters.update(overrides)
    return LowEntryV6Config(**parameters)


def truth(value: object) -> bool:
    return value is not None and str(value) == "True"


def _n(f: dict, key: str) -> float:
    value = number(f.get(key))
    return float("nan") if value is None else value


def veto_evidence_complete(f: dict) -> bool:
    masks = f.get("score_availability", {})
    return (all(number(f.get(key)) is not None and (key not in masks or truth(masks[key]))
                for key in VETO_NUMERIC_FIELDS)
            and all(str(f.get(key)) in {"True", "False"} for key in VETO_BOOLEAN_FIELDS)
            and f.get("market_regime") in {"RISK_ON", "NEUTRAL", "RISK_OFF"})


def hard_vetoes(f: dict) -> tuple[str, ...]:
    n = lambda key: _n(f, key)
    conditions = {
        "PERSISTENT_NEW_LOW": n("new_lows3") >= 2 and not truth(f.get("undercut_reclaim")) and n("clv") < 0,
        "ACCELERATING_DECLINE": n("ema5_slope") < 0 and n("ema20_slope") < 0 and n("atr_expansion") > 1.2 and n("r5") < -1.5 * n("atr_percent"),
        "HIGH_VOLUME_WEAK_SUPPORT_BREAK": n("close") < n("open") and n("clv") < -.5 and n("volume_ratio") > 1.5 and truth(f.get("support_break")),
        "UNRECOVERED_GAP": n("open") < n("support") and n("gap") < 0 and n("close") < n("support"),
        "MARKET_RELATIVE_BREAKDOWN": f.get("market_regime") == "RISK_OFF" and n("rs20") < 0 and n("rs_slope5") < 0 and n("resilience") < 0 and truth(f.get("new_low")),
    }
    return tuple(name for name, active in conditions.items() if active)


def _confirmed_retest(f: dict) -> bool:
    support, level, age = (number(f.get(key)) for key in
                           ("breakout_retest_support", "breakout_retest_level", "breakout_retest_age"))
    return (truth(f.get("breakout_retest_confirmed")) and age is not None and age.is_integer() and 1 <= age <= 5
            and support is not None and level is not None and 0 < support <= level
            and _n(f, "low") >= support and _n(f, "close") >= level)


def _confirmed_consolidation(f: dict) -> bool:
    floor = number(f.get("consolidation_support"))
    return (truth(f.get("consolidation_confirmed")) and floor is not None
            and 0 < _n(f, "support") < floor <= _n(f, "low") < _n(f, "close")
            and truth(f.get("higher_low")) and truth(f.get("support_intact")))


def trade_plan(f: dict, entry_price: float | None = None, *, config: LowEntryV6Config,
               setup: str | None = None) -> dict:
    n = lambda key: _n(f, key)
    entry = number(entry_price) if entry_price is not None else number(f.get("high"))
    support, atr = number(f.get("stop_support", f.get("support"))), number(f.get("atr14"))
    stop = support - config.stop_buffer_atr * atr if support is not None and atr is not None and atr > 0 else None
    risk = entry - stop if entry is not None and stop is not None and 0 < stop < entry else None
    stop_source = "CONFIRMED_SWING_LOW"
    anchor_age = number(f.get("swing_low_confirmation_age"))
    if truth(f.get("undercut_reclaim")):
        stop_source = "CONFIRMED_UNDERCUT_LOW"
        anchor_age = 1.
    # Anchor selection uses the completed signal bar, never the live fill. A
    # high ATR must not hide an over-wide percentage stop. Keep a valid old stop
    # otherwise; a new anchor still requires independently confirmed structure.
    anchor_atr_limit = min(1.8, config.starter_maximum_stop_atr) if setup == "LEADERSHIP_CONTINUATION" else 1.8
    # Use the risk gate's numerical equality convention here too: roundoff at
    # the boundary must not replace a valid stop and force a starter-only tier.
    needs_new_anchor = (stop is not None and atr is not None and atr > 0 and 0 < stop < n("high")
                        and (not _within_stop_limit((n("high") - stop) / atr, anchor_atr_limit)
                             or not _within_stop_limit((n("high") - stop) / n("high"), config.starter_maximum_stop_fraction)))
    # Only a completed breakout followed by a held retest can establish a newer
    # structural stop. Use it for leaders whose old swing stop is too far away.
    # Reversal/pullback use the separate short-base policy below. No target is
    # raised to obtain RR in any setup.
    retest_support = number(f.get("breakout_retest_support"))
    if (setup == "LEADERSHIP_CONTINUATION" and risk is not None and atr is not None
            and needs_new_anchor
            and _confirmed_retest(f) and support is not None and support < retest_support):
        support = retest_support
        stop = support - config.stop_buffer_atr * atr
        risk = entry - stop if 0 < stop < entry else None
        stop_source = "CONFIRMED_BREAKOUT_RETEST"
        anchor_age = number(f.get("breakout_retest_age"))
    # A held short base is a distinct, starter-only invalidation point. Select
    # it from completed signal-day evidence, never from the live entry price.
    if (setup in {"REVERSAL", "TREND_PULLBACK"} and risk is not None and atr is not None
            and needs_new_anchor and _confirmed_consolidation(f)
            and support < n("consolidation_support")):
        support = n("consolidation_support")
        stop = support - config.stop_buffer_atr * atr
        risk = entry - stop if 0 < stop < entry else None
        stop_source = "CONFIRMED_CONSOLIDATION_LOW"
        sessions = number(f.get("consolidation_sessions"))
        anchor_age = sessions - 1 if sessions is not None else None
    # Known resistance only, never manufacture reward for a leader in price discovery.
    levels = [number(f.get(key)) for key in ("swing_high", "hh20", "hh60")]
    targets = [v for v in levels if v is not None and entry is not None and v > entry]
    target = min(targets) if targets else None
    rr = ratio(target - entry, risk) if target is not None and entry is not None else None
    spread_ok = 0 <= n("spread") <= config.maximum_spread
    amount_ok = n("amount_ma20") >= config.minimum_amount_ma20
    unrestricted = f.get("restricted") is not None and str(f["restricted"]) == "False"
    plan = {
        "entry_price": entry, "stop_price": stop, "target_price": target,
        "stop_source": stop_source, "stop_support": support,
        "stop_anchor_age_sessions": anchor_age,
        "stop_distance_atr": ratio(risk, atr), "stop_distance_fraction": ratio(risk, entry), "reward_risk": rr,
        "spread_pass": spread_ok, "amount_pass": amount_ok,
        "liquidity_pass": spread_ok and amount_ok and unrestricted,
        "unrestricted": unrestricted,
    }
    plan.update(net_reward_risk=net_reward_risk(plan, config), rr_basis="NET_OF_CONFIGURED_COSTS")
    return plan


def net_reward_risk(plan: dict, config: LowEntryV6Config) -> float | None:
    """Recompute from prices, never trust a cached or caller-provided RR."""
    return payoff_scenarios(plan, buy_cost_rate=config.buy_fee_rate,
                            sell_cost_rate=config.sell_cost_rate)["net_reward_risk"]


def _rr_pass(rr: float | None, minimum: float) -> bool:
    # Numerical equality tolerance only; not a configurable admission discount.
    return rr is not None and (rr >= minimum or isclose(rr, minimum, rel_tol=0., abs_tol=1e-12))


def _within_stop_limit(value: float | None, maximum: float) -> bool:
    return value is not None and value > 0 and (value <= maximum or isclose(value, maximum, rel_tol=0., abs_tol=1e-12))


def checked_stop_distances(plan: dict, atr: object) -> tuple[float | None, float | None]:
    """Price-derived risk; cached ratios must agree, never grant extra capacity."""
    entry, stop, volatility = (number(value) for value in
                               (plan.get("entry_price"), plan.get("stop_price"), atr))
    if entry is None or stop is None or volatility is None or not 0 < stop < entry or volatility <= 0:
        return None, None
    measured = ((entry - stop) / volatility, (entry - stop) / entry)
    for key, actual in zip(("stop_distance_atr", "stop_distance_fraction"), measured):
        saved = number(plan.get(key))
        if saved is None or not isclose(saved, actual, rel_tol=0., abs_tol=1e-12):
            return None, None
    return measured


def stop_fraction_limit(signal: str, config: LowEntryV6Config) -> float:
    return config.starter_maximum_stop_fraction if signal == "BUY_SMALL" else config.buy_maximum_stop_fraction


def score_setups(f: dict, plan: dict, *, config: LowEntryV6Config | None = None,
                 score_exclusions: frozenset[str] = frozenset()) -> dict:
    config = config or LowEntryV6Config()
    if config.decision_policy_version in {3, 4}:
        raise ValueError("HYBRID_REQUIRES_CONTEXT_FIRST_PIPELINE")
    n = lambda key: _n(f, key)
    b = lambda key: truth(f.get(key))
    above = lambda key: n("close") > n(key)
    rs60 = number(f.get("rs60"))
    trend = n("ema10") > n("ema20") and n("ema20_slope") > 0
    demand = [n("clv") > .4, b("cmf_improving"), n("obv_slope5") > 0,
              n("up_volume_ratio") > 1, above("vwap20")]
    # Only explicit policy 1 evaluates these legacy, setup-specific scales.
    def legacy_execution(setup: str, minimum: float) -> list:
        if setup == "LEADERSHIP_CONTINUATION":
            distance, _ = checked_stop_distances(leader_plan, f.get("atr14"))
            return [(_within_stop_limit(distance, min(config.starter_maximum_stop_atr, 1.8)), 4),
                    (plan["spread_pass"], 2), (plan["amount_pass"], 2), (plan["unrestricted"], 2)]
        base_plan = trade_plan(f, plan["entry_price"], config=config, setup=setup)
        active = base_plan if base_plan["stop_source"] == "CONFIRMED_CONSOLIDATION_LOW" else plan
        distance, _ = checked_stop_distances(active, f.get("atr14"))
        rr = net_reward_risk(active, config)
        return [(_within_stop_limit(distance, min(config.starter_maximum_stop_atr, 1.8)), 4),
                (_rr_pass(rr, minimum), 3), (active["spread_pass"], 1), (active["amount_pass"], 2)]
    retest_plan = trade_plan(f, plan["entry_price"], config=config, setup="LEADERSHIP_CONTINUATION")
    leader_plan = retest_plan if retest_plan["stop_source"] == "CONFIRMED_BREAKOUT_RETEST" else plan
    retest = leader_plan.get("stop_source") == "CONFIRMED_BREAKOUT_RETEST"
    consolidation = _confirmed_consolidation(f)
    stable_low = b("no_new_low2") or b("undercut_reclaim") or consolidation
    reversal = {
        "price_location": [(n("new_lows3") < 2 and -.3 <= n("drawdown60") <= -.1, 5),
                           (n("new_lows3") < 2 and 0 <= n("position60") <= .3, 5)],
        "stabilization": [(b("no_new_low2") or consolidation, 7), (b("higher_low"), 7), (above("ema5"), 5),
                          (n("ema5_slope") > 0, 4), (b("break_previous_high"), 4), (b("atr_stable"), 3)],
        "demand": list(zip(demand, (6, 5, 5, 4, 5))),
        "momentum": [(b("rsi_cross40"), 6), (b("macd_improving"), 5), (n("macd_strict") > n("macd_signal_strict"), 4)],
        "relative_strength": [(b("rs_improving"), 4), (n("rs_slope5") > 0, 3), (b("resilience_improving"), 3)],
    }
    pullback = {
        "primary_trend": [(n("ema10") > n("ema20"), 7), (n("ema20_slope") > 0, 7),
                          (n("ema20_extension") >= -.5, 5), (b("higher_low") and b("support_intact"), 3), (n("rs20") > 0, 3)],
        "pullback_quality": [(-.12 <= n("drawdown20") <= -.03, 5),
                             (min(abs(n("close") - n("ema10")), abs(n("close") - n("ema20"))) <= n("atr14"), 4),
                             (abs(n("close") - n("vwap20")) <= n("atr14"), 3),
                             (b("down_volume_contracts"), 4), (b("atr_stable"), 3),
                             (b("support_intact"), 3), (n("new_lows3") < 2, 3)],
        "relative_strength": [(n("rs20") > 0, 5), (rs60 > 0 if rs60 is not None else None, 5),
                              (n("rs_slope5") >= 0, 5), (n("resilience") > 0, 5)],
        "demand": [(b("reclaim10") or b("reclaim20"), 4), (n("clv") > .3, 3),
                   (b("cmf_improving"), 3), (n("obv_slope5") > 0, 3), (n("up_volume_ratio") >= 1, 2)],
        "momentum": [(b("rsi_pullback"), 2), (b("macd_improving"), 2), (b("break_previous_high"), 1)],
    }
    extension, breakout = n("ema20_extension"), n("breakout_extension")
    extension_points = 7 if extension <= 1 else 4 if extension <= 1.5 else 1 if extension <= 2 else 0
    breakout_points = 8 if 0 <= breakout <= .75 else 5 if .75 < breakout <= 1.25 else 2 if 1.25 < breakout <= 1.5 else 0
    leadership = {
        "relative_strength": [(n("rs20") > 0, 8), (rs60 > 0 if rs60 is not None else None, 7),
                              (b("rs_near_high"), 8), (n("resilience") > 0, 7)],
        "trend": [(n("ema10") > n("ema20"), 7), (n("ema20_slope") > 0, 7), (above("ema20"), 5),
                  (b("higher_high") and b("higher_low") and b("support_intact"), 6)],
        "demand": [(n("clv") > .5, 5), (n("volume_ratio") >= 1.1, 5), (n("amount_ratio") > 1, 3),
                   (n("cmf20") > 0, 3), (n("obv_slope5") > 0, 4)],
        "extension": [(extension_points / 7, 7), (breakout_points / 8, 8)],
    }
    leadership_dependencies = None
    if config.decision_policy_version == 2 and config.leadership_policy_version == 2:
        # Replace existing slots, never add a second bonus budget. Optional
        # industry/correction evidence remains unobserved rather than neutral.
        from domain.strategies.low_entry_v6_score_evidence import DEPENDENCIES
        leadership["relative_strength"][0] = (n("industry_rs20") > 0, 8)
        industry60 = number(f.get("industry_rs60"))
        leadership["relative_strength"][1] = (industry60 > 0 if industry60 is not None else None, 7)
        leadership["relative_strength"][3] = (f.get("downside_resilience"), 7)
        leadership["trend"][3] = (f.get("base_compression"), 6)
        leadership["demand"][0] = (number(f.get("clv_breakout_quality")), 5)
        leadership["demand"][1] = (n("breakout_volume_ratio") >= 1.1, 5)
        leadership["demand"][2] = (n("breakout_amount_ratio") >= 1.2, 3)
        leadership_dependencies = dict(DEPENDENCIES["LEADERSHIP_CONTINUATION"],
            relative_strength=(("industry_rs20",), ("industry_rs60",), ("rs_near_high",), ("downside_resilience",)),
            trend=DEPENDENCIES["LEADERSHIP_CONTINUATION"]["trend"][:3] + (("base_compression",),),
            demand=(("clv_breakout_quality",), ("breakout_volume_ratio",), ("breakout_amount_ratio",),
                    ("cmf20",), ("obv_slope5",)))
    # Absolute price recovery need not already outperform a rising benchmark.
    # This alternative requires stronger base/demand evidence and cannot grant BUY.
    held_higher_floor = (b("no_new_low2") and b("higher_low") and b("support_intact"))
    absolute_recovery = ((consolidation or held_higher_floor) and f.get("market_regime") in {"RISK_ON", "NEUTRAL"}
                         and above("ema5") and above("vwap20") and n("ema5_slope") > 0
                         and b("macd_improving") and n("r5") > 0 and n("clv") > .4
                         and sum(demand) >= 3)
    # A held-floor rebound can precede a higher pivot, VWAP reclaim and
    # benchmark outperformance. Require a held floor, contracting volatility
    # and actual accumulation instead; price location, score and risk still gate it.
    floor_rebound = (f.get("market_regime") in {"RISK_ON", "NEUTRAL"}
                     and b("no_new_low2") and b("support_intact") and above("support")
                     and above("ema5") and n("ema5_slope") > 0 and n("r5") > 0
                     and 0 < n("atr_expansion") <= 1 and b("macd_improving")
                     and n("clv") > .4 and (b("cmf_improving") or n("obv_slope5") > 0))
    strict_valid = (
        (n("drawdown60") <= -.08 or n("position60") <= .3 or b("undercut_reclaim"))
        and stable_low and n("ema5_slope") > 0 and b("atr_stable")
        and any(demand) and (b("rs_improving") or n("rs_slope5") > 0),
        trend and b("support_intact") and -.12 <= n("drawdown20") <= -.03,
        trend and n("rs20") > 0 and n("rs_slope5") >= 0,
    )
    # A recovery is an additional starter-only path, never a replacement for
    # confirmed trend evidence. Missing numbers fail these comparisons closed.
    recovering_trend = (n("ema10") > n("ema20") and n("ema20_slope") >= -config.recovery_ema20_slope_tolerance
                        and n("close") >= n("ema20") and n("ema5_slope") > 0
                        and b("macd_improving") and n("rs_slope5") > 0)
    # An intact longer trend may recover before EMA10 crosses EMA20. This
    # alternative requires an independently confirmed higher floor and price
    # reclaim; it cannot establish the confirmed/full-size BUY tier.
    longer_trend_reclaim = (n("ema20") > n("ema50") and n("ema20_slope") >= 0
                           and held_higher_floor and n("close") >= n("ema20")
                           and above("ema5") and n("ema5_slope") > 0
                           and (b("reclaim5") or b("break_previous_high"))
                           and 0 < n("atr_expansion") <= config.recovery_maximum_atr_expansion)
    # MACD is a lagging confirmation, not the sole proof of recovery. A RSI
    # recovery or strong completed close above yesterday's high can substitute;
    # award no MACD points when it has not actually improved.
    recovery_momentum = (n("ema5_slope") > 0 and above("ema5")
                         and (b("macd_improving") or b("rsi_cross40")
                              or (b("break_previous_high") and n("clv") > .4)))
    # Separate improving medium-horizon leadership from the five-day RS slope:
    # a held near-high range may pause briefly while still leading over 20/60d.
    held_leadership = (trend and held_higher_floor and b("rs_near_high")
                       and n("rs20") > 0 and rs60 is not None and rs60 > 0
                       and b("rs_improving") and n("close") >= n("ema20")
                       and 0 < n("atr_expansion") <= config.recovery_maximum_atr_expansion)
    recovery_checks = (
        {
            "REVERSAL_LOCATION": n("drawdown60") <= -config.reversal_minimum_drawdown or n("position60") <= config.reversal_maximum_position or b("undercut_reclaim"),
            "STABLE_LOW": stable_low,
            "RECOVERY_MOMENTUM": recovery_momentum,
            "CONTROLLED_VOLATILITY": 0 < n("atr_expansion") <= config.recovery_maximum_atr_expansion,
            "SUPPORT_RECOVERED": above("support"),
            "DEMAND_CONFIRMATION": sum(demand) >= 2,
            "RELATIVE_OR_ABSOLUTE_RECOVERY": b("rs_improving") or n("rs_slope5") > 0 or absolute_recovery or floor_rebound,
        },
        {
            "TREND_OR_RECOVERY": trend or recovering_trend or longer_trend_reclaim,
            "INTACT_SUPPORT": b("support_intact"),
            "PULLBACK_DEPTH": -config.pullback_maximum_drawdown <= n("drawdown20") <= -config.pullback_minimum_drawdown,
            "SHALLOW_PULLBACK_LOCATION": (n("drawdown20") <= -.02 or (
                -.5 <= n("ema20_extension") <= 1
                and min(abs(n("close") - n("ema10")), abs(n("close") - n("ema20")),
                        abs(n("close") - n("vwap20"))) <= n("atr14"))),
            "STABLE_LOW": stable_low,
            "DEMAND_CONFIRMATION": sum(demand) >= 2,
            "RECOVERY_MOMENTUM": recovery_momentum,
            "RELATIVE_IMPROVEMENT": n("rs_slope5") > 0 or held_leadership,
        },
        {
            "TREND_OR_RECOVERY": trend or recovering_trend,
            "RELATIVE_LEADERSHIP": n("rs20") > 0 or (n("rs20") >= -config.leadership_rs20_tolerance and rs60 is not None and rs60 > 0 and b("rs_near_high")),
            "RELATIVE_IMPROVEMENT": n("rs_slope5") > 0 or held_leadership,
            "INTACT_SUPPORT": b("support_intact"),
            "DEMAND_CONFIRMATION": sum(demand) >= 2,
            "NOT_EXTENDED": n("ema20_extension") <= 1 and n("breakout_extension") <= 1.25,
        },
    )
    # Shape recognition is distinct from entry eligibility. A held low-zone
    # base or healthy trend pullback need not have today's buying trigger yet.
    # Preserve `valid` for executable setup eligibility; FORMING never enters
    # the BUY/BUY_SMALL evaluator and cannot hide an eligible setup.
    formation_base = (b("history_ready") and b("market_ready") and stable_low
                      and n("atr14") > 0 and n("volume") > 0
                      and b("support_intact") and 0 < n("support") <= n("low") < n("close")
                      and 0 < n("atr_expansion") <= config.recovery_maximum_atr_expansion
                      and n("new_lows3") < 2)
    recognition_checks = (
        {
            "HELD_SUPPORT": formation_base,
            "LOW_ZONE": recovery_checks[0]["REVERSAL_LOCATION"],
            "NEAR_SHORT_TREND": n("close") >= n("ema5") - .5 * n("atr14"),
            "BASE_EVIDENCE": b("higher_low") or b("macd_improving") or b("down_volume_contracts"),
        },
        {
            "HELD_SUPPORT": formation_base,
            "PRIMARY_TREND": trend or (n("ema20") > n("ema50") and n("ema20_slope") >= 0),
            "PULLBACK_DEPTH": recovery_checks[1]["PULLBACK_DEPTH"],
            "NEAR_PRIMARY_TREND": -.5 <= n("ema20_extension") <= 1,
        },
        {
            "HELD_SUPPORT": formation_base,
            "PRIMARY_TREND": trend and n("close") >= n("ema20"),
            "MEDIUM_TERM_LEADERSHIP": n("rs20") > 0 and rs60 is not None and rs60 > 0 and b("rs_near_high"),
            "NOT_EXTENDED": recovery_checks[2]["NOT_EXTENDED"],
        },
    )
    result = {}
    for name, groups, confirmed, checks, shape in zip(SETUPS, (reversal, pullback, leadership), strict_valid, recovery_checks, recognition_checks):
        recovered = all(checks.values())
        base_plan = trade_plan(f, plan["entry_price"], config=config, setup=name)
        execution_policy = None
        if config.decision_policy_version == 2:
            groups["execution"], execution_policy = execution_conditions(
                f, base_plan, distances=checked_stop_distances(base_plan, f.get("atr14")), config=config)
            validate_score_budget(groups)
        else:
            groups["execution"] = legacy_execution(name, 1.8 if name == "TREND_PULLBACK" else 2.)
        evidence = score_evidence(name, groups, f, base_plan,
            execution_dependencies=EXECUTION_DEPENDENCIES if execution_policy is not None else None,
            setup_dependencies=leadership_dependencies if name == "LEADERSHIP_CONTINUATION" else None,
            excluded_channels=score_exclusions)
        earned = {key: value["earned_points"] for key, value in evidence["components"].items()}
        starter_only = ((retest and name == "LEADERSHIP_CONTINUATION")
                        or base_plan["stop_source"] == "CONFIRMED_CONSOLIDATION_LOW"
                        or (name == "REVERSAL" and consolidation
                            and not (b("no_new_low2") or b("undercut_reclaim"))))
        recognized = confirmed or recovered or all(shape.values())
        result[name] = {"score": evidence["formal_score"],
                        "formal_score": evidence["formal_score"],
                        "estimated_score": evidence["estimated_score"],
                        "score_coverage": evidence["coverage_percent"],
                        "valid": confirmed or recovered, "strict_valid": confirmed and not starter_only,
                        "recognized": recognized, "recognition_checks": shape,
                        "entry_tier": "CONFIRMED" if confirmed and not starter_only else "STARTER_ONLY" if confirmed or recovered else "FORMING" if recognized else "NONE",
                        "recovery_checks": checks,
                        "blocking_conditions": [] if confirmed or recovered else [key for key, passed in checks.items() if not passed],
                        "components": earned, "available_weight": evidence["available_weight"],
                        "score_evidence": evidence}
        if execution_policy is not None:
            result[name]["execution_score"] = dict(execution_policy, **evidence["components"]["execution"])
    return result


def triggers(setup: str, f: dict) -> tuple[bool, bool, int]:
    n = lambda key: _n(f, key)
    b = lambda key: truth(f.get(key))
    demand = sum([n("clv") > .4, b("cmf_improving"), n("obv_slope5") > 0,
                  n("up_volume_ratio") > 1, n("close") > n("vwap20")])
    if setup == "REVERSAL":
        improvements = sum([n("clv") > .4, b("cmf_improving"), n("obv_slope5") > 0,
                            b("rsi_cross40"), b("macd_improving"), n("rs_slope5") > 0])
        starter = (b("no_new_low2") or b("undercut_reclaim") or _confirmed_consolidation(f)) and n("ema5_slope") > 0 and improvements >= 2
        confirmed = starter and (b("higher_low") or b("undercut_reclaim")) and b("break_previous_high") and demand >= 2 and (b("rs_improving") or n("rs_slope5") > 0)
    elif setup == "TREND_PULLBACK":
        starter = b("support_intact") and (n("close") >= n("ema20") or b("reclaim10")) and demand >= 1
        confirmed = starter and (b("reclaim10") or b("break_previous_high")) and (n("rs20") >= 0 or (b("rs_improving") and n("rs_slope5") > 0))
    else:
        # Peak-to-current resilience is not the same as relative leadership.
        # A controlled pullback may have negative resilience while outperforming
        # over both horizons. Require independent evidence, not just a high score.
        relative_leader = (n("rs20") > 0 and n("rs60") > 0 and n("rs_slope5") > 0
                           and b("rs_near_high") and b("support_intact") and b("no_new_low2"))
        starter = (n("resilience") > 0 or relative_leader) and n("ema20_extension") <= 1 and n("breakout_extension") <= 1.25 and (n("clv") > .3 or n("obv_slope5") > 0 or n("cmf20") > 0)
        held_retest = (_confirmed_retest(f) and b("support_intact") and n("ema10") > n("ema20")
                       and n("ema20_slope") > 0 and n("rs20") > 0 and n("rs_slope5") >= 0
                       and n("ema20_extension") <= 1.5 and n("breakout_extension") <= 1.25 and demand >= 2)
        starter = starter or held_retest
        confirmed = (b("break_previous_high") or n("close") > n("hh20")) and demand >= 2 and n("breakout_extension") <= 1.25
        # A confirmed breakout is also sufficient timing evidence for a smaller
        # entry. Risk/extension, Quality and sizing are still checked separately.
        starter = starter or confirmed
    return starter, confirmed, demand


def _leader_location_pass(f: dict, plan: dict) -> bool:
    extension_limit = 1.5 if plan.get("stop_source") == "CONFIRMED_BREAKOUT_RETEST" else 1.
    return (_n(f, "ema20_extension") <= extension_limit and _n(f, "breakout_extension") <= 1.25
            and truth(f.get("support_intact")))


def risk_pass(setup: str, signal: str, f: dict, plan: dict, valuation: str, config: LowEntryV6Config) -> tuple[bool, float, float | None]:
    stop_limit = config.starter_maximum_stop_atr if signal == "BUY_SMALL" else config.buy_maximum_stop_atr
    risk_off = f.get("market_regime") == "RISK_OFF"
    if risk_off:
        # Preserve the pre-relaxation stress budget for both entry tiers.
        stop_limit = min(stop_limit, 1.8 if signal == "BUY_SMALL" else 1.5)
        stop_limit *= config.risk_off_stop_multiplier
    minimum_rr = 2. if setup == "REVERSAL" else 1.8
    if signal == "BUY_SMALL":
        minimum_rr = max(minimum_rr, config.starter_minimum_net_rr) if risk_off else config.starter_minimum_net_rr
    if valuation == "EXPENSIVE":
        minimum_rr += config.expensive_rr_increment
    distance, fraction = checked_stop_distances(plan, f.get("atr14"))
    stop_ok = _within_stop_limit(distance, stop_limit)
    stop_ok = stop_ok and _within_stop_limit(fraction, stop_fraction_limit(signal, config))
    if setup == "LEADERSHIP_CONTINUATION" and plan["target_price"] is None:
        # No invented resistance: stricter extension and support requirements.
        valid = stop_ok and _leader_location_pass(f, plan)
        return valid, stop_limit, None
    # A confirmed breakout can qualify for a starter, but nearby resistance is
    # not permission to chase. Preserve the original starter extension limits.
    if setup == "LEADERSHIP_CONTINUATION" and signal == "BUY_SMALL" and not _leader_location_pass(f, plan):
        return False, stop_limit, minimum_rr
    return stop_ok and _rr_pass(net_reward_risk(plan, config), minimum_rr), stop_limit, minimum_rr


def _starter_threshold(setup: str, config: LowEntryV6Config) -> float:
    return dict(zip(SETUPS, (config.reversal_starter_score_minimum, config.pullback_starter_score_minimum,
                            config.leadership_starter_score_minimum)))[setup]


def technical_observation(quality: QualityResult, f: dict, decision: dict, *, valuation: str,
                          config: LowEntryV6Config) -> dict:
    """Retain internal technical metadata without authorizing entry or rendering it.

    Quality/liquidity early returns must not hide relative strength or downstream
    risk failures in stored audit evidence. Observations have no executable
    signal or position size; the email renderer ignores this legacy metadata.
    """
    result = {"patterns": [], "is_entry_signal": False, "blocking_reasons": [], "checks": {}}
    if (not truth(f.get("history_ready")) or not truth(f.get("market_ready"))
            or not _n(f, "volume") > 0 or not _n(f, "close") > 0
            or not _n(f, "amount_ma20") >= config.minimum_amount_ma20):
        return result
    if _n(f, "rs20") >= .03 and _n(f, "close") > _n(f, "ema20") and _n(f, "ema20_slope") > 0:
        result["patterns"].append("RELATIVE_STRENGTH")
    if (decision["setups"].get("REVERSAL", {}).get("valid") and _n(f, "ema5_slope") > 0
            and _n(f, "close") > _n(f, "ema5")):
        result["patterns"].append("LOW_ZONE_RECOVERY")
    if not result["patterns"] or decision["signal"] in EXECUTABLE_SIGNALS:
        return result
    blockers = list(decision["reasons"]) + list(decision["hard_vetoes"])
    plan, setup = decision["plan"], decision["setup"]
    if not plan["spread_pass"]:
        blockers.append("MISSING_TWO_SIDED_QUOTE" if number(f.get("spread")) is None else "SPREAD_EXCEEDS_LIMIT")
    if not plan["unrestricted"]:
        blockers.append("TRADING_RESTRICTION" if truth(f.get("restricted")) else "TRADING_RESTRICTION_UNKNOWN")
    if setup is not None:
        starter, _, _ = triggers(setup, f)
        passed, stop_limit, minimum_rr = risk_pass(setup, "BUY_SMALL", f, plan, valuation, config)
        threshold = _starter_threshold(setup, config) + (config.risk_off_score_increment if f.get("market_regime") == "RISK_OFF" else 0)
        distance, rr = number(plan["stop_distance_atr"]), net_reward_risk(plan, config)
        if decision["score"] is None or decision["score"] < threshold:
            blockers.append("SCORE_BELOW_STARTER")
        if quality.status == "PASS" and (quality.score is None or quality.score < MINIMUM_ENTRY_QUALITY):
            blockers.append("QUALITY_BELOW_ENTRY_MINIMUM")
        if not starter:
            blockers.append("TRIGGER_NOT_CONFIRMED")
        if distance is None:
            blockers.append("NO_VALID_STRUCTURAL_STOP")
        elif not _within_stop_limit(distance, stop_limit):
            blockers.append("STOP_DISTANCE_EXCEEDS_LIMIT")
        fraction = number(plan.get("stop_distance_fraction"))
        if not _within_stop_limit(fraction, config.starter_maximum_stop_fraction):
            blockers.append("STOP_PERCENTAGE_EXCEEDS_LIMIT")
        if minimum_rr is not None and not _rr_pass(rr, minimum_rr):
            blockers.append("RR_BELOW_MINIMUM")
        elif minimum_rr is None and not passed and _within_stop_limit(distance, stop_limit):
            blockers.append("PRICE_DISCOVERY_EXTENSION_OR_SUPPORT_FAIL")
        if setup == "LEADERSHIP_CONTINUATION":
            extension_limit = 1.5 if plan.get("stop_source") == "CONFIRMED_BREAKOUT_RETEST" else 1.
            if _n(f, "ema20_extension") > extension_limit or _n(f, "breakout_extension") > 1.25:
                blockers.append("WAIT_FOR_CONTROLLED_RETEST")
        result["checks"] = {"score": decision["score"], "required_score": threshold, "starter_trigger": starter,
                            "stop_distance_atr": distance, "maximum_stop_atr": stop_limit,
                            "reward_risk": plan["reward_risk"], "net_reward_risk": rr,
                            "rr_basis": "NET_OF_CONFIGURED_COSTS", "minimum_rr": minimum_rr, "risk_pass": passed}
    result["blocking_reasons"] = list(dict.fromkeys(blockers))
    return result


def decide(quality: QualityResult, f: dict, *, valuation: str = "NOT_AVAILABLE", config: LowEntryV6Config,
           entry_price: float | None = None, industry_mode: str = "ACTIVE",
           convex_weights: dict | None = None, as_of: str | None = None) -> dict:
    if convex_weights is not None and config.decision_policy_version != 3:
        raise ValueError("CONVEX_WEIGHTS_REQUIRE_HYBRID_POLICY")
    if config.decision_policy_version in {3, 4}:
        from domain.strategies.low_entry_v6_hybrid import decide_hybrid
        result = decide_hybrid(quality, f, valuation=valuation, config=config,
                               entry_price=entry_price, industry_mode=industry_mode,
                               convex_weights=convex_weights, as_of=as_of)
        result["technical_observation"] = dict(patterns=[], is_entry_signal=False,
            blocking_reasons=list(result["reasons"]), checks={"context": result["setup"]})
        return result
    if config.decision_policy_version == 2:
        from domain.strategies.low_entry_v6_decision import decide_unified
        result = decide_unified(quality, f, valuation=valuation, config=config,
                                entry_price=entry_price, industry_mode=industry_mode)
    else:
        # Explicit compatibility policy for retained v1 callers; daily defaults
        # use v2. Never silently upgrade an archived score or decision.
        result = _decide(quality, f, valuation=valuation, config=config, entry_price=entry_price)
    result["technical_observation"] = technical_observation(quality, f, result, valuation=valuation, config=config)
    return result


def _decide(quality: QualityResult, f: dict, *, valuation: str, config: LowEntryV6Config, entry_price: float | None = None) -> dict:
    plan = trade_plan(f, entry_price, config=config)
    setups = score_setups(f, plan, config=config)
    candidates = [(name, values) for name, values in setups.items() if values["valid"]]
    recognized = [(name, values) for name, values in setups.items() if values.get("recognized")]
    selectable = candidates or recognized
    ranking_score = lambda values: number(values["score"]) if number(values["score"]) is not None else -1.
    selected, best = max(selectable, key=lambda item: ranking_score(item[1])) if selectable else (None, None)
    # A setup's measured score exists even when its shape is ineligible. Keep
    # the scoring reference separate from `setup`, which controls admission.
    # Eligible/forming setups retain their own score, not a higher rejected one.
    score_setup, measured = (selected, best) if selected else max(setups.items(), key=lambda item: ranking_score(item[1]))
    score_ready = truth(f.get("history_ready")) and truth(f.get("market_ready")) and valid_price_anchor(f)
    score_basis = ("SELECTED_SETUP" if candidates else "FORMING_SETUP" if selected else "UNQUALIFIED_SETUP")
    if not score_ready:
        # Use the most supported reference, not the largest ratio from a tiny
        # denominator. Fixed setup order breaks ties deterministically.
        score_setup, measured = max(setups.items(), key=lambda item: (
            item[1]["score_evidence"]["earned_points"], item[1]["score_evidence"]["available_weight"]))
    evidence = measured["score_evidence"]
    display_score = measured["score"] if score_ready else evidence["estimated_score"]
    vetoes = hard_vetoes(f)
    result = {"signal": "AVOID", "setup": selected, "score": display_score, "setups": setups,
              "formal_score": evidence["formal_score"], "estimated_score": evidence["estimated_score"],
              "score_coverage": evidence["coverage_percent"],
              "score_setup": score_setup if display_score is not None else None,
              "score_basis": score_basis if score_ready else "PARTIAL_DATA" if display_score is not None else "INSUFFICIENT_HISTORY",
              "score_evidence": evidence,
              "plan": trade_plan(f, entry_price, config=config, setup=selected),
              "hard_vetoes": list(vetoes), "reasons": [], "risk_gate": "NOT_EVALUATED",
              "starter_trigger": False, "confirmed_trigger": False, "risk_fraction": 0.,
              "position_multiplier": 1., "required_score": None,
              "entry_checks": {},
              "entry_tier": best["entry_tier"] if best else "NONE"}

    def finish(signal: str, reason: str):
        result.update(signal=signal, reasons=[reason])
        return result

    if quality.integrity == "FAIL":
        return finish("NO_BUY", ";".join(quality.reasons))
    if quality.status == "REJECT":
        return finish("NO_BUY", "QUALITY_REJECT")
    if quality.status == "DATA_INSUFFICIENT":
        return finish("DATA_INSUFFICIENT", "QUALITY_DATA_INSUFFICIENT")
    if quality.status != "PASS":
        return finish("WATCH_ONLY", quality.status)
    quality_score = number(quality.score)
    if quality_score is None:
        return finish("DATA_INSUFFICIENT", "QUALITY_SCORE_MISSING")
    if not MINIMUM_ENTRY_QUALITY <= quality_score <= 100:
        return finish("WATCH_ONLY", "QUALITY_BELOW_ENTRY_MINIMUM")
    if not truth(f.get("history_ready")) or not truth(f.get("market_ready")) or number(f.get("rs_slope5")) is None:
        return finish("DATA_INSUFFICIENT", "INCOMPLETE_ALIGNED_HISTORY")
    if vetoes:
        return finish("NO_BUY", ";".join(vetoes))
    if not valid_price_anchor(f):
        return finish("DATA_INSUFFICIENT", "INVALID_CURRENT_OHLCV")
    if not veto_evidence_complete(f):
        return finish("DATA_INSUFFICIENT", "INCOMPLETE_VETO_EVIDENCE")
    if selected is None:
        return finish("AVOID", "NO_VALID_SETUP")
    if not candidates:
        # This is a recognized structure, not a pending buy order. Do not let a
        # high numeric score turn missing entry confirmation into BUY_SMALL.
        result["position_multiplier"] = 0.
        return finish("WATCH" if result["score"] >= 55 else "AVOID", "SETUP_FORMING_WAIT_CONFIRMATION")
    # Evaluate each eligible setup's trigger and risk separately. A high-scoring
    # but non-executable setup must not hide another genuinely qualified one.
    outcomes = [_decide_setup(dict(result, setup=name, score=values["score"], formal_score=values["formal_score"], estimated_score=values["estimated_score"], score_coverage=values["score_coverage"], score_setup=name, score_basis="SELECTED_SETUP", score_evidence=values["score_evidence"], entry_tier=values["entry_tier"], entry_checks={},
                                  plan=trade_plan(f, entry_price, config=config, setup=name)),
                              quality, f, valuation=valuation, config=config)
                for name, values in candidates]
    executable = [outcome for outcome in outcomes if outcome["signal"] in EXECUTABLE_SIGNALS]
    for outcome in outcomes:
        outcome["comparison"] = setup_comparison(outcome)
    # Admission is evaluated BEFORE comparison; no rank can upgrade a signal.
    chosen = max(executable, key=setup_rank) if executable else max(outcomes, key=lambda outcome: ranking_score(outcome))
    chosen["setup_decisions"] = {outcome["setup"]: {key: outcome[key] for key in
                                ("signal", "score", "formal_score", "estimated_score", "score_coverage",
                                 "entry_tier", "risk_gate", "reasons", "required_score", "entry_checks", "comparison")}
                                for outcome in outcomes}
    return chosen


def setup_comparison(outcome: dict) -> dict:
    """Common risk/evidence axes, not a calibrated cross-setup probability."""
    plan = outcome["plan"]
    check = outcome.get("entry_checks", {}).get(outcome["signal"], {})
    rr, minimum = number(plan.get("net_reward_risk")), number(check.get("minimum_rr"))
    return {"policy_version": "coverage-net-risk-surplus/v1",
            "coverage_percent": outcome["score_evidence"]["coverage_percent"],
            "stop_fraction": number(plan.get("stop_distance_fraction")),
            "net_rr": rr, "required_net_rr": minimum,
            "net_rr_surplus": rr - minimum if rr is not None and minimum is not None else None,
            "rr_status": "KNOWN" if rr is not None and minimum is not None else "PRICE_DISCOVERY_OR_UNKNOWN",
            "allocation_value_status": "NOT_EVALUATED_USE_PAE"}


def setup_rank(outcome: dict) -> tuple:
    comparison = outcome["comparison"]
    surplus, stop = comparison["net_rr_surplus"], comparison["stop_fraction"]
    return (outcome["signal"] == "BUY", comparison["coverage_percent"],
            surplus is not None, surplus if surplus is not None else 0.,
            -stop if stop is not None else float("-inf"))


def _decide_setup(result: dict, quality: QualityResult, f: dict, *, valuation: str, config: LowEntryV6Config) -> dict:
    selected, plan = result["setup"], result["plan"]
    best = result["setups"][selected]

    def finish(signal: str, reason: str):
        result.update(signal=signal, reasons=[reason])
        return result

    score = result["score"]
    if not best["valid"] or result["entry_tier"] not in {"CONFIRMED", "STARTER_ONLY"}:
        result.update(risk_fraction=0., position_multiplier=0.)
        return finish("WATCH" if score >= 55 else "AVOID", "SETUP_FORMING_WAIT_CONFIRMATION")
    if score < 55:
        return finish("AVOID", "SCORE_BELOW_55")
    if score < 65:
        return finish("WATCH", "SCORE_BELOW_65")
    small_threshold = _starter_threshold(selected, config)
    if score < small_threshold:
        return finish("ARMED", "SCORE_BELOW_STARTER")
    if plan["stop_distance_atr"] is None:
        return finish("SKIP", "NO_VALID_STRUCTURAL_STOP")
    if not plan["liquidity_pass"]:
        return finish("SKIP", "LIQUIDITY_OR_RESTRICTION_FAIL")
    if selected == "LEADERSHIP_CONTINUATION" and _n(f, "breakout_extension") > 1.5:
        return finish("WAIT_RETEST", "OVEREXTENDED_BREAKOUT")
    starter, confirmed, demand = triggers(selected, f)
    result.update(starter_trigger=starter, confirmed_trigger=confirmed)
    risk_off = f.get("market_regime") == "RISK_OFF"
    if risk_off and not (_n(f, "rs20") > 0 and _n(f, "rs_slope5") > 0 and _n(f, "resilience") > 0 and demand >= 3):
        return finish("ARMED", "RISK_OFF_CONFIRMATION_INSUFFICIENT")
    extra = config.risk_off_score_increment if risk_off else 0
    for signal, threshold, trigger in (("BUY", config.buy_score_minimum, confirmed), ("BUY_SMALL", small_threshold, starter)):
        passed, stop_limit, rr_min = risk_pass(selected, signal, f, plan, valuation, config)
        blockers = []
        if signal == "BUY" and not best["strict_valid"]:
            blockers.append("STARTER_ONLY_SETUP")
        if score < threshold + extra:
            blockers.append("SCORE_BELOW_THRESHOLD")
        if not trigger:
            blockers.append("TRIGGER_NOT_CONFIRMED")
        eligible = not blockers
        distance, rr = number(plan["stop_distance_atr"]), net_reward_risk(plan, config)
        if not _within_stop_limit(distance, stop_limit):
            blockers.append("STOP_DISTANCE_EXCEEDS_LIMIT")
        fraction = number(plan.get("stop_distance_fraction"))
        if not _within_stop_limit(fraction, stop_fraction_limit(signal, config)):
            blockers.append("STOP_PERCENTAGE_EXCEEDS_LIMIT")
        if signal == "BUY_SMALL":
            if selected == "LEADERSHIP_CONTINUATION" and not _leader_location_pass(f, plan):
                blockers.append("LEADERSHIP_EXTENSION_OR_SUPPORT_FAIL")
        if rr_min is not None and not _rr_pass(rr, rr_min):
            blockers.append("RR_BELOW_MINIMUM")
        if rr_min is None and not passed and _within_stop_limit(distance, stop_limit):
            blockers.append("PRICE_DISCOVERY_EXTENSION_OR_SUPPORT_FAIL")
        result["entry_checks"][signal] = {
            "score": score, "required_score": threshold + extra, "trigger": trigger,
            "stop_distance_atr": distance, "maximum_stop_atr": stop_limit,
            "stop_distance_fraction": plan.get("stop_distance_fraction"),
            "maximum_stop_fraction": stop_fraction_limit(signal, config),
            "reward_risk": plan["reward_risk"], "net_reward_risk": rr,
            "rr_basis": "NET_OF_CONFIGURED_COSTS", "minimum_rr": rr_min, "risk_pass": passed,
            "blockers": blockers,
        }
        if eligible:
            result.update(risk_gate="PASS" if passed else "FAIL", maximum_stop_atr=stop_limit,
                          minimum_rr=rr_min, required_score=threshold + extra)
            if passed:
                multiplier = config.risk_off_position_multiplier if risk_off else 1.
                if valuation == "EXPENSIVE":
                    multiplier *= config.expensive_position_multiplier
                result.update(risk_fraction=config.small_risk_fraction if signal == "BUY_SMALL" else config.buy_risk_fraction,
                              position_multiplier=multiplier)
                return finish(signal, "QUALITY_SETUP_TRIGGER_RISK_LIQUIDITY_PASS")
    finish("ARMED", "TRIGGER_SCORE_OR_RISK_INSUFFICIENT")
    result["reasons"].extend(f"BUY_SMALL:{reason}" for reason in result["entry_checks"]["BUY_SMALL"]["blockers"])
    return result


def size_position(decision: dict, *, portfolio_value: object, available_cash: object,
                  intended_weight: object, portfolio_capacity_weight: object,
                  amount_ma20: object, config: LowEntryV6Config) -> int:
    """All caps constrain incremental capital. Missing portfolio evidence means zero."""
    if decision["signal"] not in EXECUTABLE_SIGNALS:
        return 0
    inputs = [number(v) for v in (portfolio_value, available_cash, intended_weight, portfolio_capacity_weight, amount_ma20)]
    if any(v is None or v <= 0 for v in inputs):
        return 0
    equity, cash, intended, capacity, adv = inputs
    if intended > 1 or capacity > 1:
        return 0
    plan = decision["plan"]
    entry, stop = number(plan["entry_price"]), number(plan["stop_price"])
    if entry is None or stop is None or not 0 < stop < entry:
        return 0
    starter = config.starter_fraction if decision["signal"] == "BUY_SMALL" else 1.
    multiplier, risk_fraction = (number(decision.get(key)) for key in ("position_multiplier", "risk_fraction"))
    if multiplier is None or not 0 < multiplier <= 1 or risk_fraction is None or not 0 < risk_fraction <= 1:
        return 0
    payoff = payoff_scenarios(plan, buy_cost_rate=config.buy_fee_rate, sell_cost_rate=config.sell_cost_rate)
    net_risk, entry_cost = payoff["net_risk_per_share"], payoff["entry_cost_per_share"]
    if net_risk is None or net_risk <= 0 or entry_cost is None:
        return 0
    budget = min(cash, equity * config.maximum_position_weight, equity * capacity,
                 equity * intended * starter * multiplier, adv * config.maximum_order_adv)
    # Use the same after-cost loss as RR, including sale costs at the stop.
    # Neither this scenario budget nor the stop guarantees a fill during a gap.
    return max(0, floor(min(equity * risk_fraction * multiplier / net_risk, budget / entry_cost)))


def evaluate_exit(f: dict, *, initial_stop: float | None, previous_trailing_stop: float | None = None,
                  highest_close_since_entry: float | None = None, hard_risk: bool = False,
                  holding_sessions: int | None = None, mfe_r: float | None = None) -> dict:
    """Optional v6 exit assessment; does not overwrite the existing EREM/PAE engines."""
    close, atr = number(f.get("close")), number(f.get("atr14"))
    trailing = previous_trailing_stop
    if highest_close_since_entry is not None and atr is not None and atr > 0:
        trailing = max(previous_trailing_stop or 0, highest_close_since_entry - 2.5 * atr)
    stops = [number(v) for v in (initial_stop, trailing, f.get("support"))]
    if hard_risk:
        signal, reason = "EXIT", "HARD_FUNDAMENTAL_RISK"
    elif close is None:
        signal, reason = "NO_ACTION", "MISSING_FRESH_PRICE"
    elif any(stop is not None and stop > 0 and close < stop for stop in stops):
        signal, reason = "EXIT", "STRUCTURAL_STOP"
    elif truth(f.get("restricted")):
        signal, reason = "EXIT_REVIEW", "TRADING_RESTRICTION_NO_ASSUMED_FILL"
    elif _n(f, "cmf20") < 0 and _n(f, "obv_slope5") < 0 and truth(f.get("down_volume_expands")):
        signal, reason = "REDUCE", "DEMAND_FAILURE"
    elif (close < _n(f, "ema20") and _n(f, "ema20_slope") < 0 and _n(f, "rs_slope5") < 0) or (_n(f, "ema5") < _n(f, "ema10") < _n(f, "ema20") and _n(f, "rs20") < 0):
        signal, reason = "EXIT", "TREND_FAILURE"
    elif holding_sessions is not None and holding_sessions >= 10 and mfe_r is not None and mfe_r < .5 and _n(f, "rs_slope5") <= 0:
        signal, reason = "REDUCE", "TIME_STOP"
    else:
        signal, reason = "HOLD", "NO_CONFIRMED_EXIT"
    return {"signal": signal, "reason": reason, "trailing_stop": trailing}

"""Opt-in recovery/base-breakout hypotheses; never a capital-allocation signal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import pandas as pd

from domain.features import FeatureFrame, SharedFeaturePipeline
from domain.strategy import Strategy
from domain.strategies.low_entry_score_v6 import LowEntryScoreV6Strategy, _clean
from domain.strategies.low_entry_v6_features import build_features
from domain.strategies.low_entry_v6_payoff import payoff_scenarios
from domain.strategies.low_entry_v6_quality import number
from domain.strategies.low_entry_v6_signals import LowEntryV6Config


RESEARCH_COLUMN = "Recovery Breakout Research"
RESEARCH_VERSION = "2026-09-25-recovery-base-v1"


@dataclass(frozen=True)
class RecoveryBreakoutConfig:
    """Uncalibrated research assumptions, identical for every symbol/sector."""

    correction_window: int = 60
    base_window: int = 10
    stabilization_window: int = 3
    min_correction: float = .12
    min_rebound: float = .05
    max_base_width_atr: float = 3.
    max_extension_atr: float = 2.
    breakout_volume_ratio: float = 1.2
    min_amount_ma20: float = 30_000_000.
    stop_buffer_atr: float = .3

    def __post_init__(self):
        for key in ("correction_window", "base_window", "stabilization_window"):
            value = getattr(self, key)
            if type(value) is not int or value < 2:
                raise ValueError(f"{key} must be an integer >= 2")
        if self.base_window > self.correction_window:
            raise ValueError("base_window must not exceed correction_window")
        for key, value in asdict(self).items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.min_correction >= 1 or self.min_rebound >= 1:
            raise ValueError("correction/rebound must be fractions below 1")


def build_research_features(f: pd.DataFrame, config: RecoveryBreakoutConfig) -> pd.DataFrame:
    """Freeze range and ATR before the signal bar; no future highs or pivots."""
    result = f.copy(deep=True)
    c, h, l = (pd.to_numeric(f[key], errors="coerce") for key in ("close", "high", "low"))
    prior = c.shift(1).rolling(config.correction_window)

    def drawdowns(values):
        return values / np.maximum.accumulate(values) - 1

    result["prior_max_drawdown"] = prior.apply(lambda v: drawdowns(v).min(), raw=True)
    result["correction_trough_close"] = prior.apply(lambda v: v[drawdowns(v).argmin()], raw=True)
    result["rebound_from_trough"] = c / result["correction_trough_close"] - 1
    result["prior_floor"] = l.shift(1).rolling(config.correction_window).min()
    result["base_floor"] = l.shift(1).rolling(config.base_window).min()
    result["base_ceiling"] = h.shift(1).rolling(config.base_window).max()
    # Freeze volatility before the entire base, so a wide base cannot inflate
    # its own allowance and today's gap cannot manufacture consolidation.
    result["base_atr"] = f["atr14"].shift(config.base_window + 1)
    result["base_width_atr"] = (result["base_ceiling"] - result["base_floor"]) / result["base_atr"]
    result["prior_volume_ma20"] = f["volume"].shift(1).rolling(20).mean()
    result["research_volume_ratio"] = f["volume"] / result["prior_volume_ma20"]
    result["prior_atr"] = f["atr14"].shift(1)
    prior_floor20 = l.shift(1).rolling(20).min()
    held = (l >= prior_floor20) & prior_floor20.notna()
    result["held_floor"] = held.rolling(config.stabilization_window).sum().eq(config.stabilization_window)
    o, v = (pd.to_numeric(f[key], errors="coerce") for key in ("open", "volume"))
    valid = (np.isfinite(pd.concat([o, h, l, c, v], axis=1)).all(axis=1)
             & (l > 0) & (v > 0) & (h >= pd.concat([o, c], axis=1).max(axis=1))
             & (l <= pd.concat([o, c], axis=1).min(axis=1)))
    history_size = max(60, config.correction_window + 1, config.base_window + 15,
                       config.stabilization_window + 20)
    result["research_history_ready"] = valid.rolling(history_size).sum().eq(history_size)
    benchmark = pd.to_numeric(f["taiex"], errors="coerce")
    result["research_market_ready"] = (np.isfinite(benchmark) & benchmark.gt(0)).rolling(21).sum().eq(21)
    return result


def evaluate_research(f: dict, production: dict, config: RecoveryBreakoutConfig) -> dict:
    """Separate pattern recognition from evidence readiness and live eligibility."""
    required = ("close", "high", "low", "previous_high", "ema5_slope", "ema20", "clv",
                "rs20", "prior_atr", "prior_max_drawdown", "rebound_from_trough",
                "prior_floor", "base_floor", "base_ceiling", "base_atr", "base_width_atr",
                "research_volume_ratio", "amount_ma20")
    missing = [key for key in required if number(f.get(key)) is None]
    if not f.get("research_history_ready"):
        missing.append("COMPLETE_RESEARCH_HISTORY")
    if not f.get("research_market_ready"):
        missing.append("ALIGNED_BENCHMARK_HISTORY")
    if number(f.get("prior_atr")) is not None and f["prior_atr"] <= 0:
        missing.append("POSITIVE_ATR")
    if number(f.get("base_atr")) is not None and f["base_atr"] <= 0:
        missing.append("POSITIVE_BASE_ATR")
    patterns = {}
    levels = None
    if not missing:
        extension = (f["close"] - f["ema20"]) / f["prior_atr"]
        common = {
            "relative_strength_nonnegative": f["rs20"] >= 0,
            "above_ema20_without_chasing": 0 <= extension <= config.max_extension_atr,
            "liquid_history": f["amount_ma20"] >= config.min_amount_ma20,
        }
        recovery = dict(common, correction_observed=f["prior_max_drawdown"] <= -config.min_correction,
                        rebound_observed=f["rebound_from_trough"] >= config.min_rebound,
                        floor_held=bool(f["held_floor"]), short_trend_improving=f["ema5_slope"] > 0)
        recovery_trigger = (f["close"] > f["previous_high"] and f["clv"] >= .25
                            and f["research_volume_ratio"] >= 1.)
        base = dict(common, compact_prior_base=0 < f["base_width_atr"] <= config.max_base_width_atr,
                    floor_held=f["low"] >= f["base_floor"],
                    near_or_above_base_midpoint=f["close"] >= (f["base_floor"] + f["base_ceiling"]) / 2)
        base_trigger = (f["close"] > f["base_ceiling"] and f["clv"] >= .25
                        and f["research_volume_ratio"] >= config.breakout_volume_ratio)
        for name, checks, trigger in (("PRICE_RECOVERY", recovery, recovery_trigger),
                                      ("BASE_BREAKOUT", base, base_trigger)):
            matched = all(checks.values())
            patterns[name] = {"state": "CONFIRMED" if matched and trigger else "FORMING" if matched else "NOT_DETECTED",
                              "checks": checks, "trigger_confirmed": bool(trigger)}
        selected = next((key for state in ("CONFIRMED", "FORMING") for key in ("BASE_BREAKOUT", "PRICE_RECOVERY")
                         if patterns[key]["state"] == state), None)
        if selected:
            floor = f["base_floor"] if selected == "BASE_BREAKOUT" else f["prior_floor"]
            # This is an observed range floor, NOT a newly confirmed swing low.
            stop = floor - config.stop_buffer_atr * f["prior_atr"]
            resistance = [number(f.get(k)) for k in ("swing_high", "hh20", "hh60")]
            resistance = [value for value in resistance if value is not None and value > f["close"]]
            levels = {"setup": selected, "entry_price": f["close"], "stop_price": stop if 0 < stop < f["close"] else None,
                      "target_price": min(resistance) if resistance else None,
                      "basis": "SIGNAL_CLOSE_SCENARIO_NOT_FILL", "floor": floor,
                      "floor_source": "PRIOR_RANGE_LOW_NOT_CONFIRMED_PIVOT",
                      "breakout_level": f["base_ceiling"] if selected == "BASE_BREAKOUT" else f["previous_high"],
                      "as_of": production["decision_time"]}
    else:
        patterns = {key: {"state": "DATA_INSUFFICIENT", "checks": {}, "trigger_confirmed": False}
                    for key in ("PRICE_RECOVERY", "BASE_BREAKOUT")}

    quality = production["quality"]
    blocks = list(production["downtrend_gate"]["reasons"])
    if quality["status"] == "REJECT":
        blocks.append("QUALITY_HARD_REJECT")
    restricted = f.get("restricted")
    if str(restricted) == "True":
        blocks.append("TRADING_RESTRICTION")
    elif str(restricted) != "False":
        missing.append("TRADING_RESTRICTION_EVIDENCE")
    if quality["status"] not in {"PASS", "WATCH_QUALITY", "REJECT"}:
        missing.append("POINT_IN_TIME_QUALITY_EVIDENCE")
    if blocks:
        signal = "RESEARCH_BLOCKED"
    elif missing:
        signal = "RESEARCH_DATA_INSUFFICIENT"
    elif any(p["state"] == "CONFIRMED" for p in patterns.values()):
        signal = "RESEARCH_CONFIRMED"
    elif any(p["state"] == "FORMING" for p in patterns.values()):
        signal = "RESEARCH_WATCH"
    else:
        signal = "RESEARCH_NO_SETUP"
    costs = LowEntryV6Config()
    return _clean({
        "model": "recovery_breakout_research", "parameter_version": RESEARCH_VERSION,
        "parameters": asdict(config), "research_only": True, "executable": False,
        "calibration_status": "UNVALIDATED", "symbol": production["symbol"],
        "market": production["market"]["exchange"],
        "signal_date": production["signal_date"], "decision_time": production["decision_time"],
        "research_signal": signal, "patterns": patterns, "missing_evidence": sorted(set(missing)),
        "hard_blocks": blocks, "quality": quality,
        "production_reference": {"model_version": production["model_version"],
                                 "parameter_version": production["parameter_version"],
                                 "signal": production["decision"]["signal"],
                                 "score": production["decision"]["score"],
                                 "risk_gate": production["decision"]["risk_gate"]},
        "evidence": {key: f.get(key) for key in required},
        "scenario": levels,
        "payoff_scenarios": payoff_scenarios(levels or {}, buy_cost_rate=costs.buy_fee_rate,
                                             sell_cost_rate=costs.sell_cost_rate),
        "sizing": {"indicative_shares": 0, "target_weight": 0.},
    })


class RecoveryBreakoutResearchStrategy(Strategy):
    """Explicit offline plugin; deliberately absent from the production registry."""

    name = "recovery_breakout_research"
    version = "0.1.0"
    description = "Research-only price recovery / prior-base breakout; no orders"

    def __init__(self, config: RecoveryBreakoutConfig | None = None):
        self.config = config or RecoveryBreakoutConfig()

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.run_with_features(data, SharedFeaturePipeline().build(data))

    def run_latest(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.run_with_features(data, SharedFeaturePipeline().build(data), latest_only=True)

    def run_with_features(self, data: pd.DataFrame, features: FeatureFrame, *, latest_only: bool = False) -> pd.DataFrame:
        # Reuse production PIT accounting, bar validation and risk evidence,
        # without publishing or altering its decisions/position sizing.
        baseline = LowEntryScoreV6Strategy().run_with_features(data, features, latest_only=latest_only)
        source = data.tail(1) if latest_only else data
        # An already-enriched frame must not carry old actionable report or
        # sizing columns through this research-only plugin.
        result = source.reindex(columns=[key for key in (
            "Date", "Open", "High", "Low", "Close", "Volume", "Amount",
        ) if key in source]).copy(deep=True)
        if data.empty:
            result[RESEARCH_COLUMN] = pd.Series(index=result.index, dtype=object)
            return result
        research = build_research_features(build_features(data, features), self.config)
        reports = []
        for index, production in baseline["LES v6 Report"].items():
            f = research.loc[index].to_dict()
            # The production report may replace raw Restricted with dated,
            # market-bound evidence. Never undo that evidence check here.
            f["restricted"] = production["technical"]["restricted"]
            reports.append(evaluate_research(f, production, self.config))
        result[RESEARCH_COLUMN] = reports
        return result

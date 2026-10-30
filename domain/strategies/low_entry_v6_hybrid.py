"""Context first, one fixed-budget ledger, one native plan and one decision.

These frozen curves/weights are research candidates, not calibrated probabilities.
No market ranks, per-context score competition or missing-budget redistribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from math import fsum, isclose
from types import MappingProxyType
from typing import TYPE_CHECKING

from domain.strategies.low_entry_v6_execution_score import execution_conditions, validate_score_budget
from domain.strategies.low_entry_v6_quality import QualityResult, number
from domain.strategies.low_entry_v6_score_evidence import valid_price_anchor

if TYPE_CHECKING:
    from domain.strategies.low_entry_v6_signals import LowEntryV6Config

POLICY = "context-aware-unified-score/v1"
PARAMETER_VERSION = "2026-10-04-v6-hybrid-reported-period-quality-v8"
NO_CLEAR_CONTEXT = "NO_CLEAR_CONTEXT"
CONTEXT_ORDER = ("LEADERSHIP_CONTINUATION", "TREND_PULLBACK", "REVERSAL")
FACTOR_BUDGETS = (("structure", 25), ("trend", 15), ("relative_strength", 20),
                  ("demand", 20), ("momentum", 10), ("execution", 10))
RELAXED_POLICY = "context-aware-unified-score/v2"
RELAXED_PARAMETER_VERSION = "2026-10-07-v6-hybrid-72-62-reported-period-quality-v8"
RELAXED_FACTOR_BUDGETS = (("structure", 30), ("trend", 10), ("relative_strength", 20),
                          ("demand", 25), ("momentum", 5), ("execution", 10))
CURVES = MappingProxyType({
    "recovery_slope": (-.05, .20), "continuation_slope": (-.05, .15),
    "extension": (.5, 2.), "atr_contraction": (.8, 1.25),
    "base_width_atr": (1., 4.), "base_width_ratio": (.5, 1.),
    "base_volume_ratio": (.5, 1.), "support_tests": (1., 4.),
    "close_break": (0., .5), "market_rs20": (0., .05), "market_rs60": (0., .10),
    "industry_rs20": (0., .05), "industry_rs60": (0., .10),
    "downside_capture": (.25, 1.), "clv": (0., .8), "cmf": (0., .20),
    "amount_ratio": (1., 1.5), "obv_volume_slope": (0., .20),
    "rsi_change": (0., 5.), "rsi_recovery": (25., 40.), "rsi_extension": (65., 80.),
})


def factor_budgets(config: LowEntryV6Config | None = None) -> tuple:
    return RELAXED_FACTOR_BUDGETS if config is not None and config.decision_policy_version == 4 else FACTOR_BUDGETS


def model_spec(config: LowEntryV6Config | None = None) -> dict:
    relaxed = config is not None and config.decision_policy_version == 4
    return dict(policy_version=RELAXED_POLICY if relaxed else POLICY,
        parameter_version=RELAXED_PARAMETER_VERSION if relaxed else PARAMETER_VERSION,
        factor_budgets=dict(factor_budgets(config)), curves={key: list(value) for key, value in CURVES.items()},
        source_pooling="MAX_WITH_FIXED_COVERAGE_NO_SECOND_PENALTY",
        context_priority=list(CONTEXT_ORDER), calibration_status="UNCALIBRATED_RESEARCH_CANDIDATE",
        threshold_basis=("72_62_RELAXED_CANDIDATE_NOT_EQUIVALENT_TO_BASELINE_SCORES" if relaxed else
                         "75_65_COMPARISON_BASELINE_NOT_EQUIVALENT_TO_PREVIOUS_SCORES"))


def _ramp(value: float | None, curve: str, *, inverse: bool = False) -> float | None:
    if value is None:
        return None
    low, high = CURVES[curve]
    fraction = min(1., max(0., (value - low) / (high - low)))
    return 1. - fraction if inverse else fraction


def _all(*values: bool | None) -> bool | None:
    return False if False in values else None if None in values else True


def _any(*values: bool | None) -> bool | None:
    return True if True in values else None if None in values else False


class _Facts:
    def __init__(self, features: dict):
        self.features = features

    def known(self, key: str) -> bool:
        masks = self.features.get("score_availability", {})
        if key in masks and str(masks[key]) != "True":
            return False
        if key.startswith(("rs", "resilience", "downside_")) and not key.startswith("rsi"):
            return str(self.features.get("market_ready")) == "True"
        if key.startswith("industry_rs"):
            periods = self.features.get("industry_period_status")
            if isinstance(periods, Mapping):
                return periods.get(key.removeprefix("industry_rs")) == "PASS"
            return self.features.get("industry_status") == "PASS"
        return True

    def n(self, key: str) -> float | None:
        return number(self.features.get(key)) if self.known(key) else None

    def b(self, key: str) -> bool | None:
        value = self.features.get(key)
        return str(value) == "True" if self.known(key) and str(value) in {"True", "False"} else None

    def compare(self, keys: tuple[str, ...], predicate) -> bool | None:
        values = [self.n(key) for key in keys]
        return predicate(*values) if all(value is not None for value in values) else None

    def distance(self, price: str) -> float | None:
        close, anchor, atr = (self.n(key) for key in ("close", price, "atr14"))
        return number((close - anchor) / atr) if close is not None and anchor is not None and atr is not None and atr > 0 else None


def known_features(features: dict) -> dict:
    """Unknown evidence cannot leak into the native plan/risk path as stale values."""
    facts = _Facts(features)
    return {key: value if facts.known(key) else None for key, value in features.items()}


@dataclass(frozen=True)
class ContextSelection:
    context: str
    evidence: tuple[tuple[str, tuple[tuple[str, bool | None], ...]], ...]

    def record(self) -> dict:
        matches = [name for name, checks in self.evidence if all(value is True for _, value in checks)]
        return dict(policy_version="causal-structure-context/v1", mode="STRUCTURE_BEFORE_SCORING",
            selected_setup=self.context, matches=matches, priority=list(CONTEXT_ORDER),
            status="NO_CLEAR_CONTEXT" if not matches else "MULTIPLE_MATCHES_FIXED_PRIORITY" if len(matches) > 1 else "CLEAR",
            evidence={name: dict(checks) for name, checks in self.evidence},
            risk_fallback="REVERSAL_CONSERVATIVE_LIMITS" if not matches else None,
            score_or_buy_used_for_selection=False)


def select_context(features: dict) -> ContextSelection:
    """Only completed structural evidence; no quality, quote, RR, score or account."""
    f = _Facts(features)
    held = _all(f.b("support_intact"), f.compare(("low", "support", "close"), lambda low, support, close: 0 < support <= low < close))
    recovery_location = _any(f.compare(("drawdown60",), lambda value: value <= -.06),
                            f.compare(("position60",), lambda value: value <= .4), f.b("undercut_reclaim"))
    checks = (
        ("LEADERSHIP_CONTINUATION", (
            ("HELD_SUPPORT", held), ("PRIOR_BASE", f.b("base_compression")),
            ("RESILIENCE", f.b("downside_resilience")),
            ("BREAKOUT_OR_RETEST", _any(f.b("breakout_quality"), f.b("breakout_retest_confirmed"))),
            ("HIGH_ZONE", f.compare(("position60",), lambda value: value >= .6)),
            ("CONTINUING_TREND", f.compare(("ema10", "ema20", "ema20_slope"), lambda short, medium, slope: short > medium and slope > 0)),
        )),
        ("TREND_PULLBACK", (
            ("HELD_SUPPORT", held), ("HELD_LOW", f.b("no_new_low2")),
            ("UPTREND", f.compare(("ema10", "ema20", "ema20_slope"), lambda short, medium, slope: short > medium and slope >= 0)),
            ("PULLBACK", f.compare(("drawdown20",), lambda value: -.15 <= value <= -.01)),
            ("NEAR_TREND", f.compare(("close", "ema20", "atr14"), lambda close, ema, atr: atr > 0 and close >= ema - .5 * atr)),
        )),
        ("REVERSAL", (
            ("HELD_SUPPORT", held), ("LOW_ZONE", recovery_location),
            ("HELD_LOW", _any(f.b("no_new_low2"), f.b("higher_low"), f.b("consolidation_confirmed"))),
            ("RECOVERING_SHORT_TREND", f.compare(("ema5_slope",), lambda value: value > 0)),
        )),
    )
    selected = next((name for name, items in checks if all(value is True for _, value in items)), NO_CLEAR_CONTEXT)
    return ContextSelection(selected, checks)


class _Ledger:
    def __init__(self, features: dict, config: LowEntryV6Config):
        self.anchor = valid_price_anchor(features)
        self.conditions: list[dict] = []
        self.budgets = factor_budgets(config)
        self.scales = {group: budget / dict(FACTOR_BUDGETS)[group] for group, budget in self.budgets}
        self.policy = model_spec(config)["policy_version"]

    def add(self, group: str, name: str, weight: int, fraction: float | None, dependencies: tuple[str, ...],
            *, rule: str, coverage: float | None = None, sources: list[dict] | None = None) -> None:
        # Preserve each condition's relative budget within its factor. Missing
        # sources still earn zero; their points are never redistributed.
        if self.scales[group] != 1:
            weight = weight * self.scales[group]
        observed = self.anchor and fraction is not None
        available = (1. if coverage is None else coverage) if observed else 0.
        if observed and (not 0 <= fraction <= available <= 1):
            raise ValueError("INVALID_HYBRID_ATTAINMENT")
        self.conditions.append(dict(id=f"{group}:{name}", group=group, weight=weight,
            dependencies=list(dependencies), rule=rule, observed=observed,
            status="UNKNOWN" if not observed else "PARTIAL" if available < 1 else "OBSERVED",
            fraction=fraction if observed else None, earned_points=weight * fraction if observed else 0.,
            available_weight=weight * available, sources=sources or []))

    def pool(self, group: str, name: str, weight: int, sources: list[tuple[str, float | None]]) -> None:
        known = [value for _, value in sources if value is not None]
        coverage = len(known) / len(sources)
        fraction = max(known) * coverage if known else None
        self.add(group, name, weight, fraction, tuple(key for key, _ in sources),
            rule="MAX_WITH_FIXED_COVERAGE", coverage=coverage,
            sources=[dict(field=key, fraction=value, status="UNKNOWN" if value is None else "OBSERVED") for key, value in sources])

    def finish(self) -> dict:
        components = {}
        for group, budget in self.budgets:
            conditions = [condition for condition in self.conditions if condition["group"] == group]
            if not isclose(fsum(condition["weight"] for condition in conditions), budget, rel_tol=0., abs_tol=1e-10):
                raise ValueError("HYBRID_FACTOR_BUDGET_MISMATCH")
            components[group] = dict(total_weight=budget,
                earned_points=fsum(condition["earned_points"] for condition in conditions),
                available_weight=fsum(condition["available_weight"] for condition in conditions))
        if set(components) != {condition["group"] for condition in self.conditions}:
            raise ValueError("HYBRID_UNKNOWN_FACTOR")
        validate_score_budget({group: [(c["fraction"], c["weight"]) for c in self.conditions if c["group"] == group]
                               for group in components})
        available = fsum(item["available_weight"] for item in components.values())
        raw = fsum(item["earned_points"] for item in components.values()) if available else None
        return dict(policy_version=self.policy, conditions=self.conditions, components=components,
            formal_score=raw, earned_points=raw, estimated_score=None, total_weight=100,
            available_weight=available, coverage_percent=available,
            status="COMPLETE" if available == 100 else "PARTIAL" if available else "UNAVAILABLE",
            missing_fields=sorted({key for c in self.conditions if c["status"] != "OBSERVED"
                                  for key in (tuple(s["field"] for s in c["sources"] if s["fraction"] is None)
                                              if c["sources"] else c["dependencies"])}))


def score_factors(features: dict, selection: ContextSelection, plan: dict, *, config: LowEntryV6Config,
                  legacy_resilience: bool = False) -> dict:
    """Interpret ONLY the preselected context. Every slot retains its fixed budget."""
    from domain.strategies.low_entry_v6_signals import checked_stop_distances
    features = known_features(features)
    f, ledger, context = _Facts(features), _Ledger(features, config), selection.context
    support = _all(f.b("support_intact"),
        f.compare(("support", "low", "close"), lambda support, low, close: 0 < support <= low < close))
    ledger.add("structure", "confirmed_support", 8, float(support) if support is not None else None,
               ("support", "low", "close"), rule="CONFIRMED_FLOOR_HELD")
    if context == "LEADERSHIP_CONTINUATION":
        low = _ramp(f.n("base_support_tests"), "support_tests")
        compression_values = [_ramp(f.n(key), curve, inverse=True) for key, curve in (
            ("base_range_atr10", "base_width_atr"), ("base_range_ratio5_10", "base_width_ratio"), ("base_volume_ratio5_20", "base_volume_ratio"))]
        compression = min(compression_values) if all(value is not None for value in compression_values) else None
        close_fraction = _ramp(f.n("breakout_extension"), "extension", inverse=True)
        low_fields, compression_fields, close_fields = (("base_support_tests",),
            ("base_range_atr10", "base_range_ratio5_10", "base_volume_ratio5_20"), ("breakout_extension",))
    else:
        higher = f.b("higher_low")
        low = float(higher) if higher is not None else None
        if context == "TREND_PULLBACK":
            dry = f.b("down_volume_contracts")
            compression = float(dry) if dry is not None else None
            compression_fields = ("down_volume_contracts",)
        else:
            compression = _ramp(f.n("atr_expansion"), "atr_contraction", inverse=True)
            compression_fields = ("atr_expansion",)
        close_fraction = _ramp(f.distance("previous_high"), "close_break")
        low_fields, close_fields = ("higher_low",), ("close", "previous_high", "atr14")
    ledger.add("structure", "low_structure", 7, low, low_fields, rule=context + "_LOW_STRUCTURE")
    ledger.add("structure", "contraction", 6, compression, compression_fields, rule=context + "_CONTRACTION")
    ledger.add("structure", "close_confirmation", 4, close_fraction, close_fields, rule=context + "_CLOSE_CONFIRMATION")
    slope_key = "ema5_slope" if context == "REVERSAL" else "ema20_slope"
    slope_curve = "recovery_slope" if context == "REVERSAL" else "continuation_slope"
    ledger.add("trend", "ema_progress", 10, _ramp(f.n(slope_key), slope_curve), (slope_key,), rule=slope_curve)
    extension_key = "ema5" if context == "REVERSAL" else "ema20"
    extension = f.distance(extension_key)
    ledger.add("trend", "extension", 5, _ramp(abs(extension) if extension is not None else None, "extension", inverse=True),
               ("close", extension_key, "atr14"),
               rule=f"EMA_SOURCE_POOL_{dict(factor_budgets(config))['trend']}_NO_MACD_OR_BREAKOUT_BONUS")
    for key, weight, curve in (("rs20", 6, "market_rs20"), ("rs60", 4, "market_rs60"),
                               ("industry_rs20", 4, "industry_rs20"), ("industry_rs60", 2, "industry_rs60")):
        ledger.add("relative_strength", key, weight, _ramp(f.n(key), curve), (key,), rule="ALIGNED_" + curve)
    market_ready = f.b("market_ready")
    correction = f.n("market_correction20")
    resilience_dependencies = ("market_ready", "market_correction20")
    if legacy_resilience:
        # Archived R18 reports predate the observed-zero coverage correction.
        # Reproduce their complete old arithmetic; never use for new reports.
        capture = f.n("downside_capture20") if correction is not None and correction <= -.03 else None
        resilience = _ramp(capture, "downside_capture", inverse=True)
        held = f.compare(("correction_support", "low"), lambda support, low: 0 < support <= low)
        resilience = resilience if held is True else 0. if held is False and resilience is not None else None
        resilience_dependencies = ("market_correction20", "downside_capture20", "correction_support", "low")
    elif market_ready is not True or correction is None:
        # A market-resilience observation needs a complete aligned benchmark
        # window. Missing market evidence stays UNKNOWN; it is never a zero.
        resilience = None
    elif correction > -.03:
        # No qualifying market correction occurred, so resilience was not
        # tested. This is an observed zero, not missing downside-capture data.
        resilience = 0.
    else:
        resilience_dependencies += ("downside_capture20", "correction_support", "low")
        held = f.compare(("correction_support", "low"), lambda support, low: 0 < support <= low)
        if held is False:
            resilience = 0.
        elif held is True:
            capture = f.n("downside_capture20")
            resilience = _ramp(capture, "downside_capture", inverse=True)
        else:
            resilience = None
    ledger.add("relative_strength", "correction_resilience", 4, resilience,
               resilience_dependencies, rule="ACTUAL_CORRECTION_WITH_FROZEN_SUPPORT")
    ledger.pool("demand", "money_flow", 12, [("clv", _ramp(f.n("clv"), "clv")), ("cmf20", _ramp(f.n("cmf20"), "cmf"))])
    obv, volume = f.n("obv_slope5"), f.n("volume_ma20_strict")
    obv_ratio = number(obv / volume) if obv is not None and volume is not None and volume > 0 else None
    ledger.pool("demand", "participation", 8, [("amount_ratio", _ramp(f.n("amount_ratio"), "amount_ratio")),
                                             ("obv_volume_slope", _ramp(obv_ratio, "obv_volume_slope"))])
    change, rsi = f.n("rsi_change"), f.n("rsi14")
    direction = _ramp(change, "rsi_change")
    zone = min(_ramp(rsi, "rsi_recovery"), _ramp(rsi, "rsi_extension", inverse=True)) if rsi is not None else None
    ledger.add("momentum", "rsi_direction", 6, direction, ("rsi_change",), rule="POSITIVE_RSI_DIRECTION_ONLY")
    ledger.add("momentum", "rsi_recovery", 4, min(direction, zone) if direction is not None and zone is not None else None,
               ("rsi_change", "rsi14"), rule="RECOVERY_OR_CONTINUATION_NO_OVERSOLD_BONUS")
    tests, execution = execution_conditions(features, plan,
        distances=checked_stop_distances(plan, f.n("atr14")), config=config)
    stop_known = all(number(plan.get(key)) is not None for key in ("entry_price", "stop_price")) and f.n("atr14") is not None
    target_known = stop_known and number(plan.get("target_price")) is not None and any(
        f.n(key) == plan["target_price"] for key in ("swing_high", "hh20", "hh60"))
    known = (stop_known, target_known, f.n("spread") is not None, f.n("amount_ma20") is not None)
    for (condition, weight), available, name, dependencies in zip(tests, known,
            ("stop", "net_rr", "spread", "liquidity"),
            (("@stop",), ("@net_reward",), ("spread",), ("amount_ma20",)), strict=True):
        ledger.add("execution", name, weight, float(condition) if available and condition is not None else None,
                   dependencies, rule="COMMON_EXECUTION_RISK_SCALE")
    evidence = ledger.finish()
    execution.update(evidence["components"]["execution"])
    return dict(score_evidence=evidence, execution_score=execution,
        factor_budgets={group: dict(budget=values["total_weight"], earned=values["earned_points"],
            available_weight=values["available_weight"], condition_ids=[c["id"] for c in evidence["conditions"] if c["group"] == group])
            for group, values in evidence["components"].items()},
        valid=context != NO_CLEAR_CONTEXT, recognized=context != NO_CLEAR_CONTEXT, recovery_checks={})


def decide_hybrid(quality: QualityResult, features: dict, *, valuation: str, config: LowEntryV6Config,
                  entry_price: float | None = None, industry_mode: str = "ACTIVE",
                  convex_weights: dict | None = None, as_of: str | None = None) -> dict:
    from domain.strategies.low_entry_v6_decision import common_safety_caps, score_outcome
    from domain.strategies.low_entry_v6_signals import hard_vetoes, trade_plan
    features = known_features(features)
    selection = select_context(features)
    plan = trade_plan(features, entry_price, config=config,
                      setup=None if selection.context == NO_CLEAR_CONTEXT else selection.context)
    scored = score_factors(features, selection, plan, config=config)
    weighting = None
    if convex_weights is not None:
        from domain.strategies.low_entry_v6_weights import FrozenWeights, reweight_factors
        frozen = FrozenWeights.from_record(convex_weights)
        frozen.require_compatible(config, as_of)
        scored, weighting = reweight_factors(scored, selection.context, frozen)
    penalties = [dict(reason="RISK_OFF", points=config.risk_off_score_increment)] if features.get("market_regime") == "RISK_OFF" else []
    result = score_outcome(selection.context, scored, features, plan, valuation=valuation,
        config=config, common_caps=common_safety_caps(quality, features, industry_mode=industry_mode, config=config),
        vetoes=list(hard_vetoes(features)), market_penalties=penalties, setups={},
        overlap=dict(policy_version="factor-internal-pooling/v1", penalties=[], removed_points=0.))
    spec = model_spec(config)
    result.update(policy_version=spec["policy_version"], selection=selection.record(), score_model=spec,
                  score_basis="HYBRID_SINGLE_LEDGER" if result["raw_score"] is not None else "INSUFFICIENT_HISTORY")
    if weighting is not None:
        from domain.strategies.low_entry_v6_weights import POLICY as convex_policy
        result.update(policy_version=convex_policy, weighting=weighting,
            score_basis="CONVEX_SINGLE_LEDGER" if result["raw_score"] is not None else "INSUFFICIENT_HISTORY",
            research_only=False, execution_policy="NATIVE_GATES", requires_execution_check=True,
            performance_validated=False)
    return result

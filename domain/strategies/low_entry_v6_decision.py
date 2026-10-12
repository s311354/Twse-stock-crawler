"""Fixed-budget score -> explicit penalties -> minimum safety cap -> one decision.

Indicator formulas, condition weights and causal trade plans remain owned by
signals/features. Setup/close-trigger composites are diagnostics, not a second
admission path. Intraday execution and portfolio capacity remain separate.
"""
from __future__ import annotations

from math import fsum, isclose

from domain.strategies.low_entry_v6_quality import MINIMUM_ENTRY_QUALITY, QualityResult, number, quality_evidence_complete
from domain.strategies.low_entry_v6_correlation import correlation_adjustment
from domain.strategies.low_entry_v6_signals import (
    EXECUTABLE_SIGNALS, SETUPS, LowEntryV6Config, _starter_threshold, _within_stop_limit,
    _rr_pass, _leader_location_pass, hard_vetoes,
    risk_pass, score_setups, setup_comparison, setup_rank, stop_fraction_limit,
    trade_plan, triggers, truth, valid_price_anchor, veto_evidence_complete,
)

POLICY = "unified-penalty-safety-cap/v2"


def compose_score(evidence: dict, penalties: list[dict], caps: list[dict]) -> dict:
    """Caps are limits, not additive penalties; an unobserved score stays null."""
    conditions = evidence["conditions"]
    if len({c["id"] for c in conditions}) != len(conditions):
        raise ValueError("DUPLICATE_SCORE_CONDITION")
    for condition in conditions:
        weight, fraction = number(condition["weight"]), number(condition["fraction"])
        zero_allowed = evidence.get("policy_version") == "context-aware-convex-score/v1"
        if weight is None or weight < 0 or (weight == 0 and not zero_allowed) or (condition["observed"] and (fraction is None or not 0 <= fraction <= 1)):
            raise ValueError("INVALID_SCORE_CONDITION")
    if not isclose(fsum(c["weight"] for c in conditions), 100., rel_tol=0., abs_tol=1e-10):
        raise ValueError("LES_POSITIVE_BUDGET_MUST_EQUAL_100")
    for items, field in ((penalties, "points"), (caps, "maximum")):
        if len({item["reason"] for item in items}) != len(items):
            raise ValueError("DUPLICATE_SCORE_ADJUSTMENT")
        for item in items:
            value = number(item[field])
            if value is None or value < 0 or (field == "maximum" and value > 100):
                raise ValueError("INVALID_SCORE_ADJUSTMENT")
    observed = [c for c in conditions if c["observed"] and c["weight"] > 0]
    raw = fsum(c["weight"] * c["fraction"] for c in observed) if observed else None
    penalty = fsum(item["points"] for item in penalties)
    adjusted = min(100., max(0., raw - penalty)) if raw is not None else None
    cap = min((item["maximum"] for item in caps), default=100.)
    return dict(policy_version=POLICY, raw_score=raw, penalty=penalty, adjusted_score=adjusted,
                safety_cap=cap, final_score=min(adjusted, cap) if adjusted is not None else None,
                penalties=penalties, safety_caps=caps)


def decision_from_score(final_score: float | None, *, buy_minimum: float = 75.,
                        small_minimum: float = 65., decision_policy_version: int = 2) -> str:
    """The shared label function. Never round before comparing thresholds."""
    floor = 62 if decision_policy_version == 4 else 65
    if not floor <= small_minimum <= buy_minimum <= 100 or (decision_policy_version == 4 and buy_minimum < 72):
        raise ValueError("INVALID_DECISION_THRESHOLDS")
    if final_score is None:
        return "WATCH"
    value = number(final_score)
    if value is None or not 0 <= value <= 100:
        raise ValueError("INVALID_FINAL_SCORE")
    if value >= buy_minimum:
        return "BUY"
    if value >= small_minimum:
        return "BUY_SMALL"
    return "WATCH" if value >= 55 else "AVOID"


def _budget_groups(evidence: dict) -> dict:
    """Diagnostic ownership tags; these are not additional earned points.

    The budget is the sum of EXISTING weights assigned to each family, not an
    additional score. Source-family pooling separately removes overlap through
    explicit penalties. Changing allocations requires separate calibration.
    """
    groups = {}
    for condition in evidence["conditions"]:
        fields = condition["dependencies"]
        if condition["group"] == "execution":
            group = "execution"
        elif any(k.startswith(("ema", "macd", "reclaim")) or k == "break_previous_high" for k in fields):
            group = "trend"
        elif any(k in {"clv", "cmf20", "cmf_improving", "obv_slope5", "up_volume_ratio",
                        "volume_ratio", "amount_ratio", "vwap20"} for k in fields):
            group = "demand"
        elif any(k.startswith("rsi") for k in fields):
            group = "momentum_recovery"
        elif any(k.startswith(("rs", "resilience")) for k in fields):
            group = "relative_strength"
        else:
            group = "structure_location"
        item = groups.setdefault(group, dict(budget=0., earned=0., condition_ids=[]))
        item["budget"] += condition["weight"]
        item["earned"] += condition["earned_points"]
        item["condition_ids"].append(condition["id"])
    return groups


def safety_limits(config: LowEntryV6Config | None = None) -> tuple[float, float]:
    """Keep all blocking caps below every executable tier, including overrides."""
    if config is not None and config.decision_policy_version == 4:
        return (min(config.reversal_starter_score_minimum, config.pullback_starter_score_minimum,
                    config.leadership_starter_score_minimum) - 1., config.buy_score_minimum - 1.)
    return 64., 74.


def common_safety_caps(quality: QualityResult, f: dict, *, industry_mode: str,
                       config: LowEntryV6Config | None = None) -> list[dict]:
    """Shared non-compensable evidence caps for versioned unified score policies."""
    common = []
    blocked_cap, _ = safety_limits(config)
    def cap(reason: str, maximum: float = blocked_cap) -> None:
        common.append(dict(reason=reason, maximum=maximum))
    if quality.integrity == "FAIL" or quality.status == "REJECT":
        cap("QUALITY_REJECT_OR_INTEGRITY_FAIL", 0.)
    elif quality.status != "PASS" or number(quality.score) is None or not MINIMUM_ENTRY_QUALITY <= quality.score <= 100:
        cap("QUALITY_NOT_VERIFIED_FOR_ENTRY")
    if not quality_evidence_complete(quality):
        cap("QUALITY_EVIDENCE_INCOMPLETE")
    if not truth(f.get("history_ready")) or not truth(f.get("market_ready")) or number(f.get("rs_slope5")) is None:
        cap("INCOMPLETE_ALIGNED_HISTORY")
    if not valid_price_anchor(f):
        cap("INVALID_CURRENT_OHLCV")
    if not veto_evidence_complete(f):
        cap("INCOMPLETE_VETO_EVIDENCE")
    for reason in hard_vetoes(f):
        cap(reason, 0.)
    if f.get("restricted") is None:
        cap("RESTRICTION_UNKNOWN")
    elif truth(f.get("restricted")):
        cap("TRADING_RESTRICTED", 0.)
    if industry_mode != "ACTIVE":
        cap("INDUSTRY_PROFILE_SHADOW_MODE" if industry_mode == "SHADOW" else "SECTOR_UNSUPPORTED")
    return common


def score_outcome(name: str, setup: dict, f: dict, plan: dict, *, valuation: str,
                  config: LowEntryV6Config, common_caps: list[dict], vetoes: list[str],
                  market_penalties: list[dict], setups: dict, overlap: dict | None = None) -> dict:
    """Apply native risk and the sole decision function to ONE supplied ledger/plan."""
    risk_setup = name if name in SETUPS else "REVERSAL"
    small_ok, small_stop, small_rr = risk_pass(risk_setup, "BUY_SMALL", f, plan, valuation, config)
    buy_ok, buy_stop, buy_rr = risk_pass(risk_setup, "BUY", f, plan, valuation, config)
    caps = list(common_caps)
    blocked_cap, buy_cap = safety_limits(config)
    def limit(reason: str, maximum: float = blocked_cap) -> None:
        caps.append(dict(reason=reason, maximum=maximum))
    # Unknown resistance is not price discovery. The original policy only
    # applies when measured 20/60-day resistance is below entry. A monotonic
    # trend need not have a confirmed swing high, but both windows must exist.
    levels = [number(f.get(k)) for k in ("swing_high", "hh20", "hh60")]
    discovery = (name == "LEADERSHIP_CONTINUATION" and plan["target_price"] is None
                 and number(plan["entry_price"]) is not None
                 and all(v is not None and 0 < v <= plan["entry_price"] for v in levels[1:])
                 and (levels[0] is None or 0 < levels[0] <= plan["entry_price"]))
    if plan["net_reward_risk"] is None and not discovery:
        small_ok = buy_ok = False
        limit("NET_RR_UNAVAILABLE")
    if not plan["liquidity_pass"]:
        limit("LIQUIDITY_OR_RESTRICTION_FAIL")
    if not small_ok:
        limit("STARTER_RISK_FAIL")
    if not buy_ok:
        limit("BUY_RISK_FAIL", buy_cap)
    # Retain structural stop policies, not soft shape eligibility.
    starter_only = plan["stop_source"] in {"CONFIRMED_CONSOLIDATION_LOW", "CONFIRMED_BREAKOUT_RETEST"}
    if starter_only:
        limit("STRUCTURAL_ANCHOR_STARTER_ONLY", buy_cap)
    if name == "LEADERSHIP_CONTINUATION" and number(f.get("breakout_extension")) is not None and f["breakout_extension"] > 1.5:
        limit("OVEREXTENDED_BREAKOUT")
    small_threshold = (_starter_threshold(name, config) if name in SETUPS else max(
        config.reversal_starter_score_minimum, config.pullback_starter_score_minimum,
        config.leadership_starter_score_minimum))
    overlap = overlap if overlap is not None else (correlation_adjustment(name, setup["score_evidence"])
               if config.correlation_policy_version == 2 else
               dict(policy_version="additive-source-confirmations/v1", penalties=[], removed_points=0.))
    penalties = market_penalties + overlap["penalties"]
    scoring = compose_score(setup["score_evidence"], penalties, caps)
    signal = decision_from_score(scoring["final_score"], buy_minimum=config.buy_score_minimum,
                                 small_minimum=small_threshold, decision_policy_version=config.decision_policy_version)
    starter, confirmed, _ = triggers(risk_setup, f)
    checks = {}
    for tier, passed, stop, rr, threshold in (
            ("BUY", buy_ok, buy_stop, buy_rr, config.buy_score_minimum),
            ("BUY_SMALL", small_ok, small_stop, small_rr, small_threshold)):
        blockers = [item["reason"] for item in caps if item["maximum"] < threshold]
        if scoring["final_score"] is None or scoring["final_score"] < threshold:
            blockers.append("FINAL_SCORE_BELOW_THRESHOLD")
        if not _within_stop_limit(number(plan["stop_distance_atr"]), stop):
            blockers.append("STOP_DISTANCE_EXCEEDS_LIMIT")
        if not _within_stop_limit(number(plan["stop_distance_fraction"]), stop_fraction_limit(tier, config)):
            blockers.append("STOP_PERCENTAGE_EXCEEDS_LIMIT")
        if rr is not None and not _rr_pass(number(plan["net_reward_risk"]), rr):
            blockers.append("RR_BELOW_MINIMUM")
        if name == "LEADERSHIP_CONTINUATION" and not _leader_location_pass(f, plan) and (tier == "BUY_SMALL" or rr is None):
            blockers.append("LEADERSHIP_EXTENSION_OR_SUPPORT_FAIL")
        checks[tier] = dict(score=scoring["final_score"], required_score=threshold,
            trigger=confirmed if tier == "BUY" else starter, trigger_role="SCORED_DIAGNOSTIC_ONLY",
            risk_pass=passed, maximum_stop_atr=stop, maximum_stop_fraction=stop_fraction_limit(tier, config),
            stop_distance_atr=plan["stop_distance_atr"], stop_distance_fraction=plan["stop_distance_fraction"],
            reward_risk=plan["reward_risk"], rr_basis=plan["rr_basis"],
            minimum_rr=rr, net_reward_risk=plan["net_reward_risk"], blockers=blockers)
    executable = signal in EXECUTABLE_SIGNALS
    multiplier = config.risk_off_position_multiplier if market_penalties else 1.
    if valuation == "EXPENSIVE":
        multiplier *= config.expensive_position_multiplier
    evidence = setup["score_evidence"]
    outcome = dict(signal=signal, setup=name, score=scoring["final_score"], **scoring,
        formal_score=scoring["final_score"], estimated_score=evidence["estimated_score"],
        score_evidence=evidence, score_coverage=evidence["coverage_percent"],
        score_setup=name, score_basis="UNIFIED_FINAL" if scoring["raw_score"] is not None else "INSUFFICIENT_HISTORY",
        evidence_status=evidence["status"], budget_groups=setup.get("factor_budgets", _budget_groups(evidence)),
        execution_score=setup["execution_score"],
        correlation_adjustment=overlap,
        setups=setups, plan=plan, hard_vetoes=vetoes,
        reasons=[item["reason"] for item in caps] or ["FINAL_SCORE_" + signal],
        risk_gate="PASS" if (buy_ok if signal == "BUY" else small_ok) else "FAIL",
        starter_trigger=starter, confirmed_trigger=confirmed,
        required_score=config.buy_score_minimum if signal == "BUY" else small_threshold,
        maximum_stop_atr=buy_stop if signal == "BUY" else small_stop,
        minimum_rr=buy_rr if signal == "BUY" else small_rr,
        risk_fraction=(config.buy_risk_fraction if signal == "BUY" else config.small_risk_fraction) if executable else 0.,
        position_multiplier=multiplier if executable else 0.,
        entry_tier="STARTER_ONLY" if starter_only or not buy_ok else "CONFIRMED", entry_checks=checks,
        rr_policy="KNOWN_RESISTANCE" if plan["target_price"] is not None else
            "PRICE_DISCOVERY_EXTENSION_SUPPORT" if discovery else "UNKNOWN",
        soft_conditions=dict(setup_valid=setup["valid"], setup_recognized=setup["recognized"],
            recovery_checks=setup["recovery_checks"], starter_trigger=starter, confirmed_trigger=confirmed,
            role="COMPOSITES_OF_SCORED_CONDITIONS_NO_ADDITIONAL_POINTS_OR_GATES"))
    outcome["comparison"] = setup_comparison(outcome)
    return outcome


def decide_unified(quality: QualityResult, f: dict, *, valuation: str,
                   config: LowEntryV6Config, entry_price: float | None = None,
                   industry_mode: str = "ACTIVE", score_exclusions: frozenset[str] = frozenset()) -> dict:
    # Score-only interventions are used by the non-executable research audit;
    # prices, features, native hard vetoes and risk evidence are never removed.
    setups = score_setups(f, trade_plan(f, entry_price, config=config), config=config,
                          score_exclusions=score_exclusions)
    vetoes = list(hard_vetoes(f))
    common = common_safety_caps(quality, f, industry_mode=industry_mode)
    market_penalties = ([dict(reason="RISK_OFF", points=config.risk_off_score_increment)]
                 if f.get("market_regime") == "RISK_OFF" else [])
    outcomes = []
    for name, setup in setups.items():
        plan = trade_plan(f, entry_price, config=config, setup=name)
        outcomes.append(score_outcome(name, setup, f, plan, valuation=valuation,
            config=config, common_caps=common, vetoes=vetoes,
            market_penalties=market_penalties, setups=setups))
    executable = [o for o in outcomes if o["signal"] in EXECUTABLE_SIGNALS]
    chosen = (max(executable, key=setup_rank) if executable else
              max(outcomes, key=lambda o: (o["score"] if o["score"] is not None else -1, o["score_coverage"])))
    chosen["setup_decisions"] = {o["setup"]: {k: o[k] for k in
        ("signal", "raw_score", "penalty", "adjusted_score", "safety_cap", "final_score", "score", "reasons",
         "formal_score", "estimated_score", "score_coverage", "entry_tier", "risk_gate", "required_score",
         "entry_checks", "comparison", "rr_policy", "execution_score", "correlation_adjustment")}
        for o in outcomes}
    measured = [o for o in outcomes if o["final_score"] is not None]
    highest = max((o["final_score"] for o in measured), default=None)
    chosen["selection"] = dict(policy_version="executable-risk-first-selection/v1",
        mode="EXECUTABLE_RISK_FIRST" if executable else "DIAGNOSTIC_FINAL_SCORE",
        selected_setup=chosen["setup"], selected_final_score=chosen["final_score"],
        highest_final_score=highest,
        highest_score_setups=[o["setup"] for o in measured if o["final_score"] == highest],
        selected_is_highest_score=highest is not None and chosen["final_score"] == highest,
        priority=(["EXECUTABLE_ONLY", "BUY_BEFORE_BUY_SMALL", "EVIDENCE_COVERAGE",
                   "KNOWN_NET_RR_SURPLUS", "LOWER_STOP_FRACTION", "FIXED_SETUP_ORDER"]
                  if executable else ["FINAL_SCORE", "EVIDENCE_COVERAGE", "FIXED_SETUP_ORDER"]))
    if score_exclusions:
        chosen.update(research_only=True, executable=False, score_exclusions=sorted(score_exclusions),
                      risk_fraction=0., position_multiplier=0.)
    return chosen

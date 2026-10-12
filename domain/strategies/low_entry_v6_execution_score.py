"""Shared ten-point execution budget; setup structure and admission stay separate."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isclose
from typing import TYPE_CHECKING

from domain.strategies.low_entry_v6_payoff import payoff_scenarios
from domain.strategies.low_entry_v6_quality import number

if TYPE_CHECKING:
    from domain.strategies.low_entry_v6_signals import LowEntryV6Config

POLICY = "common-execution-risk-score/v1"
MINIMUM_NET_RR = 2.
SCORE_BUDGET = 100
STRUCTURE_BUDGET = 90
EXECUTION_BUDGET = 10
Condition = tuple[bool | float | None, float]


@dataclass(frozen=True)
class ExecutionScoreRule:
    name: str
    weight: int
    dependencies: tuple[str, ...]


EXECUTION_RULES = (
    ExecutionScoreRule("stop", 4, ("@stop",)),
    ExecutionScoreRule("net_rr", 3, ("@net_reward",)),
    ExecutionScoreRule("spread", 1, ("spread",)),
    ExecutionScoreRule("liquidity", 2, ("amount_ma20",)),
)
DEPENDENCIES = tuple(rule.dependencies for rule in EXECUTION_RULES)


def validate_score_budget(groups: Mapping[str, Sequence[Condition]]) -> None:
    """Reject scale drift before scoring; never normalize a changed budget to 100."""
    execution = tuple(number(weight) for _, weight in groups.get("execution", ()))
    if (execution != tuple(rule.weight for rule in EXECUTION_RULES)
            or fsum(execution) != EXECUTION_BUDGET):
        raise ValueError("LES_EXECUTION_SCORE_SCALE_MISMATCH")
    structure = [number(weight) for group, tests in groups.items() if group != "execution"
                 for _, weight in tests]
    if any(weight is None or weight <= 0 for weight in structure):
        raise ValueError("LES_INVALID_STRUCTURE_WEIGHT")
    if not isclose(fsum(structure), STRUCTURE_BUDGET, rel_tol=0., abs_tol=1e-10):
        raise ValueError("LES_STRUCTURE_BUDGET_MUST_EQUAL_90")
    if not isclose(fsum(structure) + fsum(execution), SCORE_BUDGET, rel_tol=0., abs_tol=1e-10):
        raise ValueError("LES_POSITIVE_BUDGET_MUST_EQUAL_100")


def execution_conditions(f: dict, plan: dict, *, distances: tuple[float | None, float | None],
                         config: LowEntryV6Config) -> tuple[list[Condition], dict]:
    """Score native prices, not cached RR/pass flags. Missing reward stays null.

    Stop/RR/spread/liquidity weights are 4/3/1/2 for EVERY setup. Restriction
    evidence remains a non-compensable safety cap, not extra execution points.
    Price discovery gets no substitute RR points or redistributed budget.
    """
    atr_distance, fraction = distances
    max_atr = min(config.starter_maximum_stop_atr, config.buy_maximum_stop_atr)
    max_fraction = min(config.starter_maximum_stop_fraction, config.buy_maximum_stop_fraction)

    def within(value: float | None, maximum: float) -> bool:
        return value is not None and value > 0 and (
            value <= maximum or isclose(value, maximum, rel_tol=0., abs_tol=1e-12))

    payoff = payoff_scenarios(plan, buy_cost_rate=config.buy_fee_rate, sell_cost_rate=config.sell_cost_rate)
    rr = payoff["net_reward_risk"]
    # A valid target consumed by costs is a known failed reward test, not missing evidence.
    reward_known = payoff["status"] in {"AVAILABLE", "COST_EXCEEDS_TARGET_UPSIDE", "NO_VALID_STOP"}
    reward_pass = (rr >= MINIMUM_NET_RR or isclose(rr, MINIMUM_NET_RR, rel_tol=0., abs_tol=1e-12)) if rr is not None else False
    spread, amount = number(f.get("spread")), number(f.get("amount_ma20"))
    checks = (within(atr_distance, max_atr) and within(fraction, max_fraction),
              reward_pass if reward_known else None,
              0 <= spread <= config.maximum_spread if spread is not None else None,
              amount >= config.minimum_amount_ma20 if amount is not None else None)
    tests = [(condition, rule.weight) for condition, rule in zip(checks, EXECUTION_RULES, strict=True)]
    return tests, dict(policy_version=POLICY, total_weight=EXECUTION_BUDGET,
        weights={rule.name: rule.weight for rule in EXECUTION_RULES},
        limits=dict(maximum_stop_atr=max_atr, maximum_stop_fraction=max_fraction,
                    minimum_net_rr=MINIMUM_NET_RR, maximum_spread=config.maximum_spread,
                    minimum_amount_ma20=config.minimum_amount_ma20),
        measured=dict(stop_distance_atr=atr_distance, stop_distance_fraction=fraction,
                      net_reward_risk=rr, spread=spread, amount_ma20=amount),
        payoff_status=payoff["status"], rr_basis="NET_OF_CONFIGURED_COSTS",
        buy_cost_rate=config.buy_fee_rate, sell_cost_rate=config.sell_cost_rate)

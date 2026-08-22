"""Focused contracts for the composable Portfolio Allocation policies."""

import pytest

from domain.strategies.portfolio_allocation_v1 import AllocationLineDecision
from domain.strategies.portfolio_allocation_v1 import AllocationSecurity
from domain.strategies.portfolio_allocation_v1 import DominancePolicy
from domain.strategies.portfolio_allocation_v1 import ExitPolicy
from domain.strategies.portfolio_allocation_v1 import LifecyclePolicy
from domain.strategies.portfolio_allocation_v1 import PortfolioAction
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationConfig
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationEngine
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationRequest
from domain.strategies.portfolio_allocation_v1 import PortfolioMode
from domain.strategies.portfolio_allocation_v1 import WeightAllocator


def security(
    symbol: str,
    *,
    quality_score: float = 90,
    bear: float = 0.14,
    base: float = 0.18,
    bull: float = 0.22,
    maximum_weight: float | None = None,
    fundamental_hard_reject: bool = False,
) -> AllocationSecurity:
    return AllocationSecurity(
        symbol=symbol,
        sector="24",
        correlation_cluster=symbol,
        quality_status="QUALITY_CONFIRMED",
        quality_score=quality_score,
        data_confidence=100,
        valuation_status="COMPLETE",
        valuation_gate="PASS",
        bear_cagr=bear,
        base_cagr=base,
        bull_cagr=bull,
        valuation_confidence=100,
        les_decision="BUY",
        risk_gate="PASS",
        falling_knife_state="CLEAR",
        executable_shares=100,
        erem_holding_confidence=75,
        erem_data_sufficient=True,
        current_price=100,
        maximum_weight=maximum_weight,
        fundamental_hard_reject=fundamental_hard_reject,
    )


def test_engine_composes_independent_portfolio_policies() -> None:
    engine = PortfolioAllocationEngine()

    assert isinstance(engine.lifecycle_policy, LifecyclePolicy)
    assert isinstance(engine.dominance_policy, DominancePolicy)
    assert isinstance(engine.exit_policy, ExitPolicy)
    assert isinstance(engine.weight_allocator, WeightAllocator)
    assert engine.lifecycle_policy.dominance_policy is engine.dominance_policy
    assert engine.lifecycle_policy.exit_policy is engine.exit_policy
    assert engine.lifecycle_policy.weight_allocator is engine.weight_allocator


@pytest.mark.parametrize(
    ("holding_count", "expected"),
    (
        (0, PortfolioMode.COLD_START),
        (1, PortfolioMode.BUILDING),
        (2, PortfolioMode.STEADY_STATE),
    ),
)
def test_lifecycle_policy_resolves_holding_count_only(
    holding_count: int,
    expected: PortfolioMode,
) -> None:
    assert LifecyclePolicy.resolve(holding_count) is expected


def test_dominance_policy_requires_quality_and_cer_advantage() -> None:
    engine = PortfolioAllocationEngine()
    incumbent = security(
        "1000",
        quality_score=80,
        bear=0.06,
        base=0.08,
        bull=0.10,
    )
    challenger = security("2000", quality_score=90)
    required_return = 0.10

    assessment = engine.dominance_policy.dominance_assessment(
        incumbent,
        engine.candidate_evaluator.evaluate(incumbent, required_return),
        challenger,
        engine.candidate_evaluator.evaluate(challenger, required_return),
    )

    assert assessment.quality_gate == "PASS"
    assert assessment.cer_gate == "PASS"
    assert assessment.dual_gate == "PASS"


def test_exit_policy_keeps_hard_reject_independent_from_lifecycle() -> None:
    engine = PortfolioAllocationEngine()
    holding = security("1000", fundamental_hard_reject=True)
    evaluation = engine.candidate_evaluator.evaluate(holding, 0.10)

    action, reason, assessment = engine.exit_policy.base_holding_action(
        holding,
        evaluation,
        has_alternative=True,
    )

    assert action is PortfolioAction.EXIT_AND_REALLOCATE
    assert reason == "FUNDAMENTAL_HARD_REJECT"
    assert assessment.strategy_stop_breached is False


def test_weight_allocator_applies_security_cap_without_recomputing_cer() -> None:
    config = PortfolioAllocationConfig()
    engine = PortfolioAllocationEngine(config)
    candidate = security("2000", maximum_weight=0.20)
    request = PortfolioAllocationRequest(
        account_equity=1_000_000,
        current_cash=1_000_000,
        candidates=(candidate,),
    )
    evaluation = engine.candidate_evaluator.evaluate(candidate, 0.10)
    line = AllocationLineDecision(
        symbol=candidate.symbol,
        role="CANDIDATE",
        action=PortfolioAction.ADD,
        evaluation=evaluation,
        current_weight=0.0,
        target_weight=0.0,
    )

    allocated = engine.weight_allocator.allocate_weights(request, [line])

    assert allocated[0].target_weight == pytest.approx(0.20)
    assert allocated[0].evaluation is evaluation

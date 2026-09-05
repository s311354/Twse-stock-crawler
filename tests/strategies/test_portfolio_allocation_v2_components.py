"""Focused tests for the composable PAE v2 domain policies."""

from __future__ import annotations

import datetime as dt

import pytest

from domain.strategies.portfolio_allocation_v2_contracts import (
    PersistenceEvidenceV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioActionV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioAllocationRequestV2,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioAllocationV2Config,
)
from domain.strategies.portfolio_allocation_v2_contracts import (
    PortfolioSecurityV2,
)
from domain.strategies.portfolio_allocation_v2_dominance_policy import (
    PortfolioDominancePolicyV2,
)
from domain.strategies.portfolio_allocation_v2_evaluator import (
    PortfolioSecurityEvaluatorV2,
)
from domain.strategies.portfolio_allocation_v2_exit_policy import (
    PortfolioExitPolicyV2,
)
from domain.strategies.portfolio_allocation_v2_weight_allocator import (
    PortfolioWeightAllocatorV2,
)


NOW = dt.datetime(2026, 8, 30, 13, 30, tzinfo=dt.timezone.utc)


def security(
    stock_id: str,
    *,
    current_weight: float = 0.0,
    erem_data_sufficient: bool = True,
    hard_exit: bool = False,
    caps: bool = True,
) -> PortfolioSecurityV2:
    return PortfolioSecurityV2(
        stock_id=stock_id,
        sector_id=stock_id,
        correlation_cluster_id=stock_id,
        quality_status="PASS",
        quality_score=85.0,
        quality_confidence=90.0,
        data_confidence=90.0,
        valuation_status="COMPLETE",
        valuation_gate="PASS",
        bear_cagr=0.12,
        base_cagr=0.18,
        bull_cagr=0.24,
        valuation_confidence=90.0,
        les_decision="BUY",
        risk_gate="PASS",
        falling_knife_state="CLEAR",
        liquidity_status="PASS",
        executable_shares=1000,
        les_position_cap=0.10 if caps else None,
        risk_budget_weight=0.10 if caps else None,
        liquidity_weight_cap=0.10 if caps else None,
        erem_decision="HOLD",
        erem_holding_confidence=75.0,
        erem_data_sufficient=erem_data_sufficient,
        current_weight=current_weight,
        fundamental_hard_exit=hard_exit,
    )


def request(
    *,
    holdings: tuple[PortfolioSecurityV2, ...] = (),
    candidates: tuple[PortfolioSecurityV2, ...] = (),
    persistence: tuple[PersistenceEvidenceV2, ...] = (),
) -> PortfolioAllocationRequestV2:
    cash_weight = 1.0 - sum(item.current_weight for item in holdings)
    return PortfolioAllocationRequestV2(
        portfolio_id="component-test",
        decision_time=NOW,
        account_equity=1_000_000,
        available_cash=cash_weight * 1_000_000,
        holdings=holdings,
        candidates=candidates,
        persistence=persistence,
    )


def test_security_evaluator_owns_cer_and_eligibility_only() -> None:
    config = PortfolioAllocationV2Config()
    evaluator = PortfolioSecurityEvaluatorV2(config)
    payload = request(candidates=(security("2330"),))

    required_return = evaluator.required_return(payload)
    result = evaluator.evaluate(payload.candidates[0], required_return)

    assert required_return == pytest.approx(0.10)
    assert result.cer == pytest.approx(0.174)
    assert result.strategic_candidate is True
    assert result.executable_candidate is True


def test_exit_policy_keeps_unknown_erem_and_honors_hard_exit() -> None:
    config = PortfolioAllocationV2Config()
    evaluator = PortfolioSecurityEvaluatorV2(config)
    exits = PortfolioExitPolicyV2()
    unknown = security("2330", current_weight=0.60, erem_data_sufficient=False)
    hard_exit = security("2317", current_weight=0.30, hard_exit=True)
    required_return = evaluator.required_return(
        request(holdings=(unknown, hard_exit))
    )

    unknown_line = exits.holding_line(
        unknown,
        evaluator.evaluate(unknown, required_return),
    )
    exit_line = exits.holding_line(
        hard_exit,
        evaluator.evaluate(hard_exit, required_return),
    )

    assert unknown_line.action is PortfolioActionV2.HOLD_DATA_INSUFFICIENT
    assert unknown_line.target_weight == pytest.approx(0.60)
    assert exit_line.action is PortfolioActionV2.EXIT_TO_CASH
    assert exit_line.target_weight == 0.0


def test_weight_allocator_fails_closed_when_mandatory_caps_are_missing() -> None:
    allocator = PortfolioWeightAllocatorV2(PortfolioAllocationV2Config())
    candidate = security("2330", caps=False)

    target, blocking = allocator.target_weight(
        request(candidates=(candidate,)),
        candidate,
        available_weight=0.90,
        excluded_holding=None,
    )

    assert target == 0.0
    assert blocking == (
        "LES_POSITION_CAP_MISSING",
        "RISK_BUDGET_WEIGHT_MISSING",
        "LIQUIDITY_WEIGHT_CAP_MISSING",
    )


def test_dominance_policy_applies_persistence_as_pair_evidence() -> None:
    config = PortfolioAllocationV2Config()
    evaluator = PortfolioSecurityEvaluatorV2(config)
    dominance = PortfolioDominancePolicyV2(config)
    incumbent = security("2317", current_weight=0.80)
    challenger = security("2330")
    payload = request(
        holdings=(incumbent,),
        candidates=(challenger,),
        persistence=(
            PersistenceEvidenceV2(
                incumbent_stock_id="2317",
                challenger_stock_id="2330",
                recent_passes=(True, False, True),
            ),
        ),
    )
    required_return = evaluator.required_return(payload)
    evaluations = {
        item.stock_id: evaluator.evaluate(item, required_return)
        for item in payload.holdings + payload.candidates
    }

    observations = dominance.persistence_observations(payload, evaluations)

    assert dominance.persistence_passed(payload, "2317", "2330") is True
    assert len(observations) == 1
    assert observations[0].incumbent_stock_id == "2317"

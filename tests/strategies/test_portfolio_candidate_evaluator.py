"""Focused contracts for the decomposed PAE candidate evaluator."""

import pytest

from domain.strategies.portfolio_allocation_v1 import AllocationSecurity
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationConfig
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationEngine
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationRequest
from domain.strategies.portfolio_candidate_evaluator import (
    PortfolioCandidateEvaluator,
)


def _candidate(symbol: str = "2330", *, quality_score: float = 90) -> AllocationSecurity:
    return AllocationSecurity(
        symbol=symbol,
        sector="24",
        correlation_cluster="SEMI",
        quality_status="QUALITY_CONFIRMED",
        quality_score=quality_score,
        data_confidence=100,
        valuation_status="COMPLETE",
        valuation_gate="PASS",
        bear_cagr=0.10,
        base_cagr=0.16,
        bull_cagr=0.22,
        valuation_confidence=100,
        les_decision="BUY",
        risk_gate="PASS",
        falling_knife_state="CLEAR",
        executable_shares=100,
    )


def test_candidate_evaluator_calculates_cer_and_eligibility() -> None:
    evaluator = PortfolioCandidateEvaluator(PortfolioAllocationConfig())

    result = evaluator.evaluate(_candidate(), required_return=0.10)

    assert result.scenario_return == pytest.approx(0.16)
    assert result.cer == pytest.approx(0.16)
    assert result.opportunity_edge == pytest.approx(0.06)
    assert result.absolute_strategic_candidate is True
    assert result.executable_candidate is True
    assert result.capital_claim == pytest.approx(0.06)


def test_candidate_evaluator_requires_a_clear_falling_knife_gate() -> None:
    evaluator = PortfolioCandidateEvaluator(PortfolioAllocationConfig())
    candidate = _candidate()
    watch = AllocationSecurity(
        **{**candidate.__dict__, "falling_knife_state": "REVERSAL_WATCH"}
    )

    result = evaluator.evaluate(watch, required_return=0.10)

    assert result.absolute_strategic_candidate is True
    assert result.executable_candidate is False
    assert result.capital_claim == 0


def test_buy_small_accepts_supportive_valuation_and_quality_watch() -> None:
    evaluator = PortfolioCandidateEvaluator(PortfolioAllocationConfig())
    candidate = _candidate()
    starter = AllocationSecurity(
        **{
            **candidate.__dict__,
            "quality_status": "QUALITY_WATCH",
            "valuation_gate": "SUPPORTIVE",
            "les_decision": "BUY_SMALL",
        }
    )

    result = evaluator.evaluate(starter, required_return=0.10)

    assert result.strategic_candidate is True
    assert result.executable_candidate is True
    assert result.capital_claim == pytest.approx(0.024)


def test_standard_buy_still_requires_valuation_pass() -> None:
    evaluator = PortfolioCandidateEvaluator(PortfolioAllocationConfig())
    candidate = _candidate()
    supportive = AllocationSecurity(
        **{**candidate.__dict__, "valuation_gate": "SUPPORTIVE"}
    )

    result = evaluator.evaluate(supportive, required_return=0.10)

    assert result.strategic_candidate is False
    assert result.executable_candidate is False


def test_entry_quality_reject_still_blocks_new_capital() -> None:
    evaluator = PortfolioCandidateEvaluator(PortfolioAllocationConfig())
    candidate = _candidate()
    rejected = AllocationSecurity(
        **{**candidate.__dict__, "quality_status": "HARD_NO_BUY"}
    )

    result = evaluator.evaluate(rejected, required_return=0.10)

    assert result.hard_reject is True
    assert result.strategic_candidate is False
    assert result.executable_candidate is False


def test_engine_preserves_candidate_evaluation_compatibility_delegates() -> None:
    engine = PortfolioAllocationEngine()
    security = _candidate()
    request = PortfolioAllocationRequest(
        account_equity=1_000_000,
        current_cash=1_000_000,
        benchmark_expected_return=0.08,
    )

    direct_required = engine.candidate_evaluator.required_return(request)
    compatibility_required = engine._required_return(request)
    direct = engine.candidate_evaluator.evaluate(security, direct_required)
    compatibility = engine._evaluate(security, compatibility_required)

    assert compatibility_required == direct_required
    assert compatibility == direct


def test_candidate_ranking_remains_quality_first_and_deterministic() -> None:
    evaluator = PortfolioCandidateEvaluator(PortfolioAllocationConfig())
    lower_quality = _candidate("2317", quality_score=85)
    higher_quality = _candidate("2330", quality_score=90)
    evaluations = {
        item.symbol: evaluator.evaluate(item, required_return=0.10)
        for item in (lower_quality, higher_quality)
    }

    ranked = evaluator.rank(
        (lower_quality, higher_quality),
        evaluations,
        predicate="executable_candidate",
    )

    assert [item.symbol for item in ranked] == ["2330", "2317"]

from __future__ import annotations

import datetime as dt

import pytest
import pandas as pd

from domain.strategies.portfolio_allocation_v2 import PAE_V2_MODEL_VERSION
from domain.strategies.portfolio_allocation_v2 import PAE_V2_SCHEMA_VERSION
from domain.strategies.portfolio_allocation_v2 import PersistenceEvidenceV2
from domain.strategies.portfolio_allocation_v2 import PortfolioActionV2
from domain.strategies.portfolio_allocation_v2 import (
    PortfolioAllocationEngineV2,
)
from domain.strategies.portfolio_allocation_v2 import (
    PortfolioAllocationRequestV2,
)
from domain.strategies.portfolio_allocation_v2 import PortfolioBindingV2
from domain.strategies.portfolio_allocation_v2 import PortfolioModeV2
from domain.strategies.portfolio_allocation_v2 import PortfolioSecurityV2
from domain.strategies.portfolio_allocation_v2 import PortfolioAllocationV2Strategy
from domain.strategies.portfolio_allocation_v2 import resolve_portfolio_mode_v2


NOW = dt.datetime(2026, 8, 28, 13, 30, tzinfo=dt.timezone.utc)


def security(
    stock_id: str,
    *,
    current_weight: float = 0.0,
    quality_score: float = 85.0,
    quality_status: str = "PASS",
    data_confidence: float = 90.0,
    bear_cagr: float = 0.12,
    base_cagr: float = 0.18,
    bull_cagr: float = 0.24,
    les_decision: str = "BUY",
    erem_decision: str = "HOLD",
    erem_data_sufficient: bool = True,
    hard_exit: bool = False,
) -> PortfolioSecurityV2:
    return PortfolioSecurityV2(
        stock_id=stock_id,
        sector_id=f"SECTOR-{stock_id}",
        correlation_cluster_id=f"CLUSTER-{stock_id}",
        quality_status=quality_status,
        quality_score=quality_score,
        quality_confidence=90.0,
        data_confidence=data_confidence,
        valuation_status="COMPLETE",
        valuation_gate="PASS",
        bear_cagr=bear_cagr,
        base_cagr=base_cagr,
        bull_cagr=bull_cagr,
        valuation_confidence=90.0,
        les_decision=les_decision,
        risk_gate="PASS",
        falling_knife_state="CLEAR",
        liquidity_status="PASS",
        executable_shares=1000,
        les_position_cap=0.10,
        risk_budget_weight=0.10,
        liquidity_weight_cap=0.10,
        erem_decision=erem_decision,
        erem_holding_confidence=75.0,
        erem_data_sufficient=erem_data_sufficient,
        current_weight=current_weight,
        current_price=100.0,
        stop_price=90.0,
        fundamental_hard_exit=hard_exit,
    )


def request(
    *,
    holdings: tuple[PortfolioSecurityV2, ...] = (),
    candidates: tuple[PortfolioSecurityV2, ...] = (),
    cash_weight: float | None = None,
    persistence: tuple[PersistenceEvidenceV2, ...] = (),
) -> PortfolioAllocationRequestV2:
    if cash_weight is None:
        cash_weight = 1.0 - sum(item.current_weight for item in holdings)
    return PortfolioAllocationRequestV2(
        portfolio_id="owner-1",
        decision_time=NOW,
        account_equity=1_000_000,
        available_cash=1_000_000 * cash_weight,
        holdings=holdings,
        candidates=candidates,
        persistence=persistence,
    )


@pytest.mark.parametrize(
    ("count", "mode"),
    (
        (0, PortfolioModeV2.DIRECT_SELECTION),
        (1, PortfolioModeV2.DIRECT_SELECTION),
        (2, PortfolioModeV2.DIRECT_SELECTION),
        (3, PortfolioModeV2.PAE_ACTIVE_OPEN_SLOT),
        (4, PortfolioModeV2.PAE_ACTIVE_FULL),
        (5, PortfolioModeV2.POLICY_BREACH_REDUCE_ONLY),
    ),
)
def test_v2_mode_is_derived_only_from_effective_holdings(
    count: int,
    mode: PortfolioModeV2,
) -> None:
    assert resolve_portfolio_mode_v2(count) is mode


def test_direct_selection_is_non_binding_and_halves_buy_small_sizing() -> None:
    candidate = security("2330", les_decision="BUY_SMALL")

    decision = PortfolioAllocationEngineV2().allocate(
        request(candidates=(candidate,))
    )

    line = next(item for item in decision.lines if item.stock_id == "2330")
    assert decision.schema_version == PAE_V2_SCHEMA_VERSION
    assert decision.model_version == PAE_V2_MODEL_VERSION
    assert decision.mode is PortfolioModeV2.DIRECT_SELECTION
    assert decision.binding is PortfolioBindingV2.NON_BINDING
    assert line.action is PortfolioActionV2.STARTER
    assert line.target_weight == pytest.approx(0.05)
    assert line.persistence_pass is None
    assert decision.holding_count_after == 1


def test_early_buy_uses_point_65_execution_factor() -> None:
    candidate = security("2330", les_decision="EARLY_BUY")

    decision = PortfolioAllocationEngineV2().allocate(
        request(candidates=(candidate,))
    )

    line = next(item for item in decision.lines if item.stock_id == "2330")
    assert line.action is PortfolioActionV2.STARTER
    assert line.target_weight == pytest.approx(0.065)


def test_direct_selection_does_not_make_pae_cer_diagnostic_binding() -> None:
    weak_cer = security(
        "2330",
        bear_cagr=0.01,
        base_cagr=0.03,
        bull_cagr=0.05,
    )

    decision = PortfolioAllocationEngineV2().allocate(
        request(candidates=(weak_cer,))
    )

    line = next(item for item in decision.lines if item.stock_id == "2330")
    assert decision.mode is PortfolioModeV2.DIRECT_SELECTION
    assert decision.binding is PortfolioBindingV2.NON_BINDING
    assert line.evaluation.strategic_candidate is False
    assert line.evaluation.executable_candidate is False
    assert "CER_BELOW_REQUIRED_RETURN" in line.evaluation.blocking_reasons
    assert line.action is PortfolioActionV2.STARTER
    assert line.target_weight == pytest.approx(0.10)


def test_open_slot_admits_improving_candidate_without_dominance() -> None:
    holdings = tuple(
        security(str(1000 + index), current_weight=0.20, quality_score=80)
        for index in range(3)
    )
    candidate = security("2330", quality_score=85)

    decision = PortfolioAllocationEngineV2().allocate(
        request(holdings=holdings, candidates=(candidate,), cash_weight=0.40)
    )

    line = next(item for item in decision.lines if item.stock_id == "2330")
    assert decision.mode is PortfolioModeV2.PAE_ACTIVE_OPEN_SLOT
    assert decision.binding is PortfolioBindingV2.BINDING
    assert line.action is PortfolioActionV2.STARTER
    assert line.portfolio_improvement is True
    assert line.quality_dominance is None
    assert line.persistence_pass is None
    assert decision.holding_count_after == 4
    assert decision.cash_weight_after >= 0.10


def test_open_slot_is_not_filled_by_weak_candidate() -> None:
    holdings = tuple(
        security(str(1000 + index), current_weight=0.20, quality_score=80)
        for index in range(3)
    )
    weak = security(
        "2330",
        bear_cagr=0.01,
        base_cagr=0.03,
        bull_cagr=0.05,
    )

    decision = PortfolioAllocationEngineV2().allocate(
        request(holdings=holdings, candidates=(weak,), cash_weight=0.40)
    )

    line = next(item for item in decision.lines if item.stock_id == "2330")
    assert line.action is PortfolioActionV2.NO_ACTION
    assert "CER_BELOW_REQUIRED_RETURN" in line.blocking_reasons
    assert decision.holding_count_after == 3
    assert decision.cash_weight_after == pytest.approx(0.40)


def test_unknown_quality_never_equals_pass() -> None:
    unknown = security("2330", quality_status="UNKNOWN")

    decision = PortfolioAllocationEngineV2().allocate(
        request(candidates=(unknown,))
    )

    line = next(item for item in decision.lines if item.stock_id == "2330")
    assert line.evaluation.strategic_candidate is False
    assert line.evaluation.executable_candidate is False
    assert line.action is PortfolioActionV2.NO_ACTION


def test_falling_knife_soft_block_never_becomes_executable() -> None:
    candidate = security("2330")
    soft_blocked = PortfolioSecurityV2(
        **{**candidate.__dict__, "falling_knife_state": "SOFT_BLOCK"}
    )

    decision = PortfolioAllocationEngineV2().allocate(
        request(candidates=(soft_blocked,))
    )

    line = next(item for item in decision.lines if item.stock_id == "2330")
    assert line.evaluation.executable_candidate is False
    assert line.action is PortfolioActionV2.NO_ACTION


def test_missing_erem_does_not_force_holding_exit() -> None:
    holding = security(
        "2330",
        current_weight=0.80,
        erem_decision="DATA_INSUFFICIENT",
        erem_data_sufficient=False,
    )

    decision = PortfolioAllocationEngineV2().allocate(
        request(holdings=(holding,), cash_weight=0.20)
    )

    line = next(item for item in decision.lines if item.stock_id == "2330")
    assert line.action is PortfolioActionV2.HOLD_DATA_INSUFFICIENT
    assert line.target_weight == pytest.approx(0.80)


def test_hard_exit_does_not_require_replacement_candidate() -> None:
    holding = security("2330", current_weight=0.80, hard_exit=True)

    decision = PortfolioAllocationEngineV2().allocate(
        request(holdings=(holding,), cash_weight=0.20)
    )

    line = next(item for item in decision.lines if item.stock_id == "2330")
    assert line.action is PortfolioActionV2.EXIT_TO_CASH
    assert line.target_weight == 0
    assert decision.cash_weight_after == pytest.approx(1.0)


def test_full_portfolio_blocks_rotation_without_persistence() -> None:
    holdings = tuple(
        security(
            str(1000 + index),
            current_weight=0.20,
            quality_score=80,
            bear_cagr=0.05,
            base_cagr=0.08,
            bull_cagr=0.11,
        )
        for index in range(4)
    )
    challenger = security(
        "2330",
        quality_score=84,
        data_confidence=80,
        bear_cagr=0.18,
        base_cagr=0.25,
        bull_cagr=0.32,
    )

    decision = PortfolioAllocationEngineV2().allocate(
        request(holdings=holdings, candidates=(challenger,), cash_weight=0.20)
    )

    line = next(item for item in decision.lines if item.stock_id == "2330")
    assert decision.mode is PortfolioModeV2.PAE_ACTIVE_FULL
    assert line.action is PortfolioActionV2.NO_ACTION
    assert decision.holding_count_after == 4


def test_full_portfolio_uses_atomic_replacement_after_persistence() -> None:
    holdings = tuple(
        security(
            str(1000 + index),
            current_weight=0.20,
            quality_score=80,
            bear_cagr=0.05,
            base_cagr=0.08,
            bull_cagr=0.11,
        )
        for index in range(4)
    )
    challenger = security(
        "2330",
        quality_score=84,
        data_confidence=80,
        bear_cagr=0.18,
        base_cagr=0.25,
        bull_cagr=0.32,
    )
    persistence = tuple(
        PersistenceEvidenceV2(
            incumbent_stock_id=item.stock_id,
            challenger_stock_id="2330",
            recent_passes=(True, False, True),
        )
        for item in holdings
    )

    decision = PortfolioAllocationEngineV2().allocate(
        request(
            holdings=holdings,
            candidates=(challenger,),
            cash_weight=0.20,
            persistence=persistence,
        )
    )

    candidate_line = next(
        item for item in decision.lines if item.stock_id == "2330"
    )
    incumbent_line = next(
        item
        for item in decision.lines
        if item.stock_id == candidate_line.incumbent_stock_id
    )
    assert candidate_line.action is PortfolioActionV2.ROTATE_75
    assert candidate_line.persistence_pass is True
    assert candidate_line.atomic_group_id
    assert incumbent_line.action is PortfolioActionV2.EXIT_AND_REALLOCATE
    assert incumbent_line.atomic_group_id == candidate_line.atomic_group_id
    assert incumbent_line.target_weight == 0
    assert decision.holding_count_after == 4


def test_policy_breach_never_adds_a_new_name() -> None:
    holdings = tuple(
        security(str(1000 + index), current_weight=0.18)
        for index in range(5)
    )
    candidate = security("2330")

    decision = PortfolioAllocationEngineV2().allocate(
        request(holdings=holdings, candidates=(candidate,), cash_weight=0.10)
    )

    assert decision.mode is PortfolioModeV2.POLICY_BREACH_REDUCE_ONLY
    assert all(line.role == "HOLDING" for line in decision.lines)
    assert all(
        line.action is PortfolioActionV2.POLICY_BREACH_REDUCE_ONLY
        for line in decision.lines
    )


def test_same_inputs_produce_identical_decision_and_audit_hashes() -> None:
    candidate = security("2330")
    engine = PortfolioAllocationEngineV2()
    payload = request(candidates=(candidate,))

    first = engine.allocate(payload)
    second = engine.allocate(payload)

    assert first == second
    assert first.input_hash == second.input_hash
    assert first.config_hash == second.config_hash


def test_v2_dataframe_plugin_preserves_legacy_strategy_boundary() -> None:
    data = pd.DataFrame(
        {
            "Date": ["2026-08-28"],
            "Stock Code": ["2330"],
            "Portfolio Role": ["CANDIDATE"],
            "Account Equity": [1_000_000],
            "Current Cash": [1_000_000],
            "Sector": ["SEMI"],
            "Correlation Cluster": ["FOUNDRY"],
            "Quality Status": ["PASS"],
            "Quality Score": [85],
            "Quality Confidence": [90],
            "Data Confidence": [90],
            "Valuation Status": ["COMPLETE"],
            "Valuation Gate": ["PASS"],
            "Bear CAGR": [0.12],
            "Base CAGR": [0.18],
            "Bull CAGR": [0.24],
            "Valuation Confidence": [90],
            "LES Decision": ["BUY_SMALL"],
            "Risk Gate": ["PASS"],
            "Falling Knife State": ["CLEAR"],
            "Liquidity Status": ["PASS"],
            "Executable Shares": [1000],
            "Maximum Weight": [0.10],
            "Risk Budget Weight": [0.10],
            "Liquidity Weight Limit": [0.10],
        }
    )

    result = PortfolioAllocationV2Strategy().run(data)

    assert result.loc[0, "PAE Mode"] == "DIRECT_SELECTION"
    assert result.loc[0, "PAE Binding"] == "NON_BINDING"
    assert result.loc[0, "PAE Decision"] == "STARTER"
    assert result.loc[0, "PAE Model Version"] == "2.0.0"

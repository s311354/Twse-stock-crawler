import pandas as pd
import pytest

from domain.strategies.portfolio_allocation_v1 import AllocationSecurity
from domain.strategies.portfolio_allocation_v1 import PersistenceEvidence
from domain.strategies.portfolio_allocation_v1 import PortfolioAction
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationConfig
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationEngine
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationRequest
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationV1Strategy
from domain.strategies.portfolio_allocation_v1 import PortfolioMode
from domain.strategies.portfolio_allocation_v1 import resolve_portfolio_mode
from domain.strategy import available_strategies
from domain.strategy import get_strategy


def _candidate(
    symbol: str,
    *,
    bear: float = 0.12,
    base: float = 0.16,
    bull: float = 0.20,
    quality_score: float = 90,
    data_confidence: float = 90,
    valuation_confidence: float = 90,
    les_decision: str = "BUY",
    sector: str = "24",
    cluster: str | None = None,
    hard_reject: bool = False,
) -> AllocationSecurity:
    return AllocationSecurity(
        symbol=symbol,
        sector=sector,
        correlation_cluster=cluster or symbol,
        quality_status="QUALITY_CONFIRMED",
        quality_score=quality_score,
        data_confidence=data_confidence,
        valuation_status="COMPLETE",
        valuation_gate="PASS",
        bear_cagr=bear,
        base_cagr=base,
        bull_cagr=bull,
        valuation_confidence=valuation_confidence,
        les_score=82,
        les_decision=les_decision,
        risk_gate="PASS",
        falling_knife_state="CLEAR",
        executable_shares=100,
        current_price=100,
        fundamental_hard_reject=hard_reject,
    )


def _holding(
    symbol: str,
    *,
    bear: float = 0.08,
    base: float = 0.10,
    bull: float = 0.12,
    weight: float = 0.30,
    confidence: float | None = 75,
    erem_data_sufficient: bool = True,
    les_decision: str = "WATCH",
    hard_reject: bool = False,
    quality_score: float = 80,
    data_confidence: float = 90,
    valuation_confidence: float = 90,
    exit_streak: int = 0,
    ema20: float | None = 90,
    ema20_slope: float | None = 0.01,
    distribution_warning: bool | None = False,
    higher_low: bool | None = True,
    strategy_stop: float | None = None,
    sector: str = "24",
    cluster: str | None = None,
) -> AllocationSecurity:
    item = _candidate(
        symbol,
        bear=bear,
        base=base,
        bull=bull,
        les_decision=les_decision,
        sector=sector,
        cluster=cluster,
        hard_reject=hard_reject,
        quality_score=quality_score,
        data_confidence=data_confidence,
        valuation_confidence=valuation_confidence,
    )
    return AllocationSecurity(
        **{
            **item.__dict__,
            "current_shares": 3000,
            "market_value": weight * 1_000_000,
            "current_weight": weight,
            "erem_score": confidence,
            "erem_holding_confidence": confidence,
            "erem_exit_streak": exit_streak,
            "erem_ema20": ema20,
            "erem_ema20_slope": ema20_slope,
            "erem_distribution_warning": distribution_warning,
            "erem_higher_low": higher_low,
            "erem_decision": "HOLD" if confidence is None or confidence >= 55 else "EXIT",
            "erem_data_sufficient": erem_data_sufficient,
            "strategy_stop": strategy_stop,
        }
    )


def _line(decision, symbol: str):
    return next(item for item in decision.lines if item.symbol == symbol)


@pytest.mark.parametrize(
    ("holding_count", "expected"),
    (
        (0, PortfolioMode.COLD_START),
        (1, PortfolioMode.BUILDING),
        (2, PortfolioMode.STEADY_STATE),
        (4, PortfolioMode.STEADY_STATE),
    ),
)
def test_portfolio_mode_is_derived_from_active_holding_count(
    holding_count: int,
    expected: PortfolioMode,
) -> None:
    assert resolve_portfolio_mode(holding_count) is expected


def test_portfolio_mode_rejects_negative_holding_count() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        resolve_portfolio_mode(-1)


def test_single_holding_uses_building_mode() -> None:
    result = PortfolioAllocationEngine().allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=700_000,
            holdings=(_holding("1000"),),
        )
    )

    assert result.mode is PortfolioMode.BUILDING


def test_cold_start_keeps_cash_when_no_candidate_clears_hurdle() -> None:
    engine = PortfolioAllocationEngine()
    weak = _candidate("1001", bear=0.02, base=0.05, bull=0.08)

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=1_000_000,
            candidates=(weak,),
        )
    )

    assert result.mode is PortfolioMode.COLD_START
    assert result.required_return == pytest.approx(0.10)
    assert result.target_cash_weight == pytest.approx(1.0)
    assert _line(result, "1001").action is PortfolioAction.WATCH
    assert result.reason == "CASH_ONLY_NO_EXECUTABLE_CANDIDATE"


def test_legacy_early_buy_is_not_an_executable_candidate() -> None:
    engine = PortfolioAllocationEngine()
    legacy = _candidate("1002", les_decision="EARLY_BUY")

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=1_000_000,
            candidates=(legacy,),
        )
    )

    line = _line(result, "1002")
    assert line.evaluation.executable_candidate is False
    assert line.target_weight == 0
    assert line.action is PortfolioAction.STRATEGIC_WATCH


def test_cold_start_establishes_only_the_first_position() -> None:
    engine = PortfolioAllocationEngine()
    candidates = tuple(
        _candidate(
            str(2000 + index),
            base=0.20 - index * 0.005,
            bull=0.24 - index * 0.005,
            sector=str(20 + index),
        )
        for index in range(5)
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=1_000_000,
            candidates=candidates,
        )
    )

    invested = [line for line in result.lines if line.target_weight > 0]
    assert len(invested) == 1
    assert invested[0].symbol == "2000"
    assert sum(line.target_weight for line in invested) <= 0.35 + 1e-9
    assert result.target_cash_weight >= 0.10 - 1e-9


def test_default_single_industry_profile_does_not_cap_sector_at_half() -> None:
    engine = PortfolioAllocationEngine(
        PortfolioAllocationConfig(
            single_stock_weight_limit=0.90,
            correlation_cluster_weight_limit=0.90,
        )
    )
    candidate = _candidate("2050", sector="24", cluster="ELECTRONICS")

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=1_000_000,
            candidates=(candidate,),
        )
    )

    assert engine.config.sector_weight_limit == pytest.approx(1.0)
    assert sum(line.target_weight for line in result.lines) == pytest.approx(0.90)
    assert dict(result.sector_weights)["24"] == pytest.approx(0.90)
    assert result.target_cash_weight == pytest.approx(0.10)


def test_building_adds_second_position_without_pairwise_dominance() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding("2150", quality_score=90, cluster="INCUMBENT")
    challenger = _candidate(
        "2151",
        quality_score=80,
        cluster="COMPLEMENT",
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=700_000,
            holdings=(holding,),
            candidates=(challenger,),
        )
    )

    line = _line(result, "2151")
    assert result.mode is PortfolioMode.BUILDING
    assert line.evaluation.absolute_strategic_candidate is True
    assert line.evaluation.dual_dominance_candidate is None
    assert line.dual_dominance_gate == "NOT_APPLICABLE"
    assert line.action is PortfolioAction.ADD
    assert line.target_weight > 0


def test_building_still_enforces_correlation_limit() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding(
        "2160",
        weight=0.50,
        cluster="SHARED_CLUSTER",
    )
    challenger = _candidate("2161", cluster="SHARED_CLUSTER")

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=500_000,
            holdings=(holding,),
            candidates=(challenger,),
        )
    )

    line = _line(result, "2161")
    assert result.mode is PortfolioMode.BUILDING
    assert line.evaluation.executable_candidate is True
    assert line.target_weight == 0
    assert line.action is PortfolioAction.WATCH
    assert line.portfolio_risk_gate == "FAIL"
    assert result.reason == "BUILDING_NO_PORTFOLIO_FEASIBLE_SECOND_POSITION"


def test_cold_start_slot_ranking_uses_capital_claim_not_cer_alone() -> None:
    engine = PortfolioAllocationEngine(PortfolioAllocationConfig(max_stocks=1))
    high_cer_small_execution = _candidate(
        "2101",
        bear=0.18,
        base=0.22,
        bull=0.26,
        les_decision="BUY_SMALL",
    )
    lower_cer_full_execution = _candidate(
        "2102",
        bear=0.15,
        base=0.19,
        bull=0.23,
        les_decision="BUY",
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=1_000_000,
            candidates=(high_cer_small_execution, lower_cer_full_execution),
        )
    )

    assert _line(result, "2102").target_weight > 0
    assert _line(result, "2101").target_weight == 0


def test_cer_applies_configured_confidence_penalties() -> None:
    engine = PortfolioAllocationEngine()
    confident = _candidate("3001", data_confidence=100, valuation_confidence=100)
    uncertain = _candidate("3002", data_confidence=50, valuation_confidence=50)

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=1_000_000,
            candidates=(confident, uncertain),
        )
    )

    confident_cer = _line(result, "3001").evaluation.cer
    uncertain_cer = _line(result, "3002").evaluation.cer
    assert confident_cer == pytest.approx(0.16)
    assert uncertain_cer is not None
    assert uncertain_cer < confident_cer


def test_missing_holding_data_never_forces_liquidation() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding(
        "4001",
        confidence=None,
        erem_data_sufficient=False,
    )
    holding = AllocationSecurity(
        **{
            **holding.__dict__,
            "valuation_status": "UNKNOWN",
            "bear_cagr": None,
            "base_cagr": None,
            "bull_cagr": None,
        }
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=700_000,
            holdings=(holding,),
        )
    )

    line = _line(result, "4001")
    assert line.action is PortfolioAction.HOLD_DATA_INSUFFICIENT
    assert line.target_weight == pytest.approx(0.30)


def test_single_erem_exit_warning_is_held_for_confirmation() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding("4101", confidence=35, exit_streak=1)

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=700_000,
            holdings=(holding,),
        )
    )

    line = _line(result, "4101")
    assert line.action is PortfolioAction.HOLD_REVIEW
    assert line.target_weight == pytest.approx(0.30)
    assert line.erem_exit_persistent is False
    assert line.exit_confirmation_count == 0


def test_high_quality_holding_reduces_when_two_exit_signals_confirm() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding(
        "4102",
        bear=-0.06,
        base=-0.03,
        bull=0.0,
        confidence=35,
        exit_streak=2,
        quality_score=85,
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=700_000,
            holdings=(holding,),
        )
    )

    line = _line(result, "4102")
    assert line.action is PortfolioAction.REDUCE_25
    assert line.high_quality_protected is True
    assert line.erem_exit_persistent is True
    assert line.valuation_broken is True
    assert line.trend_broken is False
    assert line.exit_confirmation_count == 2
    assert line.target_weight == pytest.approx(0.225)


def test_all_three_soft_exit_signals_confirm_full_exit() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding(
        "4103",
        bear=-0.06,
        base=-0.03,
        bull=0.0,
        confidence=35,
        exit_streak=2,
        quality_score=85,
        ema20=110,
        ema20_slope=-0.10,
        distribution_warning=True,
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=700_000,
            holdings=(holding,),
        )
    )

    line = _line(result, "4103")
    assert line.action is PortfolioAction.EXIT_TO_CASH
    assert line.exit_confirmation_count == 3
    assert line.reason == "CONFIRMED_EREM_EXIT"
    assert line.target_weight == 0


def test_strategy_stop_remains_an_immediate_hard_exit() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding(
        "4104",
        confidence=75,
        strategy_stop=101,
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=700_000,
            holdings=(holding,),
        )
    )

    line = _line(result, "4104")
    assert line.action is PortfolioAction.EXIT_TO_CASH
    assert line.strategy_stop_breached is True
    assert line.reason == "STRATEGY_STOP_BREACHED"


def test_high_quality_moderate_erem_weakness_holds_without_adding() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding("4105", confidence=45, quality_score=85)

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=700_000,
            holdings=(holding,),
        )
    )

    line = _line(result, "4105")
    assert line.action is PortfolioAction.HOLD_NO_ADD
    assert line.reason == "QUALITY_PROTECTED_HOLD_NO_ADD"


def test_fundamental_hard_reject_exits_before_opportunity_comparison() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding("5001", hard_reject=True)
    challenger = _candidate("5002")

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=700_000,
            holdings=(holding,),
            candidates=(challenger,),
        )
    )

    holding_line = _line(result, "5001")
    challenger_line = _line(result, "5002")
    assert holding_line.action is PortfolioAction.EXIT_AND_REALLOCATE
    assert holding_line.target_weight == 0
    assert challenger_line.target_weight > 0


def test_normal_rotation_requires_two_of_three_persistent_sessions() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding("6001")
    companion = _holding("6000", weight=0.20, cluster="COMPANION")
    challenger = _candidate("6002", bear=0.15, base=0.19, bull=0.23)
    evidence = PersistenceEvidence(
        incumbent_symbol="6001",
        challenger_symbol="6002",
        recent_excess_opportunities=(-0.01, 0.02, 0.03),
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=500_000,
            holdings=(holding, companion),
            candidates=(challenger,),
            persistence=(evidence,),
        )
    )

    holding_line = _line(result, "6001")
    assert holding_line.persistence_passed is True
    assert holding_line.action in {
        PortfolioAction.ROTATE_25,
        PortfolioAction.ROTATE_50,
        PortfolioAction.ROTATE_75,
    }
    assert _line(result, "6002").target_weight > 0


def test_one_session_edge_does_not_rotate() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding("6101")
    companion = _holding("6100", weight=0.20, cluster="COMPANION")
    challenger = _candidate("6102", bear=0.15, base=0.19, bull=0.23)
    evidence = PersistenceEvidence(
        incumbent_symbol="6101",
        challenger_symbol="6102",
        recent_excess_opportunities=(-0.01, -0.01, 0.03),
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=500_000,
            holdings=(holding, companion),
            candidates=(challenger,),
            persistence=(evidence,),
        )
    )

    line = _line(result, "6101")
    assert line.action is PortfolioAction.HOLD_NO_ADD
    assert line.persistence_passed is False
    assert "PERSISTENCE_NOT_CONFIRMED" in line.reason


def test_steady_state_higher_cer_without_quality_advantage_cannot_receive_capital() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding("6201", quality_score=85)
    companion = _holding(
        "6200",
        weight=0.20,
        quality_score=86,
        cluster="COMPANION",
    )
    challenger = _candidate(
        "6202",
        bear=0.20,
        base=0.24,
        bull=0.28,
        quality_score=85,
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=500_000,
            holdings=(holding, companion),
            candidates=(challenger,),
        )
    )

    candidate_line = _line(result, "6202")
    assert candidate_line.evaluation.absolute_strategic_candidate is True
    assert candidate_line.evaluation.dual_dominance_candidate is False
    assert candidate_line.quality_dominance_gate == "FAIL"
    assert candidate_line.cer_dominance_gate == "PASS"
    assert candidate_line.target_weight == 0
    assert result.best_strategic_alternative is None


def test_steady_state_quality_advantage_without_cer_advantage_cannot_receive_capital() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding("6301", quality_score=80)
    companion = _holding(
        "6300",
        weight=0.20,
        quality_score=80,
        cluster="COMPANION",
    )
    challenger = _candidate(
        "6302",
        bear=0.11,
        base=0.13,
        bull=0.15,
        quality_score=90,
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=500_000,
            holdings=(holding, companion),
            candidates=(challenger,),
        )
    )

    candidate_line = _line(result, "6302")
    assert candidate_line.quality_dominance_gate == "PASS"
    assert candidate_line.cer_dominance_gate == "FAIL"
    assert candidate_line.dual_dominance_gate == "FAIL"
    assert candidate_line.target_weight == 0


def test_best_strategic_alternative_is_quality_first_after_dual_dominance() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding("6401", quality_score=75)
    companion = _holding(
        "6400",
        weight=0.20,
        quality_score=75,
        cluster="COMPANION",
    )
    higher_quality = _candidate(
        "6402",
        bear=0.14,
        base=0.18,
        bull=0.22,
        quality_score=95,
        les_decision="WATCH",
    )
    higher_cer = _candidate(
        "6403",
        bear=0.20,
        base=0.24,
        bull=0.28,
        quality_score=85,
        les_decision="WATCH",
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=500_000,
            holdings=(holding, companion),
            candidates=(higher_quality, higher_cer),
        )
    )

    assert result.best_strategic_alternative == "6402"
    assert result.best_executable_alternative is None


def test_obvious_opportunity_can_bypass_persistence_but_not_hard_gates() -> None:
    engine = PortfolioAllocationEngine()
    holding = _holding("7001")
    companion = _holding(
        "7000",
        weight=0.20,
        quality_score=95,
        cluster="COMPANION",
    )
    obvious = _candidate(
        "7002",
        bear=0.20,
        base=0.24,
        bull=0.28,
        quality_score=95,
        data_confidence=95,
        valuation_confidence=95,
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=500_000,
            holdings=(holding, companion),
            candidates=(obvious,),
        )
    )

    line = _line(result, "7001")
    assert line.action is PortfolioAction.FAST_TRACK_ROTATE
    assert line.obvious_opportunity is True

    blocked = AllocationSecurity(
        **{**obvious.__dict__, "symbol": "7003", "risk_gate": "FAIL"}
    )
    blocked_result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=500_000,
            holdings=(holding, companion),
            candidates=(blocked,),
        )
    )
    assert blocked_result.best_executable_alternative is None
    assert _line(blocked_result, "7001").action is PortfolioAction.HOLD_NO_ADD


def test_full_portfolio_does_not_create_a_fifth_partial_rotation_position() -> None:
    engine = PortfolioAllocationEngine()
    holdings = tuple(
        _holding(
            str(8000 + index),
            weight=0.20,
            sector=str(20 + index),
            cluster=str(8000 + index),
        )
        for index in range(4)
    )
    challenger = _candidate("9000", bear=0.18, base=0.22, bull=0.26, sector="40")
    evidence = tuple(
        PersistenceEvidence(
            incumbent_symbol=holding.symbol,
            challenger_symbol="9000",
            recent_excess_opportunities=(0.03, 0.03, 0.03),
        )
        for holding in holdings
    )

    result = engine.allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=200_000,
            holdings=holdings,
            candidates=(challenger,),
            persistence=evidence,
        )
    )

    assert _line(result, "9000").target_weight == 0
    assert sum(line.target_weight > 0 for line in result.lines) == 4
    assert all(
        line.action not in {
            PortfolioAction.ROTATE_25,
            PortfolioAction.ROTATE_50,
            PortfolioAction.ROTATE_75,
            PortfolioAction.FAST_TRACK_ROTATE,
        }
        for line in result.lines
    )


def test_dataframe_adapter_and_plugin_registry_are_compatible() -> None:
    frame = pd.DataFrame(
        {
            "Stock Code": ["9901"],
            "Portfolio Role": ["CANDIDATE"],
            "Quality Status": ["QUALITY_CONFIRMED"],
            "Quality Score": [90],
            "Data Confidence": [90],
            "Valuation Status": ["COMPLETE"],
            "Valuation Gate": ["PASS"],
            "Bear CAGR": [0.12],
            "Base CAGR": [0.16],
            "Bull CAGR": [0.20],
            "Valuation Confidence": [90],
            "LES Decision": ["BUY"],
            "Risk Gate": ["PASS"],
            "Falling Knife State": ["CLEAR"],
            "Executable Shares": [100],
            "Sector": ["24"],
            "Correlation Cluster": ["ELECTRONICS"],
            "Account Equity": [1_000_000],
            "Current Cash": [1_000_000],
        }
    )

    result = PortfolioAllocationV1Strategy().run(frame)

    assert "portfolio_allocation_v1" in available_strategies()
    assert get_strategy("portfolio_allocation_v1").version == "1.1.5"
    assert result.loc[0, "PAE Decision"] == "ADD"
    assert result.loc[0, "PAE Target Weight"] > 0
    assert result.loc[0, "PAE Model Version"] == "1.1.5"


def test_invalid_scenario_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        PortfolioAllocationConfig(
            bear_probability=0.5,
            base_probability=0.5,
            bull_probability=0.5,
        )


def test_portfolio_state_must_reconcile_to_investable_equity() -> None:
    engine = PortfolioAllocationEngine()
    with pytest.raises(ValueError, match="reconcile"):
        engine.allocate(
            PortfolioAllocationRequest(
                account_equity=1_000_000,
                current_cash=100_000,
                holdings=(_holding("9999", weight=0.30),),
            )
        )

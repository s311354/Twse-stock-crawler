"""Contracts for the isolated PAE Pandas compatibility boundary."""

import pandas as pd

from domain.strategies.portfolio_allocation_dataframe import (
    PortfolioAllocationDataFrameAdapter,
)
from domain.strategies.portfolio_allocation_dataframe import decision_to_frame
from domain.strategies.portfolio_allocation_v1 import AllocationSecurity
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationEngine
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationRequest
from domain.strategies.portfolio_allocation_v1 import PortfolioAllocationV1Strategy


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Stock Code": ["2330"],
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
            "Correlation Cluster": ["SEMI"],
            "Account Equity": [1_000_000],
            "Current Cash": [1_000_000],
        }
    )


def _candidate() -> AllocationSecurity:
    return AllocationSecurity(
        symbol="2330",
        sector="24",
        correlation_cluster="SEMI",
        quality_status="QUALITY_CONFIRMED",
        quality_score=90,
        data_confidence=90,
        valuation_status="COMPLETE",
        valuation_gate="PASS",
        bear_cagr=0.12,
        base_cagr=0.16,
        bull_cagr=0.20,
        valuation_confidence=90,
        les_decision="BUY",
        risk_gate="PASS",
        falling_knife_state="CLEAR",
        executable_shares=100,
    )


def test_strategy_dataframe_api_delegates_without_output_drift() -> None:
    strategy = PortfolioAllocationV1Strategy()
    frame = _candidate_frame()

    direct = PortfolioAllocationDataFrameAdapter(strategy.engine).run(frame)
    compatibility = strategy.run(frame)

    pd.testing.assert_frame_equal(compatibility, direct)


def test_decision_to_frame_compatibility_method_has_schema_parity() -> None:
    decision = PortfolioAllocationEngine().allocate(
        PortfolioAllocationRequest(
            account_equity=1_000_000,
            current_cash=1_000_000,
            candidates=(_candidate(),),
        )
    )

    direct = decision_to_frame(decision)
    compatibility = decision.to_frame()

    pd.testing.assert_frame_equal(compatibility, direct)
    assert compatibility.columns.tolist()[-1] == "PAE Model Version"

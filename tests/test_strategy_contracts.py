import datetime
import logging
from dataclasses import FrozenInstanceError
from dataclasses import replace

import pandas as pd
import pytest

from application.contracts import AnalysisContext
from application.contracts import MarketStatus
from application.contracts import StockAnalysisRequest
from application.market_history import MarketHistoryInput
from application.strategy_contracts import StrategyContractError
from application.strategy_contracts import StrategyDecisionAssembler
from domain.model_contracts import FieldSource
from domain.model_contracts import FieldUnit
from infrastructure.report.html_renderer import RendererFactory
from infrastructure.stock_analysis_adapters import CrawlerNotificationAdapter
from infrastructure.stock_analysis_adapters import CrawlerReportAdapter
from infrastructure.stock_analysis_adapters import CrawlerStrategyAdapter


def market_input(
    *,
    signal_date: str = "2026-07-25T13:30:00+08:00",
    fundamental_available_at: str = "2026-07-24T18:00:00+08:00",
) -> MarketHistoryInput:
    return MarketHistoryInput(
        stock_count=1,
        stock_numbers=["2330"],
        stock_names=["台積電"],
        dates=[["2026-07-25"]],
        opens=[[990_000.0]],
        highs=[[1_010_000.0]],
        lows=[[980_000.0]],
        closes=[[1_000_000.0]],
        volumes=[[20_000.0]],
        trade_counts=[[100]],
        trade_values=[[20_000_000.0]],
        bids=[[999_000.0]],
        asks=[[1_001_000.0]],
        pe_ratios=[[20.0]],
        taiex_closes=[[23_000.0]],
        annual_fundamentals={},
        fundamental_snapshots={
            "2330": {
                "available_at": fundamental_available_at,
                "twse_financial_observed_at": (
                    fundamental_available_at
                ),
                "base_quality_feature_sources": (
                    "revenue_growth=TWSE_OPENAPI_HISTORY"
                ),
            }
        },
        fundamental_statuses={},
        fundamental_status_details={},
        signal_dates={"2330": signal_date},
        account_equity=1_000_000.0,
    )


def valid_row(**overrides: object) -> dict[str, object]:
    row = {
        "日期": "2026-07-25",
        "證券代號": "2330",
        "證券名稱": "台積電",
        "收盤": 1_000_000.0,
        "LES v5 Final Quality Status": "QUALITY_CONFIRMED",
        "Base Quality Status": "BASE_PASS",
        "Base Data Confidence": 90.0,
        "Base Available Weight": 80.0,
        "Valuation Gate": "PASS",
        "Valuation Model Status": "COMPLETE",
        "Valuation Model Version": "les-v5-valuation-1.0.0",
        "Valuation Confidence": 88.0,
        "Bear Case Expected CAGR": 0.06,
        "Base Case Expected CAGR": 0.13,
        "Bull Case Expected CAGR": 0.20,
        "Base Case Fair Value": 1_120_000.0,
        "Bear Case Fair Value": 880_000.0,
        "Bear Case Permanent Loss": 0.12,
        "LES Model Version": "5.0.0",
        "低點決策": "BUY",
        "低點理由": "all gates pass",
        "EREM 決策訊號": "HOLD",
        "EREM 理由": "trend intact",
    }
    row.update(overrides)
    return row


def test_assembler_builds_frozen_versioned_field_evidence() -> None:
    decision = StrategyDecisionAssembler().assemble(
        pd.DataFrame([valid_row()]),
        market_input(),
    )[0]

    close = decision.field("收盤")
    valuation = decision.field("Base Case Fair Value")

    assert decision.schema_version == "1.0.0"
    assert close.source is FieldSource.TWSE_MARKET
    assert close.unit is FieldUnit.TWD_SCALED_1000
    assert close.confidence == 100
    assert close.model_version == "twse-daily-v1"
    assert valuation.source is FieldSource.LES_V5_VALUATION
    assert valuation.model_version == "les-v5-valuation-1.0.0"
    assert valuation.available_at <= decision.evaluated_at
    with pytest.raises(FrozenInstanceError):
        close.value = 0


def test_assembler_skips_slot_without_valid_market_history(caplog) -> None:
    market = market_input()
    market = replace(
        market,
        stock_count=2,
        stock_numbers=["2330", ""],
        stock_names=["台積電", ""],
        dates=[["2026-07-25"], []],
        opens=[[990_000.0], []],
        highs=[[1_010_000.0], []],
        lows=[[980_000.0], []],
        closes=[[1_000_000.0], []],
        volumes=[[20_000.0], []],
        trade_counts=[[100], []],
        trade_values=[[20_000_000.0], []],
        bids=[[999_000.0], []],
        asks=[[1_001_000.0], []],
        pe_ratios=[[20.0], []],
        taiex_closes=[[23_000.0], []],
    )
    invalid = valid_row(
        **{
            "證券代號": "",
            "證券名稱": "",
            "收盤": None,
            "Base Available Weight": None,
            "Valuation Gate": None,
            "Valuation Model Status": None,
            "Valuation Model Version": None,
        }
    )

    with caplog.at_level(logging.DEBUG):
        decisions = StrategyDecisionAssembler().assemble(
            pd.DataFrame([valid_row(), invalid]),
            market,
        )

    assert len(decisions) == 1
    assert decisions[0].value("證券代號") == "2330"
    assert "no valid TWSE market history" in caplog.text
    assert "Skipping stock <unresolved>" in caplog.text
    assert "analysis slot" not in caplog.text
    assert caplog.records[-1].levelno == logging.DEBUG
    assert caplog.records[-1].status == "ANALYSIS_SLOT_SKIPPED"


def test_contract_rejects_future_point_in_time_evidence() -> None:
    with pytest.raises(
        StrategyContractError,
        match="future point-in-time evidence",
    ):
        StrategyDecisionAssembler().assemble(
            pd.DataFrame([valid_row()]),
            market_input(
                fundamental_available_at=(
                    "2026-07-26T18:00:00+08:00"
                )
            ),
        )


def test_mixed_base_lineage_uses_latest_source_availability() -> None:
    market = market_input()
    snapshot = market.fundamental_snapshots["2330"]
    snapshot["twse_financial_observed_at"] = (
        "2026-07-23T13:30:00+08:00"
    )
    snapshot["base_quality_feature_sources"] = (
        "revenue_growth=TWSE_OPENAPI_HISTORY;"
        "roe_change=MOPS_POINT_IN_TIME_ANNUAL_FALLBACK"
    )

    decision = StrategyDecisionAssembler().assemble(
        pd.DataFrame([valid_row()]),
        market,
    )[0]
    base = decision.field("Base Quality Status")

    assert base.source is FieldSource.MIXED_FUNDAMENTALS
    assert base.available_at.isoformat().startswith("2026-07-24T10:00:00")


def test_contract_rejects_base_pass_without_required_coverage() -> None:
    with pytest.raises(
        StrategyContractError,
        match="below 70",
    ):
        StrategyDecisionAssembler().assemble(
            pd.DataFrame([valid_row(**{"Base Available Weight": 65})]),
            market_input(),
        )


def test_contract_rejects_complete_valuation_with_missing_scenario() -> None:
    with pytest.raises(
        StrategyContractError,
        match="missing scenario evidence",
    ):
        StrategyDecisionAssembler().assemble(
            pd.DataFrame([valid_row(**{"Bear Case Fair Value": None})]),
            market_input(),
        )


def test_contract_rejects_buy_without_complete_valuation() -> None:
    with pytest.raises(
        StrategyContractError,
        match="complete valuation evidence",
    ):
        StrategyDecisionAssembler().assemble(
            pd.DataFrame(
                [
                    valid_row(
                        **{
                            "Valuation Model Status": "UNKNOWN",
                            "Valuation Gate": "UNKNOWN",
                        }
                    )
                ]
            ),
            market_input(),
        )


def test_renderer_rejects_dataframe_boundary_input() -> None:
    renderer = RendererFactory.get_renderer("low_entry_score_v5")

    with pytest.raises(
        TypeError,
        match="StrategyDecision objects only",
    ):
        renderer.render(pd.DataFrame([valid_row()]))


def test_projector_preserves_validated_values_for_csv_compatibility() -> None:
    assembler = StrategyDecisionAssembler()
    decisions = assembler.assemble(
        pd.DataFrame([valid_row()]),
        market_input(),
    )

    projected = assembler.project(decisions)

    assert projected.loc[0, "證券代號"] == "2330"
    assert projected.loc[0, "Valuation Gate"] == "PASS"
    assert projected.loc[0, "EREM 決策訊號"] == "HOLD"


def test_production_adapters_reuse_one_typed_evaluation() -> None:
    decisions = StrategyDecisionAssembler().assemble(
        pd.DataFrame([valid_row()]),
        market_input(),
    )

    class Crawler:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def cal_max_profit(self):
            self.calls.append("profit")
            return [[1, 1, 1, 1, 0.01]]

        def build_strategy_decisions(self, maxprofits):
            self.calls.append("decisions")
            return decisions

        def record_strategy_decisions(self, **kwargs):
            assert kwargs["decisions"] is decisions
            self.calls.append("report")

        def smtp_strategy_email(self, **kwargs):
            assert kwargs["decisions"] is decisions
            self.calls.append("mail")

        def record_analysis_dataset(self, **kwargs):
            raise AssertionError("legacy DataFrame path must not run")

        def smtp_email(self, **kwargs):
            raise AssertionError("legacy email path must not run")

    crawler = Crawler()
    context = AnalysisContext(
        resource=crawler,
        market_date=datetime.date(2026, 7, 25),
        market_status=MarketStatus.CLOSED,
    )
    request = StockAnalysisRequest(
        stocklist=["2330"],
        stocktype="ELEC",
        output_file_names="shirong",
        endbacktrack=3,
        mail=True,
    )

    evaluation = CrawlerStrategyAdapter().evaluate(context)
    CrawlerReportAdapter().generate(context, request, evaluation)
    CrawlerNotificationAdapter().send(context, request, evaluation)

    assert evaluation.decisions is decisions
    assert crawler.calls == ["profit", "decisions", "report", "mail"]

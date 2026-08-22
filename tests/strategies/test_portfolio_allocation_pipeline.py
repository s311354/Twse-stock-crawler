import datetime

from application.portfolio_allocation import PortfolioAllocationPipeline
from application.portfolio_allocation import PortfolioRuntimeContext
from application.contracts import AnalysisContext
from application.contracts import MarketStatus
from domain.model_contracts import DecisionField
from domain.model_contracts import FieldSource
from domain.model_contracts import FieldUnit
from domain.model_contracts import StrategyDecision
from domain.models import PortfolioPosition
from infrastructure.stock_analysis_adapters import CrawlerStrategyAdapter


EVALUATED_AT = datetime.datetime(
    2026,
    8,
    10,
    13,
    30,
    tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
)


def daily_decision(symbol: str = "2330") -> StrategyDecision:
    values = {
        "日期": "2026-08-10",
        "證券代號": symbol,
        "證券名稱": "台積電",
        "收盤": 1_000_000.0,
        "LES v5 Final Quality Status": "QUALITY_CONFIRMED",
        "LES v5 Final Quality Score": 90.0,
        "Base Data Confidence": 92.0,
        "Extended Data Confidence": 88.0,
        "Quality Hard Rejects": "NONE",
        "Valuation Model Status": "COMPLETE",
        "Valuation Gate": "PASS",
        "Valuation Confidence": 90.0,
        "Bear Case Expected CAGR": 0.12,
        "Base Case Expected CAGR": 0.18,
        "Bull Case Expected CAGR": 0.24,
        "低點分數": 84.0,
        "低點決策": "BUY",
        "Risk Gate": "PASS",
        "Falling Knife State": "REVERSAL_CONFIRMED",
        "建議股數": 1000.0,
        "Position Fraction": 1 / 3,
        "EREM 分數": 78.0,
        "EREM Holding Confidence": 80.0,
        "EREM 決策訊號": "HOLD",
        "EREM 理由": "趨勢完整；決策=HOLD",
    }
    fields = []
    for name, value in values.items():
        fields.append(
            DecisionField(
                name=name,
                value=value,
                source=(
                    FieldSource.TWSE_MARKET
                    if name == "收盤"
                    else FieldSource.LES_V5
                ),
                available_at=EVALUATED_AT,
                confidence=100,
                unit=(
                    FieldUnit.TWD_SCALED_1000
                    if name == "收盤"
                    else FieldUnit.TEXT
                ),
                model_version="test-v1",
            )
        )
    return StrategyDecision(
        strategy_name="daily_analysis_v1",
        evaluated_at=EVALUATED_AT,
        fields=tuple(fields),
    )


def test_pipeline_allocates_an_eligible_cold_start_candidate() -> None:
    enriched = PortfolioAllocationPipeline().enrich(
        (daily_decision(),),
        PortfolioRuntimeContext(
            account_equity=1_000_000,
            default_sector="24",
        ),
    )

    decision = enriched[0]
    assert decision.value("PAE Status") == "SUCCESS"
    assert decision.value("PAE Mode") == "COLD_START"
    assert decision.value("PAE Portfolio Input") == "COLD_START_ALL_CASH"
    assert decision.value("PAE Decision") == "ADD"
    assert decision.value("PAE CER") > decision.value("PAE Required Return")
    assert decision.value("PAE Target Weight") > 0
    assert decision.value("PAE Target Cash Weight") >= 0.10
    assert decision.field("PAE Decision").source is FieldSource.PAE_V1


def test_pipeline_values_symbol_only_holdings_as_equal_weight_reference() -> None:
    symbols = ("2449", "3008", "3016", "6271")
    enriched = PortfolioAllocationPipeline().enrich(
        tuple(daily_decision(symbol) for symbol in symbols),
        PortfolioRuntimeContext(
            account_equity=1_000_000,
            positions=tuple(PortfolioPosition(symbol=symbol) for symbol in symbols),
            sector_by_symbol={
                symbol: "SECTOR_{}".format(index)
                for index, symbol in enumerate(symbols)
            },
        ),
    )

    assert all(item.value("PAE Status") == "SUCCESS" for item in enriched)
    assert all(item.value("PAE Mode") == "STEADY_STATE" for item in enriched)
    assert all(
        item.value("PAE Portfolio Input") == "SYMBOL_ONLY_EQUAL_WEIGHT"
        for item in enriched
    )
    assert all(
        abs(float(item.value("PAE Current Weight")) - 0.225) < 1e-9
        for item in enriched
    )
    assert all(
        abs(float(item.value("PAE Target Cash Weight")) - 0.10) < 1e-9
        for item in enriched
    )


def test_pipeline_reports_building_for_one_effective_holding() -> None:
    enriched = PortfolioAllocationPipeline().enrich(
        (daily_decision(),),
        PortfolioRuntimeContext(
            account_equity=1_000_000,
            positions=(PortfolioPosition(symbol="2330"),),
            default_sector="24",
        ),
    )

    assert enriched[0].value("PAE Mode") == "BUILDING"


def test_pipeline_marks_missing_configured_holding_as_data_insufficient() -> None:
    enriched = PortfolioAllocationPipeline().enrich(
        (daily_decision(),),
        PortfolioRuntimeContext(
            account_equity=1_000_000,
            positions=(PortfolioPosition(symbol="2317", shares=1000),),
            default_sector="24",
        ),
    )

    assert enriched[0].value("PAE Status") == "DATA_INSUFFICIENT"
    assert enriched[0].value("PAE Mode") == "BUILDING"
    assert enriched[0].value("PAE Decision") == "DATA_INSUFFICIENT"
    assert "CONFIGURED_HOLDINGS_MISSING_FROM_ANALYSIS=2317" in str(
        enriched[0].value("PAE Reason")
    )


def test_crawler_strategy_adapter_runs_pae_after_daily_strategies() -> None:
    class Crawler:
        analysis_universe = "ELEC"
        fundamental_snapshots_by_stock = {"2330": {"sector": "24"}}

        def cal_max_profit(self):
            return [[1, 1, 1, 1, 0.1]]

        def build_strategy_decisions(self, maxprofits):
            assert maxprofits
            return (daily_decision(),)

    evaluation = CrawlerStrategyAdapter(
        portfolio_allocator=PortfolioAllocationPipeline(),
        account_equity=1_000_000,
    ).evaluate(
        AnalysisContext(
            resource=Crawler(),
            market_date=datetime.date(2026, 8, 10),
            market_status=MarketStatus.OPEN,
        )
    )

    assert evaluation.decisions[0].value("PAE Status") == "SUCCESS"
    assert evaluation.decisions[0].value("PAE Decision") == "ADD"

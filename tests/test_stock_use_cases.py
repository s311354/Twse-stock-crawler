import datetime
from pathlib import Path

import pytest

from application.contracts import AnalysisContext
from application.contracts import MarketStatus
from application.contracts import StockAnalysisRequest
from application.contracts import StrategyEvaluation
from application.stock_service import StockAnalysisService
from application.use_cases import EvaluateStrategies
from application.use_cases import FetchMarketData
from application.use_cases import GenerateReport
from application.use_cases import RefreshFundamentals
from application.use_cases import SendNotification


TODAY = datetime.date(2026, 7, 20)
NOW = datetime.datetime(
    2026,
    7,
    20,
    13,
    30,
    tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
)


def request(**overrides) -> StockAnalysisRequest:
    values = {
        "stocklist": ["2330"],
        "stocktype": "ELEC",
        "output_file_names": "shirong",
        "endbacktrack": 3,
    }
    values.update(overrides)
    return StockAnalysisRequest(**values)


def test_service_sequences_the_five_use_cases() -> None:
    calls: list[str] = []
    context = AnalysisContext(object(), TODAY, MarketStatus.OPEN)
    evaluation = StrategyEvaluation(maxprofits=[])

    class UseCase:
        def __init__(self, name, result=None):
            self.name = name
            self.result = result

        def execute(self, *args):
            calls.append(self.name)
            return self.result

    class LineChart:
        def generate(self, analysis_request):
            calls.append("line_chart")

    service = StockAnalysisService(
        fetch_market_data=UseCase("fetch", context),
        refresh_fundamentals=UseCase("fundamentals"),
        evaluate_strategies=UseCase("strategies", evaluation),
        generate_report=UseCase("report"),
        send_notification=UseCase("notification"),
        line_chart=LineChart(),
    )

    assert service.run(request()) is True
    assert calls == [
        "fetch",
        "fundamentals",
        "strategies",
        "report",
        "notification",
    ]


def test_fetch_market_data_preserves_closed_market_context(capsys) -> None:
    class Calendar:
        def is_trading_day(self, target_date):
            return False

    class MarketData:
        def fetch(
            self,
            analysis_request,
            *,
            calendar,
            market_date,
            is_trading_day,
        ):
            return AnalysisContext(
                resource="history",
                market_date=market_date,
                market_status=(
                    MarketStatus.OPEN
                    if is_trading_day
                    else MarketStatus.CLOSED
                ),
            )

    result = FetchMarketData(MarketData(), Calendar(), lambda: TODAY).execute(
        request()
    )

    assert result.market_status is MarketStatus.CLOSED
    assert result.resource == "history"
    assert "TWSE market status for 2026-07-20: CLOSED" in capsys.readouterr().out


def test_optional_ports_are_not_called_when_disabled() -> None:
    calls: list[str] = []
    context = AnalysisContext(object(), TODAY, MarketStatus.OPEN)
    evaluation = StrategyEvaluation(maxprofits=[])

    class Fundamentals:
        def refresh(self, *args, **kwargs):
            calls.append("fundamentals")

    class Notifications:
        def send(self, *args, **kwargs):
            calls.append("notification")

    RefreshFundamentals(Fundamentals(), lambda: NOW).execute(
        context,
        request(mops_enabled=False),
    )
    SendNotification(Notifications()).execute(
        context,
        request(mail=False),
        evaluation,
    )

    assert calls == []


def test_use_cases_delegate_without_mutating_results() -> None:
    context = AnalysisContext(object(), TODAY, MarketStatus.OPEN)
    expected = StrategyEvaluation(maxprofits=[[1, 2, 3]])
    captured: list[object] = []

    class Strategies:
        def evaluate(self, actual_context):
            captured.append(actual_context)
            return expected

    class Reports:
        def generate(self, actual_context, actual_request, actual_evaluation):
            captured.extend(
                [actual_context, actual_request, actual_evaluation]
            )

    actual = EvaluateStrategies(Strategies()).execute(context)
    analysis_request = request()
    GenerateReport(Reports()).execute(context, analysis_request, actual)

    assert actual is expected
    assert captured == [context, context, analysis_request, expected]


def test_application_workflow_has_no_infrastructure_imports() -> None:
    application_root = Path(__file__).parents[1] / "application"
    for module_path in sorted(application_root.glob("*.py")):
        source = module_path.read_text()
        assert "from infrastructure" not in source
        assert "import infrastructure" not in source


def test_service_emits_failed_summary_when_a_use_case_raises() -> None:
    observer_calls: list[str] = []

    class Observer:
        def start(self, analysis_request):
            observer_calls.append("STARTED")

        def finish(self, status):
            observer_calls.append(status)

        def record(self, *args, **kwargs):
            pass

    class FailingFetch:
        def execute(self, analysis_request):
            raise RuntimeError("market unavailable")

    class Unused:
        def execute(self, *args):
            raise AssertionError("later use cases must not execute")

    class LineChart:
        def generate(self, analysis_request):
            raise AssertionError("line chart must not execute")

    service = StockAnalysisService(
        fetch_market_data=FailingFetch(),
        refresh_fundamentals=Unused(),
        evaluate_strategies=Unused(),
        generate_report=Unused(),
        send_notification=Unused(),
        line_chart=LineChart(),
        observer=Observer(),
    )

    with pytest.raises(RuntimeError, match="market unavailable"):
        service.run(request())

    assert observer_calls == ["STARTED", "FAILED"]

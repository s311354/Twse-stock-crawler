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
        include_reference = None

        def fetch(
            self,
            analysis_request,
            *,
            calendar,
            market_date,
            is_trading_day,
            include_reference,
        ):
            self.include_reference = include_reference
            return AnalysisContext(
                resource="history",
                market_date=market_date,
                market_status=(
                    MarketStatus.OPEN
                    if is_trading_day
                    else MarketStatus.CLOSED
                ),
            )

    market_data = MarketData()
    result = FetchMarketData(market_data, Calendar(), lambda: TODAY).execute(
        request()
    )

    assert result.market_status is MarketStatus.CLOSED
    assert result.resource == "history"
    assert market_data.include_reference is True
    assert "TWSE market status for 2026-07-20: CLOSED" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("analysis_time", "expected"),
    [
        (datetime.time(13, 44, 59), False),
        (datetime.time(13, 45), True),
    ],
)
def test_fetch_market_data_waits_for_daily_publication_buffer(
    analysis_time,
    expected,
) -> None:
    captured: list[bool] = []

    class Calendar:
        def is_trading_day(self, target_date):
            return True

    class MarketData:
        def fetch(self, analysis_request, **kwargs):
            captured.append(kwargs["include_reference"])
            return AnalysisContext(
                resource="history",
                market_date=kwargs["market_date"],
                market_status=MarketStatus.OPEN,
            )

    now = datetime.datetime.combine(
        TODAY,
        analysis_time,
        tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
    )
    FetchMarketData(
        MarketData(),
        Calendar(),
        lambda: TODAY,
        lambda: now,
    ).execute(request())

    assert captured == [expected]


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


@pytest.mark.parametrize("changed_at", ["fetch", "fundamentals", "strategies", "report", "notification", None])
def test_version_drift_stops_at_stage_boundary_and_never_reports_success(changed_at):
    from application.daily_work import DailyWorkContext, current_work
    from application.work_version import WorkVersionChanged

    calls, statuses, events = [], [], []
    state = {"changed": False}

    def check():
        if state["changed"]:
            raise WorkVersionChanged({"source": {"modified": {"count": 1, "paths": ["synthetic.py"]}}})

    work = DailyWorkContext("SYNTHETIC", check_version=check)

    class UseCase:
        def __init__(self, name):
            self.name = name

        def execute(self, *args):
            calls.append(self.name)
            state["changed"] = self.name == changed_at
            return object()

    class Observer:
        def start(self, request):
            statuses.append("STARTED")

        def finish(self, status):
            statuses.append(status)

        def record(self, name, **metrics):
            events.append((name, metrics))

    # The native port returns its active work; nullcontext callers remain
    # compatible. A real notification provider is never used in this test.
    from contextlib import contextmanager

    @contextmanager
    def scope():
        with work.activate():
            yield work

    service = StockAnalysisService(
        fetch_market_data=UseCase("fetch"), refresh_fundamentals=UseCase("fundamentals"),
        evaluate_strategies=UseCase("strategies"), generate_report=UseCase("report"),
        send_notification=UseCase("notification"), line_chart=None,
        observer=Observer(), work_scope=scope)
    sequence = ["fetch", "fundamentals", "strategies", "report", "notification"]
    if changed_at is None:
        assert service.run(request(mail=True)) is True
        assert statuses == ["STARTED", "SUCCESS"]
        assert calls == sequence
        assert not [v for name, v in events if name == "work_version_changed"]
    else:
        with pytest.raises(WorkVersionChanged):
            service.run(request(mail=True))
        assert statuses == ["STARTED", "FAILED"]
        assert calls == sequence[:sequence.index(changed_at) + 1]
        error = next(v for name, v in events if name == "work_version_changed")
        assert error["status"] == "FAILED" and error["changes"]["source"]
    assert service.last_performance["status"] == statuses[-1]
    assert current_work() is None


def test_scope_entry_failure_is_observed_without_starting_analysis():
    from contextlib import contextmanager
    from types import SimpleNamespace
    from application.work_version import WorkVersionChanged

    statuses = []

    @contextmanager
    def failed_scope():
        raise WorkVersionChanged({"runtime": ["gil_enabled"]})
        yield

    service = StockAnalysisService(
        fetch_market_data=None, refresh_fundamentals=None, evaluate_strategies=None,
        generate_report=None, send_notification=None, line_chart=None, work_scope=failed_scope,
        observer=SimpleNamespace(start=lambda _: statuses.append("STARTED"), finish=statuses.append,
                                 record=lambda *args, **kwargs: None))
    with pytest.raises(WorkVersionChanged):
        service.run(request())
    assert statuses == ["STARTED", "FAILED"]
    assert service.last_performance["status"] == "FAILED"

"""Application facade for the TWSE stock-analysis workflow."""

from __future__ import annotations

from contextlib import nullcontext

from application.contracts import StockAnalysisRequest
from application.observability import NullRunObserver
from application.daily_performance import DailyPerformance, measure_stage
from application.work_version import WorkVersionChanged
from application.ports import LineChartPort
from application.ports import RunObserverPort
from application.use_cases import EvaluateStrategies
from application.use_cases import FetchMarketData
from application.use_cases import GenerateReport
from application.use_cases import RefreshFundamentals
from application.use_cases import SendNotification


class StockAnalysisService:
    """Sequence use cases without depending on crawler infrastructure."""

    def __init__(
        self,
        *,
        fetch_market_data: FetchMarketData,
        refresh_fundamentals: RefreshFundamentals,
        evaluate_strategies: EvaluateStrategies,
        generate_report: GenerateReport,
        send_notification: SendNotification,
        line_chart: LineChartPort,
        observer: RunObserverPort | None = None,
        work_scope=None,
        performance_sink=None,
        performance_stage_callback=None,
    ) -> None:
        self._fetch_market_data = fetch_market_data
        self._refresh_fundamentals = refresh_fundamentals
        self._evaluate_strategies = evaluate_strategies
        self._generate_report = generate_report
        self._send_notification = send_notification
        self._line_chart = line_chart
        self._observer = observer or NullRunObserver()
        self._work_scope = work_scope or nullcontext
        self._performance_sink = performance_sink
        self._performance_stage_callback = performance_stage_callback
        self.last_performance = None
        self._last_context = None

    def run(self, request: StockAnalysisRequest) -> bool:
        probe = DailyPerformance(stage_callback=self._performance_stage_callback)
        status = "FAILED"
        self._last_context = None
        with probe.activate():
            self._observer.start(request)
            try:
                with measure_stage("total"):
                    with self._work_scope() as work:
                        result = self._run(request, work=work)
                    status = "SUCCESS"
                    return result
            except WorkVersionChanged as error:
                self._observer.record("work_version_changed", source="APPLICATION", status="FAILED",
                                      **error.diagnostics)
                raise
            finally:
                # Context exit performs the final version check. Never publish
                # SUCCESS before it has passed (including line-chart jobs).
                self._observer.finish(status)
                self.last_performance = dict(status=status, **probe.snapshot())
                if self._performance_sink is not None:
                    self._performance_sink(request, self._last_context, self.last_performance)
                self._observer.record("daily_performance", source="APPLICATION", **self.last_performance)

    @staticmethod
    def _check_version(work, checkpoint):
        if work is None:
            return
        with measure_stage("version_check"):
            try:
                work.check_version()
            except WorkVersionChanged as error:
                error.diagnostics["checkpoint"] = checkpoint
                error.add_note("Version checkpoint: " + checkpoint)
                raise

    def _run(self, request: StockAnalysisRequest, *, work=None) -> bool:
        if request.linechart:
            self._line_chart.generate(request)
            return True

        with measure_stage("market_data"):
            context = self._fetch_market_data.execute(request)
            self._last_context = context
        self._check_version(work, "before_fundamentals")
        with measure_stage("fundamentals"):
            self._refresh_fundamentals.execute(context, request)
        self._check_version(work, "before_strategies")
        with measure_stage("strategies"):
            evaluation = self._evaluate_strategies.execute(context)
        self._check_version(work, "before_report")
        with measure_stage("report"):
            self._generate_report.execute(context, request, evaluation)
        self._check_version(work, "before_notification")
        if request.notification_enabled:
            with measure_stage("notification"):
                self._send_notification.execute(context, request, evaluation)
        else:
            self._send_notification.execute(context, request, evaluation)
        return True


__all__ = ["StockAnalysisRequest", "StockAnalysisService"]

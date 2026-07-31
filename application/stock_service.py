"""Application facade for the TWSE stock-analysis workflow."""

from __future__ import annotations

from application.contracts import StockAnalysisRequest
from application.observability import NullRunObserver
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
    ) -> None:
        self._fetch_market_data = fetch_market_data
        self._refresh_fundamentals = refresh_fundamentals
        self._evaluate_strategies = evaluate_strategies
        self._generate_report = generate_report
        self._send_notification = send_notification
        self._line_chart = line_chart
        self._observer = observer or NullRunObserver()

    def run(self, request: StockAnalysisRequest) -> bool:
        self._observer.start(request)
        try:
            if request.linechart:
                self._line_chart.generate(request)
                self._observer.finish("SUCCESS")
                return True

            context = self._fetch_market_data.execute(request)
            self._refresh_fundamentals.execute(context, request)
            evaluation = self._evaluate_strategies.execute(context)
            self._generate_report.execute(context, request, evaluation)
            self._send_notification.execute(context, request, evaluation)
        except Exception:
            self._observer.finish("FAILED")
            raise

        self._observer.finish("SUCCESS")
        return True


__all__ = ["StockAnalysisRequest", "StockAnalysisService"]

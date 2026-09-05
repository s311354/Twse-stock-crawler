"""Strategy-execution runtime adapter."""

from __future__ import annotations

from application.contracts import AnalysisContext
from application.contracts import StrategyEvaluation
from application.observability import NullRunObserver
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from application.ports import RunObserverPort
from domain.models import PortfolioPosition


class CrawlerStrategyAdapter:
    def __init__(
        self,
        *,
        observer: RunObserverPort | None = None,
        portfolio_allocator: PortfolioAllocationV2Pipeline | None = None,
        portfolio_positions: tuple[PortfolioPosition, ...] = (),
        portfolio_current_cash: float | None = None,
        account_equity: float = 1_000_000.0,
    ) -> None:
        self._observer = observer or NullRunObserver()
        self._portfolio_allocator = portfolio_allocator
        self._portfolio_positions = portfolio_positions
        self._portfolio_current_cash = portfolio_current_cash
        self._account_equity = account_equity

    def evaluate(self, context: AnalysisContext) -> StrategyEvaluation:
        crawler = context.resource
        if self._portfolio_allocator is not None:
            crawler.portfolio_entry_les_by_stock = {
                position.symbol: position.entry_les
                for position in self._portfolio_positions
                if position.entry_les is not None
            }
            crawler.portfolio_entry_dates_by_stock = {
                position.symbol: position.entry_date
                for position in self._portfolio_positions
                if position.entry_date is not None
            }
            crawler.portfolio_entry_decisions_by_stock = {
                position.symbol: position.entry_decision
                for position in self._portfolio_positions
                if position.entry_decision is not None
            }
        maxprofits = crawler.cal_max_profit()
        decision_builder = getattr(
            crawler,
            "build_strategy_decisions",
            None,
        )
        decisions = (
            tuple(decision_builder(maxprofits))
            if callable(decision_builder)
            else ()
        )
        if self._portfolio_allocator is not None and decisions:
            snapshots = getattr(
                crawler,
                "fundamental_snapshots_by_stock",
                {},
            )
            sector_by_symbol = {
                str(symbol): str(snapshot.get("sector", "")).strip()
                for symbol, snapshot in snapshots.items()
                if isinstance(snapshot, dict)
                and str(snapshot.get("sector", "")).strip()
            }
            decisions = self._portfolio_allocator.enrich(
                decisions,
                PortfolioRuntimeContext(
                    account_equity=self._account_equity,
                    positions=self._portfolio_positions,
                    current_cash=self._portfolio_current_cash,
                    sector_by_symbol=sector_by_symbol,
                    default_sector=str(
                        getattr(crawler, "analysis_universe", "UNKNOWN")
                    ),
                ),
            )
        evaluation = StrategyEvaluation(
            maxprofits=maxprofits,
            decisions=decisions,
        )
        self._observer.record(
            "strategy_evaluation_completed",
            trade_date=context.market_date.isoformat(),
            source="STRATEGY",
            status="SUCCESS",
            result_count=len(evaluation.maxprofits),
            pae_status=(
                evaluation.decisions[0].value("PAE Status")
                if evaluation.decisions
                else None
            ),
            pae_mode=(
                evaluation.decisions[0].value("PAE Mode")
                if evaluation.decisions
                else None
            ),
        )
        return evaluation

"""Strategy-execution runtime adapter."""

from __future__ import annotations

import time
import datetime
from collections.abc import Callable

from application.contracts import AnalysisContext
from application.contracts import StrategyEvaluation
from application.observability import NullRunObserver
from application.daily_performance import measure_stage
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
        restriction_refresh: Callable[[AnalysisContext], None] | None = None,
        now_provider: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        self._observer = observer or NullRunObserver()
        self._portfolio_allocator = portfolio_allocator
        self._portfolio_positions = portfolio_positions
        self._portfolio_current_cash = portfolio_current_cash
        self._account_equity = account_equity
        self._restriction_refresh = restriction_refresh
        self._now_provider = now_provider or (lambda: datetime.datetime.now(datetime.timezone.utc))

    def evaluate(self, context: AnalysisContext) -> StrategyEvaluation:
        started = time.perf_counter()
        crawler = context.resource
        allocator = self._portfolio_allocator
        if context.as_of is not None and allocator is not None:
            # Never read or backdate writes into live pairwise persistence.
            # A one-shot replay has no confirmed multi-session persistence.
            allocator = PortfolioAllocationV2Pipeline(engine=allocator.engine, portfolio_id="historical_replay")
        # Daily sizing is indicative. Missing cash or ambiguous holding quantities
        # must not manufacture executable shares; the execution layer sizes again.
        cash = self._portfolio_current_cash
        if cash is None:
            prices = {str(symbol): values[-1] / 1000.0 for symbol, values in zip(
                getattr(crawler, "stocknumbers", ()), getattr(crawler, "daily_closes", ())
            ) if values and values[-1] is not None}
            if all(position.shares is not None and position.symbol in prices for position in self._portfolio_positions):
                cash = max(0.0, self._account_equity - sum(position.shares * prices[position.symbol] for position in self._portfolio_positions))
        crawler.les_v6_portfolio = {
            "value": self._account_equity, "available_cash": cash,
            "intended_weight": .25, "capacity_weight": .25,
        }
        if getattr(getattr(crawler, "app_settings", None), "daily_entry_model", "v6") == "v7":
            crawler.les_v6_portfolio["positions"] = [
                {"symbol": position.symbol, "shares": position.shares}
                for position in self._portfolio_positions
            ]
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
        with measure_stage("profit_diagnostics"):
            maxprofits = crawler.cal_max_profit()
        settings = getattr(crawler, "app_settings", None)
        if (context.as_of is None
                and getattr(settings, "daily_entry_model", "v6") == "v7"
                and getattr(settings, "les_v7_native_evidence", False)):
            # Freeze one actual clock after collection for every stock/model.
            # Replay and legacy daily paths retain their existing cutoffs.
            if self._restriction_refresh is not None:
                self._restriction_refresh(context)
            cutoff = self._now_provider()
            if not isinstance(cutoff, datetime.datetime) or cutoff.utcoffset() is None:
                raise ValueError("Daily decision cutoff requires a timezone-aware clock")
            crawler.les_v7_decision_time = cutoff.astimezone(datetime.timezone.utc).isoformat()
            # This account was constructed for this native daily assessment;
            # bind it to the same frozen clock, not an inferred historical time.
            crawler.les_v6_portfolio["as_of"] = crawler.les_v7_decision_time
            capture = getattr(crawler, "les_v7_source_capture", None)
            if capture is not None:
                crawler.les_v7_source_selections = capture.selections(
                    crawler, cutoff=crawler.les_v7_decision_time)
            self._observer.record(
                "v7_daily_cutoff_fixed", source="STRATEGY", status="SUCCESS",
                decision_time=crawler.les_v7_decision_time,
                cutoff_policy="AFTER_COLLECTION_CURRENT_DAILY_ONLY",
                market_observed_at=getattr(crawler, "market_observed_at", None),
            )
        decision_builder = getattr(
            crawler,
            "build_strategy_decisions",
            None,
        )
        with measure_stage("strategy_decision_assembly"):
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
            from application.snapshot_retention import snapshot_metadata
            sector_by_symbol = {}
            for symbol in snapshots:
                metadata = snapshot_metadata(snapshots, symbol)
                if not isinstance(metadata, dict):
                    continue  # Preserve the legacy sector projection contract.
                sector = str(metadata.get("sector", "")).strip()
                if sector:
                    sector_by_symbol[str(symbol)] = sector
            with measure_stage("pae"):
                decisions = allocator.enrich(
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
            duration_seconds=round(time.perf_counter() - started, 3),
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

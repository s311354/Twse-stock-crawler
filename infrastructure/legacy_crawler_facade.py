"""Backward-compatible analysis/report facade for the legacy crawler API."""

from __future__ import annotations

import html
import json
import logging
from collections.abc import Sequence
from typing import Protocol

import pandas as pd
from tabulate import tabulate

from application.analysis_dataset import AnalysisDatasetBuilder
from application.market_history import MarketHistoryInput
from domain.features import FeatureFrame
from domain.model_contracts import StrategyDecision
from domain.models import ProfitRow
from domain.models import SignalRow
from domain.models import StockRows
from domain.services import max_profit
from domain.services import max_profit_k_transactions
from domain.services import max_profit_unlimited
from domain.services import max_profit_with_fee
from infrastructure.notification.mail import SMTPEmail
from infrastructure.report.html_renderer import RendererFactory
from infrastructure.storage.chart_repository import save_profit_ratio_chart
from infrastructure.storage.csv_repository import CsvRepository


class CsvRepositoryPort(Protocol):
    def write_analysis_dataset(
        self,
        file_name: str,
        dataset: pd.DataFrame,
    ) -> None: ...

    def write_daily_rows(
        self,
        file_name: str,
        scheduled_time: str,
        row_data: StockRows,
    ) -> None: ...


class LegacyCrawlerFacade:
    """Preserve TwseCrawker methods while delegating extracted services."""

    def _initialize_legacy_facade(
        self,
        *,
        csv_repository: CsvRepositoryPort | None = None,
        app_settings: object | None = None,
        analysis_dataset_builder: AnalysisDatasetBuilder | None = None,
    ) -> None:
        self.csv_repository = csv_repository or CsvRepository(
            workspace_root=getattr(app_settings, "local_workspace_dir", None))
        self.app_settings = app_settings
        encoded_industry = getattr(app_settings, "industry_benchmarks_json", None)
        self._industry_benchmarks = json.loads(encoded_industry) if encoded_industry is not None else {}
        stock_executor = None
        workers = getattr(app_settings, "daily_strategy_workers", 1)
        backend = getattr(app_settings, "daily_executor", "process")
        if analysis_dataset_builder is None:
            if backend == "thread":
                from infrastructure.daily_threaded import DailyThreadExecutor
                stock_executor = DailyThreadExecutor(workers)
            elif backend == "process" and type(workers) is int and workers > 1:
                from infrastructure.daily_parallel import DailyProcessExecutor
                stock_executor = DailyProcessExecutor(workers)
            elif backend == "serial" and workers != 1:
                raise ValueError("DAILY_SERIAL_EXECUTOR_REQUIRES_ONE_WORKER")
            elif backend not in {"serial", "process"}:
                raise ValueError("DAILY_EXECUTOR_MUST_BE_SERIAL_PROCESS_OR_THREAD")
        self._analysis_dataset_builder = (
            analysis_dataset_builder or AnalysisDatasetBuilder(stock_executor=stock_executor,
                valuation_history_verifier_scope=self._valuation_history_verifier_scope)
        )

    @staticmethod
    def _valuation_history_verifier_scope():
        from infrastructure.storage.valuation_source_verification import valuation_history_verifier_scope
        return valuation_history_verifier_scope()

    def _market_history_input(self) -> MarketHistoryInput:
        signal_dates = self.low_entry_v4_signal_dates_by_stock
        native_daily = (getattr(self.app_settings, "daily_entry_model", "v6") == "v7"
                        and getattr(self.app_settings, "les_v7_native_evidence", False)
                        and not getattr(self, "replay_as_of", None))
        decision_time = getattr(self, "les_v7_decision_time", None)
        if native_daily and decision_time:
            signal_dates = {str(symbol): decision_time for symbol in self.stocknumbers if symbol}
        return MarketHistoryInput(
            stock_count=self.stocklistsize,
            stock_numbers=self.stocknumbers,
            stock_names=self.stocknames,
            dates=self.daily_dates,
            opens=self.daily_stocks,
            highs=self.daily_highs,
            lows=self.daily_lows,
            closes=self.daily_closes,
            volumes=self.daily_volumes,
            trade_counts=self.daily_trade_counts,
            trade_values=self.daily_trade_values,
            bids=self.daily_bid_prices,
            asks=self.daily_ask_prices,
            pe_ratios=self.daily_pe_ratios,
            taiex_closes=self.daily_taiex_closes,
            annual_fundamentals=self.annual_fundamentals_by_stock,
            fundamental_snapshots=self.fundamental_snapshots_by_stock,
            fundamental_statuses=self.fundamental_status_by_stock,
            fundamental_status_details=(
                self.fundamental_status_detail_by_stock
            ),
            signal_dates=signal_dates,
            account_equity=self.low_entry_v4_account_equity,
            entry_model=getattr(self.app_settings, "daily_entry_model", "v6"),
            les_v7_bundle_json=getattr(self.app_settings, "les_v7_bundle_json", None),
            les_v7_research_metadata_json=getattr(self.app_settings, "les_v7_research_metadata_json", None),
            les_v7_strategy_policy_json=getattr(self.app_settings, "les_v7_strategy_policy_json", None),
            les_v7_diagnostics_only=getattr(self.app_settings, "les_v7_diagnostics_only", False),
            les_v7_native_evidence=native_daily,
            les_v7_rule_baseline=native_daily and getattr(self.app_settings, "les_v7_rule_baseline", False),
            market_observed_at=getattr(self, "market_observed_at", None) if native_daily else None,
            les_v7_source_repair=(native_daily and getattr(self.app_settings, "les_v7_source_repair", False)
                and getattr(self.app_settings, "les_v7_rule_baseline", False)),
            native_source_selections=getattr(self, "les_v7_source_selections", {}) if native_daily else {},
            fundamental_capture_cutoffs=(dict(self.low_entry_v4_signal_dates_by_stock)
                                         if native_daily and decision_time else {}),
            default_sector=str(getattr(self, "analysis_universe", "UNKNOWN")),
            market_source=getattr(self, "market_source", "TWSE"),
            replay_as_of=getattr(self, "replay_as_of", None),
            les_v6_portfolio=getattr(self, "les_v6_portfolio", {}),
            trading_restrictions=getattr(self, "trading_restrictions", {}),
            industry_benchmarks=getattr(self, "_industry_benchmarks", {}),
            les_v6_research_context=getattr(self, "les_v6_research_context", {}),
            fallback_date=(
                self.iso_scheduled_times[-1]
                if self.iso_scheduled_times
                else ""
            ),
            entry_les_by_stock=getattr(
                self,
                "portfolio_entry_les_by_stock",
                {},
            ),
            entry_dates_by_stock=getattr(
                self,
                "portfolio_entry_dates_by_stock",
                {},
            ),
            entry_decisions_by_stock=getattr(
                self,
                "portfolio_entry_decisions_by_stock",
                {},
            ),
            market_breadth_snapshots=getattr(
                self,
                "market_breadth_snapshots_by_date",
                {},
            ),
        )

    def smtp_email(
        self,
        subject: str,
        ccreceiver: str | None,
        stocktype: object,
        maxprofits: list[ProfitRow],
    ) -> None:
        self.smtp_strategy_email(
            subject=subject,
            ccreceiver=ccreceiver,
            stocktype=stocktype,
            decisions=self.build_strategy_decisions(maxprofits),
        )

    def smtp_strategy_email(
        self,
        subject: str,
        ccreceiver: str | None,
        stocktype: object,
        decisions: Sequence[StrategyDecision],
    ) -> None:
        entryanalysis = self.build_entry_signal_analysis(
            decisions
        )
        smtpemail = SMTPEmail(
            subject,
            ccreceiver,
            settings=self.app_settings,
        )
        smtpemail.smtpauthentication()
        smtpemail.textstockprofittable(
            iso_scheduled_times=self.iso_scheduled_times,
            transactiondays=self.transactiondays,
            stocktype=("TPEx 上櫃半導體 SEMI" if getattr(self, "market_source", "TWSE") == "TPEX" else stocktype),
            stockprofittable="",
            entryanalysis=entryanalysis,
            les_model_version=(decisions[0].value("LES Model Version", "5.0.0") if decisions else "6.0.0"),
        )

    def smtp_img_email(
        self,
        subject: str,
        ccreceiver: str | None,
        stocktype: object,
        backtrack: str,
    ) -> None:
        smtpemail = SMTPEmail(
            subject,
            ccreceiver,
            settings=self.app_settings,
        )
        smtpemail.smtpauthentication()
        smtpemail.imgstockprofittable(
            backtrack=backtrack,
            stocktype=stocktype,
        )

    def draw_linechart(
        self,
        duration: int,
        maxprofitratios: list[list[float]],
    ) -> None:
        save_profit_ratio_chart(
            duration=duration,
            maxprofitratios=maxprofitratios,
            stocknumbers=self.stocknumbers,
            stocksprice=self.stocksprice,
        )

    def cal_max_profit(self) -> list[ProfitRow]:
        maxprofits: list[ProfitRow] = [
            [] for _ in range(self.stocklistsize)
        ]
        missing_symbols: list[str] = []
        unresolved_count = 0
        for item, daily_stock in enumerate(self.daily_stocks):
            if not daily_stock:
                tracked_numbers = getattr(self, "tracked_stock_numbers", ())
                tracked_symbol = (
                    tracked_numbers[item]
                    if item < len(tracked_numbers)
                    else None
                )
                known_symbol = self.stocknumbers[item] or tracked_symbol
                if known_symbol:
                    missing_symbols.append(str(known_symbol))
                else:
                    unresolved_count += 1
                logging.debug(
                    "Skipping max profit calculation for stock %s because "
                    "no valid prices were collected.",
                    known_symbol or "UNKNOWN",
                    extra={
                        "symbol": known_symbol,
                        "source": "TWSE",
                        "status": "MAX_PROFIT_PRICE_MISSING",
                    },
                )
                maxprofits[item].extend([0, 0, 0, 0, None])
                self.stocksprice.append(None)
                continue

            profit_with_fee = max_profit_with_fee(daily_stock, 300)
            maxprofits[item].append(
                round(max_profit_unlimited(daily_stock), 2)
            )
            maxprofits[item].append(round(max_profit(daily_stock), 2))
            maxprofits[item].append(
                round(max_profit_k_transactions(5, daily_stock), 2)
            )
            maxprofits[item].append(round(profit_with_fee, 2))
            maxprofits[item].append(
                round(profit_with_fee / daily_stock[-1], 2)
            )
            self.stocksprice.append(round(daily_stock[-1], 2))

        if missing_symbols or unresolved_count:
            symbol_text = ",".join(dict.fromkeys(missing_symbols)) or "NONE"
            logging.warning(
                "Max profit unavailable summary stocks=%s symbols=%s "
                "unresolved=%s; no valid prices were collected.",
                len(missing_symbols) + unresolved_count,
                symbol_text,
                unresolved_count,
                extra={
                    "source": "TWSE",
                    "status": "MAX_PROFIT_PRICE_MISSING_SUMMARY",
                },
            )
        return maxprofits

    def cal_signal_features(self) -> list[SignalRow]:
        return self._analysis_dataset_builder.build_signal_features(
            self._market_history_input()
        )

    def build_strategy_history(self, item: int) -> pd.DataFrame:
        return self._analysis_dataset_builder.build_history(
            self._market_history_input(),
            item,
        )

    def build_strategy_screen_history(self, item: int) -> pd.DataFrame:
        return self._analysis_dataset_builder.history_assembler.build_screen(
            self._market_history_input(), item,
        )

    def set_low_entry_v4_fundamentals(
        self,
        stock_no: str,
        annual_fundamentals: pd.DataFrame,
        fundamental_snapshot: dict[str, object],
        account_equity: float | None = None,
        signal_date: str | None = None,
    ) -> None:
        normalized_stock_no = str(stock_no).strip()
        if not normalized_stock_no:
            raise ValueError("stock_no cannot be empty")
        self.annual_fundamentals_by_stock[normalized_stock_no] = (
            annual_fundamentals.copy()
        )
        from application.daily_work import current_work
        from infrastructure.snapshot_retention import SnapshotRetentionMap
        work = current_work()
        retention = getattr(work, "snapshot_retention", None)
        if retention is None:
            self.fundamental_snapshots_by_stock[normalized_stock_no] = dict(fundamental_snapshot)
        else:
            existing = self.fundamental_snapshots_by_stock
            if not isinstance(existing, SnapshotRetentionMap) or existing.owner is not retention:
                retained = retention.snapshots()
                for symbol, snapshot in existing.items():
                    retained.retain(symbol, snapshot,
                        cutoff=self.low_entry_v4_signal_dates_by_stock.get(symbol),
                        market=getattr(self, "market_source", "TWSE"))
                self.fundamental_snapshots_by_stock = retained
            self.fundamental_snapshots_by_stock.retain(normalized_stock_no, fundamental_snapshot,
                cutoff=signal_date, market=getattr(self, "market_source", "TWSE"))
        self.fundamental_status_by_stock.pop(normalized_stock_no, None)
        self.fundamental_status_detail_by_stock.pop(
            normalized_stock_no,
            None,
        )
        if signal_date is not None:
            self.low_entry_v4_signal_dates_by_stock[
                normalized_stock_no
            ] = signal_date
        if account_equity is not None:
            if account_equity <= 0:
                raise ValueError("account_equity must be positive")
            self.low_entry_v4_account_equity = float(account_equity)

    def set_low_entry_v4_fundamentals_unavailable(
        self,
        stock_no: str,
        *,
        status: str,
        detail: str,
        signal_date: str,
        account_equity: float | None = None,
    ) -> None:
        normalized_stock_no = str(stock_no).strip()
        if not normalized_stock_no:
            raise ValueError("stock_no cannot be empty")
        self.annual_fundamentals_by_stock.pop(normalized_stock_no, None)
        self.fundamental_snapshots_by_stock.pop(
            normalized_stock_no,
            None,
        )
        self.fundamental_status_by_stock[normalized_stock_no] = str(
            status
        )
        self.fundamental_status_detail_by_stock[normalized_stock_no] = str(
            detail
        )
        self.low_entry_v4_signal_dates_by_stock[
            normalized_stock_no
        ] = signal_date
        if account_equity is not None:
            if account_equity <= 0:
                raise ValueError("account_equity must be positive")
            self.low_entry_v4_account_equity = float(account_equity)

    def cal_low_entry_strategy(
        self,
        histories: list[pd.DataFrame] | None = None,
        feature_frames: list[FeatureFrame] | None = None,
    ) -> pd.DataFrame:
        prepared = (
            self._analysis_dataset_builder.prepare_strategy_inputs(
                self._market_history_input(),
                histories,
                feature_frames,
            )
        )
        return self._analysis_dataset_builder.evaluate_low_entry(
            prepared
        )

    def cal_expected_return_exit_strategy(
        self,
        histories: list[pd.DataFrame] | None = None,
        feature_frames: list[FeatureFrame] | None = None,
    ) -> pd.DataFrame:
        prepared = (
            self._analysis_dataset_builder.prepare_strategy_inputs(
                self._market_history_input(),
                histories,
                feature_frames,
            )
        )
        return (
            self._analysis_dataset_builder
            .evaluate_expected_return_exit(prepared)
        )

    def build_analysis_dataset(
        self,
        maxprofits: list[ProfitRow],
    ) -> pd.DataFrame:
        return self.project_strategy_decisions(
            self.build_strategy_decisions(maxprofits)
        )

    def build_strategy_decisions(
        self,
        maxprofits: list[ProfitRow],
    ) -> tuple[StrategyDecision, ...]:
        return self._analysis_dataset_builder.build_decisions(
            self._market_history_input(),
            maxprofits,
        )

    def project_strategy_decisions(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> pd.DataFrame:
        return self._analysis_dataset_builder.project_decisions(
            decisions
        )

    def cal_max_profit_ratio_data(self) -> list[list[float]]:
        ratios: list[list[float]] = [
            [] for _ in range(self.stocklistsize)
        ]
        for item, daily_stock in enumerate(self.daily_stocks):
            if not daily_stock:
                continue
            ratios[item].append(
                round(
                    max_profit_with_fee(daily_stock, 300)
                    / daily_stock[-1],
                    2,
                )
            )
            self.stocksprice.append(round(daily_stock[-1], 2))
        logging.debug("Max Profit Ratio: %s", ratios)
        return ratios

    @staticmethod
    def record_to_html_tablefmt(
        decisions: Sequence[StrategyDecision],
    ) -> str:
        max_profit_columns = [
            "日期",
            "證券代號",
            "證券名稱",
            "開盤",
            "收盤",
            "無限次交易(手續費$NTD300)",
            "利潤比",
        ]
        return tabulate(
            [
                [decision.value(column) for column in max_profit_columns]
                for decision in decisions
            ],
            headers=max_profit_columns,
            tablefmt="html",
        )

    @staticmethod
    def format_signal_value(
        value: object,
        precision: int = 4,
    ) -> str:
        if pd.isna(value):
            return "N/A"
        if isinstance(value, str):
            return html.escape(value)
        return str(round(float(value), precision))

    @staticmethod
    def build_entry_signal_analysis(
        decisions: Sequence[StrategyDecision],
    ) -> str:
        renderer = RendererFactory.get_renderer(
            "daily_strategy_email"
        )
        return renderer.render(decisions)

    def record_strategy_decisions(
        self,
        file_name: str,
        decisions: Sequence[StrategyDecision],
    ) -> None:
        self.csv_repository.write_analysis_dataset(
            file_name,
            self.project_strategy_decisions(decisions),
        )

    def record_analysis_dataset(
        self,
        file_name: str,
        maxprofits: list[ProfitRow],
    ) -> None:
        self.record_strategy_decisions(
            file_name,
            self.build_strategy_decisions(maxprofits),
        )

    def record(
        self,
        file_name: str,
        scheduled_time: str,
        row_data: StockRows,
    ) -> None:
        if not any(row_data):
            logging.warning(
                "Skipping CSV write for %s because row_data is empty.",
                scheduled_time,
            )
            return
        self.csv_repository.write_daily_rows(
            file_name,
            scheduled_time,
            row_data,
        )


__all__ = [
    "CsvRepositoryPort",
    "LegacyCrawlerFacade",
]

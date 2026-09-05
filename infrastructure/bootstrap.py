"""Composition root for the stock-analysis application."""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

from application.fundamental_service import MopsFundamentalService
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.stock_service import StockAnalysisService
from application.use_cases import EvaluateStrategies
from application.use_cases import FetchMarketData
from application.use_cases import GenerateReport
from application.use_cases import RefreshFundamentals
from application.use_cases import SendNotification
from config.settings import AppSettings
from infrastructure.calendar.twse_calendar import TwseCalendar
from infrastructure.crawler.twse_http_client import TwseHttpClient
from infrastructure.crawler.twse_http_client import TwseHttpConfig
from infrastructure.fundamentals.ixbrl_parser import IxbrlFilingParser
from infrastructure.fundamentals.mops_client import MopsClient
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.fundamentals.twse_financial_batch import TwseFinancialBatchClient
from infrastructure.observability import StructuredRunObserver
from infrastructure.storage.market_history_cache import SQLiteMarketHistoryCache
from infrastructure.storage.portfolio_persistence import (
    SQLitePortfolioPersistence,
)
from infrastructure.runtime.fundamental_adapter import CrawlerFundamentalAdapter
from infrastructure.runtime.market_data_adapter import CrawlerMarketDataAdapter
from infrastructure.runtime.output_adapters import CrawlerLineChartAdapter
from infrastructure.runtime.output_adapters import CrawlerNotificationAdapter
from infrastructure.runtime.output_adapters import CrawlerReportAdapter
from infrastructure.runtime.strategy_adapter import CrawlerStrategyAdapter


TAIPEI_TIMEZONE = ZoneInfo("Asia/Taipei")


@dataclass(frozen=True)
class StockAnalysisRuntime:
    service: StockAnalysisService
    http_client: TwseHttpClient
    market_calendar: Any
    crawler_factory: Callable[[int], Any]
    fundamental_service: Any
    settings: AppSettings
    observer: StructuredRunObserver


def build_stock_analysis_runtime(
    *,
    crawler_factory: Callable[[int], Any] | None = None,
    market_calendar: Any | None = None,
    today_provider: Callable[[], datetime.date] | None = None,
    now_provider: Callable[[], datetime.datetime] | None = None,
    http_client: TwseHttpClient | None = None,
    fundamental_service: Any | None = None,
    settings: AppSettings | None = None,
    run_id_provider: Callable[[], str] | None = None,
) -> StockAnalysisRuntime:
    effective_settings = settings or AppSettings.from_env()
    shared_http_client = http_client or TwseHttpClient(
        config=TwseHttpConfig(
            timeout_seconds=effective_settings.http_timeout_seconds,
            max_attempts=effective_settings.http_max_attempts,
            backoff_seconds=effective_settings.http_backoff_seconds,
            min_request_interval_seconds=(
                effective_settings.http_min_interval_seconds
            ),
            allow_insecure_ssl_fallback=(
                effective_settings.allow_insecure_ssl_fallback
            ),
        )
    )
    observer = StructuredRunObserver(
        (run_id_provider or (lambda: uuid.uuid4().hex))(),
    )
    observer.record(
        "runtime_configuration_validated",
        source="CONFIG",
        status="SUCCESS",
        runtime_mode=effective_settings.runtime_mode,
        http_timeout_seconds=effective_settings.http_timeout_seconds,
        http_max_attempts=effective_settings.http_max_attempts,
        ssl_fallback_enabled=(
            effective_settings.allow_insecure_ssl_fallback
        ),
        mops_cache_configured=bool(effective_settings.mops_cache_dir),
        smtp_configured=bool(
            effective_settings.smtp_email
            and effective_settings.smtp_password
            and effective_settings.smtp_receivers
        ),
        account_equity_valid=effective_settings.account_equity > 0,
        pae_holding_count=len(effective_settings.pae_positions),
        pae_persistence_configured=bool(
            effective_settings.pae_persistence_path
        ),
    )

    if now_provider is not None:
        effective_now_provider = now_provider
    elif today_provider is not None:
        effective_now_provider = lambda: datetime.datetime.combine(
            today_provider(),
            datetime.time(13, 30),
            tzinfo=TAIPEI_TIMEZONE,
        )
    else:
        effective_now_provider = lambda: datetime.datetime.now(TAIPEI_TIMEZONE)

    effective_today_provider = today_provider or (
        lambda: effective_now_provider().astimezone(TAIPEI_TIMEZONE).date()
    )
    effective_calendar = market_calendar or TwseCalendar(
        http_client=shared_http_client,
    )

    if crawler_factory is None:
        from stockanalysis import TwseCrawker

        market_history_cache = SQLiteMarketHistoryCache(
            effective_settings.market_history_cache_path
        )

        effective_crawler_factory = lambda size: TwseCrawker(
            size,
            http_client=shared_http_client,
            app_settings=effective_settings,
            market_history_cache=market_history_cache,
        )
    else:
        effective_crawler_factory = crawler_factory

    if fundamental_service is not None:
        effective_fundamental_service = fundamental_service
    else:
        mops_repository = MopsRepository(
            root=effective_settings.mops_cache_dir,
            connect_attempts=(
                effective_settings.mops_sqlite_connect_attempts
            ),
        )
        filing_catalog = MopsClient(
            http_client=shared_http_client,
            now_provider=effective_now_provider,
            network_rounds=(
                effective_settings.mops_network_retry_rounds
            ),
            network_backoff_seconds=(
                effective_settings.mops_network_round_backoff_seconds
            ),
            catalog_empty_attempts=(
                effective_settings.mops_catalog_empty_attempts
            ),
            catalog_empty_backoff_seconds=(
                effective_settings.mops_catalog_empty_backoff_seconds
            ),
            min_request_interval_seconds=(
                effective_settings.mops_min_request_interval_seconds
            ),
        )
        effective_fundamental_service = MopsFundamentalService(
            filing_catalog,
            repository=mops_repository,
            filing_documents=filing_catalog,
            filing_parser=IxbrlFilingParser(),
            include_current_observations=True,
            now_provider=effective_now_provider,
            financial_batch_client=TwseFinancialBatchClient(
                shared_http_client
            ),
        )

    market_data_adapter = CrawlerMarketDataAdapter(
        effective_crawler_factory,
        observer=observer,
    )
    fundamental_adapter = CrawlerFundamentalAdapter(
        effective_fundamental_service,
        observer=observer,
    )
    service = StockAnalysisService(
        fetch_market_data=FetchMarketData(
            market_data_adapter,
            effective_calendar,
            effective_today_provider,
            effective_now_provider,
        ),
        refresh_fundamentals=RefreshFundamentals(
            fundamental_adapter,
            effective_now_provider,
        ),
        evaluate_strategies=EvaluateStrategies(
            CrawlerStrategyAdapter(
                observer=observer,
                portfolio_allocator=PortfolioAllocationV2Pipeline(
                    persistence=SQLitePortfolioPersistence(
                        effective_settings.pae_persistence_path
                    )
                ),
                portfolio_positions=effective_settings.pae_positions,
                portfolio_current_cash=effective_settings.pae_current_cash,
                account_equity=effective_settings.account_equity,
            )
        ),
        generate_report=GenerateReport(
            CrawlerReportAdapter(observer=observer)
        ),
        send_notification=SendNotification(
            CrawlerNotificationAdapter(observer=observer)
        ),
        line_chart=CrawlerLineChartAdapter(
            effective_crawler_factory,
            effective_calendar,
            now_provider=effective_now_provider,
            observer=observer,
        ),
        observer=observer,
    )
    return StockAnalysisRuntime(
        service=service,
        http_client=shared_http_client,
        market_calendar=effective_calendar,
        crawler_factory=effective_crawler_factory,
        fundamental_service=effective_fundamental_service,
        settings=effective_settings,
        observer=observer,
    )


def build_stock_analysis_service(**kwargs: Any) -> StockAnalysisService:
    return build_stock_analysis_runtime(**kwargs).service

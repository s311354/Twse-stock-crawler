"""Architecture guards for the focused runtime adapters."""

from __future__ import annotations

import ast
import datetime
from pathlib import Path

from application.contracts import AnalysisContext
from application.contracts import MarketStatus
from application.contracts import StockAnalysisRequest
from infrastructure import stock_analysis_adapters as compatibility
from infrastructure.runtime.fundamental_adapter import (
    CrawlerFundamentalAdapter,
)
from infrastructure.runtime.market_data_adapter import CrawlerMarketDataAdapter
from infrastructure.runtime.output_adapters import CrawlerLineChartAdapter
from infrastructure.runtime.output_adapters import CrawlerNotificationAdapter
from infrastructure.runtime.output_adapters import CrawlerReportAdapter
from infrastructure.runtime.strategy_adapter import CrawlerStrategyAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPATIBILITY_FACADE = (
    PROJECT_ROOT / "infrastructure/stock_analysis_adapters.py"
)
BOOTSTRAP = PROJECT_ROOT / "infrastructure/bootstrap.py"


def _import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }


def test_legacy_runtime_adapter_imports_are_direct_aliases() -> None:
    expected = {
        "CrawlerFundamentalAdapter": CrawlerFundamentalAdapter,
        "CrawlerLineChartAdapter": CrawlerLineChartAdapter,
        "CrawlerMarketDataAdapter": CrawlerMarketDataAdapter,
        "CrawlerNotificationAdapter": CrawlerNotificationAdapter,
        "CrawlerReportAdapter": CrawlerReportAdapter,
        "CrawlerStrategyAdapter": CrawlerStrategyAdapter,
    }

    assert compatibility.__all__ == list(expected)
    for name, implementation in expected.items():
        assert getattr(compatibility, name) is implementation


def test_fundamental_adapter_only_delegates_to_refresh_coordinator() -> None:
    class RecordingCoordinator:
        def __init__(self) -> None:
            self.calls = []

        def refresh(self, context, request, *, analysis_time) -> None:
            self.calls.append((context, request, analysis_time))

    coordinator = RecordingCoordinator()
    service = object()
    adapter = CrawlerFundamentalAdapter(
        service,
        coordinator=coordinator,
    )
    context = AnalysisContext(
        resource=object(),
        market_date=datetime.date(2026, 8, 30),
        market_status=MarketStatus.CLOSED,
    )
    request = StockAnalysisRequest(
        stocklist=["2330"],
        stocktype="ELEC",
        output_file_names="shirong",
        endbacktrack=3,
    )
    analysis_time = datetime.datetime(
        2026,
        8,
        30,
        13,
        45,
        tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
    )

    adapter.refresh(context, request, analysis_time=analysis_time)

    assert adapter.fundamental_service is service
    assert coordinator.calls == [(context, request, analysis_time)]


def test_runtime_compatibility_module_remains_a_thin_facade() -> None:
    source = COMPATIBILITY_FACADE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(COMPATIBILITY_FACADE))

    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)
    assert len(source.splitlines()) <= 30


def test_production_bootstrap_uses_focused_runtime_modules() -> None:
    modules = _import_modules(BOOTSTRAP)

    assert "infrastructure.stock_analysis_adapters" not in modules
    assert "infrastructure.runtime.fundamental_adapter" in modules
    assert "infrastructure.runtime.market_data_adapter" in modules
    assert "infrastructure.runtime.output_adapters" in modules
    assert "infrastructure.runtime.strategy_adapter" in modules

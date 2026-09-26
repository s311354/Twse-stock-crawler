"""Trading strategy abstraction and plugin registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from domain.features import FeatureFrame


class Strategy(ABC):
    """Base class for deterministic stock-analysis strategies."""

    name: str
    description: str = ""
    version: str = ""

    @abstractmethod
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of data with strategy output columns appended."""

    def run_with_features(
        self,
        data: pd.DataFrame,
        features: FeatureFrame,
    ) -> pd.DataFrame:
        """Use shared features when supported, otherwise preserve plugin API."""

        del features
        return self.run(data)


class StrategyRegistry:
    """Small plugin registry for built-in and externally registered strategies."""

    def __init__(self) -> None:
        self._strategies: dict[str, Strategy] = {}

    def register(self, strategy: Strategy, replace: bool = False) -> Strategy:
        if not strategy.name:
            raise ValueError("Strategy must define a non-empty name.")
        if strategy.name in self._strategies and not replace:
            raise ValueError("Strategy '{}' is already registered.".format(strategy.name))
        self._strategies[strategy.name] = strategy
        return strategy

    def get(self, name: str) -> Strategy:
        try:
            return self._strategies[name]
        except KeyError as error:
            raise KeyError("Unknown strategy '{}'. Available: {}".format(name, self.names())) from error

    def names(self) -> list[str]:
        return sorted(self._strategies)

    def all(self) -> list[Strategy]:
        """Return all registered strategies."""
        return [self._strategies[name] for name in self.names()]


strategy_registry = StrategyRegistry()
_BUILTINS_LOADED: set[str] = set()

_BUILTIN_STRATEGIES = {
    "expected_return_exit": "domain.strategies.expected_return_exit.ExpectedReturnExitStrategy",
    "expected_return_exit_v2": "domain.strategies.expected_return_exit_v2.ExpectedReturnExitV2Strategy",
    "expected_return_exit_v3": "domain.strategies.expected_return_exit_v3.ExpectedReturnExitV3Strategy",
    "low_entry_score": "domain.strategies.low_entry_score.LowEntryScoreStrategy",
    "low_entry_score_v2": "domain.strategies.low_entry_score_v2.LowEntryScoreV2Strategy",
    "low_entry_score_v3": "domain.strategies.low_entry_score_v3.LowEntryScoreV3Strategy",
    "low_entry_score_v4": "domain.strategies.low_entry_score_v4.LowEntryScoreV4Strategy",
    "low_entry_score_v4_1": "domain.strategies.low_entry_score_v4_1.LowEntryScoreV41Strategy",
    "low_entry_score_v5": "domain.strategies.low_entry_score_v5.LowEntryScoreV5Strategy",
    "low_entry_score_v6": "domain.strategies.low_entry_score_v6.LowEntryScoreV6Strategy",
    "portfolio_allocation_v1": "domain.strategies.portfolio_allocation_v1.PortfolioAllocationV1Strategy",
    "portfolio_allocation_v2": "domain.strategies.portfolio_allocation_v2.PortfolioAllocationV2Strategy",
}

def load_builtin_strategies(name: str | None = None) -> None:
    """Register bundled strategies once.

    External plugins can call register_strategy() without importing application
    code. Built-ins are lazily loaded to keep CLI help/imports lightweight.
    Passing a name loads only that strategy; omitting it loads every built-in.
    """

    names = [name] if name is not None else sorted(_BUILTIN_STRATEGIES)

    for strategy_name in names:
        strategy_path = _BUILTIN_STRATEGIES.get(strategy_name)
        if strategy_path is None:
            raise ValueError("Unknown builtin strategy: {}".format(strategy_name))
        if strategy_name in _BUILTINS_LOADED:
            continue

        module_name, class_name = strategy_path.rsplit(".", 1)
        module = import_module(module_name)
        strategy_cls = getattr(module, class_name)
        strategy_registry.register(strategy_cls(), replace=True)
        _BUILTINS_LOADED.add(strategy_name)


def register_strategy(strategy: Strategy, replace: bool = False) -> Strategy:
    return strategy_registry.register(strategy, replace=replace)


def get_strategy(name: str) -> Strategy:
    load_builtin_strategies(name)
    return strategy_registry.get(name)


def available_strategies() -> list[str]:
    load_builtin_strategies()
    return strategy_registry.names()


__all__ = [
    "Strategy",
    "StrategyRegistry",
    "available_strategies",
    "get_strategy",
    "load_builtin_strategies",
    "register_strategy",
    "strategy_registry",
]

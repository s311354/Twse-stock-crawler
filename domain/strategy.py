"""Trading strategy abstraction and plugin registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class Strategy(ABC):
    """Base class for deterministic stock-analysis strategies."""

    name: str

    @abstractmethod
    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of data with strategy output columns appended."""


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


strategy_registry = StrategyRegistry()
_BUILTINS_LOADED = False

_BUILTIN_STRATEGIES = {
    "low_entry_score": "domain.strategies.low_entry_score.LowEntryScoreStrategy",
    "low_entry_score_v2": "domain.strategies.low_entry_score_v2.LowEntryScoreV2Strategy",
    "low_entry_score_v3": "domain.strategies.low_entry_score_v3.LowEntryScoreV3Strategy",
}

def load_builtin_strategies(name: str) -> None:
    """Register bundled strategies once.

    External plugins can call register_strategy() without importing application
    code. Built-ins are lazily loaded to keep CLI help/imports lightweight.
    """

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    strategy_path = _BUILTIN_STRATEGIES.get(name)
    if strategy_path is None:
        raise ValueError(f"Unknown builtin strategy: {name}")
    
    module_name, class_name = strategy_path.rsplit(".", 1)
    module = __import__(
        module_name,
        fromlist=[class_name],
    )

    strategy_cls = getattr(module, class_name)
    strategy_registry.register(strategy_cls(), replace=True)

    _BUILTINS_LOADED = True


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

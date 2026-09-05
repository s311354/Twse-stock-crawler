"""Version-neutral HTML rendering contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from domain.model_contracts import StrategyDecision


@dataclass(frozen=True)
class MetricColumn:
    label: str
    column: str
    precision: int = 2


class HtmlRenderer(ABC):
    """Base renderer for strategy output embedded in email reports."""

    @abstractmethod
    def render(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> str:
        """Render validated decisions as an HTML fragment."""


__all__ = ["HtmlRenderer", "MetricColumn"]

"""Small pandas pipeline abstraction for the TWSE analysis flow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


DEFAULT_PIPELINE_STAGE_NAMES = ("raw", "clean", "feature_engineering", "signal", "report")


@dataclass(frozen=True)
class PipelineStage:
    """Named DataFrame transform used by PandasPipeline."""

    name: str
    transform: Callable[[pd.DataFrame], pd.DataFrame]


class PandasPipeline:
    """Run deterministic DataFrame transforms from raw data to report data."""

    def __init__(self, stages: list[PipelineStage]) -> None:
        if not stages:
            raise ValueError("PandasPipeline requires at least one stage.")
        self._stages = tuple(stages)

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self._stages)

    def run(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        current = raw_data.copy()

        for stage in self._stages:
            next_frame = stage.transform(current.copy())
            if not isinstance(next_frame, pd.DataFrame):
                raise TypeError("Pipeline stage '{}' must return a pandas DataFrame.".format(stage.name))
            current = next_frame

        return current

    @classmethod
    def from_transforms(
        cls,
        *,
        clean: Callable[[pd.DataFrame], pd.DataFrame],
        feature_engineering: Callable[[pd.DataFrame], pd.DataFrame],
        signal: Callable[[pd.DataFrame], pd.DataFrame],
        report: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> PandasPipeline:
        """Build the standard Raw -> Clean -> Feature -> Signal -> Report flow."""

        return cls(
            [
                PipelineStage("raw", lambda data: data),
                PipelineStage("clean", clean),
                PipelineStage("feature_engineering", feature_engineering),
                PipelineStage("signal", signal),
                PipelineStage("report", report),
            ]
        )

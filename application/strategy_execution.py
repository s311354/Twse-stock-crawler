"""Feature-aware execution of registered strategy plugins."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from domain.features import FeatureFrame
from domain.features import SharedFeaturePipeline
from domain.strategy import Strategy
from domain.strategy import get_strategy


class StrategyOutputSchemaError(ValueError):
    """Raised when a strategy plugin violates its declared output schema."""


@dataclass(frozen=True)
class PreparedStrategyInputs:
    histories: tuple[pd.DataFrame, ...]
    feature_frames: tuple[FeatureFrame, ...]

    def __post_init__(self) -> None:
        if len(self.histories) != len(self.feature_frames):
            raise ValueError(
                "Strategy histories and FeatureFrames must have equal length"
            )


class StrategyExecutionService:
    """Prepare common features once and execute any registered plugin."""

    def __init__(
        self,
        *,
        strategy_resolver: Callable[[str], Strategy] = get_strategy,
        feature_pipeline_factory: Callable[
            [],
            SharedFeaturePipeline,
        ] = SharedFeaturePipeline,
    ) -> None:
        self._strategy_resolver = strategy_resolver
        self._feature_pipeline_factory = feature_pipeline_factory

    def prepare(
        self,
        histories: Sequence[pd.DataFrame],
        feature_frames: Sequence[FeatureFrame] | None = None,
    ) -> PreparedStrategyInputs:
        prepared_histories = tuple(histories)
        prepared_features = (
            tuple(feature_frames)
            if feature_frames is not None
            else self._build_features(prepared_histories)
        )
        return PreparedStrategyInputs(
            histories=prepared_histories,
            feature_frames=prepared_features,
        )

    def _build_features(
        self,
        histories: Sequence[pd.DataFrame],
    ) -> tuple[FeatureFrame, ...]:
        pipeline = self._feature_pipeline_factory()
        return tuple(pipeline.build(history) for history in histories)

    def evaluate(
        self,
        strategy_name: str,
        output_columns: Sequence[str],
        prepared: PreparedStrategyInputs,
    ) -> pd.DataFrame:
        strategy = self._strategy_resolver(strategy_name)
        rows: list[dict[str, object]] = []
        for history, features in zip(
            prepared.histories,
            prepared.feature_frames,
        ):
            result = strategy.run_with_features(history, features)
            missing_columns = [
                column
                for column in output_columns
                if column not in result.columns
            ]
            if missing_columns:
                raise StrategyOutputSchemaError(
                    "Strategy '{}' omitted declared output columns: {}"
                    .format(
                        strategy_name,
                        ", ".join(missing_columns),
                    )
                )
            latest = (
                result.iloc[-1].to_dict()
                if not result.empty
                else {}
            )
            rows.append(
                {
                    column: latest.get(column)
                    for column in output_columns
                }
            )
        return pd.DataFrame(rows, columns=list(output_columns))


__all__ = [
    "PreparedStrategyInputs",
    "StrategyExecutionService",
    "StrategyOutputSchemaError",
]

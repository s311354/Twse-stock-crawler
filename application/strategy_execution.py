"""Feature-aware execution of registered strategy plugins."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from domain.features import FeatureFrame
from domain.features import SharedFeaturePipeline
from domain.features import feature_cache_identity
from domain.strategy import Strategy
from domain.strategy import get_strategy
from application.daily_performance import count_work, measure_stage
from application.evidence_context import StockEvidence
from application.daily_work import current_work


class StrategyOutputSchemaError(ValueError):
    """Raised when a strategy plugin violates its declared output schema."""


@dataclass(frozen=True)
class PreparedStrategyInputs:
    histories: tuple[pd.DataFrame, ...]
    feature_frames: tuple[FeatureFrame, ...]
    normalized_feature_frames: tuple[FeatureFrame, ...] | None = None
    watches: tuple[dict, ...] | None = None
    evidence: tuple[StockEvidence, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.histories) != len(self.feature_frames):
            raise ValueError(
                "Strategy histories and FeatureFrames must have equal length"
            )
        if self.normalized_feature_frames is not None and len(self.histories) != len(self.normalized_feature_frames):
            raise ValueError("Normalized features must match strategy histories")
        if self.watches is not None and len(self.watches) != len(self.histories):
            raise ValueError("Watch observations must match strategy histories")
        if self.evidence is not None:
            if len(self.evidence) != len(self.histories):
                raise ValueError("Evidence must match strategy histories")
            for history, packet in zip(self.histories, self.evidence, strict=True):
                packet.validate(history)


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
        *,
        include_normalized: bool = False,
    ) -> PreparedStrategyInputs:
        prepared_histories = tuple(histories)
        normalized = None
        if include_normalized and feature_frames is None:
            pipeline = self._feature_pipeline_factory()
            pair_builder = getattr(pipeline, "build_price_pair", None)
            if callable(pair_builder):
                with measure_stage("features"):
                    def build(history):
                        def compute():
                            count_work("feature_pair_builds")
                            return pair_builder(history)
                        work = current_work()
                        if work is not None and type(pipeline) is SharedFeaturePipeline:
                            return work.memo_immutable("feature_pair", feature_cache_identity(history, pipeline.config),
                                compute, lambda pair: sum(48 * len(f._index) * len(f._columns) for f in pair))
                        return compute()
                    pairs = [build(history) for history in prepared_histories]
                prepared_features = tuple(pair[0] for pair in pairs)
                normalized = tuple(pair[1] for pair in pairs)
            else:
                # Existing injected pipelines only promise build(). Keep their
                # contract; v6 retains its independent, unit-safe fallback.
                prepared_features = self._build_features(prepared_histories)
        else:
            prepared_features = (
                tuple(feature_frames)
                if feature_frames is not None
                else self._build_features(prepared_histories)
            )
        return PreparedStrategyInputs(
            histories=prepared_histories,
            feature_frames=prepared_features,
            normalized_feature_frames=normalized,
        )

    def _build_features(
        self,
        histories: Sequence[pd.DataFrame],
    ) -> tuple[FeatureFrame, ...]:
        pipeline = self._feature_pipeline_factory()
        with measure_stage("features"):
            count_work("feature_single_builds", len(histories))
            return tuple(pipeline.build(history) for history in histories)

    def evaluate(
        self,
        strategy_name: str,
        output_columns: Sequence[str],
        prepared: PreparedStrategyInputs,
        *,
        latest_only: bool = True,
    ) -> pd.DataFrame:
        strategy = self._strategy_resolver(strategy_name)
        rows: list[dict[str, object]] = []
        for history, features in zip(
            prepared.histories,
            prepared.feature_frames,
        ):
            latest_runner = getattr(strategy, "run_latest_with_features", None)
            result = (latest_runner(history, features) if latest_only and callable(latest_runner)
                      else strategy.run_with_features(history, features))
            count_work("strategy_output_rows", len(result))
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

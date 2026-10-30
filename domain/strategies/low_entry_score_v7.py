"""Independent v7 plugin; existing v1--v6 defaults and models remain untouched."""
from __future__ import annotations

from copy import deepcopy
import pandas as pd

from domain.features import FeatureFrame
from domain.strategy import Strategy
from domain.les_v7.engine import evaluate

V7_CONTEXT_ATTR = "les_v7_context"
V7_OUTPUT_COLUMNS = ("LES v7 Score", "LES v7 Decision", "LES v7 Context", "LES v7 Report")


def attach_low_entry_v7_context(data: pd.DataFrame, **context) -> pd.DataFrame:
    result = data.copy(deep=True)
    result.attrs[V7_CONTEXT_ATTR] = deepcopy(context)
    return result


class LowEntryScoreV7Strategy(Strategy):
    name = "low_entry_score_v7"
    version = "7.0.0"
    description = "Fractal-assisted bottom localization; one context / seed / score, fail-closed artifacts"

    def evaluate(self, data: pd.DataFrame, *, context: dict | None = None, features=None,
                 receipt_producer=None) -> dict:
        options = {} if receipt_producer is None else {"receipt_producer": receipt_producer}
        return evaluate(data, context if context is not None else data.attrs.get(V7_CONTEXT_ATTR, {}),
                        shared=features, **options)

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.run_with_features(data, None)

    def run_with_features(self, data: pd.DataFrame, features: FeatureFrame | None) -> pd.DataFrame:
        """Evaluate the explicit snapshot once, never backfill today's evidence into warmup rows."""
        report = self.evaluate(data, features=features)
        result = data.copy(deep=True)
        for column in V7_OUTPUT_COLUMNS:
            result[column] = pd.Series(None, index=result.index, dtype=object)
        selected_date = report["signal_date"]
        if selected_date is not None:
            candidates = result.index[pd.to_datetime(result["Date"]).dt.date.astype(str) == selected_date]
            position = candidates[-1]
            result.at[position, "LES v7 Score"] = report["envelope"]["score"]["final_score"]
            result.at[position, "LES v7 Decision"] = report["envelope"]["strategy"]["strategy_decision"]
            result.at[position, "LES v7 Context"] = report["envelope"]["event"]["context"]
            result.at[position, "LES v7 Report"] = report
        return result

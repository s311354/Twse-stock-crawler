"""Compose selected native entry and independent valuation, with explicit ownership."""
from __future__ import annotations

import pandas as pd

from application.daily_valuation import DailyValuationProvider
from application.low_entry_v6_daily import build_daily_v6
from application.low_entry_v7_daily import build_daily_v7
from application.strategy_execution import PreparedStrategyInputs
from application.daily_performance import measure_stage
from domain.numeric_history import metadata_free_frame


class DailyEntryService:
    """No v5 entry dependency; sidecar cannot overwrite native entry fields."""

    def __init__(self, *, valuation_provider=None, entry_evaluator=None, reuse_watch=True):
        self.valuation_provider = valuation_provider or DailyValuationProvider()
        self.entry_evaluator = entry_evaluator or build_daily_v6
        self.reuse_watch = reuse_watch

    def evaluate(self, prepared: PreparedStrategyInputs) -> pd.DataFrame:
        with measure_stage("les"):
            models = {history.attrs.get("daily_entry_model", "v6") for history in prepared.histories}
            if len(models) > 1: raise ValueError("MIXED_DAILY_ENTRY_MODELS")
            if models == {"v7"} and self.entry_evaluator is build_daily_v6:
                entry = build_daily_v7(prepared.histories, feature_frames=prepared.normalized_feature_frames,
                                       evidence=prepared.evidence)
            elif self.entry_evaluator is build_daily_v6:
                entry = self.entry_evaluator(prepared.histories, feature_frames=prepared.normalized_feature_frames,
                                             watches=prepared.watches, reuse_watch=self.reuse_watch,
                                             evidence=prepared.evidence)
            else:
                legacy = tuple(self._legacy_history(history, prepared.evidence[item] if prepared.evidence else None)
                               for item, history in enumerate(prepared.histories))
                entry = self.entry_evaluator(legacy)
                del legacy
        with measure_stage("valuation"):
            rows = []
            for item, (history, features) in enumerate(zip(prepared.histories, prepared.feature_frames, strict=True)):
                packet = prepared.evidence[item] if prepared.evidence is not None else None
                if type(self.valuation_provider) is DailyValuationProvider:
                    rows.append(self.valuation_provider.evaluate(history, features, evidence=packet))
                else:
                    # Preserve the two-argument injected provider contract.
                    legacy = self._legacy_history(history, packet)
                    rows.append(self.valuation_provider.evaluate(legacy, features))
                    del legacy
            sidecar = pd.DataFrame(rows, index=entry.index)
        collisions = set(entry.columns) & set(sidecar.columns)
        if collisions:
            raise ValueError("DAILY_ENTRY_FIELD_OWNERSHIP_CONFLICT:" + ",".join(sorted(collisions)))
        return pd.concat([entry, sidecar], axis=1)

    @staticmethod
    def _legacy_history(history, packet):
        if packet is None:
            return history
        legacy = metadata_free_frame(history)
        legacy.attrs = dict(history.attrs)
        legacy.attrs.update(packet.legacy_attrs())
        return legacy

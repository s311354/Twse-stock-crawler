"""TWSE batch snapshot resolution and Base Quality assembly."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from application.fundamental_ports import FundamentalRepositoryPort
from application.fundamental_ports import FundamentalUnitOfWork
from domain.base_quality_features import BaseQualityFeatureProducer
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_models import FundamentalSnapshot
from domain.strategies.low_entry_v6_quality import combine_filing_sources, snapshot_record


@dataclass(frozen=True)
class BatchSnapshotResolution:
    snapshot: FundamentalSnapshot | None
    source: str
    cache_hit: bool


class SnapshotAssembler:
    """Resolve PIT-eligible TWSE snapshots and attach auditable evidence."""

    def __init__(
        self,
        repository: FundamentalRepositoryPort,
        feature_producer: BaseQualityFeatureProducer,
    ) -> None:
        self._repository = repository
        self._feature_producer = feature_producer

    @staticmethod
    def eligible_batch_snapshot(
        daily_financials: dict[str, FundamentalSnapshot],
        symbol: str,
        signal_time: pd.Timestamp,
    ) -> FundamentalSnapshot | None:
        snapshot = daily_financials.get(symbol)
        if snapshot is None:
            return None
        observed_at = pd.to_datetime(
            snapshot.observed_at,
            utc=True,
            errors="coerce",
        )
        if pd.isna(observed_at) or observed_at > signal_time:
            return None
        return snapshot

    def resolve(
        self,
        daily_financials: dict[str, FundamentalSnapshot],
        symbol: str,
        signal_time: pd.Timestamp,
        *,
        uow: FundamentalUnitOfWork,
        history: list[FundamentalSnapshot] | None = None,
    ) -> BatchSnapshotResolution:
        current = self.eligible_batch_snapshot(
            daily_financials,
            symbol,
            signal_time,
        )
        if current is not None:
            return BatchSnapshotResolution(
                snapshot=current,
                source="TWSE_OPENAPI_LATEST",
                cache_hit=False,
            )
        if history is None:
            history = self._repository.load_financial_batch_snapshots(
                symbol,
                as_of=signal_time.isoformat(),
                uow=uow,
            )
        if not history:
            return BatchSnapshotResolution(
                snapshot=None,
                source="MOPS_POINT_IN_TIME_BASE_FALLBACK",
                cache_hit=False,
            )
        cached = max(
            history,
            key=lambda item: (
                item.period,
                item.observed_at,
                item.content_sha256,
            ),
        )
        logging.debug(
            "TWSE financial batch symbol=%s status=CACHE_FALLBACK "
            "period=%sQ%s observed_at=%s cache_hit=true",
            symbol,
            cached.fiscal_year,
            cached.quarter,
            cached.observed_at,
        )
        return BatchSnapshotResolution(
            snapshot=cached,
            source="TWSE_OPENAPI_CACHE_FALLBACK",
            cache_hit=True,
        )

    def save_batch_state(
        self,
        symbol: str,
        snapshot: FundamentalSnapshot,
        *,
        uow: FundamentalUnitOfWork,
    ) -> None:
        self._repository.save_financial_batch_state(
            symbol,
            snapshot.fiscal_year,
            snapshot.quarter,
            snapshot.content_sha256,
            snapshot.observed_at,
            uow=uow,
        )

    def attach_base_evidence(
        self,
        bundle: FundamentalBundle,
        resolution: BatchSnapshotResolution,
        *,
        symbol: str,
        signal_time: pd.Timestamp,
        uow: FundamentalUnitOfWork,
        history: list[FundamentalSnapshot] | None = None,
        persist_snapshot: bool = True,
    ) -> FundamentalBundle:
        if resolution.snapshot is not None:
            return self._attach_batch_snapshot(
                bundle,
                resolution.snapshot,
                source=resolution.source,
                signal_time=signal_time,
                uow=uow,
                history=history,
                persist_snapshot=persist_snapshot,
            )
        features = self._feature_producer.produce(
            current=None,
            history=[],
            annual_fundamentals=bundle.annual_fundamentals,
            existing_snapshot=bundle.snapshot,
            signal_date=signal_time,
        )
        snapshot = dict(bundle.snapshot)
        snapshot.update(features.as_snapshot_fields())
        snapshot["base_quality_evidence_mode"] = resolution.source
        snapshot["les_v6_filings"] = list(snapshot.get("les_v6_mops_filings", []))
        logging.debug(
            "Base quality symbol=%s status=PROVISIONAL source=%s "
            "feature_count=%s cache_hit=false",
            symbol,
            resolution.source,
            len(features.values),
        )
        return FundamentalBundle(
            annual_fundamentals=bundle.annual_fundamentals,
            snapshot=snapshot,
            as_of=bundle.as_of,
        )

    def persist_daily_financials(
        self,
        daily_financials: dict[str, FundamentalSnapshot],
    ) -> None:
        save = getattr(
            self._repository,
            "save_financial_batch_snapshot",
            None,
        )
        unit_of_work = getattr(self._repository, "unit_of_work", None)
        if not callable(save) or not callable(unit_of_work):
            return
        with unit_of_work() as uow:
            for snapshot in daily_financials.values():
                save(snapshot, uow=uow)

    def _attach_batch_snapshot(
        self,
        bundle: FundamentalBundle,
        batch_snapshot: FundamentalSnapshot,
        *,
        source: str,
        signal_time: pd.Timestamp,
        uow: FundamentalUnitOfWork,
        history: list[FundamentalSnapshot] | None,
        persist_snapshot: bool,
    ) -> FundamentalBundle:
        if persist_snapshot:
            self._repository.save_financial_batch_snapshot(
                batch_snapshot,
                uow=uow,
            )
        if history is None:
            history = self._repository.load_financial_batch_snapshots(
                batch_snapshot.symbol,
                as_of=signal_time.isoformat(),
                uow=uow,
            )
        elif not any(
            item.content_sha256 == batch_snapshot.content_sha256
            and item.period == batch_snapshot.period
            for item in history
        ):
            history = sorted(
                [*history, batch_snapshot],
                key=lambda item: (
                    item.period,
                    item.observed_at,
                    item.content_sha256,
                ),
            )
        features = self._feature_producer.produce(
            current=batch_snapshot,
            history=history,
            annual_fundamentals=bundle.annual_fundamentals,
            existing_snapshot=bundle.snapshot,
            signal_date=signal_time,
        )
        snapshot = dict(bundle.snapshot)
        batch_fields = batch_snapshot.as_snapshot_fields()
        batch_fields["twse_financial_source"] = source
        snapshot.update(batch_fields)
        snapshot.update(features.as_snapshot_fields())
        snapshot["base_quality_evidence_mode"] = source
        # Preserve raw period values and their observation time, not annual
        # approximations or a v5 score. v6 performs its own PIT/YTD normalization.
        # Keep TWSE core precedence; separately dated MOPS quarter EPS is usable
        # only after v6 ties the entire period and checks its own PIT cutoff.
        twse = [snapshot_record(item) for item in history]
        snapshot["les_v6_filings"] = combine_filing_sources(
            snapshot.get("les_v6_mops_filings", []), twse, as_of=signal_time,
        )
        return FundamentalBundle(
            annual_fundamentals=bundle.annual_fundamentals,
            snapshot=snapshot,
            as_of=bundle.as_of,
        )


__all__ = ["BatchSnapshotResolution", "SnapshotAssembler"]

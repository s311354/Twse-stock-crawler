"""Versioned derivation and snapshot-cache activation for fundamentals."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import pandas as pd

from application.fundamental_ports import FundamentalRepositoryPort
from application.fundamental_ports import FundamentalUnitOfWork
from domain.fundamental_derivation import FiledFinancials
from domain.fundamental_derivation import FUNDAMENTAL_DERIVATION_VERSION
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_derivation import derive_fundamentals


class FundamentalDeriver:
    """Reuse or derive one versioned bundle from PIT-eligible filings."""

    def __init__(
        self,
        repository: FundamentalRepositoryPort,
        derive_bundle: Callable[..., FundamentalBundle] = derive_fundamentals,
    ) -> None:
        self._repository = repository
        self._derive_bundle = derive_bundle

    def load_or_derive(
        self,
        symbol: str,
        *,
        parsed_signal: pd.Timestamp,
        resolved_sector: str,
        filed_financials: list[FiledFinancials],
        uow: FundamentalUnitOfWork,
    ) -> tuple[FundamentalBundle, bool]:
        filing_set_sha256 = self._repository.filing_set_hash(
            filed_financials
        )
        snapshot_cache_key = hashlib.sha256(
            f"{FUNDAMENTAL_DERIVATION_VERSION}:{filing_set_sha256}".encode(
                "ascii"
            )
        ).hexdigest()
        bundle = self._repository.load_fundamental_bundle(
            symbol,
            snapshot_cache_key,
            resolved_sector,
            uow=uow,
        )
        cache_hit = bundle is not None
        if bundle is None:
            bundle = self._derive_bundle(
                filed_financials,
                signal_date=parsed_signal,
                sector=resolved_sector,
            )
            self._repository.save_fundamental_bundle(
                symbol,
                snapshot_cache_key,
                resolved_sector,
                bundle,
                uow=uow,
            )
        latest = max(
            filed_financials,
            key=lambda filing: (
                filing.fiscal_year,
                filing.quarter,
                filing.available_at,
                filing.revision_number,
            ),
        )
        available_at = max(
            filed_financials,
            key=lambda filing: pd.to_datetime(
                filing.available_at,
                utc=True,
                errors="raise",
            ),
        ).available_at
        self._repository.mark_active_fundamental_bundle(
            symbol,
            filing_set_sha256=filing_set_sha256,
            snapshot_cache_key=snapshot_cache_key,
            derivation_version=FUNDAMENTAL_DERIVATION_VERSION,
            available_at=available_at,
            latest_period=(latest.fiscal_year, latest.quarter),
            annual_history_years=self.fast_path_history_years(
                filed_financials
            ),
            uow=uow,
        )
        return bundle, cache_hit

    @staticmethod
    def fast_path_history_years(filings: list[FiledFinancials]) -> int:
        periods = {(item.fiscal_year, item.quarter) for item in filings}
        if not periods:
            return 0
        latest_year, latest_quarter = max(periods)
        if latest_quarter < 4:
            comparable_periods = {
                (latest_year, latest_quarter),
                (latest_year - 1, latest_quarter),
                (latest_year - 1, 4),
            }
            if not comparable_periods.issubset(periods):
                return 0
        return len({year for year, quarter in periods if quarter == 4})


__all__ = ["FundamentalDeriver"]

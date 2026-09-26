"""Focused contracts for the decomposed fundamental refresh pipeline."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from application.fundamental_deriver import FundamentalDeriver
from application.fundamental_point_in_time import PointInTimeResolver
from application.fundamental_refresh import RefreshPlanner
from application.fundamental_snapshot import SnapshotAssembler
from domain.base_quality_features import BaseQualityFeatureProducer
from domain.fundamental_derivation import FiledFinancials
from domain.fundamental_derivation import FundamentalBundle
from domain.fundamental_models import FilingCorrection
from domain.fundamental_models import FilingRecord
from domain.fundamental_models import FundamentalSnapshot
from domain.strategies.low_entry_v6_filings import V6_FILING_SCHEMA_VERSION


def test_refresh_planner_detects_only_new_revision_events() -> None:
    cached = _filing(correction_announced_on="20260210")
    corrections = {
        "2330": {
            (2025, 4): FilingCorrection(
                symbol="2330",
                fiscal_year=2025,
                quarter=4,
                announced_on="20260211",
                description="更正財務報告",
            )
        }
    }

    assert RefreshPlanner.event_corrected_periods(
        "2330",
        [cached],
        corrections,
    ) == {(2025, 4)}
    assert RefreshPlanner.event_corrected_periods(
        "2317",
        [cached],
        corrections,
    ) == set()


def test_refresh_planner_detects_new_quarter_without_future_lookahead() -> None:
    batch = _snapshot(
        fiscal_year=2026,
        quarter=2,
        observed_at="2026-08-10T00:00:00+00:00",
    )
    eligible = _filing(
        fiscal_year=2026,
        quarter=1,
        available_at="2026-05-15T00:00:00+00:00",
    )
    future = _filing(
        fiscal_year=2026,
        quarter=4,
        available_at="2027-03-01T00:00:00+00:00",
    )

    assert RefreshPlanner.batch_has_new_period(
        batch,
        [eligible, future],
        as_of=pd.Timestamp("2026-08-10T00:00:00+00:00"),
    ) is True


def test_point_in_time_resolver_never_backdates_live_observations() -> None:
    observed = dt.datetime(2026, 8, 10, 6, tzinfo=dt.timezone.utc)
    resolver = PointInTimeResolver(
        include_current_observations=True,
        now_provider=lambda: observed,
    )
    requested = pd.Timestamp("2026-08-09T00:00:00+00:00")
    bundle = FundamentalBundle(pd.DataFrame(), {"symbol": "2330"})

    effective = resolver.effective_signal_time(requested)
    stamped = resolver.bundle_as_of(bundle, effective)

    assert effective == pd.Timestamp(observed)
    assert stamped.as_of == observed.isoformat()


def test_snapshot_assembler_reuses_latest_point_in_time_cache() -> None:
    cached = _snapshot(
        fiscal_year=2026,
        quarter=1,
        observed_at="2026-05-15T00:00:00+00:00",
    )
    repository = _SnapshotRepository(cached)
    assembler = SnapshotAssembler(
        repository,
        BaseQualityFeatureProducer(),
    )

    resolution = assembler.resolve(
        {},
        "2330",
        pd.Timestamp("2026-08-10T00:00:00+00:00"),
        uow=object(),
    )

    assert resolution.snapshot == cached
    assert resolution.source == "TWSE_OPENAPI_CACHE_FALLBACK"
    assert resolution.cache_hit is True
    assert repository.as_of == "2026-08-10T00:00:00+00:00"


def test_fundamental_deriver_uses_versioned_cache_without_recalculation() -> None:
    cached = FundamentalBundle(
        pd.DataFrame([{"fiscal_year": 2025}]),
        {"symbol": "2330", "les_v6_mops_filings": [],
         "les_v6_filing_schema_version": V6_FILING_SCHEMA_VERSION},
    )
    repository = _DeriverRepository(cached)

    def fail_derivation(*args, **kwargs):
        raise AssertionError("cache hit must not recalculate fundamentals")

    result, cache_hit = FundamentalDeriver(
        repository,
        derive_bundle=fail_derivation,
    ).load_or_derive(
        "2330",
        parsed_signal=pd.Timestamp("2026-08-10T00:00:00+00:00"),
        resolved_sector="24",
        filed_financials=[
            FiledFinancials(
                fiscal_year=2025,
                quarter=4,
                available_at="2026-02-10T00:00:00+00:00",
                revision_number=1,
                facts=(),
            )
        ],
        uow=object(),
    )

    assert result is cached
    assert cache_hit is True
    assert repository.activated is True


@pytest.mark.parametrize("old_fields", [{}, {"les_v6_mops_filings": [], "les_v6_filing_schema_version": "old"}])
def test_legacy_bundle_only_adds_v6_records_without_recalculating_old_metrics(old_fields):
    cached = FundamentalBundle(
        pd.DataFrame([{"fiscal_year": 2025}]),
        {"symbol": "2330", "legacy_quality": 72, **old_fields}, "2026-02-10T00:00:00+00:00",
    )
    class MigratingRepository(_DeriverRepository):
        def save_fundamental_bundle(self, symbol, cache_key, sector, bundle, **kwargs):
            self.bundle = bundle
    repository = MigratingRepository(cached)

    def unexpected_derivation(*args, **kwargs):
        raise AssertionError("format-only migration must preserve derived metrics")

    result, cache_hit = FundamentalDeriver(repository, unexpected_derivation).load_or_derive(
        "2330", parsed_signal=pd.Timestamp("2026-09-13T00:00:00+00:00"), resolved_sector="24",
        filed_financials=[FiledFinancials(
            fiscal_year=2025, quarter=4, available_at="2026-02-10T00:00:00+00:00",
            revision_number=1, facts=(),
        )], uow=object(),
    )
    assert cache_hit and result.snapshot["legacy_quality"] == 72
    assert result.as_of == cached.as_of
    assert result.annual_fundamentals is cached.annual_fundamentals
    assert result.snapshot["les_v6_mops_filings"][0]["available_at"] == cached.as_of
    assert result.snapshot["les_v6_filing_schema_version"] == V6_FILING_SCHEMA_VERSION
    assert cached.snapshot.get("les_v6_filing_schema_version") != V6_FILING_SCHEMA_VERSION
    assert "basic_eps_quarter" in result.snapshot["les_v6_mops_filings"][0]
    # A second call must reuse the migrated bundle, not write or derive again.
    def unexpected_save(*args, **kwargs):
        raise AssertionError("current schema must reuse migrated records")
    repository.save_fundamental_bundle = unexpected_save
    reused, hit = FundamentalDeriver(repository, unexpected_derivation).load_or_derive(
        "2330", parsed_signal=pd.Timestamp("2026-09-13T00:00:00+00:00"), resolved_sector="24",
        filed_financials=[FiledFinancials(2025, 4, cached.as_of, 1, ())], uow=object(),
    )
    assert hit and reused is result


class _SnapshotRepository:
    def __init__(self, cached: FundamentalSnapshot) -> None:
        self.cached = cached
        self.as_of: str | None = None

    def load_financial_batch_snapshots(self, symbol, *, as_of, uow):
        self.as_of = as_of
        return [self.cached]


class _DeriverRepository:
    def __init__(self, bundle: FundamentalBundle) -> None:
        self.bundle = bundle
        self.activated = False

    def filing_set_hash(self, filings):
        return "fixed-filing-set"

    def load_fundamental_bundle(self, *args, **kwargs):
        return self.bundle

    def save_fundamental_bundle(self, *args, **kwargs):
        raise AssertionError("cache hit must not write a derived bundle")

    def mark_active_fundamental_bundle(self, *args, **kwargs):
        self.activated = True


def _filing(
    *,
    fiscal_year: int = 2025,
    quarter: int = 4,
    available_at: str = "2026-02-10T00:00:00+00:00",
    correction_announced_on: str | None = None,
) -> FilingRecord:
    return FilingRecord(
        filing_id=fiscal_year * 10 + quarter,
        symbol="2330",
        fiscal_year=fiscal_year,
        quarter=quarter,
        report_id="C",
        available_at=available_at,
        fetched_at=available_at,
        revision_number=1,
        content_sha256=f"{fiscal_year}Q{quarter}",
        source_url="https://example.test/filing.html",
        raw_path="raw/filing.html",
        correction_announced_on=correction_announced_on,
    )


def _snapshot(
    *,
    fiscal_year: int,
    quarter: int,
    observed_at: str,
) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol="2330",
        company_name="台積電",
        fiscal_year=fiscal_year,
        quarter=quarter,
        observed_at=observed_at,
        revenue_ytd=100.0,
        gross_profit_ytd=40.0,
        operating_income_ytd=20.0,
        net_income_ytd=18.0,
        parent_net_income_ytd=17.0,
        basic_eps_ytd=10.0,
        current_assets=50.0,
        total_assets=100.0,
        current_liabilities=25.0,
        total_liabilities=35.0,
        parent_equity=60.0,
        total_equity=65.0,
    )

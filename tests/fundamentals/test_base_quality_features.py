from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from domain.base_quality_features import BaseQualityFeatureProducer
from domain.base_quality_features import MOPS_ANNUAL_FALLBACK_SOURCE
from domain.base_quality_features import TWSE_HISTORY_SOURCE
from domain.data_provenance import OFFICIAL_FILING_TIMESTAMP
from domain.data_provenance import LEGACY_UNVERIFIED
from domain.fundamental_models import FundamentalSnapshot
from domain.quality_v5 import BaseQualityStatus
from domain.quality_v5 import evaluate_v5_quality
from infrastructure.fundamentals.mops_repository import MopsRepository


SIGNAL_DATE = "2026-08-02T00:00:00Z"


def test_same_quarter_twse_history_has_priority_over_annual_fallback() -> None:
    prior = _snapshot(
        fiscal_year=2025,
        quarter=2,
        observed_at="2025-08-01T00:00:00Z",
        revenue=100.0,
        operating_income=10.0,
        assets=200.0,
        equity=100.0,
    )
    current = _snapshot(
        fiscal_year=2026,
        quarter=2,
        observed_at="2026-08-01T00:00:00Z",
        revenue=120.0,
        operating_income=18.0,
        assets=220.0,
        equity=120.0,
    )
    result = BaseQualityFeatureProducer().produce(
        current=current,
        history=[prior, current],
        annual_fundamentals=_annual_rows(),
        existing_snapshot={},
        signal_date=SIGNAL_DATE,
    )

    values = dict(result.values)
    sources = dict(result.sources)
    assert values["equity_growth_yoy"] == pytest.approx(0.20)
    assert values["operating_margin_change"] == pytest.approx(0.05)
    assert values["asset_growth_yoy"] == pytest.approx(0.10)
    assert values["asset_growth_supported"] is True
    assert sources["equity_growth_yoy"] == TWSE_HISTORY_SOURCE
    assert result.source_reliability == 95.0


def test_annual_fallback_unlocks_auditable_base_pass_coverage() -> None:
    current = _snapshot(
        fiscal_year=2026,
        quarter=2,
        observed_at="2026-08-01T00:00:00Z",
        revenue=120.0,
        operating_income=18.0,
        assets=220.0,
        equity=120.0,
    )
    producer_result = BaseQualityFeatureProducer().produce(
        current=current,
        history=[current],
        annual_fundamentals=_annual_rows(),
        existing_snapshot={"gross_margin_change": 0.01},
        signal_date=SIGNAL_DATE,
    )
    snapshot = _production_shape_snapshot(current)
    snapshot.update(producer_result.as_snapshot_fields())

    quality = evaluate_v5_quality(
        _annual_rows(),
        snapshot,
        SIGNAL_DATE,
    )

    assert producer_result.source_reliability == 90.0
    assert dict(producer_result.sources)["equity_growth_yoy"] == (
        MOPS_ANNUAL_FALLBACK_SOURCE
    )
    assert quality.base_available_weight >= 70
    assert quality.base_status is BaseQualityStatus.PASS


def test_unverified_annual_rows_cannot_increase_base_coverage() -> None:
    annual = _annual_rows()
    annual["availability_source"] = LEGACY_UNVERIFIED
    current = _snapshot(
        fiscal_year=2026,
        quarter=2,
        observed_at="2026-08-01T00:00:00Z",
    )

    result = BaseQualityFeatureProducer().produce(
        current=current,
        history=[current],
        annual_fundamentals=annual,
        existing_snapshot={},
        signal_date=SIGNAL_DATE,
    )

    assert "equity_growth_yoy" not in dict(result.values)
    assert result.source_reliability is None


def test_mops_only_base_evidence_is_point_in_time_but_provisional() -> None:
    snapshot = {
        "sector": "24",
        "available_at": "2026-03-01T00:00:00Z",
        "availability_source": OFFICIAL_FILING_TIMESTAMP,
        "equity": 110.0,
        "total_equity": 110.0,
        "total_assets": 210.0,
        "total_liabilities": 100.0,
        "current_assets": 100.0,
        "current_liabilities": 50.0,
        "roe_ttm": 0.12,
        "roa_ttm": 0.06,
        "share_dilution_yoy": 0.0,
        "revenue_growth_ttm_yoy": 0.10,
        "gross_margin_change": 0.01,
    }
    features = BaseQualityFeatureProducer().produce(
        current=None,
        history=[],
        annual_fundamentals=_annual_rows(),
        existing_snapshot=snapshot,
        signal_date=SIGNAL_DATE,
    )
    snapshot.update(features.as_snapshot_fields())

    result = evaluate_v5_quality(_annual_rows(), snapshot, SIGNAL_DATE)

    assert result.base_available_weight >= 50
    assert result.base_status is BaseQualityStatus.PROVISIONAL
    assert "TWSE_BASE_SNAPSHOT_MISSING" in result.warnings
    assert "MOPS_POINT_IN_TIME_BASE_FALLBACK" in result.warnings


def test_future_feature_evidence_removes_only_that_base_field() -> None:
    current = _snapshot(
        fiscal_year=2026,
        quarter=2,
        observed_at="2026-08-01T00:00:00Z",
    )
    snapshot = _production_shape_snapshot(current)
    snapshot.update(
        {
            "equity_growth_yoy": 0.10,
            "base_quality_feature_sources": (
                "equity_growth_yoy=MOPS_POINT_IN_TIME_ANNUAL_FALLBACK"
            ),
            "base_quality_feature_available_at": (
                "equity_growth_yoy=2026-08-03T00:00:00Z"
            ),
        }
    )

    result = evaluate_v5_quality(_annual_rows(), snapshot, SIGNAL_DATE)

    metrics = {metric.name: metric for metric in result.base_metrics}
    assert metrics["equity_growth"].available is False
    assert metrics["positive_equity"].available is True
    assert "BASE_FIELDS_NOT_POINT_IN_TIME=equity_growth_yoy" in result.warnings


def test_repository_preserves_first_observation_and_selects_revision_as_of(
    tmp_path,
) -> None:
    repository = MopsRepository(tmp_path / "mops")
    original = _snapshot(
        fiscal_year=2026,
        quarter=2,
        observed_at="2026-08-01T00:00:00Z",
        revenue=100.0,
    )
    duplicate_seen_later = replace(
        original,
        observed_at="2026-08-02T00:00:00Z",
    )
    revised = replace(
        original,
        observed_at="2026-08-03T00:00:00Z",
        revenue_ytd=110.0,
    )

    repository.save_financial_batch_snapshot(original)
    repository.save_financial_batch_snapshot(duplicate_seen_later)
    repository.save_financial_batch_snapshot(revised)

    before_revision = repository.load_financial_batch_snapshots(
        "2330",
        as_of="2026-08-02T23:59:59Z",
    )
    after_revision = repository.load_financial_batch_snapshots(
        "2330",
        as_of="2026-08-03T00:00:00Z",
    )

    assert len(before_revision) == 1
    assert before_revision[0].observed_at == original.observed_at
    assert before_revision[0].revenue_ytd == 100.0
    assert after_revision[0].revenue_ytd == 110.0


def _snapshot(
    *,
    fiscal_year: int,
    quarter: int,
    observed_at: str,
    revenue: float = 100.0,
    operating_income: float = 12.0,
    assets: float = 200.0,
    equity: float = 110.0,
) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol="2330",
        company_name="台積電",
        fiscal_year=fiscal_year,
        quarter=quarter,
        observed_at=observed_at,
        revenue_ytd=revenue,
        gross_profit_ytd=revenue * 0.4,
        operating_income_ytd=operating_income,
        net_income_ytd=operating_income * 0.9,
        parent_net_income_ytd=operating_income * 0.85,
        basic_eps_ytd=operating_income / 10,
        current_assets=100.0,
        total_assets=assets,
        current_liabilities=50.0,
        total_liabilities=assets - equity,
        parent_equity=equity,
        total_equity=equity,
    )


def _annual_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fiscal_year": [2024, 2025, 2026],
            "available_at": [
                "2025-03-01T00:00:00Z",
                "2026-03-01T00:00:00Z",
                "2027-03-01T00:00:00Z",
            ],
            "availability_source": [OFFICIAL_FILING_TIMESTAMP] * 3,
            "revenue": [100.0, 110.0, 999.0],
            "net_income": [10.0, 12.0, 999.0],
            "operating_margin": [0.12, 0.13, -1.0],
            "equity": [100.0, 110.0, 1.0],
            "total_assets": [200.0, 210.0, 999.0],
            "operating_cash_flow": [12.0, 14.0, 999.0],
            "free_cash_flow": [8.0, 9.0, 999.0],
        }
    )


def _production_shape_snapshot(
    current: FundamentalSnapshot,
) -> dict[str, object]:
    snapshot = current.as_snapshot_fields()
    snapshot.update(
        {
            "sector": "24",
            "available_at": "2026-07-31T00:00:00Z",
            "availability_source": OFFICIAL_FILING_TIMESTAMP,
            "share_dilution_yoy": 0.0,
            "revenue_growth_ttm_yoy": 0.10,
            "gross_margin_change": 0.01,
            "roe_ttm": 0.15,
            "roa_ttm": 0.08,
            "audit_opinion": "REVIEWED_UNMODIFIED",
            "material_accounting_issue": False,
            "going_concern": False,
        }
    )
    return snapshot

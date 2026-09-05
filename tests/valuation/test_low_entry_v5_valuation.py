import pandas as pd
import pytest

from domain.valuation_v5 import CycleValuationReviewStatus
from domain.valuation_v5 import LowEntryV5ValuationProducer
from domain.valuation_v5 import ValuationProductionStatus


SIGNAL_DATE = "2026-07-21T06:00:00Z"


def _annual() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fiscal_year": [2022, 2023, 2024, 2025],
            "available_at": [
                "2023-03-01T00:00:00Z",
                "2024-03-01T00:00:00Z",
                "2025-03-01T00:00:00Z",
                "2026-03-01T00:00:00Z",
            ],
            "revenue": [800.0, 880.0, 968.0, 1_064.8],
            "net_income": [80.0, 92.0, 106.0, 121.0],
            "operating_margin": [0.12, 0.125, 0.13, 0.135],
            "diluted_shares": [10.0, 10.1, 10.2, 10.3],
            "dividends_paid": [30.0, 34.0, 38.0, 42.0],
        }
    )


def _snapshot() -> dict[str, object]:
    return {
        "available_at": "2026-05-15T00:00:00Z",
        "ttm_eps": 12.0,
        "revenue_growth_ttm_yoy": 0.10,
        "operating_margin_ttm": 0.135,
        "share_dilution_yoy": 0.01,
    }


def _semi_cycle_annual() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fiscal_year": [2021, 2022, 2023, 2024, 2025],
            "available_at": [
                "2022-03-01T00:00:00Z",
                "2023-03-01T00:00:00Z",
                "2024-03-01T00:00:00Z",
                "2025-03-01T00:00:00Z",
                "2026-03-01T00:00:00Z",
            ],
            "revenue": [1_000.0] * 5,
            "net_income": [120.0] * 5,
            "operating_margin": [0.15] * 5,
            "diluted_shares": [100.0] * 5,
            "dividends_paid": [0.0] * 5,
            "equity": [800.0] * 5,
        }
    )


def _semi_cycle_snapshot(*, sector: str = "24") -> dict[str, object]:
    return {
        "available_at": "2026-05-15T00:00:00Z",
        "sector": sector,
        "ttm_eps": 4.0,
        "revenue_ttm": 1_000.0,
        "revenue_growth_ttm_yoy": 0.0,
        "operating_margin_ttm": 0.15,
        "share_dilution_yoy": 0.0,
        "equity": 800.0,
    }


def test_producer_builds_deterministic_ordered_scenarios() -> None:
    producer = LowEntryV5ValuationProducer()

    first = producer.produce(
        _annual(),
        _snapshot(),
        signal_date=SIGNAL_DATE,
        current_price=100.0,
        recent_pe_median=12.0,
    )
    second = producer.produce(
        _annual(),
        _snapshot(),
        signal_date=SIGNAL_DATE,
        current_price=100.0,
        recent_pe_median=12.0,
    )

    assert first == second
    assert first.status is ValuationProductionStatus.COMPLETE
    assert first.bear is not None
    assert first.base is not None
    assert first.bull is not None
    assert first.bear.fair_value < first.base.fair_value < first.bull.fair_value
    assert first.base.expected_cagr >= first.required_return
    assert first.gate_passed is True
    assert first.as_snapshot_fields()["base_case_expected_cagr"] == pytest.approx(
        first.base.expected_cagr
    )


def test_producer_excludes_fundamentals_published_after_signal_date() -> None:
    producer = LowEntryV5ValuationProducer()
    future = pd.concat(
        [
            _annual(),
            pd.DataFrame(
                {
                    "fiscal_year": [2026],
                    "available_at": ["2026-08-01T00:00:00Z"],
                    "revenue": [100_000.0],
                    "net_income": [50_000.0],
                    "operating_margin": [0.80],
                    "diluted_shares": [1.0],
                    "dividends_paid": [20_000.0],
                }
            ),
        ],
        ignore_index=True,
    )

    baseline = producer.produce(
        _annual(),
        _snapshot(),
        signal_date=SIGNAL_DATE,
        current_price=100.0,
        recent_pe_median=12.0,
    )
    with_future = producer.produce(
        future,
        _snapshot(),
        signal_date=SIGNAL_DATE,
        current_price=100.0,
        recent_pe_median=12.0,
    )

    assert with_future == baseline


def test_producer_returns_unknown_instead_of_inventing_missing_dividend() -> None:
    annual = _annual().drop(columns="dividends_paid")

    result = LowEntryV5ValuationProducer().produce(
        annual,
        _snapshot(),
        signal_date=SIGNAL_DATE,
        current_price=100.0,
        recent_pe_median=12.0,
    )

    assert result.status is ValuationProductionStatus.UNKNOWN
    assert result.gate_passed is False
    assert "MISSING_EXPECTED_DIVIDEND" in result.reasons
    assert result.base is None


def test_producer_fails_gate_when_bear_case_loss_is_excessive() -> None:
    result = LowEntryV5ValuationProducer().produce(
        _annual(),
        _snapshot(),
        signal_date=SIGNAL_DATE,
        current_price=250.0,
        recent_pe_median=12.0,
    )

    assert result.status is ValuationProductionStatus.COMPLETE
    assert result.bear_permanent_loss is not None
    assert result.bear_permanent_loss > 0.35
    assert result.bear_permanent_loss_acceptable is False
    assert result.gate_passed is False
    assert "BASE_FAIR_VALUE_NOT_ABOVE_CURRENT_PRICE" in result.reasons
    assert "BEAR_PERMANENT_LOSS_EXCEEDS_LIMIT" in result.reasons


def test_producer_exposes_supportive_tier_without_passing_strict_gate() -> None:
    result = LowEntryV5ValuationProducer().produce(
        _annual(),
        _snapshot(),
        signal_date=SIGNAL_DATE,
        current_price=130.0,
        recent_pe_median=12.0,
    )

    assert result.status is ValuationProductionStatus.COMPLETE
    assert result.gate_passed is False
    assert result.supportive_gate_passed is True
    assert result.bear_permanent_loss is not None
    assert 0.35 < result.bear_permanent_loss <= 0.45


def test_supportive_tier_rejects_excessive_permanent_loss() -> None:
    result = LowEntryV5ValuationProducer().produce(
        _annual(),
        _snapshot(),
        signal_date=SIGNAL_DATE,
        current_price=150.0,
        recent_pe_median=12.0,
    )

    assert result.gate_passed is False
    assert result.supportive_gate_passed is False
    assert result.bear_permanent_loss is not None
    assert result.bear_permanent_loss > 0.45


def test_air_cycle_profile_allows_only_supportive_tier_near_cycle_value() -> None:
    air_snapshot = _snapshot()
    air_snapshot["sector"] = "15"
    air_snapshot["issuer_symbol"] = "2610"
    result = LowEntryV5ValuationProducer().produce(
        _annual(),
        air_snapshot,
        signal_date=SIGNAL_DATE,
        current_price=140.0,
        recent_pe_median=12.0,
    )

    review = result.cycle_review
    assert result.gate_passed is False
    assert result.supportive_gate_passed is True
    assert result.bear_permanent_loss is not None
    assert 0.45 < result.bear_permanent_loss <= 0.50
    assert review is not None
    assert review.required is True
    assert review.eps_divergence_ratio is not None
    assert review.eps_divergence_ratio < 3
    assert "AIR_CYCLICAL_VALUATION_REVIEW" in review.reasons


def test_air_cycle_profile_does_not_make_excessive_downside_supportive() -> None:
    air_snapshot = _snapshot()
    air_snapshot["sector"] = "15"
    air_snapshot["issuer_symbol"] = "2610"
    result = LowEntryV5ValuationProducer().produce(
        _annual(),
        air_snapshot,
        signal_date=SIGNAL_DATE,
        current_price=170.0,
        recent_pe_median=12.0,
    )

    assert result.gate_passed is False
    assert result.supportive_gate_passed is False
    assert result.bear_permanent_loss is not None
    assert result.bear_permanent_loss > 0.50


def test_semi_eps_divergence_runs_three_anchor_cycle_review() -> None:
    result = LowEntryV5ValuationProducer().produce(
        _semi_cycle_annual(),
        _semi_cycle_snapshot(),
        signal_date=SIGNAL_DATE,
        current_price=15.0,
        recent_pe_median=15.0,
    )

    review = result.cycle_review
    assert result.status is ValuationProductionStatus.COMPLETE
    assert result.gate_passed is False
    assert result.supportive_gate_passed is True
    assert review is not None
    assert review.status is CycleValuationReviewStatus.SUPPORTIVE
    assert review.required is True
    assert review.eps_divergence_ratio == pytest.approx(4.0 / 1.2)
    assert review.mid_cycle_operating_margin == pytest.approx(0.15)
    assert review.normalized_roe == pytest.approx(0.15)
    assert review.margin_fair_value == pytest.approx(18.0)
    assert review.pb_fair_value == pytest.approx(12.0)
    assert review.ttm_eps_fair_value == pytest.approx(42.0)
    assert review.composite_fair_value == pytest.approx(18.0)
    assert review.available_anchor_count == 3
    assert review.supporting_anchor_count == 2
    assert "NORMALIZED_EPS_DIVERGENCE" in result.reasons
    assert "CYCLE_REVIEW_SUPPORTIVE" in result.reasons
    fields = result.as_snapshot_fields()
    assert fields["cycle_valuation_review_status"] == "SUPPORTIVE"
    assert fields["normalized_eps_divergence_ratio"] == pytest.approx(4.0 / 1.2)


def test_cycle_review_is_limited_to_configured_semi_sector() -> None:
    result = LowEntryV5ValuationProducer().produce(
        _semi_cycle_annual(),
        _semi_cycle_snapshot(sector="25"),
        signal_date=SIGNAL_DATE,
        current_price=15.0,
        recent_pe_median=15.0,
    )

    assert result.cycle_review is not None
    assert (
        result.cycle_review.status
        is CycleValuationReviewStatus.NOT_APPLICABLE
    )
    assert result.cycle_review.required is False
    assert result.supportive_gate_passed is False
    assert "NORMALIZED_EPS_DIVERGENCE" not in result.reasons


def test_cycle_review_does_not_invent_missing_valuation_anchors() -> None:
    annual = _semi_cycle_annual().drop(columns=["operating_margin", "equity"])
    result = LowEntryV5ValuationProducer().produce(
        annual,
        _semi_cycle_snapshot(),
        signal_date=SIGNAL_DATE,
        current_price=15.0,
        recent_pe_median=15.0,
    )

    review = result.cycle_review
    assert review is not None
    assert review.status is CycleValuationReviewStatus.INSUFFICIENT_DATA
    assert review.available_anchor_count == 1
    assert review.supportive is False
    assert "CYCLE_REVIEW_INSUFFICIENT_ANCHORS" in result.reasons

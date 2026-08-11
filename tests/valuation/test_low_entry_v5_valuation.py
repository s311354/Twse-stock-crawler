import pandas as pd
import pytest

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

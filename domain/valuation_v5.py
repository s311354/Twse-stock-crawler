"""Point-in-time valuation scenarios for Low Entry Model v5."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


VALUATION_MODEL_VERSION = "les-v5-valuation-2026-08-18-v4-air-cycle-review"


class ValuationProductionStatus(str, Enum):
    COMPLETE = "COMPLETE"
    UNKNOWN = "UNKNOWN"


class CycleValuationReviewStatus(str, Enum):
    """Audit state for the multi-anchor cyclical valuation review."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_TRIGGERED = "NOT_TRIGGERED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    FAIL = "FAIL"
    SUPPORTIVE = "SUPPORTIVE"


@dataclass(frozen=True)
class LowEntryV5ValuationConfig:
    """Versioned assumptions used only by the valuation producer."""

    passive_benchmark_expected_return: float = 0.07
    required_excess_return: float = 0.03
    maximum_bear_permanent_loss: float = 0.35
    # The strict gate remains the only valuation state that can authorize a
    # normal BUY. Standard SUPPORTIVE remains a narrower exception for a
    # conditional BUY_SMALL: downside must stay bounded and either base fair
    # value or a near-hurdle CAGR must support spot. A separate cyclical
    # review may also return SUPPORTIVE, but never strict PASS.
    supportive_maximum_bear_permanent_loss: float = 0.45
    air_supportive_maximum_bear_permanent_loss: float = 0.50
    supportive_required_return_discount: float = 0.02
    bear_years: int = 3
    base_years: int = 5
    bull_years: int = 7
    growth_spread: float = 0.05
    bear_margin_multiplier: float = 0.85
    bull_margin_multiplier: float = 1.10
    bear_pe_multiplier: float = 0.70
    bull_pe_multiplier: float = 1.20
    minimum_exit_pe: float = 5.0
    maximum_exit_pe: float = 30.0
    minimum_growth: float = -0.10
    maximum_growth: float = 0.20
    minimum_margin: float = 0.01
    maximum_dilution: float = 0.10
    minimum_dilution: float = -0.05
    cyclical_sector_codes: frozenset[str] = frozenset({"15", "24"})
    always_review_cyclical_sector_codes: frozenset[str] = frozenset({"15"})
    normalized_eps_divergence_threshold: float = 3.0
    cycle_ttm_eps_haircut: float = 0.70
    cycle_minimum_pe: float = 5.0
    cycle_maximum_pe: float = 20.0
    cycle_minimum_justified_pb: float = 0.50
    cycle_maximum_justified_pb: float = 3.00
    cycle_default_net_income_conversion: float = 0.80
    cycle_minimum_available_anchors: int = 2
    cycle_minimum_supporting_anchors: int = 2
    model_version: str = VALUATION_MODEL_VERSION


@dataclass(frozen=True)
class ValuationScenario:
    name: str
    normalized_eps: float
    revenue_growth: float
    operating_margin: float
    competitive_advantage_years: int
    exit_pe: float
    expected_annual_dividend: float
    annual_share_dilution: float
    projected_eps: float
    terminal_value: float
    total_expected_dividends: float
    future_value: float
    fair_value: float
    expected_cagr: float


@dataclass(frozen=True)
class CyclicalValuationReview:
    """Multi-anchor review for cyclical earnings and asset-value divergence.

    This review can support only the limited ``SUPPORTIVE`` valuation tier.
    It never changes the strict Bear/Base/Bull gate and cannot override any
    technical, falling-knife, trigger, liquidity, or risk requirement.
    """

    status: CycleValuationReviewStatus
    required: bool
    sector_code: str | None
    ttm_eps: float | None
    normalized_eps: float | None
    eps_divergence_ratio: float | None
    mid_cycle_operating_margin: float | None = None
    normalized_roe: float | None = None
    book_value_per_share: float | None = None
    margin_fair_value: float | None = None
    pb_fair_value: float | None = None
    ttm_eps_fair_value: float | None = None
    composite_fair_value: float | None = None
    available_anchor_count: int = 0
    supporting_anchor_count: int = 0
    supportive: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class LowEntryV5ValuationResult:
    status: ValuationProductionStatus
    signal_date: str | None
    current_price: float | None
    passive_benchmark_expected_return: float
    required_return: float
    bear: ValuationScenario | None = None
    base: ValuationScenario | None = None
    bull: ValuationScenario | None = None
    bear_permanent_loss: float | None = None
    bear_permanent_loss_acceptable: bool | None = None
    supportive_gate_passed: bool | None = None
    cycle_review: CyclicalValuationReview | None = None
    reasons: tuple[str, ...] = ()
    model_version: str = VALUATION_MODEL_VERSION

    @property
    def gate_passed(self) -> bool:
        return bool(
            self.status is ValuationProductionStatus.COMPLETE
            and self.base is not None
            and self.base.expected_cagr >= self.required_return
            and self.current_price is not None
            and self.base.fair_value > self.current_price
            and self.bear_permanent_loss_acceptable is True
        )

    def as_snapshot_fields(self) -> dict[str, object]:
        """Expose the historical v5 consumer keys plus auditable details."""

        cycle = self.cycle_review
        return {
            "valuation_model_status": self.status.value,
            "valuation_model_version": self.model_version,
            "valuation_reasons": ";".join(self.reasons) or "NONE",
            "passive_benchmark_expected_return": (
                self.passive_benchmark_expected_return
            ),
            "required_valuation_return": self.required_return,
            "base_case_expected_cagr": (
                self.base.expected_cagr if self.base is not None else None
            ),
            "base_case_fair_value": (
                self.base.fair_value if self.base is not None else None
            ),
            "bear_case_fair_value": (
                self.bear.fair_value if self.bear is not None else None
            ),
            "bull_case_fair_value": (
                self.bull.fair_value if self.bull is not None else None
            ),
            "bear_case_permanent_loss": self.bear_permanent_loss,
            "bear_case_permanent_loss_acceptable": (
                self.bear_permanent_loss_acceptable
            ),
            "supportive_valuation_gate_passed": self.supportive_gate_passed,
            "cycle_valuation_review_status": (
                cycle.status.value if cycle is not None else None
            ),
            "cycle_valuation_review_required": (
                cycle.required if cycle is not None else None
            ),
            "cycle_valuation_reasons": (
                (";".join(cycle.reasons) or "NONE")
                if cycle is not None
                else None
            ),
            "valuation_ttm_eps": cycle.ttm_eps if cycle is not None else None,
            "normalized_eps_divergence_ratio": (
                cycle.eps_divergence_ratio if cycle is not None else None
            ),
            "cycle_mid_cycle_operating_margin": (
                cycle.mid_cycle_operating_margin if cycle is not None else None
            ),
            "cycle_normalized_roe": (
                cycle.normalized_roe if cycle is not None else None
            ),
            "cycle_book_value_per_share": (
                cycle.book_value_per_share if cycle is not None else None
            ),
            "cycle_margin_fair_value": (
                cycle.margin_fair_value if cycle is not None else None
            ),
            "cycle_pb_fair_value": (
                cycle.pb_fair_value if cycle is not None else None
            ),
            "cycle_ttm_eps_fair_value": (
                cycle.ttm_eps_fair_value if cycle is not None else None
            ),
            "cycle_composite_fair_value": (
                cycle.composite_fair_value if cycle is not None else None
            ),
            "cycle_available_anchor_count": (
                cycle.available_anchor_count if cycle is not None else None
            ),
            "cycle_supporting_anchor_count": (
                cycle.supporting_anchor_count if cycle is not None else None
            ),
        }


class LowEntryV5ValuationProducer:
    """Build deterministic Bear/Base/Bull scenarios from available facts.

    The specification identifies scenario inputs but does not define a pricing
    equation. This producer therefore keeps its assumptions explicit and
    versioned. It never substitutes zero for missing facts.
    """

    def __init__(self, config: LowEntryV5ValuationConfig | None = None) -> None:
        self.config = config or LowEntryV5ValuationConfig()

    def produce(
        self,
        annual_fundamentals: pd.DataFrame,
        snapshot: dict[str, Any],
        *,
        signal_date: object,
        current_price: object,
        recent_pe_median: object,
    ) -> LowEntryV5ValuationResult:
        signal_time = pd.to_datetime(signal_date, utc=True, errors="coerce")
        price = _positive_number(current_price)
        pe = _positive_number(recent_pe_median)
        reasons: list[str] = []

        if pd.isna(signal_time):
            reasons.append("INVALID_SIGNAL_DATE")
        if price is None:
            reasons.append("INVALID_CURRENT_PRICE")
        if pe is None:
            reasons.append("MISSING_RECENT_PE")

        annual = self._point_in_time_annual(annual_fundamentals, signal_time)
        point_in_time_snapshot = self._point_in_time_snapshot(snapshot, signal_time)
        if annual.empty:
            reasons.append("MISSING_POINT_IN_TIME_ANNUAL_FUNDAMENTALS")

        normalized_eps = self._normalized_eps(annual, point_in_time_snapshot)
        revenue_growth = self._normalized_revenue_growth(
            annual,
            point_in_time_snapshot,
        )
        operating_margin = self._normalized_operating_margin(
            annual,
            point_in_time_snapshot,
        )
        expected_dividend = self._expected_dividend_per_share(annual)
        share_dilution = self._normalized_share_dilution(
            annual,
            point_in_time_snapshot,
        )

        required_inputs = {
            "MISSING_NORMALIZED_EPS": normalized_eps,
            "MISSING_REVENUE_GROWTH": revenue_growth,
            "MISSING_OPERATING_MARGIN": operating_margin,
            "MISSING_EXPECTED_DIVIDEND": expected_dividend,
            "MISSING_SHARE_DILUTION": share_dilution,
        }
        reasons.extend(name for name, value in required_inputs.items() if value is None)
        if reasons:
            return self._unknown(signal_time, price, reasons)

        assert price is not None
        assert pe is not None
        assert normalized_eps is not None
        assert revenue_growth is not None
        assert operating_margin is not None
        assert expected_dividend is not None
        assert share_dilution is not None

        base_growth = float(
            np.clip(
                revenue_growth,
                self.config.minimum_growth,
                self.config.maximum_growth,
            )
        )
        base_margin = max(operating_margin, self.config.minimum_margin)
        base_pe = float(
            np.clip(
                pe,
                self.config.minimum_exit_pe,
                self.config.maximum_exit_pe,
            )
        )
        dilution = float(
            np.clip(
                share_dilution,
                self.config.minimum_dilution,
                self.config.maximum_dilution,
            )
        )
        required_return = (
            self.config.passive_benchmark_expected_return
            + self.config.required_excess_return
        )
        bear = self._scenario(
            name="BEAR",
            normalized_eps=normalized_eps,
            revenue_growth=max(
                self.config.minimum_growth,
                base_growth - self.config.growth_spread,
            ),
            operating_margin=max(
                self.config.minimum_margin,
                base_margin * self.config.bear_margin_multiplier,
            ),
            normalized_margin=base_margin,
            years=self.config.bear_years,
            exit_pe=max(
                self.config.minimum_exit_pe,
                base_pe * self.config.bear_pe_multiplier,
            ),
            expected_dividend=expected_dividend,
            share_dilution=dilution,
            current_price=price,
            required_return=required_return,
        )
        base = self._scenario(
            name="BASE",
            normalized_eps=normalized_eps,
            revenue_growth=base_growth,
            operating_margin=base_margin,
            normalized_margin=base_margin,
            years=self.config.base_years,
            exit_pe=base_pe,
            expected_dividend=expected_dividend,
            share_dilution=dilution,
            current_price=price,
            required_return=required_return,
        )
        bull = self._scenario(
            name="BULL",
            normalized_eps=normalized_eps,
            revenue_growth=min(
                self.config.maximum_growth,
                base_growth + self.config.growth_spread,
            ),
            operating_margin=base_margin * self.config.bull_margin_multiplier,
            normalized_margin=base_margin,
            years=self.config.bull_years,
            exit_pe=min(
                self.config.maximum_exit_pe,
                base_pe * self.config.bull_pe_multiplier,
            ),
            expected_dividend=expected_dividend,
            share_dilution=dilution,
            current_price=price,
            required_return=required_return,
        )
        cycle_review = self._cyclical_valuation_review(
            annual,
            point_in_time_snapshot,
            normalized_eps=normalized_eps,
            current_price=price,
            recent_pe_median=pe,
            required_return=required_return,
        )
        permanent_loss = max(0.0, 1.0 - (bear.fair_value / price))
        acceptable = permanent_loss <= self.config.maximum_bear_permanent_loss
        sector_code = _text(point_in_time_snapshot.get("sector"))
        supportive_loss_limit = (
            self.config.air_supportive_maximum_bear_permanent_loss
            if sector_code in self.config.always_review_cyclical_sector_codes
            else self.config.supportive_maximum_bear_permanent_loss
        )
        supportive_downside = permanent_loss <= supportive_loss_limit
        supportive_return = (
            base.expected_cagr
            >= (
                required_return
                - self.config.supportive_required_return_discount
            )
        )
        supportive_value = base.fair_value > price
        standard_supportive = bool(
            supportive_downside
            and (supportive_return or supportive_value)
        )
        supportive = bool(standard_supportive or cycle_review.supportive)
        gate_reasons: list[str] = []
        if base.expected_cagr < required_return:
            gate_reasons.append("BASE_CAGR_BELOW_REQUIRED_RETURN")
        if base.fair_value <= price:
            gate_reasons.append("BASE_FAIR_VALUE_NOT_ABOVE_CURRENT_PRICE")
        if not acceptable:
            gate_reasons.append("BEAR_PERMANENT_LOSS_EXCEEDS_LIMIT")
        if cycle_review.required:
            gate_reasons.extend(cycle_review.reasons)
        return LowEntryV5ValuationResult(
            status=ValuationProductionStatus.COMPLETE,
            signal_date=signal_time.isoformat(),
            current_price=price,
            passive_benchmark_expected_return=(
                self.config.passive_benchmark_expected_return
            ),
            required_return=required_return,
            bear=bear,
            base=base,
            bull=bull,
            bear_permanent_loss=permanent_loss,
            bear_permanent_loss_acceptable=acceptable,
            supportive_gate_passed=supportive,
            cycle_review=cycle_review,
            reasons=tuple(dict.fromkeys(gate_reasons)),
            model_version=self.config.model_version,
        )

    def _unknown(
        self,
        signal_time: pd.Timestamp,
        current_price: float | None,
        reasons: list[str],
    ) -> LowEntryV5ValuationResult:
        return LowEntryV5ValuationResult(
            status=ValuationProductionStatus.UNKNOWN,
            signal_date=None if pd.isna(signal_time) else signal_time.isoformat(),
            current_price=current_price,
            passive_benchmark_expected_return=(
                self.config.passive_benchmark_expected_return
            ),
            required_return=(
                self.config.passive_benchmark_expected_return
                + self.config.required_excess_return
            ),
            reasons=tuple(dict.fromkeys(reasons)),
            model_version=self.config.model_version,
        )

    @staticmethod
    def _point_in_time_annual(
        annual: pd.DataFrame,
        signal_time: pd.Timestamp,
    ) -> pd.DataFrame:
        if annual.empty or pd.isna(signal_time) or "available_at" not in annual:
            return pd.DataFrame()
        result = annual.copy()
        result["available_at"] = pd.to_datetime(
            result["available_at"],
            utc=True,
            errors="coerce",
        )
        result = result[
            result["available_at"].notna()
            & (result["available_at"] <= signal_time)
        ]
        if result.empty:
            return result
        if "fiscal_year" in result:
            result["fiscal_year"] = pd.to_numeric(
                result["fiscal_year"],
                errors="coerce",
            )
            result = result.sort_values(["fiscal_year", "available_at"])
            result = result.drop_duplicates("fiscal_year", keep="last")
        else:
            result = result.sort_values("available_at")
        return result.reset_index(drop=True)

    @staticmethod
    def _point_in_time_snapshot(
        snapshot: dict[str, Any],
        signal_time: pd.Timestamp,
    ) -> dict[str, Any]:
        if not snapshot or pd.isna(signal_time):
            return {}
        available = pd.to_datetime(
            snapshot.get("available_at"),
            utc=True,
            errors="coerce",
        )
        if pd.isna(available) or available > signal_time:
            return {}
        return snapshot

    @staticmethod
    def _normalized_eps(
        annual: pd.DataFrame,
        snapshot: dict[str, Any],
    ) -> float | None:
        values: list[float] = []
        ttm_eps = _positive_number(snapshot.get("ttm_eps"))
        if ttm_eps is not None:
            values.append(ttm_eps)
        if {"net_income", "diluted_shares"}.issubset(annual):
            income = pd.to_numeric(annual["net_income"], errors="coerce")
            shares = pd.to_numeric(annual["diluted_shares"], errors="coerce")
            eps = (income / shares.where(shares > 0)).replace([np.inf, -np.inf], np.nan)
            values.extend(float(value) for value in eps.tail(3).dropna() if value > 0)
        return _median(values, positive=True)

    @staticmethod
    def _normalized_revenue_growth(
        annual: pd.DataFrame,
        snapshot: dict[str, Any],
    ) -> float | None:
        values: list[float] = []
        ttm_growth = _number(snapshot.get("revenue_growth_ttm_yoy"))
        if ttm_growth is not None:
            values.append(ttm_growth)
        if "revenue" in annual:
            revenue = pd.to_numeric(annual["revenue"], errors="coerce")
            growth = revenue.where(revenue > 0).pct_change(fill_method=None)
            values.extend(float(value) for value in growth.tail(4).dropna())
        return _median(values)

    @staticmethod
    def _normalized_operating_margin(
        annual: pd.DataFrame,
        snapshot: dict[str, Any],
    ) -> float | None:
        values: list[float] = []
        for key in ("operating_margin_ttm", "twse_operating_margin_ytd"):
            value = _number(snapshot.get(key))
            if value is not None:
                values.append(value)
        ebit_ttm = _number(snapshot.get("ebit_ttm"))
        revenue_ttm = _positive_number(snapshot.get("revenue_ttm"))
        if ebit_ttm is not None and revenue_ttm is not None:
            values.append(ebit_ttm / revenue_ttm)
        if "operating_margin" in annual:
            margin = pd.to_numeric(annual["operating_margin"], errors="coerce")
            values.extend(float(value) for value in margin.tail(3).dropna())
        return _median(values, positive=True)

    @staticmethod
    def _expected_dividend_per_share(annual: pd.DataFrame) -> float | None:
        if not {"dividends_paid", "diluted_shares"}.issubset(annual):
            return None
        dividends = pd.to_numeric(annual["dividends_paid"], errors="coerce").abs()
        shares = pd.to_numeric(annual["diluted_shares"], errors="coerce")
        per_share = (dividends / shares.where(shares > 0)).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        return _median([float(value) for value in per_share.tail(3).dropna()])

    @staticmethod
    def _normalized_share_dilution(
        annual: pd.DataFrame,
        snapshot: dict[str, Any],
    ) -> float | None:
        values: list[float] = []
        snapshot_dilution = _number(snapshot.get("share_dilution_yoy"))
        if snapshot_dilution is not None:
            values.append(snapshot_dilution)
        if "diluted_shares" in annual:
            shares = pd.to_numeric(annual["diluted_shares"], errors="coerce")
            growth = shares.where(shares > 0).pct_change(fill_method=None)
            values.extend(float(value) for value in growth.tail(4).dropna())
        return _median(values)

    def _cyclical_valuation_review(
        self,
        annual: pd.DataFrame,
        snapshot: dict[str, Any],
        *,
        normalized_eps: float,
        current_price: float,
        recent_pe_median: float,
        required_return: float,
    ) -> CyclicalValuationReview:
        """Review cyclical valuations with multiple independent anchors.

        The three independent anchors are deliberately visible in the result:

        1. mid-cycle operating margin converted to EPS and valued at a capped PE;
        2. book value per share times a normalized-ROE-derived P/B multiple;
        3. TTM EPS valued at the same capped PE after a peak-cycle haircut.

        At least two available anchors and two anchors above spot are required
        before the review can mark valuation as SUPPORTIVE.
        """

        sector_code = _text(snapshot.get("sector"))
        ttm_eps = _positive_number(snapshot.get("ttm_eps"))
        divergence = (
            ttm_eps / normalized_eps
            if ttm_eps is not None and normalized_eps > 0
            else None
        )
        common = {
            "sector_code": sector_code,
            "ttm_eps": ttm_eps,
            "normalized_eps": normalized_eps,
            "eps_divergence_ratio": divergence,
        }
        if sector_code not in self.config.cyclical_sector_codes:
            return CyclicalValuationReview(
                status=CycleValuationReviewStatus.NOT_APPLICABLE,
                required=False,
                reasons=("NON_CYCLICAL_SECTOR",),
                **common,
            )
        always_review = (
            sector_code in self.config.always_review_cyclical_sector_codes
        )
        if not always_review and (
            divergence is None
            or divergence <= self.config.normalized_eps_divergence_threshold
        ):
            return CyclicalValuationReview(
                status=CycleValuationReviewStatus.NOT_TRIGGERED,
                required=False,
                reasons=("NORMALIZED_EPS_DIVERGENCE_NOT_TRIGGERED",),
                **common,
            )

        reasons = [
            "AIR_CYCLICAL_VALUATION_REVIEW"
            if always_review
            else "NORMALIZED_EPS_DIVERGENCE"
        ]
        cycle_pe = float(
            np.clip(
                recent_pe_median,
                self.config.cycle_minimum_pe,
                self.config.cycle_maximum_pe,
            )
        )
        mid_cycle_margin = self._mid_cycle_operating_margin(annual)
        normalized_roe = self._normalized_cycle_roe(annual)
        current_revenue = self._current_revenue(annual, snapshot)
        current_equity = self._current_equity(annual, snapshot)
        shares = self._current_diluted_shares(annual)
        net_income_conversion = self._mid_cycle_net_income_conversion(annual)
        if net_income_conversion is None:
            net_income_conversion = self.config.cycle_default_net_income_conversion
            reasons.append("MID_CYCLE_NET_INCOME_CONVERSION_ASSUMED")

        margin_fair_value: float | None = None
        if (
            mid_cycle_margin is not None
            and mid_cycle_margin > 0
            and current_revenue is not None
            and shares is not None
        ):
            mid_cycle_eps = (
                current_revenue
                * mid_cycle_margin
                * net_income_conversion
                / shares
            )
            margin_fair_value = mid_cycle_eps * cycle_pe

        book_value_per_share = (
            current_equity / shares
            if current_equity is not None and shares is not None
            else None
        )
        pb_fair_value: float | None = None
        if (
            normalized_roe is not None
            and normalized_roe > 0
            and book_value_per_share is not None
            and required_return > 0
        ):
            justified_pb = float(
                np.clip(
                    normalized_roe / required_return,
                    self.config.cycle_minimum_justified_pb,
                    self.config.cycle_maximum_justified_pb,
                )
            )
            pb_fair_value = book_value_per_share * justified_pb

        ttm_eps_fair_value = (
            ttm_eps * cycle_pe * self.config.cycle_ttm_eps_haircut
            if ttm_eps is not None
            else None
        )
        anchors = [
            value
            for value in (
                margin_fair_value,
                pb_fair_value,
                ttm_eps_fair_value,
            )
            if value is not None and np.isfinite(value) and value > 0
        ]
        available_count = len(anchors)
        supporting_count = sum(value >= current_price for value in anchors)
        composite = _median(anchors, positive=True)

        if available_count < self.config.cycle_minimum_available_anchors:
            status = CycleValuationReviewStatus.INSUFFICIENT_DATA
            supportive = False
            reasons.append("CYCLE_REVIEW_INSUFFICIENT_ANCHORS")
        else:
            supportive = bool(
                supporting_count
                >= self.config.cycle_minimum_supporting_anchors
                and composite is not None
                and composite >= current_price
            )
            if supportive:
                status = CycleValuationReviewStatus.SUPPORTIVE
                reasons.append("CYCLE_REVIEW_SUPPORTIVE")
            else:
                status = CycleValuationReviewStatus.FAIL
                reasons.append("CYCLE_REVIEW_NOT_SUPPORTIVE")

        return CyclicalValuationReview(
            status=status,
            required=True,
            mid_cycle_operating_margin=mid_cycle_margin,
            normalized_roe=normalized_roe,
            book_value_per_share=book_value_per_share,
            margin_fair_value=margin_fair_value,
            pb_fair_value=pb_fair_value,
            ttm_eps_fair_value=ttm_eps_fair_value,
            composite_fair_value=composite,
            available_anchor_count=available_count,
            supporting_anchor_count=supporting_count,
            supportive=supportive,
            reasons=tuple(reasons),
            **common,
        )

    @staticmethod
    def _mid_cycle_operating_margin(annual: pd.DataFrame) -> float | None:
        if "operating_margin" not in annual:
            return None
        values = pd.to_numeric(
            annual["operating_margin"],
            errors="coerce",
        ).tail(5)
        return _median([float(value) for value in values.dropna()])

    @staticmethod
    def _normalized_cycle_roe(annual: pd.DataFrame) -> float | None:
        if not {"net_income", "equity"}.issubset(annual):
            return None
        income = pd.to_numeric(annual["net_income"], errors="coerce")
        equity = pd.to_numeric(annual["equity"], errors="coerce")
        roe = (income / equity.where(equity > 0)).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        return _median([float(value) for value in roe.tail(5).dropna()])

    @staticmethod
    def _mid_cycle_net_income_conversion(annual: pd.DataFrame) -> float | None:
        if "net_income" not in annual:
            return None
        income = pd.to_numeric(annual["net_income"], errors="coerce")
        if "ebit" in annual:
            ebit = pd.to_numeric(annual["ebit"], errors="coerce")
        elif {"revenue", "operating_margin"}.issubset(annual):
            revenue = pd.to_numeric(annual["revenue"], errors="coerce")
            margin = pd.to_numeric(annual["operating_margin"], errors="coerce")
            ebit = revenue * margin
        else:
            return None
        conversion = (income / ebit.where(ebit > 0)).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        values = [
            float(value)
            for value in conversion.tail(5).dropna()
            if value > 0
        ]
        result = _median(values, positive=True)
        return float(np.clip(result, 0.50, 1.00)) if result is not None else None

    @staticmethod
    def _current_revenue(
        annual: pd.DataFrame,
        snapshot: dict[str, Any],
    ) -> float | None:
        value = _positive_number(snapshot.get("revenue_ttm"))
        if value is not None:
            return value
        return _last_positive(annual, "revenue")

    @staticmethod
    def _current_equity(
        annual: pd.DataFrame,
        snapshot: dict[str, Any],
    ) -> float | None:
        for key in ("equity", "total_equity", "twse_parent_equity"):
            value = _positive_number(snapshot.get(key))
            if value is not None:
                return value
        return _last_positive(annual, "equity")

    @staticmethod
    def _current_diluted_shares(annual: pd.DataFrame) -> float | None:
        return _last_positive(annual, "diluted_shares")

    @staticmethod
    def _scenario(
        *,
        name: str,
        normalized_eps: float,
        revenue_growth: float,
        operating_margin: float,
        normalized_margin: float,
        years: int,
        exit_pe: float,
        expected_dividend: float,
        share_dilution: float,
        current_price: float,
        required_return: float,
    ) -> ValuationScenario:
        # EPS grows with revenue and margin, then is adjusted for dilution.
        projected_eps = (
            normalized_eps
            * ((1 + revenue_growth) ** years)
            * (operating_margin / normalized_margin)
            * ((1 - share_dilution) ** years)
        )
        terminal_value = projected_eps * exit_pe
        total_dividends = expected_dividend * years
        future_value = terminal_value + total_dividends
        fair_value = future_value / ((1 + required_return) ** years)
        expected_cagr = (future_value / current_price) ** (1 / years) - 1
        return ValuationScenario(
            name=name,
            normalized_eps=normalized_eps,
            revenue_growth=revenue_growth,
            operating_margin=operating_margin,
            competitive_advantage_years=years,
            exit_pe=exit_pe,
            expected_annual_dividend=expected_dividend,
            annual_share_dilution=share_dilution,
            projected_eps=projected_eps,
            terminal_value=terminal_value,
            total_expected_dividends=total_dividends,
            future_value=future_value,
            fair_value=fair_value,
            expected_cagr=expected_cagr,
        )


def _number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _positive_number(value: object) -> float | None:
    parsed = _number(value)
    return parsed if parsed is not None and parsed > 0 else None


def _text(value: object) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None


def _last_positive(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[(values > 0) & values.notna()]
    return float(values.iloc[-1]) if not values.empty else None


def _median(values: list[float], *, positive: bool = False) -> float | None:
    clean = [
        float(value)
        for value in values
        if np.isfinite(value) and (not positive or value > 0)
    ]
    return float(np.median(clean)) if clean else None


__all__ = [
    "CycleValuationReviewStatus",
    "CyclicalValuationReview",
    "LowEntryV5ValuationConfig",
    "LowEntryV5ValuationProducer",
    "LowEntryV5ValuationResult",
    "VALUATION_MODEL_VERSION",
    "ValuationProductionStatus",
    "ValuationScenario",
]

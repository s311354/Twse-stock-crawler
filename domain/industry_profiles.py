"""Versioned industry quality profiles independent from entry-risk profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import ClassVar


class IndustryProfileMode(str, Enum):
    ACTIVE = "ACTIVE"
    SHADOW = "SHADOW"


@dataclass(frozen=True)
class MetricRule:
    name: str
    category: str
    maximum: float
    scorer: Callable[[float], float]


@dataclass(frozen=True)
class IndustryQualityProfile:
    name: str
    version: str
    sector_codes: frozenset[str]
    mode: IndustryProfileMode
    metric_rules: tuple[MetricRule, ...]
    max_net_debt_to_ebitda: float
    min_interest_coverage: float
    pass_score: float
    fail_score: float
    buy_small_quality_score: float
    buy_small_minimum_data_confidence: float
    allow_buy: bool = True
    allow_buy_small: bool = True


ELECTRONICS_SECTOR_CODES = frozenset(str(code) for code in range(24, 32))
CHEMICAL_SECTOR_CODES = frozenset({"21"})
ELECTRONICS_QUALITY_PROFILE_VERSION = "electronics-quality-2026-07-21-v2"
CHEMICAL_QUALITY_PROFILE_VERSION = "chemical-quality-2026-07-27-v2"


ELECTRONICS_PROFILE = IndustryQualityProfile(
    name="ELECTRONICS_PROFILE",
    version=ELECTRONICS_QUALITY_PROFILE_VERSION,
    sector_codes=ELECTRONICS_SECTOR_CODES,
    mode=IndustryProfileMode.ACTIVE,
    metric_rules=(
        MetricRule("operating_income_ttm", "A", 5, lambda x: 5 if x > 0 else 0),
        MetricRule("positive_operating_quarters_8", "A", 5, lambda x: 5 if x >= 7 else 4 if x >= 6 else 2 if x >= 4 else 0),
        MetricRule("roe_ttm", "A", 5, lambda x: 5 if x >= .15 else 4 if x >= .10 else 2 if x >= .05 else 0),
        MetricRule("roa_ttm", "A", 4, lambda x: 4 if x >= .08 else 3 if x >= .05 else 1 if x >= .02 else 0),
        MetricRule("roic_ttm", "A", 6, lambda x: 6 if x >= .12 else 5 if x >= .08 else 3 if x >= .05 else 0),
        MetricRule("cfo_ttm", "B", 6, lambda x: 6 if x > 0 else 0),
        MetricRule("cash_conversion", "B", 8, lambda x: 8 if x >= 1 else 6 if x >= .8 else 3 if x >= .5 else 0),
        MetricRule("fcf_ttm", "B", 6, lambda x: 6 if x > 0 else 0),
        MetricRule("accrual_ratio", "B", 5, lambda x: 5 if x <= 0 else 4 if x <= .05 else 2 if x <= .10 else 0),
        MetricRule("debt_ratio", "C", 5, lambda x: 5 if x <= .40 else 3 if x <= .60 else 0),
        MetricRule("net_debt_to_equity", "C", 6, lambda x: 6 if x <= 0 else 5 if x <= .25 else 3 if x <= .50 else 0),
        MetricRule("interest_coverage", "C", 6, lambda x: 6 if x >= 8 else 5 if x >= 5 else 3 if x >= 3 else 0),
        MetricRule("current_ratio", "C", 4, lambda x: 4 if x >= 1.5 else 3 if x >= 1.2 else 1 if x >= 1 else 0),
        MetricRule("cash_to_current_liabilities", "C", 4, lambda x: 4 if x >= .5 else 3 if x >= .3 else 1 if x >= .15 else 0),
        MetricRule("revenue_growth_ttm_yoy", "D", 4, lambda x: 4 if x > .05 else 3 if x >= 0 else 1 if x >= -.05 else 0),
        MetricRule("operating_income_growth_ttm_yoy", "D", 4, lambda x: 4 if x > .10 else 3 if x >= 0 else 1 if x >= -.10 else 0),
        MetricRule("positive_eps_quarters_8", "D", 3, lambda x: 3 if x >= 7 else 2 if x >= 6 else 1 if x >= 4 else 0),
        MetricRule("gross_margin_change", "D", 4, lambda x: 4 if x >= -.01 else 2 if x >= -.03 else 0),
        MetricRule("share_dilution_yoy", "E", 4, lambda x: 4 if x <= 0 else 3 if x <= .02 else 1 if x <= .05 else 0),
        MetricRule("distribution_coverage", "E", 3, lambda x: 3 if 0 <= x <= 1 else 1 if x <= 1.5 else 0),
        MetricRule("bvps_trend", "E", 3, lambda x: 3 if x > .02 else 1 if x >= -.02 else 0),
    ),
    max_net_debt_to_ebitda=3.0,
    min_interest_coverage=3.0,
    pass_score=70.0,
    fail_score=55.0,
    buy_small_quality_score=75.0,
    buy_small_minimum_data_confidence=60.0,
)


CHEMICAL_PROFILE = IndustryQualityProfile(
    name="CHEMICAL_PROFILE",
    version=CHEMICAL_QUALITY_PROFILE_VERSION,
    sector_codes=CHEMICAL_SECTOR_CODES,
    mode=IndustryProfileMode.ACTIVE,
    metric_rules=(
        MetricRule("operating_income_ttm", "A", 4, lambda x: 4 if x > 0 else 0),
        MetricRule("roic_5y_median", "A", 10, lambda x: 10 if x >= .08 else 7 if x >= .05 else 3 if x >= 0 else 0),
        MetricRule("positive_operating_quarters_8", "A", 5, lambda x: 5 if x >= 6 else 3 if x >= 4 else 0),
        MetricRule("operating_margin_trend", "A", 6, lambda x: 6 if x >= .01 else 4 if x >= 0 else 2 if x >= -.02 else 0),
        MetricRule("cumulative_cfo_3y", "B", 7, lambda x: 7 if x > 0 else 0),
        MetricRule("positive_cfo_year_ratio", "B", 8, lambda x: 8 if x >= .8 else 5 if x >= .6 else 2 if x >= .4 else 0),
        MetricRule("cash_conversion", "B", 6, lambda x: 6 if x >= 1 else 4 if x >= .7 else 2 if x >= .4 else 0),
        MetricRule("fcf_ttm", "B", 4, lambda x: 4 if x > 0 else 0),
        MetricRule("debt_ratio", "C", 5, lambda x: 5 if x <= .50 else 3 if x <= .65 else 0),
        MetricRule("net_debt_to_equity", "C", 5, lambda x: 5 if x <= .25 else 4 if x <= .50 else 2 if x <= .80 else 0),
        MetricRule("interest_coverage", "C", 6, lambda x: 6 if x >= 5 else 4 if x >= 3 else 0),
        MetricRule("current_ratio", "C", 4, lambda x: 4 if x >= 1.5 else 3 if x >= 1.2 else 1 if x >= 1 else 0),
        MetricRule("gross_margin_5y_percentile", "D", 6, lambda x: 6 if .2 <= x <= .6 else 4 if .6 < x <= .9 else 2 if x < .2 else 1),
        MetricRule("gross_margin_change", "D", 5, lambda x: 5 if x >= .01 else 3 if x >= 0 else 1 if x >= -.03 else 0),
        MetricRule("operating_income_growth_ttm_yoy", "D", 5, lambda x: 5 if x > .05 else 3 if x >= 0 else 1 if x >= -.10 else 0),
        MetricRule("revenue_growth_ttm_yoy", "D", 4, lambda x: 4 if x > .05 else 3 if x >= 0 else 1 if x >= -.05 else 0),
        MetricRule("share_dilution_yoy", "E", 4, lambda x: 4 if x <= 0 else 3 if x <= .02 else 1 if x <= .05 else 0),
        MetricRule("distribution_coverage", "E", 3, lambda x: 3 if 0 <= x <= 1 else 1 if x <= 1.5 else 0),
        MetricRule("bvps_trend", "E", 3, lambda x: 3 if x > .02 else 1 if x >= -.02 else 0),
    ),
    max_net_debt_to_ebitda=3.5,
    min_interest_coverage=3.0,
    pass_score=65.0,
    fail_score=50.0,
    buy_small_quality_score=60.0,
    buy_small_minimum_data_confidence=75.0,
)


class IndustryProfileRegistry:
    """Resolve one immutable quality profile from an official TWSE sector code."""

    _profiles: ClassVar = (ELECTRONICS_PROFILE, CHEMICAL_PROFILE)
    _by_sector: ClassVar = MappingProxyType(
        {
            sector: profile
            for profile in _profiles
            for sector in profile.sector_codes
        }
    )
    _by_name: ClassVar = MappingProxyType(
        {profile.name: profile for profile in _profiles}
    )

    @classmethod
    def resolve(cls, sector_code: str) -> IndustryQualityProfile | None:
        return cls._by_sector.get(str(sector_code).strip())

    @classmethod
    def profiles(cls) -> tuple[IndustryQualityProfile, ...]:
        return cls._profiles

    @classmethod
    def get(cls, profile_name: str) -> IndustryQualityProfile | None:
        return cls._by_name.get(str(profile_name).strip())


__all__ = [
    "CHEMICAL_QUALITY_PROFILE_VERSION",
    "CHEMICAL_PROFILE",
    "CHEMICAL_SECTOR_CODES",
    "ELECTRONICS_QUALITY_PROFILE_VERSION",
    "ELECTRONICS_PROFILE",
    "ELECTRONICS_SECTOR_CODES",
    "IndustryProfileMode",
    "IndustryProfileRegistry",
    "IndustryQualityProfile",
    "MetricRule",
]

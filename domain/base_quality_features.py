"""Point-in-time feature production for the Low Entry v5 Base Quality gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from domain.data_provenance import TRUSTED_AVAILABILITY_SOURCES
from domain.fundamental_models import FundamentalSnapshot


TWSE_HISTORY_SOURCE = "TWSE_OPENAPI_HISTORY"
MOPS_ANNUAL_FALLBACK_SOURCE = "MOPS_POINT_IN_TIME_ANNUAL_FALLBACK"
MOPS_TTM_FALLBACK_SOURCE = "MOPS_POINT_IN_TIME_TTM_FALLBACK"


@dataclass(frozen=True)
class BaseQualityFeatureSet:
    """Immutable result with field-level provenance for auditability."""

    values: tuple[tuple[str, object], ...]
    sources: tuple[tuple[str, str], ...]
    available_at: tuple[tuple[str, str], ...]
    source_reliability: float | None
    periods: tuple[str, ...]

    def as_snapshot_fields(self) -> dict[str, object]:
        fields = dict(self.values)
        fields.update(
            {
                "base_quality_feature_sources": ";".join(
                    "{}={}".format(name, source)
                    for name, source in self.sources
                ),
                "base_quality_feature_available_at": ";".join(
                    "{}={}".format(name, timestamp)
                    for name, timestamp in self.available_at
                ),
                "base_quality_feature_source_reliability": (
                    self.source_reliability
                ),
                "base_quality_feature_periods": ",".join(self.periods),
            }
        )
        return fields


@dataclass(frozen=True)
class BaseQualityFeatureProducer:
    """Derive Base features from comparable official reporting periods.

    Same-quarter TWSE snapshots are authoritative. Until that cache has a
    prior-year period, audited/reviewed annual MOPS rows provide an explicit
    point-in-time fallback for annual trend fields. YTD and annual flows are
    never compared with each other.
    """

    def produce(
        self,
        *,
        current: FundamentalSnapshot | None,
        history: list[FundamentalSnapshot],
        annual_fundamentals: pd.DataFrame,
        existing_snapshot: dict[str, Any],
        signal_date: str | pd.Timestamp,
    ) -> BaseQualityFeatureSet:
        signal = pd.to_datetime(signal_date, utc=True, errors="coerce")
        if pd.isna(signal):
            return BaseQualityFeatureSet((), (), (), None, ())

        values: dict[str, object] = {}
        sources: dict[str, str] = {}
        available_at: dict[str, str] = {}
        periods: set[str] = set()

        candidates = history + ([current] if current is not None else [])
        eligible = _latest_eligible_history(candidates, signal)
        active = eligible.get(current.period) if current is not None else None
        prior = (
            eligible.get((current.fiscal_year - 1, current.quarter))
            if current is not None
            else None
        )
        if active is not None and prior is not None:
            twse_values = _twse_same_quarter_features(active, prior, eligible)
            _merge_features(
                values,
                sources,
                available_at,
                twse_values,
                TWSE_HISTORY_SOURCE,
                active.observed_at,
                overwrite=True,
            )
            periods.update(
                {
                    "{}Q{}".format(active.fiscal_year, active.quarter),
                    "{}Q{}".format(prior.fiscal_year, prior.quarter),
                }
            )

        annual_rows = _eligible_annual_rows(annual_fundamentals, signal)
        if len(annual_rows) >= 2:
            previous = annual_rows.iloc[-2]
            latest = annual_rows.iloc[-1]
            annual_values = _annual_features(latest, previous)
            annual_available_at = str(latest["available_at"])
            _merge_features(
                values,
                sources,
                available_at,
                annual_values,
                MOPS_ANNUAL_FALLBACK_SOURCE,
                annual_available_at,
                overwrite=False,
            )
            periods.update(
                {
                    "{}FY".format(int(previous["fiscal_year"])),
                    "{}FY".format(int(latest["fiscal_year"])),
                }
            )

        # Existing point-in-time TTM derivations are more current than annual
        # fallback values, but never override comparable TWSE history.
        existing_available_at = pd.to_datetime(
            existing_snapshot.get("available_at"),
            utc=True,
            errors="coerce",
        )
        existing_is_eligible = (
            not pd.isna(existing_available_at)
            and existing_available_at <= signal
            and str(existing_snapshot.get("availability_source", ""))
            in TRUSTED_AVAILABILITY_SOURCES
        )
        if existing_is_eligible:
            for key in ("gross_margin_change", "revenue_growth_ttm_yoy"):
                value = _number(existing_snapshot.get(key))
                if value is not None and key not in values:
                    values[key] = value
                    sources[key] = MOPS_TTM_FALLBACK_SOURCE
                    available_at[key] = existing_available_at.isoformat()

        used_sources = set(sources.values())
        reliability = None
        if used_sources:
            reliability = min(
                95.0 if source == TWSE_HISTORY_SOURCE else 90.0
                for source in used_sources
            )
        return BaseQualityFeatureSet(
            values=tuple(sorted(values.items())),
            sources=tuple(sorted(sources.items())),
            available_at=tuple(sorted(available_at.items())),
            source_reliability=reliability,
            periods=tuple(sorted(periods)),
        )


def _latest_eligible_history(
    snapshots: list[FundamentalSnapshot],
    signal: pd.Timestamp,
) -> dict[tuple[int, int], FundamentalSnapshot]:
    latest: dict[tuple[int, int], FundamentalSnapshot] = {}
    ordered = sorted(
        snapshots,
        key=lambda item: (
            item.fiscal_year,
            item.quarter,
            item.observed_at,
            item.content_sha256,
        ),
    )
    for snapshot in ordered:
        observed = pd.to_datetime(snapshot.observed_at, utc=True, errors="coerce")
        if pd.isna(observed) or observed > signal:
            continue
        latest[snapshot.period] = snapshot
    return latest


def _twse_same_quarter_features(
    current: FundamentalSnapshot,
    prior: FundamentalSnapshot,
    history: dict[tuple[int, int], FundamentalSnapshot],
) -> dict[str, object]:
    asset_growth = _growth(current.total_assets, prior.total_assets)
    revenue_growth = _growth(current.revenue_ytd, prior.revenue_ytd)
    income_growth = _growth(
        current.parent_net_income_ytd,
        prior.parent_net_income_ytd,
    )
    return {
        "equity_growth_yoy": _growth(current.parent_equity, prior.parent_equity),
        "operating_margin_change": _difference(
            _safe_div(current.operating_income_ytd, current.revenue_ytd),
            _safe_div(prior.operating_income_ytd, prior.revenue_ytd),
        ),
        "gross_margin_change": _difference(
            _safe_div(current.gross_profit_ytd, current.revenue_ytd),
            _safe_div(prior.gross_profit_ytd, prior.revenue_ytd),
        ),
        "roe_change_4q": _difference(
            _safe_div(current.parent_net_income_ytd, current.parent_equity),
            _safe_div(prior.parent_net_income_ytd, prior.parent_equity),
        ),
        "eps_growth_ttm_yoy": _growth(current.basic_eps_ytd, prior.basic_eps_ytd),
        "asset_growth_yoy": asset_growth,
        "asset_growth_supported": _asset_growth_supported(
            asset_growth,
            revenue_growth,
            income_growth,
        ),
        "positive_operating_quarters_8": _positive_operating_quarters(history),
    }


def _eligible_annual_rows(
    annual: pd.DataFrame,
    signal: pd.Timestamp,
) -> pd.DataFrame:
    required = {"fiscal_year", "available_at", "availability_source"}
    if not isinstance(annual, pd.DataFrame) or annual.empty or not required.issubset(annual):
        return pd.DataFrame()
    rows = annual.copy()
    available = pd.to_datetime(rows["available_at"], utc=True, errors="coerce")
    trusted = rows["availability_source"].isin(TRUSTED_AVAILABILITY_SOURCES)
    rows = rows.loc[
        available.notna() & (available <= signal) & trusted
    ].copy()
    if rows.empty:
        return rows
    rows["fiscal_year"] = pd.to_numeric(rows["fiscal_year"], errors="coerce")
    rows = rows.dropna(subset=["fiscal_year"])
    return rows.sort_values("fiscal_year", kind="stable").drop_duplicates(
        "fiscal_year",
        keep="last",
    )


def _annual_features(current: pd.Series, prior: pd.Series) -> dict[str, object]:
    equity_growth = _growth(_series_number(current, "equity"), _series_number(prior, "equity"))
    asset_growth = _growth(
        _series_number(current, "total_assets"),
        _series_number(prior, "total_assets"),
    )
    revenue_growth = _growth(
        _series_number(current, "revenue"),
        _series_number(prior, "revenue"),
    )
    income_growth = _growth(
        _series_number(current, "net_income"),
        _series_number(prior, "net_income"),
    )
    current_roe = _safe_div(
        _series_number(current, "net_income"),
        _series_number(current, "equity"),
    )
    prior_roe = _safe_div(
        _series_number(prior, "net_income"),
        _series_number(prior, "equity"),
    )
    return {
        "equity_growth_yoy": equity_growth,
        "operating_margin_change": _difference(
            _series_number(current, "operating_margin"),
            _series_number(prior, "operating_margin"),
        ),
        "roe_change_4q": _difference(current_roe, prior_roe),
        "asset_growth_yoy": asset_growth,
        "asset_growth_supported": _asset_growth_supported(
            asset_growth,
            revenue_growth,
            income_growth,
        ),
    }


def _positive_operating_quarters(
    history: dict[tuple[int, int], FundamentalSnapshot],
) -> int | None:
    isolated: list[tuple[tuple[int, int], float]] = []
    for period in sorted(history):
        year, quarter = period
        current = history[period].operating_income_ytd
        if current is None:
            continue
        if quarter == 1:
            value = current
        else:
            prior = history.get((year, quarter - 1))
            if prior is None or prior.operating_income_ytd is None:
                continue
            value = current - prior.operating_income_ytd
        isolated.append((period, value))
    if len(isolated) < 8:
        return None
    return sum(value > 0 for _, value in isolated[-8:])


def _merge_features(
    values: dict[str, object],
    sources: dict[str, str],
    available_at: dict[str, str],
    incoming: dict[str, object],
    source: str,
    observed_at: str,
    *,
    overwrite: bool,
) -> None:
    for key, value in incoming.items():
        if value is None or (not overwrite and key in values):
            continue
        values[key] = value
        sources[key] = source
        available_at[key] = observed_at


def _asset_growth_supported(
    asset_growth: float | None,
    revenue_growth: float | None,
    income_growth: float | None,
) -> bool | None:
    if asset_growth is None:
        return None
    support = [value for value in (revenue_growth, income_growth) if value is not None]
    if not support:
        return None
    return max(support) >= asset_growth - 0.05


def _series_number(row: pd.Series, key: str) -> float | None:
    return _number(row.get(key))


def _number(value: object) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    return float(number)


def _growth(current: float | None, prior: float | None) -> float | None:
    if current is None or prior in {None, 0}:
        return None
    return (current - prior) / abs(prior)


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _difference(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None:
        return None
    return current - prior

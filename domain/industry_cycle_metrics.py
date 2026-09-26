"""Shared numerical primitives; industry thresholds belong to profiles.

Callers must first select annual filing revisions at the decision cutoff.
Complete windows never skip missing years, untrusted sources or invalid values.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from domain.data_provenance import TRUSTED_AVAILABILITY_SOURCES


def complete_annual_window(annual: pd.DataFrame, columns: list[str], years: int) -> pd.DataFrame | None:
    required = {"fiscal_year", "availability_source", *columns}
    if not required.issubset(annual.columns):
        return None
    window = annual.sort_values("fiscal_year").tail(years)
    fiscal = pd.to_numeric(window["fiscal_year"], errors="coerce")
    if (len(window) != years or not fiscal.mod(1).eq(0).all() or not fiscal.diff().iloc[1:].eq(1).all()
            or not window["availability_source"].isin(TRUSTED_AVAILABILITY_SOURCES).all()):
        return None
    values = window[columns].apply(pd.to_numeric, errors="coerce")
    return values if np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all() else None


def finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def round_cycle_values(values: Mapping[str, object]) -> dict[str, float | None]:
    """Stabilize decimal financial thresholds after binary floating arithmetic."""
    return {key: round(number, 12) if number is not None else None
            for key, value in values.items() for number in [finite_number(value)]}


def cash_flow_cycle_values(annual: pd.DataFrame) -> dict[str, float | None]:
    """Five consecutive years; FCF = CFO - non-negative CAPEX (same unit)."""
    values: dict[str, float | None] = dict.fromkeys(("positive_cfo_year_ratio", "positive_fcf_year_ratio", "capex_to_cfo_5y"))
    cfo = complete_annual_window(annual, ["operating_cash_flow"], 5)
    if cfo is not None:
        values["positive_cfo_year_ratio"] = float(cfo["operating_cash_flow"].gt(0).mean())
    cash = complete_annual_window(annual, ["operating_cash_flow", "capex"], 5)
    if cash is not None and cash["capex"].ge(0).all():
        values["positive_fcf_year_ratio"] = float((cash["operating_cash_flow"] - cash["capex"]).gt(0).mean())
        total_cfo = float(cash["operating_cash_flow"].sum())
        # Relative tolerance handles cancellation without depending on TWD units.
        scale = float(cash["operating_cash_flow"].abs().sum())
        if total_cfo > scale * 1e-12:
            values["capex_to_cfo_5y"] = float(cash["capex"].sum() / total_cfo)
    return values


def current_gross_margin(snapshot: Mapping[str, object]) -> float | None:
    margin = finite_number(snapshot.get("gross_margin_ttm"))
    if margin is None:
        gross_profit = finite_number(snapshot.get("gross_profit_ttm"))
        revenue = finite_number(snapshot.get("revenue_ttm"))
        if gross_profit is not None and revenue is not None and revenue > 0:
            margin = gross_profit / revenue
    return margin

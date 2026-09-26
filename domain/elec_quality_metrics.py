"""Electronics cycle metrics from PIT-selected, trusted annual statements.

Five years means five consecutive fiscal years, not five non-null observations.
Margins/returns use fractions, CAPEX is a non-negative cash outflow. Missing
years, untrusted facts and invalid denominators remain None, never zero.
"""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from domain.industry_cycle_metrics import (
    cash_flow_cycle_values, complete_annual_window, current_gross_margin,
    round_cycle_values,
)


ELEC_CYCLE_METRICS = (
    "elec_roic_5y_median", "elec_positive_cfo_year_ratio",
    "elec_positive_fcf_year_ratio", "elec_capex_to_cfo_5y",
    "elec_gross_margin_drawdown_5y", "elec_operating_margin_trend_3y",
    "elec_revenue_cagr_5y",
)


def electronics_cycle_values(
    annual: pd.DataFrame, snapshot: Mapping[str, object],
) -> dict[str, float | None]:
    """Caller filters annual revisions by cutoff; only pass a PIT-safe snapshot."""
    values: dict[str, float | None] = dict.fromkeys(ELEC_CYCLE_METRICS)
    roic = complete_annual_window(annual, ["roic"], 5)
    if roic is not None:
        values["elec_roic_5y_median"] = float(roic["roic"].median())
    values.update({f"elec_{name}": value for name, value in cash_flow_cycle_values(annual).items()})
    margins = complete_annual_window(annual, ["gross_margin"], 5)
    current_margin = current_gross_margin(snapshot)
    if margins is not None and current_margin is not None:
        values["elec_gross_margin_drawdown_5y"] = current_margin - float(margins["gross_margin"].max())
    trend = complete_annual_window(annual, ["operating_margin"], 3)
    if trend is not None:
        # OLS slope for equally spaced three-year observations: (last-first)/2.
        values["elec_operating_margin_trend_3y"] = float((trend.iloc[-1, 0] - trend.iloc[0, 0]) / 2)
    revenue = complete_annual_window(annual, ["revenue"], 5)
    if revenue is not None and revenue["revenue"].gt(0).all():
        # Five observations span four annual growth intervals.
        values["elec_revenue_cagr_5y"] = float((revenue.iloc[-1, 0] / revenue.iloc[0, 0]) ** .25 - 1)
    return round_cycle_values(values)

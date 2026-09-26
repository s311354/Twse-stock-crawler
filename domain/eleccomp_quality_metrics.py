"""Electronic-component cycle resilience, independent of entry/risk settings."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from domain.industry_cycle_metrics import (
    cash_flow_cycle_values, complete_annual_window, current_gross_margin,
    round_cycle_values,
)


ELECCOMP_CYCLE_METRICS = (
    "eleccomp_roic_5y_p25", "eleccomp_gross_margin_5y_p25",
    "eleccomp_positive_cfo_year_ratio", "eleccomp_positive_fcf_year_ratio",
    "eleccomp_capex_to_cfo_5y", "eleccomp_gross_margin_drawdown_5y",
    "eleccomp_operating_margin_trend_3y", "eleccomp_asset_turnover_trend_3y",
)


def components_cycle_values(
    annual: pd.DataFrame, snapshot: Mapping[str, object],
) -> dict[str, float | None]:
    """Use PIT-selected revisions and a trusted snapshot supplied by the gate.

    P25 uses linear interpolation (five observations: the second sorted value).
    Turnover is revenue/year-end assets, not utilization or fixed-asset turnover.
    """
    values: dict[str, float | None] = dict.fromkeys(ELECCOMP_CYCLE_METRICS)
    roic = complete_annual_window(annual, ["roic"], 5)
    if roic is not None:
        values["eleccomp_roic_5y_p25"] = float(roic["roic"].quantile(.25, interpolation="linear"))
    values.update({f"eleccomp_{name}": value for name, value in cash_flow_cycle_values(annual).items()})
    margins = complete_annual_window(annual, ["gross_margin"], 5)
    current_margin = current_gross_margin(snapshot)
    if margins is not None:
        values["eleccomp_gross_margin_5y_p25"] = float(margins["gross_margin"].quantile(.25, interpolation="linear"))
        if current_margin is not None:
            values["eleccomp_gross_margin_drawdown_5y"] = current_margin - float(margins["gross_margin"].max())
    trend = complete_annual_window(annual, ["operating_margin"], 3)
    if trend is not None:
        values["eleccomp_operating_margin_trend_3y"] = float((trend.iloc[-1, 0] - trend.iloc[0, 0]) / 2)
    asset_use = complete_annual_window(annual, ["revenue", "total_assets"], 3)
    if asset_use is not None and asset_use["total_assets"].gt(0).all() and asset_use["revenue"].gt(0).all():
        turnover = asset_use["revenue"] / asset_use["total_assets"]
        values["eleccomp_asset_turnover_trend_3y"] = float((turnover.iloc[-1] - turnover.iloc[0]) / 2)
    return round_cycle_values(values)

"""Optional cutoff-only inputs for next-session open-in-band research.

These are observed trailing statistics, not execution probabilities, fill
labels, trading permission or a replacement for the frozen reference path.
The caller owns the source/coordinate receipts and the causal base features.
Nothing is added to the legacy feature dictionary unless explicitly requested.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from domain.les_v7.contracts import ReferencePlan, finite, timestamp
from domain.les_v7.features import select_bars

EXECUTION_FEATURE_PROFILE = "V7_EXECUTION_MULTISCALE_R1"
GAP_WINDOWS = (5, 20, 60)
BAND_WINDOWS = (20, 60, 120)
REQUIRED_CLOSE_COUNT = 121
VOLATILITY_DENOMINATOR_FLOOR = 1e-8
FRACTAL_DERIVED_NAMES = (
    "dfa512_x_gap_vol_ratio5_60",
    "higuchi256_x_band_fraction20",
    "higuchi512_x_band_fraction60",
)
FEATURE_NAMES = (
    *(f"execution_gap_mean{window}" for window in GAP_WINDOWS),
    *(f"execution_gap_std{window}" for window in GAP_WINDOWS),
    *(f"execution_band_fraction{window}" for window in BAND_WINDOWS),
    "execution_gap_vol_ratio5_60",
    "execution_close_vol_ratio5_60",
    "execution_band_width_fraction",
    "execution_last_gap",
    *FRACTAL_DERIVED_NAMES,
)
FEATURE_BOUNDS = (
    *((-10., 10.),) * 3,
    *((0., 10.),) * 3,
    *((0., 1.),) * 3,
    (0., 20.), (0., 20.), (0., 10.), (-10., 10.),
    *((-200., 200.),) * 3,
)


def profile() -> dict:
    """Fresh, declared definitions; bounds are not estimated on any labels."""
    return dict(version=EXECUTION_FEATURE_PROFILE, feature_names=list(FEATURE_NAMES),
        feature_bounds=[list(bounds) for bounds in FEATURE_BOUNDS],
        required_close_count=REQUIRED_CLOSE_COUNT, gap_windows=list(GAP_WINDOWS),
        band_windows=list(BAND_WINDOWS),
        gap_coordinate="LOG_OPEN_OVER_PREVIOUS_CLOSE",
        band_reference="CURRENT_FROZEN_PLAN_BAND_OVER_CUTOFF_CLOSE",
        volatility="POPULATION_STANDARD_DEVIATION_DDOF_0",
        gap_volatility_ratio="TRAILING_5_OVER_TRAILING_60",
        close_volatility_ratio="TRAILING_5_OVER_TRAILING_60_LOG_CLOSE_RETURNS",
        volatility_ratio_denominator="MAX_TRAILING_60_STD_WITH_DECLARED_NUMERICAL_FLOOR",
        volatility_denominator_floor=VOLATILITY_DENOMINATOR_FLOOR,
        fractal_interactions=dict(zip(FRACTAL_DERIVED_NAMES, (
            "(dfa512 - 0.5) * execution_gap_vol_ratio5_60",
            "(higuchi256 - 1.5) * execution_band_fraction20",
            "(higuchi512 - 1.5) * execution_band_fraction60"))),
        missing_policy="NOT_READY_NO_IMPUTATION",
        support_policy="OUT_OF_SUPPORT_NO_CLIPPING",
        interpretation="TRAILING_FEATURES_NOT_FITTED_FILL_PROBABILITIES")


def compute_execution_features(data: pd.DataFrame, *, cutoff: str,
                               base_features: dict, plan: ReferencePlan | None) -> tuple[dict, dict]:
    """Select only complete, available bars, then compute the optional profile.

    ``base_features`` must be the caller's source-bound features for this same
    cutoff. In particular the supplied DFA/Higuchi estimators are not refitted
    from this module's shorter 121-close history. No target labels are accepted.
    """
    timestamp(cutoff)
    selected = select_bars(data, cutoff)
    return _from_selected_bars(selected, base_features=base_features, plan=plan)


def _from_selected_bars(data: pd.DataFrame, *, base_features: dict,
                        plan: ReferencePlan | None) -> tuple[dict, dict]:
    """Internal fast path for already validated, cutoff-selected native bars.

    This is not a public prepared-row/source-proof intake. Runtime composition
    and verified offline research may avoid reselecting an existing native
    cutoff view. Only trailing 121 closes are needed; no estimator refitting,
    future path, account state, labels or mutable cache is involved.
    """
    if not isinstance(base_features, dict):
        raise ValueError("EXECUTION_BASE_FEATURES_REQUIRED")
    if plan is not None and not isinstance(plan, ReferencePlan):
        raise ValueError("EXECUTION_REFERENCE_PLAN_REQUIRED")
    values = dict.fromkeys(FEATURE_NAMES)
    audit = dict(profile=EXECUTION_FEATURE_PROFILE, status="NOT_READY",
        available_close_count=len(data), required_close_count=REQUIRED_CLOSE_COUNT,
        cutoff_session=(pd.Timestamp(data["Date"].iloc[-1]).date().isoformat()
                        if not data.empty else None), reasons=[])
    if len(data) < REQUIRED_CLOSE_COUNT:
        audit["reasons"].append("INSUFFICIENT_EXECUTION_FEATURE_HISTORY")
        return values, audit
    if plan is None:
        audit["reasons"].append("EXECUTION_REFERENCE_PLAN_NOT_READY")
        return values, audit
    close = data["Close"].iloc[-REQUIRED_CLOSE_COUNT:].to_numpy(dtype=float, copy=True)
    opened = data["Open"].iloc[-REQUIRED_CLOSE_COUNT:].to_numpy(dtype=float, copy=True)
    if (not np.isfinite(close).all() or not np.isfinite(opened).all()
            or (close <= 0).any() or (opened <= 0).any()):
        raise ValueError("EXECUTION_POSITIVE_FINITE_PRICES_REQUIRED")
    latest = finite(base_features.get("close"), "execution_cutoff_close", 0)
    if latest != float(close[-1]) or latest <= 0:
        raise ValueError("EXECUTION_BASE_FEATURE_CUTOFF_MISMATCH")
    atr = finite(base_features.get("atr"), "execution_cutoff_atr", 0)
    if atr != plan.atr:
        raise ValueError("EXECUTION_REFERENCE_ATR_MISMATCH")
    # Log differences avoid an intermediate Open/Close overflow and use the
    # identical relative-price coordinate for history and the frozen band.
    log_close = np.log(close)
    gap = np.log(opened[1:]) - log_close[:-1]
    returns = np.diff(log_close)
    for window in GAP_WINDOWS:
        values[f"execution_gap_mean{window}"] = float(np.mean(gap[-window:]))
        values[f"execution_gap_std{window}"] = float(np.std(gap[-window:], ddof=0))
    lower, upper = (math.log(bound) - math.log(latest) for bound in plan.band)
    for window in BAND_WINDOWS:
        sample = gap[-window:]
        values[f"execution_band_fraction{window}"] = float(np.mean((sample >= lower) & (sample <= upper)))
    gap_long = values["execution_gap_std60"]
    close_short, close_long = (float(np.std(returns[-window:], ddof=0)) for window in (5, 60))
    # Zero observed volatility is data, not missing history. A declared fixed
    # numerical denominator floor gives a finite zero ratio for 0/0 and never
    # supplies an unobserved return or fill label. The floor is not data fitted.
    values["execution_gap_vol_ratio5_60"] = values["execution_gap_std5"] / max(
        gap_long, VOLATILITY_DENOMINATOR_FLOOR)
    values["execution_close_vol_ratio5_60"] = close_short / max(
        close_long, VOLATILITY_DENOMINATOR_FLOOR)
    audit["volatility_denominator_floor"] = VOLATILITY_DENOMINATOR_FLOOR
    audit["variance_floor_applied"] = [name for name, value in (("gap60", gap_long), ("close60", close_long))
                                       if value < VOLATILITY_DENOMINATOR_FLOOR]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        width = float(np.divide(plan.band[1] - plan.band[0], latest))
    values["execution_band_width_fraction"] = width if math.isfinite(width) else None
    values["execution_last_gap"] = float(gap[-1])
    for name, base_name, center, other_name in (
        (FRACTAL_DERIVED_NAMES[0], "dfa512", .5, "execution_gap_vol_ratio5_60"),
        (FRACTAL_DERIVED_NAMES[1], "higuchi256", 1.5, "execution_band_fraction20"),
        (FRACTAL_DERIVED_NAMES[2], "higuchi512", 1.5, "execution_band_fraction60"),
    ):
        fractal, other = base_features.get(base_name), values[other_name]
        if fractal is not None and other is not None:
            values[name] = (finite(fractal, base_name) - center) * other
        else:
            audit["reasons"].append("MISSING_EXECUTION_INTERACTION_INPUT:" + name)
    missing = [name for name, value in values.items() if value is None or not math.isfinite(value)]
    if missing:
        audit["missing_features"] = missing
        if not audit["reasons"]:
            audit["reasons"].append("NONFINITE_EXECUTION_FEATURE")
        return values, audit
    outside = [name for name, bounds in zip(FEATURE_NAMES, FEATURE_BOUNDS)
               if not bounds[0] <= values[name] <= bounds[1]]
    if outside:
        audit.update(status="OOD", reasons=["EXECUTION_FEATURE_OUTSIDE_DECLARED_SUPPORT"],
                     outside_support=outside)
        return values, audit
    audit.update(status="COMPUTED", used_close_count=REQUIRED_CLOSE_COUNT,
                 used_gap_count=REQUIRED_CLOSE_COUNT - 1)
    return values, audit

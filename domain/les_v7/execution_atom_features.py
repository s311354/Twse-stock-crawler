"""Opt-in atom-aware gap distribution coordinates for fill-only research.

These are fixed statistical inputs, never fills, fitted target probabilities,
entry permission or a replacement for native reference execution.  Exact zero
overnight gaps remain discrete atoms; only observed nonzero gaps are smoothed.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import math

import numpy as np
import pandas as pd

from domain.les_v7.contracts import ReferencePlan, finite, timestamp
from domain.les_v7.features import select_bars, wilder
from domain.les_v7.reference_path import _GRID

EXECUTION_FEATURE_PROFILE = "V7_EXECUTION_ATOM_CDF_R2"
WINDOWS = (20, 60, 120)
REQUIRED_CLOSE_COUNT = 135
JEFFREYS_ALPHA = .5
SILVERMAN_FACTOR = 1.06
MAX_LEGAL_BAND_PRICES = 1000000
FRACTAL_DERIVED_NAMES = (
    "dfa512_x_execution_atr_band_logit60",
    "higuchi256_x_execution_zero_gap_logit60",
    "higuchi512_x_execution_atom_band_logit60",
)
FEATURE_NAMES = (
    *(f"execution_atom_band_logit{window}" for window in WINDOWS),
    *(f"execution_atr_band_logit{window}" for window in WINDOWS),
    "execution_zero_gap_logit60", "execution_legal_band_log1p_prices",
    *FRACTAL_DERIVED_NAMES,
)
FEATURE_BOUNDS = (
    *((-math.log(2 * window + 1), math.log(2 * window + 1)) for window in WINDOWS),
    *((-math.log(2 * window + 1), math.log(2 * window + 1)) for window in WINDOWS),
    (-math.log(121), math.log(121)), (0., math.log(MAX_LEGAL_BAND_PRICES + 1)),
    *((-200., 200.),) * 3,
)


def profile() -> dict:
    """Exact label-independent model assumptions, names, order and bounds."""
    return dict(version=EXECUTION_FEATURE_PROFILE, feature_names=list(FEATURE_NAMES),
        feature_bounds=[list(bounds) for bounds in FEATURE_BOUNDS], windows=list(WINDOWS),
        required_close_count=REQUIRED_CLOSE_COUNT, zero_gap="EXACT_OBSERVED_OPEN_EQUALS_PREVIOUS_CLOSE",
        log_gap="LOG_OPEN_MINUS_LOG_PREVIOUS_CLOSE",
        atr_gap="OPEN_MINUS_PREVIOUS_CLOSE_OVER_PREVIOUS_NATIVE_WILDER_ATR14",
        atr_seed="FIRST_14_COMPLETE_TRUE_RANGES_WITH_PREVIOUS_CLOSE_THEN_NATIVE_WILDER_RECURSION",
        atr_history="ALL_SELECTED_CUTOFF_HISTORY_NO_RESEED_AT_TRAILING_WINDOW",
        band_coordinates="CURRENT_FROZEN_PLAN_BAND_OVER_CUTOFF_CLOSE_OR_CUTOFF_ATR",
        kernel="NONZERO_GAUSSIAN_CDF_INTERVAL_MASS_PLUS_EXACT_ZERO_ATOM",
        bandwidth="1.06*POPULATION_STD_NONZERO*n_nonzero**(-0.2)", silverman_factor=SILVERMAN_FACTOR,
        degenerate_nonzero="EMPIRICAL_POINT_MASSES_WHEN_ALL_NONZERO_VALUES_IDENTICAL_OR_STD_IS_ZERO",
        interval="CLOSED_INTERVAL_FOR_ATOMS_CONTINUOUS_GAUSSIAN_MASS_FOR_NONZERO",
        probability_prior="FIXED_JEFFREYS_BETA_HALF_HALF_EFFECTIVE_MASS_COUNT",
        jeffreys_alpha=JEFFREYS_ALPHA, probability="(window*mass+0.5)/(window+1)",
        export="LOG_PROBABILITY_MINUS_LOG_ONE_MINUS_PROBABILITY",
        legal_prices="EXACT_NATIVE_ORDINARY_STOCK_TICK_R1_CLOSED_BAND_COUNT",
        max_legal_band_prices=MAX_LEGAL_BAND_PRICES,
        fractal_interactions=dict(zip(FRACTAL_DERIVED_NAMES, (
            "(dfa512-0.5)*execution_atr_band_logit60",
            "(higuchi256-1.5)*execution_zero_gap_logit60",
            "(higuchi512-1.5)*execution_atom_band_logit60"))),
        missing_policy="NOT_READY_NO_IMPUTATION", support_policy="OUT_OF_SUPPORT_NO_CLIPPING",
        interpretation="CAUSAL_STATISTICAL_FEATURES_NOT_FITTED_TARGET_PROBABILITIES")


def _legal_price_count(band: tuple[float, float]) -> int:
    """Count the shared native grid exactly, including both legal boundaries."""
    with localcontext() as context:
        context.prec = 50
        lower, upper = (Decimal(str(value)) for value in band)
        if not lower.is_finite() or not upper.is_finite() or not 0 < lower <= upper:
            raise ValueError("EXECUTION_ATOM_POSITIVE_FINITE_BAND_REQUIRED")
        count, segment_lower = 0, Decimal(0)
        for segment_upper, tick in (*_GRID, (Decimal("Infinity"), Decimal(5))):
            low = max(lower, segment_lower)
            if low <= upper and low < segment_upper:
                first = (low / tick).to_integral_value(rounding=ROUND_CEILING)
                last = (upper / tick).to_integral_value(rounding=ROUND_FLOOR)
                if segment_upper.is_finite():
                    last = min(last, (segment_upper / tick).to_integral_value(rounding=ROUND_CEILING) - 1)
                count += max(0, int(last - first + 1))
            segment_lower = segment_upper
        return count


def _interval_gaussian_mass(lower: float, upper: float, center: float, bandwidth: float) -> float:
    a, b = (lower - center) / bandwidth, (upper - center) / bandwidth
    root = math.sqrt(2.)
    # erfc avoids cancellation when both boundaries are in a positive tail.
    return (.5 * (math.erfc(a / root) - math.erfc(b / root)) if a >= 0
            else .5 * (math.erf(b / root) - math.erf(a / root)))


def _mass(sample: np.ndarray, zero: np.ndarray, lower: float, upper: float) -> tuple[float, dict]:
    nonzero = sample[~zero]
    zero_count, count = int(zero.sum()), len(sample)
    zero_mass = zero_count / count if lower <= 0 <= upper else 0.
    deviation = float(np.std(nonzero, ddof=0)) if len(nonzero) else None
    bandwidth = None
    if not len(nonzero):
        nonzero_mass, mode = 0., "NO_NONZERO_GAPS"
    elif np.all(nonzero == nonzero[0]) or deviation == 0:
        nonzero_mass = float(np.sum((nonzero >= lower) & (nonzero <= upper))) / count
        mode = "DEGENERATE_NONZERO_EMPIRICAL_ATOMS"
    else:
        bandwidth = SILVERMAN_FACTOR * deviation * len(nonzero) ** (-.2)
        if not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("EXECUTION_ATOM_NONFINITE_KERNEL_BANDWIDTH")
        nonzero_mass = math.fsum(_interval_gaussian_mass(lower, upper, float(value), bandwidth)
                                 for value in nonzero) / count
        mode = "NONZERO_GAUSSIAN_KERNEL"
    mass = zero_mass + nonzero_mass
    if not math.isfinite(mass) or not 0 <= mass <= 1:
        raise ValueError("EXECUTION_ATOM_KERNEL_MASS_OUTSIDE_UNIT_INTERVAL")
    return mass, dict(zero_count=zero_count, nonzero_count=len(nonzero), nonzero_std=deviation,
        bandwidth=bandwidth, mode=mode, mass=mass)


def _logit_mass(mass: float, window: int) -> float:
    # Declared prior on an observed effective mass count, not missing-data fill.
    return math.log((window * mass + JEFFREYS_ALPHA) /
                    (window * (1 - mass) + JEFFREYS_ALPHA))


def compute_execution_features(data: pd.DataFrame, *, cutoff: str, base_features: dict,
                               plan: ReferencePlan | None) -> tuple[dict, dict]:
    timestamp(cutoff)
    return _from_selected_bars(select_bars(data, cutoff), base_features=base_features, plan=plan)


def _from_selected_bars(data: pd.DataFrame, *, base_features: dict,
                        plan: ReferencePlan | None) -> tuple[dict, dict]:
    """Trusted native selected-history path; never accepts target labels."""
    if not isinstance(base_features, dict) or plan is not None and not isinstance(plan, ReferencePlan):
        raise ValueError("EXECUTION_ATOM_BASE_AND_REFERENCE_PLAN_REQUIRED")
    values = dict.fromkeys(FEATURE_NAMES)
    audit = dict(profile=EXECUTION_FEATURE_PROFILE, status="NOT_READY", available_close_count=len(data),
        required_close_count=REQUIRED_CLOSE_COUNT, reasons=[],
        cutoff_session=pd.Timestamp(data["Date"].iloc[-1]).date().isoformat() if len(data) else None)
    if len(data) < REQUIRED_CLOSE_COUNT or plan is None:
        audit["reasons"].append("INSUFFICIENT_ATOM_HISTORY" if len(data) < REQUIRED_CLOSE_COUNT else "REFERENCE_PLAN_NOT_READY")
        return values, audit
    close, opened, high, low = (data[name].to_numpy(dtype=float, copy=True)
                               for name in ("Close", "Open", "High", "Low"))
    if any(not np.isfinite(array).all() or (array <= 0).any() for array in (close, opened, high, low)):
        raise ValueError("EXECUTION_ATOM_POSITIVE_FINITE_PRICES_REQUIRED")
    latest, current_atr = finite(base_features.get("close")), finite(base_features.get("atr"))
    if latest != float(close[-1]) or current_atr != plan.atr:
        raise ValueError("EXECUTION_ATOM_BASE_FEATURE_BINDING_MISMATCH")
    previous = np.r_[np.nan, close[:-1]]
    ranges = np.maximum.reduce((high - low, np.abs(high - previous), np.abs(low - previous)))
    ranges[0] = np.nan
    atr = wilder(pd.Series(ranges, index=data.index)).to_numpy()
    if atr[-1] != current_atr:
        raise ValueError("EXECUTION_ATOM_NATIVE_ATR_BINDING_MISMATCH")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        log_gap = np.log(opened[1:]) - np.log(close[:-1])
        atr_gap = (opened[1:] - close[:-1]) / atr[:-1]
    zero = opened[1:] == close[:-1]
    if (not np.isfinite(log_gap[-120:]).all() or not np.isfinite(atr_gap[-120:]).all()
            or (atr[-121:] <= 0).any()):
        audit["reasons"].append("PREVIOUS_NATIVE_ATR_OR_GAP_NOT_READY")
        return values, audit
    log_band = tuple(math.log(bound) - math.log(latest) for bound in plan.band)
    atr_band = tuple((bound - latest) / current_atr for bound in plan.band)
    kernel_audit = {}
    for coordinate, sample, band in (("atom", log_gap, log_band), ("atr", atr_gap, atr_band)):
        kernel_audit[coordinate] = {}
        for window in WINDOWS:
            mass, proof = _mass(sample[-window:], zero[-window:], *band)
            values[f"execution_{coordinate}_band_logit{window}"] = _logit_mass(mass, window)
            kernel_audit[coordinate][str(window)] = proof
    values["execution_zero_gap_logit60"] = _logit_mass(float(np.mean(zero[-60:])), 60)
    legal_count = _legal_price_count(plan.band)
    values["execution_legal_band_log1p_prices"] = math.log(legal_count + 1)
    for name, base_name, center, other in (
        (FRACTAL_DERIVED_NAMES[0], "dfa512", .5, "execution_atr_band_logit60"),
        (FRACTAL_DERIVED_NAMES[1], "higuchi256", 1.5, "execution_zero_gap_logit60"),
        (FRACTAL_DERIVED_NAMES[2], "higuchi512", 1.5, "execution_atom_band_logit60")):
        if base_features.get(base_name) is not None:
            values[name] = (finite(base_features[base_name], base_name) - center) * values[other]
    audit.update(kernel_windows=kernel_audit, legal_band_price_count=legal_count,
        native_atr=dict(window=14, selected_history_count=len(data), last_value=float(atr[-1]),
                        previous_atr_min=float(np.min(atr[-121:-1]))))
    missing = [name for name, value in values.items() if value is None or not math.isfinite(value)]
    if missing:
        audit.update(reasons=["MISSING_ATOM_FEATURE_INPUT"], missing_features=missing)
        return values, audit
    outside = [name for name, bounds in zip(FEATURE_NAMES, FEATURE_BOUNDS)
               if not bounds[0] <= values[name] <= bounds[1]]
    if outside:
        audit.update(status="OOD", reasons=["ATOM_FEATURE_OUTSIDE_DECLARED_SUPPORT"], outside_support=outside)
        return values, audit
    audit.update(status="COMPUTED", used_gap_count=120)
    return values, audit

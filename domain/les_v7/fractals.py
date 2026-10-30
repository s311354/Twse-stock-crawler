"""Bounded deterministic trailing estimators. Estimates are not trading odds."""
from __future__ import annotations

from dataclasses import dataclass
from math import pi
import numpy as np


DFA_WINDOWS = (512, 1024)
HIGUCHI_WINDOWS = (256, 512)
DFA_CHANGE_LAG = 5


def is_active_fractal_feature(name: str) -> bool:
    """Include derived DFA inputs in both artifact and inference validation."""
    return name.startswith(("dfa", "delta_dfa", "higuchi"))


def fractal_history_requirements(close_count: int) -> dict:
    """Required cutoff-valid closes, not calendar days or estimator validity.

    Differencing N+1 log closes produces N returns. The five-session DFA
    change also needs the earlier N-return window, hence N+1+5 closes.
    """
    if type(close_count) is not int or close_count < 0:
        raise ValueError("NONNEGATIVE_CLOSE_COUNT_REQUIRED")
    required = {f"dfa{n}": n + 1 for n in DFA_WINDOWS}
    required.update({f"delta_dfa{n}_{DFA_CHANGE_LAG}": n + 1 + DFA_CHANGE_LAG
                     for n in DFA_WINDOWS})
    required.update({f"higuchi{n}": n for n in HIGUCHI_WINDOWS})
    return {name: dict(available_close_count=close_count, required_close_count=count,
                       missing_close_count=max(0, count - close_count),
                       status="SUFFICIENT_HISTORY" if close_count >= count else "INSUFFICIENT_HISTORY")
            for name, count in required.items()}


@dataclass(frozen=True)
class Estimate:
    value: float | None
    status: str
    window: int
    grid: tuple[int, ...] = ()
    residual: float | None = None
    segments: tuple[int, ...] = ()
    validation_status: str = "NULL_ENVELOPE_NOT_PROVIDED"


def _series(values, minimum):
    x = np.asarray(values, dtype=float)
    return x if x.ndim == 1 and len(x) >= minimum and np.isfinite(x).all() else None


def _variances(values, scales):
    profile = np.cumsum(values - np.mean(values))
    result = []
    for scale in scales:
        count = len(profile) // scale
        positions = np.arange(scale, dtype=float)
        design = np.column_stack((np.ones(scale), positions))
        segments = np.vstack((profile[:count * scale].reshape(count, scale),
                              profile[len(profile) - count * scale:].reshape(count, scale)))
        fitted = np.linalg.lstsq(design, segments.T, rcond=None)[0]
        result.append(np.mean((segments - (design @ fitted).T) ** 2, axis=1))
    return result


def dfa(values, scales=None) -> Estimate:
    x = _series(values, 64)
    n = len(values)
    if x is None:
        return Estimate(None, "INSUFFICIENT_OR_INVALID", n)
    if np.ptp(x) == 0:
        return Estimate(None, "ZERO_FLUCTUATION", n)
    grid = tuple(int(v) for v in (scales if scales is not None else np.unique(
        np.geomspace(8, n // 8, 8).astype(int))))
    if len(grid) < 6 or len(set(grid)) != len(grid) or any(s < 8 or s > n // 8 for s in grid):
        return Estimate(None, "INVALID_SCALES", n, grid)
    fluctuation = np.sqrt([v.mean() for v in _variances(x, grid)])
    if (fluctuation <= 0).any() or not np.isfinite(fluctuation).all():
        return Estimate(None, "ZERO_FLUCTUATION", n, grid)
    a = np.column_stack((np.ones(len(grid)), np.log(grid)))
    beta = np.linalg.lstsq(a, np.log(fluctuation), rcond=None)[0]
    residual = float(np.mean((np.log(fluctuation) - a @ beta) ** 2))
    return Estimate(float(beta[1]), "ESTIMATED", n, grid, residual,
                    tuple(2 * (n // s) for s in grid))


def higuchi(log_prices, grid=tuple(range(1, 17))) -> Estimate:
    x = _series(log_prices, 64)
    n = len(log_prices)
    if x is None:
        return Estimate(None, "INSUFFICIENT_OR_INVALID", n)
    grid = tuple(grid)
    if len(grid) < 3 or len(set(grid)) != len(grid) or any(type(k) is not int or k < 1 or k > n // 4 for k in grid):
        return Estimate(None, "INVALID_GRID", n, grid)
    lengths = []
    for k in grid:
        per_offset = []
        for offset in range(k):
            path = x[offset::k]
            count = len(path) - 1
            per_offset.append((n - 1) / (count * k * k) * np.abs(np.diff(path)).sum())
        lengths.append(np.mean(per_offset))
    if any(v <= 0 for v in lengths):
        return Estimate(None, "ZERO_LENGTH", n, grid)
    a = np.column_stack((np.ones(len(grid)), np.log(1 / np.array(grid))))
    beta = np.linalg.lstsq(a, np.log(lengths), rcond=None)[0]
    return Estimate(float(beta[1]), "ESTIMATED", n, grid,
                    float(np.mean((np.log(lengths) - a @ beta) ** 2)))


def mfdfa(values, q_grid=tuple(np.arange(-2., 2.01, .5))) -> dict:
    """Shadow only: no zero-variance epsilon and no financial interpretation."""
    x = _series(values, 2048)
    if x is None:
        return dict(status="INSUFFICIENT_OR_INVALID", spectrum_width=None)
    if np.ptp(x) == 0:
        return dict(status="ZERO_FLUCTUATION", spectrum_width=None)
    scales = np.unique(np.geomspace(8, len(x) // 8, 12).astype(int))
    variances = _variances(x, scales)
    if any((v <= 0).any() for v in variances):
        return dict(status="ZERO_FLUCTUATION", spectrum_width=None)
    slopes = []
    for q in q_grid:
        logs = np.array([.5 * np.mean(np.log(v)) if q == 0 else
                         np.log(np.mean(v ** (q / 2))) / q for v in variances])
        slopes.append(float(np.polyfit(np.log(scales), logs, 1)[0]))
    tau = np.array(q_grid) * slopes - 1
    alpha = np.gradient(tau, q_grid)
    return dict(status="SHADOW_ESTIMATED", q=list(q_grid), h=slopes, tau=tau.tolist(),
                spectrum_width=float(np.ptp(alpha)), validation_status="NULL_ENVELOPE_NOT_PROVIDED")


def lppl_shadow(log_prices, horizon=20) -> dict:
    """Fixed bounded grid, all fits retained, never substitute t_c for timing odds."""
    if type(horizon) is not int or horizon != 20: raise ValueError("FIXED_LPPL_H20_GRID_REQUIRED")
    x = _series(log_prices, 120)
    if x is None:
        return dict(status="INSUFFICIENT_OR_INVALID", fits=[], tc_offset_median=None)
    fits = []
    for n in (120, 250, 500):
        if len(x) < n:
            continue
        y, t = x[-n:], np.arange(n)
        for m in (.1, .3, .5, .7, .9):
            for omega in (4., 8., 12., 16., 20.):
                for offset in range(1, min(horizon, int(.375 * (n - 1))) + 1):
                    tau = n - 1 + offset - t
                    power = tau ** m
                    design = np.column_stack((np.ones(n), power, power * np.cos(omega * np.log(tau)),
                                              power * np.sin(omega * np.log(tau))))
                    beta = np.linalg.lstsq(design, y, rcond=None)[0]
                    residual = float(np.mean((y - design @ beta) ** 2))
                    simple = np.column_stack((np.ones(n), power))
                    trend = np.linalg.lstsq(simple, y, rcond=None)[0]
                    base = float(np.mean((y - simple @ trend) ** 2))
                    oscillations = omega / (2 * pi) * np.log((n - 1 + offset) / offset)
                    eligible = bool(beta[1] > 0 and trend[1] > 0 and y[-1] < y[0]
                                    and oscillations >= 1.5 and base > 0 and residual < base)
                    fits.append(dict(window=n, m=m, omega=omega, tc_offset=offset,
                                     coefficients=beta.tolist(), residual=residual, eligible=eligible,
                                     residual_improvement=1 - residual / base if base > 0 else None,
                                     boundary_hit=m in {.1, .9} or omega in {4., 20.}))
    eligible = [f for f in fits if f["eligible"]]
    offsets = [f["tc_offset"] for f in eligible]
    return dict(status="SHADOW_ESTIMATED" if eligible else "LPPL_NOT_APPLICABLE", fits=fits,
                tc_offset_median=float(np.median(offsets)) if offsets else None,
                tc_offset_iqr=float(np.ptp(np.quantile(offsets, [.25, .75]))) if offsets else None,
                eligible_fit_fraction=len(eligible) / len(fits) if fits else 0.,
                validation_status="ABLATION_NOT_PROVIDED")

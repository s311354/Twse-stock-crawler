"""Offline hierarchical convex fitting and immutable, unvalidated weights.

Only NumPy is required. A simplex-constrained convex gap certifies numerical
convergence, NOT predictive skill, evidence authenticity or trading authority.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from hashlib import sha256
import json
from math import fsum, isclose
from pathlib import Path

import numpy as np
import pandas as pd

from domain.strategies.low_entry_v6_hybrid import FACTOR_BUDGETS, model_spec
from domain.strategies.low_entry_v6_outcomes import number

POLICY = "context-aware-convex-score/v1"
SCHEMA = "les-v6-convex-weights/v1"
ALGORITHM = "projected-gradient-simplex-huber/v1"
FACTORS = tuple(name for name, _ in FACTOR_BUDGETS)
CONTEXTS = ("REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION")
WEIGHT_KEYS = (*CONTEXTS, "GLOBAL")
PRIOR = tuple(budget / 100 for _, budget in FACTOR_BUDGETS)


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@cache
def implementation_signature() -> str:
    """Pin the scoring implementation once per process; paths are portable.

    Explicit labels alone cannot detect a changed formula under the same release
    name. Restart after editing runtime source; in-process hot reload is unsupported.
    """
    domain = Path(__file__).resolve().parents[1]
    names = ["features.py", "industry_benchmark.py", "financial_taxonomy.py"] + [f"strategies/{name}.py" for name in (
        "low_entry_score_v6", "low_entry_v6_features", "low_entry_v6_leadership",
        "low_entry_v6_score_evidence", "low_entry_v6_hybrid", "low_entry_v6_execution_score",
        "low_entry_v6_signals", "low_entry_v6_decision", "low_entry_v6_quality", "low_entry_v6_quality_evidence",
        "low_entry_v6_ttm", "low_entry_v6_filings", "low_entry_v6_reported_period",
        "low_entry_v6_payoff", "low_entry_v6_weights")]
    return digest(dict(sources={name: sha256((domain/name).read_bytes()).hexdigest() for name in names},
                       numpy=np.__version__, pandas=pd.__version__))


def timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("WEIGHTS_TIMESTAMP_REQUIRES_TIMEZONE")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("WEIGHTS_TIMESTAMP_REQUIRES_TIMEZONE")
    return parsed.astimezone(timezone.utc)


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto nonnegative rows summing to one."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all():
        raise ValueError("INVALID_SIMPLEX_INPUT")
    ordered = np.sort(a, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.
    counts = np.sum(ordered - cumulative / np.arange(1, 7) > 0, axis=1)
    threshold = cumulative[np.arange(len(a)), counts - 1] / counts
    return np.maximum(a - threshold[:, None], 0.)


def objective_gradient(weights: np.ndarray, x: np.ndarray, y: np.ndarray,
                       contexts: np.ndarray, alpha: np.ndarray, *, delta: float,
                       pooling: float, prior_strength: float, prior=None) -> tuple[float, np.ndarray]:
    residual = np.sum(x * weights[contexts], axis=1) - y
    absolute = np.abs(residual)
    loss = np.where(absolute <= delta, .5 * residual ** 2,
                    delta * (absolute - .5 * delta))
    gradient = np.zeros((4, 6))
    np.add.at(gradient, contexts, (alpha * np.clip(residual, -delta, delta))[:, None] * x)
    differences = weights[:3] - weights[3]
    deviation = weights[3] - np.asarray(PRIOR if prior is None else prior)
    gradient[:3] += 2 * pooling * differences
    gradient[3] += -2 * pooling * differences.sum(axis=0) + 2 * prior_strength * deviation
    value = np.dot(alpha, loss) + pooling * np.sum(differences ** 2) + prior_strength * np.dot(deviation, deviation)
    return float(value), gradient


def fit_convex_weights(x: object, y: object, contexts: list[str], *, alpha: object = None,
                       delta: float = .1, pooling: float = 1., prior_strength: float = 1.,
                       tolerance: float = 1e-8, max_iterations: int = 20000, prior=None) -> dict:
    """Minimize the SUM (not mean) weighted Huber + pooling + prior objective.

    All hyperparameters must be fixed before evaluating the holdout. No outcome
    generation, hyperparameter search, randomness or dependence on input order.
    """
    parameters = [number(v) for v in (delta, pooling, prior_strength, tolerance)]
    if (any(v is None for v in parameters) or delta <= 0 or pooling < 0 or prior_strength < 0
            or not 0 < tolerance <= 1e-3 or type(max_iterations) is not int or not 1 <= max_iterations <= 100000):
        raise ValueError("INVALID_CONVEX_PARAMETERS")
    anchor = np.asarray(PRIOR if prior is None else prior, dtype=float)
    if (anchor.shape != (6,) or not np.isfinite(anchor).all() or np.any(anchor < 0)
            or abs(float(anchor.sum()) - 1.) > 1e-9):
        raise ValueError("INVALID_CONVEX_PRIOR")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("INVALID_CONVEX_OBSERVATIONS")
    a = np.ones(len(y)) if alpha is None else np.asarray(alpha, dtype=float)
    if (x.shape != (len(y), 6) or y.ndim != 1 or a.shape != y.shape or not len(y)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(a).all()
            or np.any((x < 0) | (x > 1)) or np.any((y < 0) | (y > 1))
            or np.any(a < 0) or not np.any(a > 0) or not np.isfinite(a.sum())
            or len(contexts) != len(y) or any(c not in CONTEXTS for c in contexts)):
        raise ValueError("INVALID_CONVEX_OBSERVATIONS")
    if number(8*pooling + 2*prior_strength + 6*float(a.sum())) is None:
        raise ValueError("CONVEX_NUMERIC_OVERFLOW")
    indices = np.array([CONTEXTS.index(c) for c in contexts])
    # Canonical ordering makes floating-point reductions repeatable after a shuffle.
    order = np.lexsort(tuple(x[:, i] for i in reversed(range(6))) + (a, y, indices))
    x, y, a, indices = x[order], y[order], a[order], indices[order]
    hessian_bound = np.zeros((24, 24))
    for c in range(3):
        selected = indices == c
        hessian_bound[c*6:(c+1)*6, c*6:(c+1)*6] = x[selected].T @ (a[selected, None] * x[selected])
        for j in range(6):
            hessian_bound[c*6+j, c*6+j] += 2 * pooling
            hessian_bound[18+j, 18+j] += 2 * pooling
            hessian_bound[c*6+j, 18+j] = hessian_bound[18+j, c*6+j] = -2 * pooling
    hessian_bound[18:, 18:] += 2 * prior_strength * np.eye(6)
    lipschitz = float(np.linalg.eigvalsh(hessian_bound)[-1])
    if not np.isfinite(lipschitz):
        raise ValueError("CONVEX_NUMERIC_OVERFLOW")
    weights = np.tile(anchor, (4, 1))
    initial = objective_gradient(weights, x, y, indices, a, delta=delta,
                                 pooling=pooling, prior_strength=prior_strength, prior=anchor)[0]
    for iteration in range(max_iterations + 1):
        value, gradient = objective_gradient(weights, x, y, indices, a, delta=delta,
                                             pooling=pooling, prior_strength=prior_strength, prior=anchor)
        # For a convex differentiable objective on four simplexes, this bounds
        # f(w)-f(w*): max feasible descent of the first-order approximation.
        gap = max(0., float(np.sum(weights * gradient) - np.sum(gradient.min(axis=1))))
        if gap <= tolerance or iteration == max_iterations:
            break
        weights = project_simplex(weights - gradient / max(lipschitz, 1e-12))
    return dict(status="CONVERGED" if gap <= tolerance else "NOT_CONVERGED",
        weights={name: weights[i].tolist() for i, name in enumerate(WEIGHT_KEYS)},
        objective=value, initial_objective=initial, convex_gap=gap, iterations=iteration,
        algorithm=ALGORITHM, parameters=dict(delta=delta, lambda_pool=pooling,
            lambda_prior=prior_strength, tolerance=tolerance, max_iterations=max_iterations))


def normalized_factors(scored: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Missing slots remain zero contribution; never divide by available weight."""
    groups = scored["factor_budgets"]
    fractions, coverage = [], []
    for name, budget in FACTOR_BUDGETS:
        group = groups[name]
        earned, available = number(group["earned"]), number(group["available_weight"])
        if (group["budget"] != budget or earned is None or available is None
                or not 0 <= earned <= available <= budget):
            raise ValueError("INVALID_NORMALIZED_FACTOR_LEDGER")
        fractions.append(earned / budget)
        coverage.append(available / budget)
    return tuple(fractions), tuple(coverage)


@dataclass(frozen=True)
class FrozenWeights:
    """A canonical immutable record; record() returns a detached mutable copy."""
    canonical_json: str
    vectors: tuple[tuple[float, ...], ...]
    sha256: str
    available_at: datetime

    @classmethod
    def from_record(cls, record: dict) -> FrozenWeights:
        if not isinstance(record, dict):
            raise ValueError("WEIGHTS_REQUIRE_OBJECT")
        data = deepcopy(record)
        checksum = data.pop("sha256", None)
        if checksum != digest(data):
            raise ValueError("WEIGHTS_INTEGRITY_MISMATCH")
        if (data.get("schema") != SCHEMA or data.get("factors") != list(FACTORS)
                or data.get("status") != "ESTIMATED_RESEARCH_ONLY" or data.get("research_only") is not True
                or data.get("executable") is not False or data.get("feature_model_sha256") != digest(model_spec())
                or data.get("implementation_sha256") != implementation_signature()
                or data.get("fit", {}).get("status") != "CONVERGED"
                or data.get("fit", {}).get("algorithm") != ALGORITHM
                or data.get("fit", {}).get("parameters", {}).get("tolerance") is None):
            raise ValueError("INCOMPATIBLE_FROZEN_WEIGHTS")
        gap = number(data["fit"].get("convex_gap"))
        tolerance = number(data["fit"]["parameters"]["tolerance"])
        if gap is None or tolerance is None or not 0 <= gap <= tolerance <= 1e-3:
            raise ValueError("UNCONVERGED_FROZEN_WEIGHTS")
        if set(data.get("weights", {})) != set(WEIGHT_KEYS):
            raise ValueError("INCOMPLETE_WEIGHT_CONTEXTS")
        vectors = tuple(tuple(number(v) for v in data["weights"][name]) for name in WEIGHT_KEYS)
        if any(len(v) != 6 or any(x is None or x < 0 for x in v)
               or not isclose(fsum(v), 1., rel_tol=0., abs_tol=1e-12) for v in vectors):
            raise ValueError("WEIGHTS_MUST_BE_ON_SIMPLEX")
        available = timestamp(data["available_at"])
        if not timestamp(data["training_boundary"]) <= timestamp(data["data_as_of"]) <= available:
            raise ValueError("INVALID_WEIGHT_TIME_ORDER")
        return cls(json.dumps(record, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False), vectors, checksum, available)

    def record(self) -> dict:
        return json.loads(self.canonical_json)

    def require_compatible(self, config: object, as_of: str) -> None:
        if digest(vars(config)) != self.record()["decision_config_sha256"]:
            raise ValueError("WEIGHTS_CONFIG_MISMATCH")
        if timestamp(as_of) < self.available_at:
            raise ValueError("WEIGHTS_UNAVAILABLE_AT_CUTOFF")

    def for_context(self, context: str) -> tuple[str, tuple[float, ...]]:
        name = context if context in CONTEXTS else "GLOBAL"
        return name, self.vectors[WEIGHT_KEYS.index(name)]


def reweight_factors(scored: dict, context: str, weights: FrozenWeights) -> tuple[dict, dict]:
    vector, coverage = normalized_factors(scored)
    name, w = weights.for_context(context)
    result = deepcopy(scored)
    evidence = result["score_evidence"]
    original = deepcopy(evidence)
    evidence.update(policy_version=POLICY, normalization_evidence=original)
    for condition in evidence["conditions"]:
        i = FACTORS.index(condition["group"])
        scale = 100 * w[i] / FACTOR_BUDGETS[i][1]
        for key in ("weight", "earned_points", "available_weight"):
            condition[key] *= scale
    for i, factor in enumerate(FACTORS):
        group = result["factor_budgets"][factor]
        group.update(budget=100*w[i], earned=100*w[i]*vector[i], available_weight=100*w[i]*coverage[i])
        evidence["components"][factor] = dict(total_weight=group["budget"],
            earned_points=group["earned"], available_weight=group["available_weight"])
    available = fsum(g["available_weight"] for g in result["factor_budgets"].values())
    raw = 100 * fsum(a*b for a, b in zip(w, vector, strict=True)) if available else None
    evidence.update(formal_score=raw, earned_points=raw, available_weight=available,
        coverage_percent=available, status="COMPLETE" if isclose(available, 100., abs_tol=1e-10)
        else "PARTIAL" if available else "UNAVAILABLE")
    result["execution_score"].update(effective_budget=100*w[-1], effective_earned=100*w[-1]*vector[-1])
    record = dict(policy_version=POLICY, artifact_sha256=weights.sha256, applied_context=name,
        factors=list(FACTORS), normalized_vector=list(vector), factor_coverage=list(coverage),
        applied_weights=list(w), raw_score=raw, feature_model_sha256=digest(model_spec()),
        execution_policy="NATIVE_GATES", performance_validated=False, artifact=weights.record())
    return result, record

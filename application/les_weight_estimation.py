"""Bind frozen Hybrid signals to purged, closed, cost-aware outcome evidence.

Does not create fills or certify user-supplied execution/market evidence. A fit
is a non-executable research artifact, never a Kelly probability calibration.
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta, timezone
from math import fsum, isclose

import numpy as np

from domain.strategies.low_entry_v6_hybrid import POLICY as HYBRID_POLICY, PARAMETER_VERSION, model_spec
from domain.strategies.low_entry_v6_outcomes import MIN_TRAINING, MIN_VALIDATION, outcome_audit, number
from domain.strategies.low_entry_v6_report import entry_gate_checks
from domain.strategies.low_entry_v6_signals import LowEntryV6Config
from domain.strategies.low_entry_v6_weights import (
    FACTORS, PRIOR, SCHEMA, FrozenWeights, digest,
    fit_convex_weights, implementation_signature, normalized_factors, timestamp,
)

EXECUTION_FIELDS = ("next_session", "transaction_costs", "spread_slippage", "liquidity",
                    "initial_stop", "price_limits", "exit_protocol")


def outcome_in_r(row: dict, *, lower: float = 1., upper: float = 2.) -> tuple[float, float]:
    """Net CASH PnL / (shares * per-share initial risk); costs counted once.

    Entry notional is actual executable fill value and includes fill slippage;
    commissions/tax are the separate recorded cash costs, not estimated rates.
    """
    if any(number(v) is None or v <= 0 for v in (lower, upper)) or number(lower+upper) is None:
        raise ValueError("INVALID_UTILITY_CLIP")
    quantity = row.get("quantity")
    fields = ("entry_notional", "exit_notional", "entry_cost", "exit_cost", "net_cash_dividends", "initial_stop")
    values = [number(row.get(k)) for k in fields]
    if type(quantity) is not int or quantity <= 0 or any(v is None or v < 0 for v in values):
        raise ValueError("INVALID_INITIAL_RISK_OR_CASH_FLOWS")
    entry, exit_value, buy_cost, sell_cost, dividends, stop = values
    risk = entry / quantity - stop
    if stop <= 0 or risk <= 0 or exit_value < sell_cost:
        raise ValueError("INVALID_INITIAL_RISK_OR_CASH_FLOWS")
    pnl = fsum((exit_value, dividends, -entry, -buy_cost, -sell_cost))
    result = number(pnl / (quantity * risk))
    if result is None:
        raise ValueError("NONFINITE_OUTCOME_IN_R")
    utility = min(upper, max(-lower, result))
    return result, (utility + lower) / (lower + upper)


def _bind(raw: dict, parsed: dict, versions: dict, reports: dict, *, lower: float, upper: float) -> dict:
    report = reports.get(raw["research_result_sha256"])
    if not isinstance(report, dict) or digest(report) != raw["research_result_sha256"]:
        raise ValueError("SIGNAL_REPORT_HASH_MISMATCH")
    decision = report.get("decision", {})
    if not isinstance(decision, dict) or not isinstance(report.get("market"), dict):
        raise ValueError("OUTCOME_SIGNAL_IDENTITY_MISMATCH")
    if (report.get("symbol") != raw["symbol"] or report.get("decision_time") is None
            or timestamp(report["decision_time"]) != timestamp(raw["signal_at"])
            or report.get("market", {}).get("exchange") != raw["market"]
            or report.get("market", {}).get("regime") != raw["market_regime"]
            or report.get("model_version") != versions["model"]
            or report.get("parameter_version") != versions["parameters"]
            or decision.get("policy_version") != HYBRID_POLICY
            or decision.get("setup") != raw["setup"] or decision.get("signal") != raw["entry_decision"]
            or decision.get("research_only") is True
            or not isclose(decision.get("formal_score", -1), raw["formal_score"], rel_tol=0., abs_tol=1e-10)):
        raise ValueError("OUTCOME_SIGNAL_IDENTITY_MISMATCH")
    if any(value != "PASS" for value in entry_gate_checks(report).values()):
        raise ValueError("SIGNAL_REPORT_NATIVE_VALIDATION_FAILED")
    config = LowEntryV6Config(**report["parameters"])
    if config.decision_policy_version != 3:
        raise ValueError("CONVEX_FIT_REQUIRES_HYBRID_FACTORS")
    vector, coverage = normalized_factors(dict(factor_budgets=decision["budget_groups"]))
    if any(v != 1 for v in coverage):
        raise ValueError("INCOMPLETE_TRAINING_FACTOR_COVERAGE")
    if number(raw.get("initial_stop")) != number(report.get("entry", {}).get("stop_price")):
        raise ValueError("INITIAL_STOP_NOT_BOUND_TO_SIGNAL_PLAN")
    evidence = raw.get("execution_evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("INCOMPLETE_EXECUTION_EVIDENCE")
    for field in EXECUTION_FIELDS:
        item = evidence.get(field, {})
        if not isinstance(item, dict):
            raise ValueError("INCOMPLETE_EXECUTION_EVIDENCE:" + field)
        source_hash = item.get("source_sha256", "")
        if (item.get("status") != "ASSESSED" or not item.get("source_id") or not item.get("detail")
                or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash)
                or timestamp(item.get("available_at")) > timestamp(parsed["available_at"])):
            raise ValueError("INCOMPLETE_EXECUTION_EVIDENCE:" + field)
    local = timezone(timedelta(hours=8))
    entry_date = timestamp(raw["entry_at"]).astimezone(local).date().isoformat()
    if (evidence["next_session"].get("trade_date") != entry_date
            or entry_date <= timestamp(raw["signal_at"]).astimezone(local).date().isoformat()
            or evidence["exit_protocol"].get("policy_version") != versions["execution_policy"]):
        raise ValueError("NEXT_SESSION_OR_EXIT_PROTOCOL_MISMATCH")
    alpha = number(raw.get("observation_weight", 1.))
    if alpha is None or alpha < 0:
        raise ValueError("INVALID_OBSERVATION_WEIGHT")
    risk_units, outcome = outcome_in_r(raw, lower=lower, upper=upper)
    return dict(episode_id=parsed["episode_id"], context=parsed["setup"], vector=list(vector),
        outcome_in_r=risk_units, normalized_outcome=outcome, alpha=alpha,
        signal_sha256=raw["research_result_sha256"], outcome_sha256=parsed["outcome_sha256"],
        partition=parsed["partition"], decision_config_sha256=digest(vars(config)))


def estimate_weights(payload: dict) -> dict:
    """Reuse native cutoff, purge, unit, duplicate and overlapping-episode checks.

    Fit only training observations. Holdout never chooses weights/parameters.
    Future or boundary-crossing labels are excluded before signal binding.
    """
    audit = outcome_audit(payload)
    versions = payload["versions"]
    if versions["model"] != "6.0.0" or versions["parameters"] != PARAMETER_VERSION:
        raise ValueError("CONVEX_FIT_REQUIRES_CURRENT_HYBRID_VERSION")
    settings = payload.get("weight_estimation", {})
    if not isinstance(settings, dict) or set(settings) - {
            "frozen_at", "lower_clip", "upper_clip", "delta", "lambda_pool", "lambda_prior",
            "tolerance", "max_iterations"}:
        raise ValueError("INVALID_WEIGHT_ESTIMATION_SETTINGS")
    frozen = timestamp(settings.get("frozen_at"))
    if frozen < timestamp(payload["as_of"]):
        raise ValueError("WEIGHTS_CANNOT_BE_BACKDATED")
    lower, upper = settings.get("lower_clip", 1.), settings.get("upper_clip", 2.)
    if any(number(v) is None or v <= 0 for v in (lower, upper)) or number(lower+upper) is None:
        raise ValueError("INVALID_UTILITY_CLIP")
    reports = payload.get("signal_reports")
    if not isinstance(reports, dict):
        raise ValueError("EXPLICIT_FROZEN_SIGNAL_REPORTS_REQUIRED")
    # A future excluded episode must not shadow a selected episode with the
    # same ID. Bind to the complete selected raw record, not a mutable ID map.
    raw_by_hash = {digest(row): row for row in payload["outcomes"]}
    bound, excluded = [], list(audit["excluded"])
    for row in audit["rows"]:
        try:
            bound.append(_bind(raw_by_hash[row["outcome_sha256"]], row, versions, reports, lower=lower, upper=upper))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            excluded.append(dict(episode_id=row["episode_id"], reason=str(error)))
    config_hashes = {r["decision_config_sha256"] for r in bound}
    if len(config_hashes) > 1:
        raise ValueError("MIXED_SIGNAL_DECISION_PARAMETERS")
    training = [r for r in bound if r["partition"] == "training"]
    holdout = [r for r in bound if r["partition"] == "validation"]
    count = sum(r["alpha"] > 0 for r in training)
    result = dict(schema="les-v6-weight-estimation/v1", status="UNKNOWN", artifact=None,
        research_only=True, executable=False, performance_validated=False, kelly_calibration_eligible=False,
        source_posture="DECLARED_OUTCOMES_NOT_INDEPENDENTLY_VERIFIED", source_kind=payload["source_kind"],
        training_count=count, training_counts=dict(Counter(r["context"] for r in training if r["alpha"] > 0)),
        holdout_count=len(holdout), exclusions=excluded, exclusion_counts=dict(Counter(r["reason"] for r in excluded)),
        minimum_training=MIN_TRAINING, minimum_holdout=MIN_VALIDATION,
        holdout_status="NOT_ASSESSABLE", holdout_huber=None, holdout_prior_huber=None,
        reasons=[], training_rows=training)
    if count < MIN_TRAINING:
        result["reasons"] = ["INSUFFICIENT_TRAINING_OUTCOMES"]
        return result
    fit = fit_convex_weights([r["vector"] for r in training], [r["normalized_outcome"] for r in training],
        [r["context"] for r in training], alpha=[r["alpha"] for r in training],
        delta=settings.get("delta", .1), pooling=settings.get("lambda_pool", 1.),
        prior_strength=settings.get("lambda_prior", 1.), tolerance=settings.get("tolerance", 1e-8),
        max_iterations=settings.get("max_iterations", 20000))
    if fit["status"] != "CONVERGED":
        result.update(fit=fit, reasons=["OPTIMIZER_NOT_CONVERGED"])
        return result
    artifact = dict(schema=SCHEMA, status="ESTIMATED_RESEARCH_ONLY", research_only=True, executable=False,
        factors=list(FACTORS), weights=fit["weights"], prior=list(PRIOR), fit={k: v for k, v in fit.items() if k != "weights"},
        feature_model_sha256=digest(model_spec()), decision_config_sha256=next(iter(config_hashes)),
        implementation_sha256=implementation_signature(),
        model_versions=versions, source_kind=payload["source_kind"], source_posture=result["source_posture"],
        data_as_of=timestamp(payload["as_of"]).isoformat(), available_at=frozen.isoformat(),
        training_boundary=timestamp(payload["validation_start"]).isoformat(),
        training_sha256=digest(training), training_count=count, training_counts=result["training_counts"],
        outcome_policy=dict(lower_clip=lower, upper_clip=upper,
            units="NET_CASH_PNL_DIVIDED_BY_SHARES_TIMES_PER_SHARE_INITIAL_RISK", definition=payload["definition"]))
    artifact["sha256"] = digest(artifact)
    FrozenWeights.from_record(artifact)
    result.update(status="ESTIMATED_RESEARCH_ONLY", artifact=artifact)
    if len(holdout) >= MIN_VALIDATION:
        def losses(prior: bool) -> float:
            residuals = [r["normalized_outcome"] - np.dot(PRIOR if prior else fit["weights"][r["context"]], r["vector"]) for r in holdout]
            delta = fit["parameters"]["delta"]
            return fsum(.5*r*r if abs(r) <= delta else delta*(abs(r)-.5*delta) for r in residuals) / len(residuals)
        result.update(holdout_status="DIAGNOSTICS_AVAILABLE_NOT_VALIDATED", holdout_huber=losses(False), holdout_prior_huber=losses(True))
    return result

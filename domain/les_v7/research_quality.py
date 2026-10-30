"""Paired forecast and H20 reference-path diagnostics, never account efficacy.

Callers must first verify prepared rows, model provenance and execution lookups
through the native archive ports. Hashes here bind an analysis; they do not
certify its source. No fitting, threshold search, fill invention or account
simulation is performed. A lower fill Brier score is not a profit claim.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation, localcontext
from math import fsum, isfinite, log
from zoneinfo import ZoneInfo

import numpy as np

from domain.les_v7.contracts import finite, identity, timestamp

SCHEMA = "les-v7-reference-quality-diagnostics/v1"
COMPARISON_SCHEMA = "les-v7-paired-fill-quality-comparison/v1"
MAX_ROWS = 100000
INFERENCE = dict(block_sessions=21, bootstrap_replicates=1999, seed=20261018,
                 confidence=.95, method="PAIRED_MOVING_OFFICIAL_SESSION_BLOCK_BOOTSTRAP")


def _number(value, name):
    if isinstance(value, bool):
        raise ValueError("QUALITY_FINITE_NUMBER_REQUIRED:" + name)
    return finite(value, name)


def _inputs(rows, predictions):
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("QUALITY_BOUNDED_PREPARED_ROWS_REQUIRED")
    if not isinstance(predictions, dict):
        raise ValueError("QUALITY_ROW_KEYED_PREDICTIONS_REQUIRED")
    result, seen = [], set()
    for row in rows:
        key = row.get("row_id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("QUALITY_UNIQUE_ROW_IDS_REQUIRED")
        seen.add(key)
        cutoff, available = timestamp(row["cutoff"]), timestamp(row["label_available_at"])
        if not cutoff < available:
            raise ValueError("QUALITY_LABEL_MUST_FOLLOW_EVENT")
        labels = row["labels"]
        fill = labels.get("fill")
        if type(fill) is not int or fill not in (0, 1):
            raise ValueError("QUALITY_OBSERVED_FILL_REQUIRED")
        rebound, joint = labels.get("rebound"), labels.get("joint")
        if fill:
            if (type(rebound) is not int or type(joint) is not int
                    or rebound not in (0, 1) or joint not in (0, 1) or joint > rebound):
                raise ValueError("QUALITY_CONDITIONAL_LABELS_INVALID")
        elif rebound is not None or joint is not None:
            raise ValueError("QUALITY_NO_FILL_CONDITIONAL_LABELS_MUST_BE_NULL")
        raw = predictions.get(key)
        if not isinstance(raw, dict):
            raise ValueError("QUALITY_PREDICTION_COVERAGE_MISMATCH")
        probabilities = [_number(raw.get(name), name) for name in ("p_fill", "p_rebound", "p_entry")]
        if any(not 0 <= p <= 1 for p in probabilities) or probabilities[2] > probabilities[1]:
            raise ValueError("QUALITY_PROBABILITIES_OR_PROJECTION_INVALID")
        close, atr, bottom, weight = (_number(v, name) for v, name in (
            (row["features"].get("close"), "close"), (row["features"].get("atr"), "atr"),
            (row.get("bottom"), "bottom"), (row.get("observation_weight", 1.), "weight")))
        if min(close, atr, bottom, weight) <= 0:
            raise ValueError("QUALITY_POSITIVE_PRICE_ATR_WEIGHT_REQUIRED")
        if bottom > close:
            raise ValueError("QUALITY_BOTTOM_ABOVE_CUTOFF_CLOSE")
        result.append(dict(row=row, key=key, date=cutoff.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat(),
            cutoff=cutoff, available=available, fill=fill, rebound=rebound, joint=joint,
            p_fill=probabilities[0], p_rebound=probabilities[1], p_entry=probabilities[2],
            close=close, atr=atr, bottom=bottom, weight=weight))
    if set(predictions) != seen:
        raise ValueError("QUALITY_PREDICTION_COVERAGE_MISMATCH")
    # Normalize before products to avoid overflowing valid finite weights.
    largest = max(item["weight"] for item in result)
    scaled = [item["weight"] / largest for item in result]
    total = fsum(scaled)
    for item, weight in zip(result, scaled):
        item["weight"] = weight / total
        if item["weight"] <= 0:
            raise ValueError("QUALITY_WEIGHT_DYNAMIC_RANGE_UNREPRESENTABLE")
    return result


def _mean(items, function):
    total = fsum(item["weight"] for item in items)
    return fsum(item["weight"] * function(item) for item in items) / total if total else None


def _auc(items):
    """Weighted pair concordance, giving half credit to exact probability ties."""
    ordered = sorted(items, key=lambda item: item["p_fill"])
    positive = fsum(item["weight"] for item in items if item["fill"])
    negative = fsum(item["weight"] for item in items if not item["fill"])
    if not positive or not negative:
        return None
    numerator, below, start = 0., 0., 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end]["p_fill"] == ordered[start]["p_fill"]:
            end += 1
        group = ordered[start:end]
        pos = fsum(item["weight"] for item in group if item["fill"])
        neg = fsum(item["weight"] for item in group if not item["fill"])
        numerator += pos * (below + .5 * neg)
        below += neg
        start = end
    return min(1., max(0., numerator / (positive * negative)))


def _forecast_metrics(items):
    filled = [item for item in items if item["fill"]]
    bins = []
    for index in range(10):
        group = [item for item in items if min(9, int(item["p_fill"] * 10)) == index]
        bins.append(dict(lower=index / 10, upper=(index + 1) / 10,
            upper_inclusive=index == 9, rows=len(group), weight_fraction=fsum(item["weight"] for item in group),
            mean_prediction=_mean(group, lambda item: item["p_fill"]),
            observed_fill_fraction=_mean(group, lambda item: item["fill"]),
            mean_squared_error=_mean(group, lambda item: (item["p_fill"] - item["fill"]) ** 2)))
    brier = _mean(items, lambda item: (item["p_fill"] - item["fill"]) ** 2)
    epsilon = 1e-15
    return dict(fill_brier=brier,
        fill_nll=_mean(items, lambda item: -log(max(epsilon, item["p_fill"] if item["fill"] else 1 - item["p_fill"]))),
        weighted_fill_auc=_auc(items), observed_fill_fraction=_mean(items, lambda item: item["fill"]),
        mean_predicted_fill=_mean(items, lambda item: item["p_fill"]),
        constant_half_probability_brier=.25, brier_skill_vs_constant_half=1 - brier / .25,
        conditional_rebound_brier=_mean(filled, lambda item: (item["p_rebound"] - item["rebound"]) ** 2),
        conditional_near_and_rebound_brier=_mean(filled, lambda item: (item["p_entry"] - item["joint"]) ** 2),
        unconditional_near_and_rebound_brier=_mean(items, lambda item: (item["p_fill"] * item["p_entry"] - (item["joint"] if item["fill"] else 0)) ** 2),
        reliability_bins=bins, reliability_scope="DESCRIPTIVE_FIXED_DECILES_NOT_CALIBRATION_FIT_OR_CONFIDENCE_INTERVAL")


def _execution(item, lookup):
    execution = lookup.get(item["key"])
    if not isinstance(execution, dict):
        return None, "MISSING_VERIFIED_EXECUTION"
    if execution.get("filled") is not True:
        raise ValueError("QUALITY_EXECUTION_LABEL_MISMATCH")
    if (execution.get("path_order_verified") is not True
            or not isinstance(execution.get("path_evidence_ids"), list) or not execution["path_evidence_ids"]
            or any(not isinstance(value, str) or not value for value in execution["path_evidence_ids"])
            or any(not isinstance(execution.get(key), str) or not execution[key]
                   for key in ("cost_policy_version", "cost_policy_id"))):
        return None, "INCOMPLETE_PATH_OR_COST_EVIDENCE"
    arithmetic = execution.get("cost_arithmetic", {})
    try:
        entry, stop, atr = (Decimal(str(execution[key])) for key in ("entry", "initial_stop", "cutoff_atr"))
        pnl, risk, debit, proceeds = (Decimal(str(arithmetic[key])) for key in
            ("realized_net_pnl", "native_gross_reference_risk", "buy_debit", "sell_proceeds"))
        shares = execution["shares"]
        numbers = (entry, stop, atr, pnl, risk, debit, proceeds)
        with localcontext() as context:
            context.prec = 50  # Same declared decimal arithmetic as reference_path.
            if (not all(value.is_finite() and isfinite(float(value)) for value in numbers)
                    or type(shares) is not int or shares <= 0
                    or not 0 < stop < entry or atr <= 0 or risk != shares * (entry - stop)
                    or pnl != proceeds - debit or pnl != Decimal(str(execution["realized_net_pnl"]))
                    or atr != Decimal(str(item["atr"]))):
                return None, "INCONSISTENT_EXACT_COST_ARITHMETIC"
            net_r = float(pnl / risk)
            if not isfinite(net_r):
                return None, "INCONSISTENT_EXACT_COST_ARITHMETIC"
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return None, "MISSING_EXACT_COST_ARITHMETIC"
    entry_error = (float(entry) - item["bottom"]) / item["atr"]
    if not isfinite(entry_error):
        return None, "INCONSISTENT_EXACT_COST_ARITHMETIC"
    if entry_error < -1e-10:
        raise ValueError("QUALITY_EXECUTION_BELOW_OBSERVED_BOTTOM")
    near = float(0 <= entry_error <= 1.)
    if int(bool(near and item["rebound"])) != item["joint"]:
        raise ValueError("QUALITY_EXECUTION_JOINT_LABEL_MISMATCH")
    return dict(entry_bottom_atr=max(0., entry_error), near=near,
                net_r=net_r, net_pnl=float(pnl), cost_policy_id=execution["cost_policy_id"]), None


def _selected_quality(items, executions, threshold):
    selected = [item for item in items if item["p_fill"] >= threshold]
    filled = [item for item in selected if item["fill"]]
    known, reasons = [], set()
    for item in filled:
        value, reason = _execution(item, executions)
        if reason:
            reasons.add(reason)
        else:
            known.append(dict(item, execution=value))
    complete = bool(filled) and len(known) == len(filled)
    if not selected:
        reasons.add("NO_SELECTED_EVENTS")
    elif not filled:
        reasons.add("NO_SELECTED_FILLED_REFERENCE_PATHS")
    path = dict(status="COMPLETE_REFERENCE_DIAGNOSTICS" if complete else "UNKNOWN",
        verified_reference_paths=len(known), required_reference_paths=len(filled), reasons=sorted(reasons),
        near_bottom_rate=_mean(known, lambda item: item["execution"]["near"]) if complete else None,
        mean_entry_bottom_atr=_mean(known, lambda item: item["execution"]["entry_bottom_atr"]) if complete else None,
        median_entry_bottom_atr=float(np.median([item["execution"]["entry_bottom_atr"] for item in known])) if complete else None,
        mean_unclipped_net_r=_mean(known, lambda item: item["execution"]["net_r"]) if complete else None,
        mean_reference_net_pnl_twd=_mean(known, lambda item: item["execution"]["net_pnl"]) if complete else None,
        net_positive_reference_trade_rate=_mean(known, lambda item: float(item["execution"]["net_pnl"] > 0)) if complete else None,
        cost_policy_ids=sorted(set(item["execution"]["cost_policy_id"] for item in known)),
        trade_size_scope="NATIVE_INDIVIDUAL_REFERENCE_SHARES_NOT_A_SHARED_ACCOUNT")
    return dict(selection=dict(rule="PREDICTED_FILL_AT_LEAST_FIXED_THRESHOLD", threshold=threshold,
            threshold_fitted=False, joint_success_probability="p_fill_times_conditional_p_entry"),
        selected_events=len(selected), selected_filled_reference_events=len(filled),
        selected_event_weight_fraction=fsum(item["weight"] for item in selected),
        observed_selected_fill_rate=_mean(selected, lambda item: item["fill"]),
        observed_rebound_given_fill=_mean(filled, lambda item: item["rebound"]),
        observed_near_and_rebound_given_fill=_mean(filled, lambda item: item["joint"]),
        observed_near_and_rebound_per_selected_event=_mean(selected, lambda item: item["joint"] if item["fill"] else 0),
        reference_paths=path)


def evaluate_reference_quality(rows, predictions, executions=None, *, fill_threshold=.5, evaluation_cutoff=None):
    """Summarize fixed predictions on all supplied verified events, without fit."""
    threshold = _number(fill_threshold, "fill_threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("QUALITY_FIXED_THRESHOLD_OUT_OF_RANGE")
    items = _inputs(rows, predictions)
    if evaluation_cutoff is not None and any(item["available"] > timestamp(evaluation_cutoff) for item in items):
        raise ValueError("QUALITY_UNMATURED_EVALUATION_LABEL")
    if executions is not None and not isinstance(executions, dict):
        raise ValueError("QUALITY_ROW_KEYED_EXECUTIONS_REQUIRED")
    return dict(schema=SCHEMA, scope="REFERENCE_H20_PATH_DIAGNOSTICS", row_count=len(items),
        event_ids_sha256=identity([item["key"] for item in items]),
        prediction_id=identity(predictions), metrics=_forecast_metrics(items),
        reference_quality=_selected_quality(items, executions or {}, threshold),
        portfolio_after_cost=dict(status="NOT_ASSESSABLE", reason="PAIRED_FOUR_ARM_SOURCE_BOUND_ACCOUNT_PATHS_REQUIRED"),
        fr2_acceptance="NOT_ASSESSED", source_authenticity="CALLER_VERIFIED_INPUTS_REQUIRED",
        execution_scope="NEXT_OFFICIAL_OPEN_REFERENCE_WITH_DECLARED_COSTS_NOT_BROKER_FILL_OR_CAPACITY_PROOF",
        inference_scope="HISTORICAL_OR_OOF_DIAGNOSTICS_NOT_FRESH_CONFIRMATORY_EVIDENCE",
        approved_for_live=False, research_only=True, execution_action="NO_ACTION")


def _paired_interval(items, deltas, frozen_sessions):
    groups = defaultdict(list)
    for index, item in enumerate(items):
        groups[item["date"]].append(index)
    dates = sorted(groups)
    metadata = dict(INFERENCE, observed_sessions=len(dates), status="UNKNOWN", interval=None)
    if len(dates) < 2 * INFERENCE["block_sessions"]:
        return dict(metadata, reason="AT_LEAST_42_OBSERVED_OFFICIAL_SESSIONS_REQUIRED")
    if not isinstance(frozen_sessions, (list, tuple)) or not frozen_sessions:
        return dict(metadata, reason="FROZEN_OFFICIAL_SESSION_ORDER_REQUIRED")
    if (len(frozen_sessions) > 4096 or list(frozen_sessions) != sorted(set(frozen_sessions))
            or any(not isinstance(day, str) for day in frozen_sessions)):
        raise ValueError("QUALITY_FROZEN_SESSION_ORDER_INVALID")
    positions = {day: index for index, day in enumerate(frozen_sessions)}
    if any(day not in positions for day in dates):
        raise ValueError("QUALITY_EVENT_OUTSIDE_FROZEN_SESSIONS")
    indices = [positions[day] for day in dates]
    if any(right != left + 1 for left, right in zip(indices, indices[1:])):
        return dict(metadata, reason="DISCONTIGUOUS_OBSERVED_SESSION_SUPPORT")
    weights = np.asarray([item["weight"] for item in items])
    day_weight = np.asarray([weights[groups[day]].sum() for day in dates])
    day_delta = np.asarray([np.dot(weights[groups[day]], deltas[groups[day]]) for day in dates])
    generator = np.random.default_rng(INFERENCE["seed"])
    estimates, n, block = [], len(dates), INFERENCE["block_sessions"]
    for _ in range(INFERENCE["bootstrap_replicates"]):
        selected = []
        while len(selected) < n:
            start = int(generator.integers(0, n - block + 1))
            selected.extend(range(start, start + block))
        selected = selected[:n]
        estimates.append(float(day_delta[selected].sum() / day_weight[selected].sum()))
    alpha = 1 - INFERENCE["confidence"]
    return dict(metadata, status="AVAILABLE_DIAGNOSTIC_INTERVAL", reason=None,
        interval=dict(lower=float(np.quantile(estimates, alpha / 2)), upper=float(np.quantile(estimates, 1 - alpha / 2))),
        correction="NONE_EXPLORATORY_DIAGNOSTIC_NOT_SELECTION_ADJUSTED_OR_FR2_ACCEPTANCE")


def compare_reference_quality(rows, baseline_predictions, candidate_predictions, executions=None, *,
                              fill_threshold=.5, evaluation_cutoff=None, frozen_sessions=None):
    """Same-event paired comparison; never drops a bad or missing prediction."""
    left = _inputs(rows, baseline_predictions)
    right = _inputs(rows, candidate_predictions)
    deltas = np.asarray([(b["p_fill"] - b["fill"]) ** 2 - (a["p_fill"] - a["fill"]) ** 2
                         for a, b in zip(left, right)])
    return dict(schema=COMPARISON_SCHEMA, scope="PAIRED_SAME_EVENT_REFERENCE_DIAGNOSTICS", paired_events=len(rows),
        baseline=evaluate_reference_quality(rows, baseline_predictions, executions,
            fill_threshold=fill_threshold, evaluation_cutoff=evaluation_cutoff),
        candidate=evaluate_reference_quality(rows, candidate_predictions, executions,
            fill_threshold=fill_threshold, evaluation_cutoff=evaluation_cutoff),
        fill_brier_difference_candidate_minus_baseline=float(np.dot([item["weight"] for item in left], deltas)),
        paired_interval=_paired_interval(left, deltas, frozen_sessions),
        interpretation="NEGATIVE_BRIER_DIFFERENCE_IS_NOT_A_PROFIT_OR_FR2_ACCEPTANCE_CLAIM",
        research_only=True, approved_for_live=False, execution_action="NO_ACTION")

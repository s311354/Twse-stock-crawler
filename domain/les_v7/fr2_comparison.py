"""Predeclared FR2 2x2 acceptance, never admission or an account simulator.

Hash-bound inputs establish integrity, not authenticity. Missing arms, account
paths or fresh independent test windows remain NOT_ASSESSABLE. No winner filter,
synthetic fills, probability changes or threshold search is performed here.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from math import fsum, isfinite, log
from zoneinfo import ZoneInfo

import numpy as np

from domain.les_v7.contracts import finite, identity, timestamp
from domain.les_v7.fractals import is_active_fractal_feature

SCHEMA = "les-v7-fr2-comparison-input/v1"
REPORT_SCHEMA = "les-v7-fr2-comparison/v1"
ARMS = ("M0_P0", "M1_P0", "M0_P1", "M1_P1")
CONDITION_KEYS = ("dataset_id", "universe_id", "split_id", "reference_policy_id",
                  "cost_policy_id", "initial_account_id", "execution_environment_id")
POLICY = dict(schema="les-v7-fr2-acceptance-policy/v1", version="FR_ACCEPT_R2",
    minimum_paired_events=2000, test_windows=3, sessions_per_window=60,
    minimum_completed_trades_per_arm=200, block_sessions=21,
    bootstrap_replicates=1999, seed=20261018, confidence=.95,
    minimum_relative_improvement=dict(median_bottom_error_atr=.02, pinball_loss=.02,
                                     timing_nll=.01, rebound_brier=.01),
    maximum_relative_worsening=dict(fill_brier=.01, joint_brier=.01),
    maximum_drawdown_worsening=.02,
    inference="PAIRED_MOVING_TEMPORAL_BLOCK_BOOTSTRAP_BONFERRONI_R1")
METRICS = ("median_bottom_error_atr", "pinball_loss", "timing_nll",
           "rebound_brier", "fill_brier", "joint_brier")


def _sealed(record, name):
    if not isinstance(record, dict) or record.get("sha256") != identity(record.get("payload")):
        raise ValueError("FR2_HASH_MISMATCH:" + name)
    return record["payload"]


def _model(record, name):
    model = _sealed(record, name)
    names = model.get("feature_names")
    if (not isinstance(names, list) or not names or len(names) != len(set(names))
            or any(not isinstance(n, str) or not n for n in names)):
        raise ValueError("FR2_MODEL_FEATURES_REQUIRED")
    has_fractal = any(is_active_fractal_feature(n) for n in names)
    if has_fractal != (name == "M1"):
        raise ValueError("FR2_TRAINED_NO_FRACTAL_COUNTERPART_REQUIRED")
    for key in ("heads_id", "calibration_id", "weights_id", "training_config_id"):
        if not model.get(key):
            raise ValueError("FR2_MODEL_PROVENANCE_REQUIRED:" + key)
    fit = timestamp(model["fit_cutoff"])
    if any(timestamp(model[k]) > fit for k in
           ("last_training_label_available_at", "calibration_last_label_available_at")):
        raise ValueError("FR2_MODEL_FUTURE_LABEL_LEAKAGE")
    return model


def _event(row, model, evaluation):
    if not isinstance(row, dict):
        raise ValueError("FR2_EVENT_OBJECT_REQUIRED")
    cutoff = timestamp(row["cutoff"])
    if (not timestamp(model["fit_cutoff"]) < cutoff < timestamp(row["label_available_at"]) <= evaluation
            or row["row_id"] != identity(dict(symbol=row["symbol"], market=row["market"], cutoff=cutoff.isoformat()))):
        raise ValueError("FR2_EVENT_ID_OR_TIME_ISOLATION_INVALID")
    labels, f = row["labels"], row["forecast"]
    if (type(labels.get("fill")) is not int or labels["fill"] not in (0, 1)
            or type(labels.get("timing")) is not int or not 0 <= labels["timing"] <= 20):
        raise ValueError("FR2_OBSERVED_LABELS_REQUIRED")
    if labels["fill"] == 1:
        if (any(type(labels.get(k)) is not int or labels[k] not in (0, 1) for k in ("rebound", "joint"))
                or labels["joint"] > labels["rebound"]):
            raise ValueError("FR2_CONDITIONAL_LABELS_INVALID")
    elif any(labels.get(k) is not None for k in ("rebound", "joint")):
        raise ValueError("FR2_NO_FILL_CONDITIONAL_LABELS_MUST_BE_NULL")
    q = [finite(v, "quantile", 0) for v in f["quantiles"]]
    timing = [finite(v, "timing_probability", 0, 1) for v in f["timing"]]
    if (len(q) != 3 or q != sorted(q) or len(timing) != 21 or abs(fsum(timing) - 1) > 1e-9
            or finite(f["p_entry"], "joint_probability", 0, 1) > finite(f["p_rebound"], "rebound_probability", 0, 1)):
        raise ValueError("FR2_FORECAST_PROJECTION_INVALID")
    finite(f["p_fill"], "fill_probability", 0, 1)
    bottom = finite(row["bottom"], "bottom", 0)
    close, atr = finite(row["close"], "close", 0), finite(row["atr"], "atr", 0)
    if not close > 0 or not atr > 0 or not bottom > 0:
        raise ValueError("FR2_POSITIVE_PRICE_COORDINATES_REQUIRED")
    weight = finite(row.get("observation_weight", 1), "observation_weight", 0)
    if weight <= 0:
        raise ValueError("FR2_POSITIVE_OBSERVATION_WEIGHT_REQUIRED")
    errors = [(bottom - p) / close for p in q]
    return dict(error=abs(q[1] - bottom) / atr,
        pinball=fsum(max(t * e, (t - 1) * e) for t, e in zip((.1, .5, .9), errors)) / 3,
        timing=-log(max(timing[labels["timing"]], 1e-15)),
        fill=(f["p_fill"] - labels["fill"]) ** 2,
        rebound=(f["p_rebound"] - labels["rebound"]) ** 2 if labels["fill"] else None,
        joint=(f["p_entry"] - labels["joint"]) ** 2 if labels["fill"] else None,
        covered=float(q[0] <= bottom <= q[2]), width=(q[2] - q[0]) / atr, weight=weight)


def _metrics(rows):
    def mean(key):
        keep = [r for r in rows if r[key] is not None]
        return fsum(r[key] * r["weight"] for r in keep) / fsum(r["weight"] for r in keep) if keep else None
    return dict(median_bottom_error_atr=float(np.median([r["error"] for r in rows])),
        pinball_loss=fsum(r["pinball"] for r in rows) / len(rows), timing_nll=mean("timing"),
        fill_brier=mean("fill"), rebound_brier=mean("rebound"), joint_brier=mean("joint"),
        interval_coverage=fsum(r["covered"] for r in rows) / len(rows),
        mean_interval_width_atr=fsum(r["width"] for r in rows) / len(rows))


def _bootstrap(groups, statistic, *, tests):
    """Keep all stocks at each date together; 21-session blocks preserve H20 overlap."""
    n, block = len(groups), POLICY["block_sessions"]
    if n < 2 * block:
        return None
    rng = np.random.default_rng(POLICY["seed"])
    estimates = []
    for _ in range(POLICY["bootstrap_replicates"]):
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, n - block + 1))
            indices.extend(range(start, start + block))
        estimate = statistic([groups[i] for i in indices[:n]])
        if estimate is None or not isfinite(estimate):
            return None
        estimates.append(estimate)
    alpha = (1 - POLICY["confidence"]) / tests
    return dict(lower=float(np.quantile(estimates, alpha / 2)), upper=float(np.quantile(estimates, 1 - alpha / 2)),
                confidence=POLICY["confidence"], family_tests=tests,
                correction="BONFERRONI", block_sessions=block,
                replicates=POLICY["bootstrap_replicates"], seed=POLICY["seed"])


def _forecast_comparison(left, right, dates):
    lm, rm = _metrics(left), _metrics(right)
    by_date = defaultdict(list)
    for index, day in enumerate(dates):
        by_date[day].append(index)
    groups = [by_date[d] for d in sorted(by_date)]
    output = {}
    for key in METRICS:
        if lm[key] is None or rm[key] is None or lm[key] <= 0:
            output[key] = dict(status="NOT_ASSESSABLE", reason="NONPOSITIVE_OR_ABSENT_REFERENCE_METRIC")
            continue
        improvement = 1 - rm[key] / lm[key]
        def statistic(selected):
            indices = [i for group in selected for i in group]
            a, b = _metrics([left[i] for i in indices])[key], _metrics([right[i] for i in indices])[key]
            return 1 - b / a if a and b is not None else None
        ci = _bootstrap(groups, statistic, tests=16)
        minimum = POLICY["minimum_relative_improvement"].get(key, -POLICY["maximum_relative_worsening"].get(key, 0))
        passes = ci is not None and improvement >= minimum and ci["lower"] > (0 if minimum > 0 else minimum)
        output[key] = dict(baseline=lm[key], fractal=rm[key], relative_improvement=improvement,
            required_minimum=minimum, confidence_interval=ci,
            status="NOT_ASSESSABLE" if ci is None else "PASS" if passes else "FAIL")
    return dict(baseline=lm, fractal=rm, criteria=output)


def _account(arm, sessions, initial_id, cost_id):
    account = arm.get("account_path")
    if account is None:
        return None
    body = _sealed(account, "account_path")
    if (body.get("initial_account_id") != initial_id or body.get("cost_policy_id") != cost_id
            or body.get("external_cashflows") != [] or body.get("net_of_costs") is not True
            or not body.get("execution_evidence_ids") or body.get("sessions") != sessions):
        raise ValueError("FR2_ACCOUNT_COST_OR_SESSION_BINDING_INVALID")
    values = [finite(v, "net_equity", 0) for v in body["net_equity"]]
    if len(values) != len(sessions) + 1 or any(v <= 0 for v in values):
        raise ValueError("FR2_INITIAL_PLUS_DAILY_NET_EQUITY_REQUIRED")
    trades = body["completed_trades"]
    if not isinstance(trades, list) or len({t["trade_id"] for t in trades}) != len(trades):
        raise ValueError("FR2_UNIQUE_COMPLETED_TRADES_REQUIRED")
    for trade in trades:
        if not trade.get("source_evidence_ids") or trade.get("cost_policy_id") != cost_id:
            raise ValueError("FR2_TRADE_EXECUTION_COST_EVIDENCE_REQUIRED")
        if not timestamp(trade["entry_at"]) < timestamp(trade["exit_at"]):
            raise ValueError("FR2_COMPLETED_TRADE_TIME_INVALID")
        if (timestamp(trade["entry_at"]) < timestamp(sessions[0] + "T13:30:00+08:00")
                or timestamp(trade["exit_at"]) > timestamp(sessions[-1] + "T13:30:00+08:00")):
            raise ValueError("FR2_COMPLETED_TRADE_OUTSIDE_TEST_WINDOW")
        finite(trade["realized_net_pnl"], "realized_net_pnl")
    running, drawdown = values[0], 0.
    for value in values:
        running = max(running, value)
        drawdown = max(drawdown, 1 - value / running)
    return dict(return_net=values[-1] / values[0] - 1, max_drawdown=drawdown,
        completed_trades=len(trades), initial_equity=values[0],
        daily_gross=[b / a for a, b in zip(values, values[1:])], account_path_id=account["sha256"])


def compare(record):
    """Evaluate all four supplied arms without running or changing any model."""
    body = _sealed(record, "input")
    if body.get("schema") != SCHEMA or body.get("research_only") is not True:
        raise ValueError("FR2_EXPLICIT_RESEARCH_COMPARISON_REQUIRED")
    conditions = body["conditions"]
    if any(not conditions.get(k) for k in CONDITION_KEYS):
        raise ValueError("FR2_SAME_CONDITION_IDENTITIES_REQUIRED")
    sessions = body["test_sessions"]
    if (not isinstance(sessions, list) or not sessions or sessions != sorted(set(sessions))
            or not body.get("calendar_evidence_ids")):
        raise ValueError("FR2_FROZEN_OFFICIAL_TEST_SESSIONS_REQUIRED")
    models = {name: _model(body["models"][name], name) for name in ("M0", "M1")}
    if (models["M0"]["fit_cutoff"] != models["M1"]["fit_cutoff"]
            or models["M0"]["training_config_id"] != models["M1"]["training_config_id"]
            or [n for n in models["M1"]["feature_names"] if not is_active_fractal_feature(n)] != models["M0"]["feature_names"]):
        raise ValueError("FR2_MATCHED_FITTED_ARCHITECTURE_REQUIRED")
    evaluation = timestamp(body["evaluation_cutoff"])
    reasons, arms, losses, events = [], {}, {}, {}
    for name in ARMS:
        raw = body.get("arms", {}).get(name)
        if raw is None:
            reasons.append("MISSING_ARM:" + name)
            arms[name] = dict(status="NOT_ASSESSABLE", reason="MISSING_ARM")
            continue
        arm = _sealed(raw, name)
        if arm.get("conditions") != conditions or arm.get("model_id") != body["models"][name[:2]]["sha256"]:
            raise ValueError("FR2_ARM_CONDITION_OR_MODEL_MISMATCH:" + name)
        expected_policy = body["entry_policy_ids"][name[-2:]]
        if arm.get("entry_policy_id") != expected_policy or arm.get("execution_policy_id") != body["execution_policy_id"]:
            raise ValueError("FR2_ARM_POLICY_MISMATCH:" + name)
        rows = arm["events"]
        if not rows or len(rows) > 100000 or len({r["row_id"] for r in rows}) != len(rows):
            raise ValueError("FR2_BOUNDED_UNIQUE_EVENTS_REQUIRED")
        rows = sorted(rows, key=lambda r: (timestamp(r["cutoff"]), r["symbol"], r["market"]))
        if any(timestamp(r["cutoff"]).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat() not in sessions for r in rows):
            raise ValueError("FR2_EVENT_OUTSIDE_TEST_SESSIONS")
        events[name] = rows
        losses[name] = [_event(r, models[name[:2]], evaluation) for r in rows]
        account = _account(arm, sessions, conditions["initial_account_id"], conditions["cost_policy_id"])
        if account is None:
            reasons.append("MISSING_AFTER_COST_ACCOUNT_PATH:" + name)
        elif account["completed_trades"] < POLICY["minimum_completed_trades_per_arm"]:
            reasons.append("INSUFFICIENT_COMPLETED_TRADES:" + name)
        arms[name] = dict(status="FORECAST_COMPLETED_ACCOUNT_NOT_ASSESSABLE" if account is None else "COMPLETED",
            forecast_status="COMPLETED", account_status="NOT_ASSESSABLE" if account is None else "COMPLETED",
            row_count=len(rows), model_id=arm["model_id"],
            entry_policy_id=expected_policy, execution_policy_id=arm["execution_policy_id"],
            metrics=_metrics(losses[name]), account=account, arm_id=raw["sha256"])
    if events:
        reference = next(iter(events.values()))
        keys = ("row_id", "symbol", "market", "cutoff", "label_available_at", "labels", "bottom", "close", "atr", "observation_weight")
        signature = [{k: r.get(k) for k in keys} for r in reference]
        if any([{k: r.get(k) for k in keys} for r in rows] != signature for rows in events.values()):
            raise ValueError("FR2_EVENT_OR_LABEL_PAIRING_MISMATCH")
        if len(reference) < POLICY["minimum_paired_events"]:
            reasons.append("INSUFFICIENT_PAIRED_FORECAST_EVENTS")
    for model in ("M0", "M1"):
        a, b = events.get(model + "_P0"), events.get(model + "_P1")
        if a and b and [r["forecast"] for r in a] != [r["forecast"] for r in b]:
            raise ValueError("FR2_ENTRY_POLICY_CANNOT_MODIFY_PREDICTIVE_PROBABILITIES")
    windows = body.get("test_windows", [])
    if (len(windows) != POLICY["test_windows"] or any(len(w.get("sessions", [])) != POLICY["sessions_per_window"] for w in windows)
            or [d for w in windows for d in w["sessions"]] != sessions):
        reasons.append("THREE_DISJOINT_60_SESSION_WINDOWS_REQUIRED")
    if body.get("holdout_use") != "FRESH_CONFIRMATORY" or body.get("predeclared_before_outcomes") is not True:
        reasons.append("DEVELOPMENT_OR_NON_PREDECLARED_HOLDOUT")
    protocol = body.get("preregistered_protocol")
    if protocol is None:
        reasons.append("MISSING_PREDECLARED_PROTOCOL_EVIDENCE")
    else:
        protocol_body = _sealed(protocol, "preregistered_protocol")
        if (protocol_body.get("acceptance_policy_id") != identity(POLICY)
                or protocol_body.get("conditions") != conditions
                or not protocol_body.get("source_evidence_ids")
                or timestamp(protocol_body["registered_at"]) >= timestamp(sessions[0] + "T00:00:00+08:00")):
            raise ValueError("FR2_PREREGISTRATION_BINDING_OR_TIME_INVALID")
    forecast = {}
    for policy in ("P0", "P1"):
        left, right = "M0_" + policy, "M1_" + policy
        if left in losses and right in losses:
            # The policy-independence invariant above proves forecasts/labels
            # identical across P0/P1. Reuse the paired model comparison rather
            # than resampling the same potentially large prediction set twice.
            forecast[policy] = forecast["P0"] if policy == "P1" and "P0" in forecast else _forecast_comparison(
                losses[left], losses[right],
                [timestamp(r["cutoff"]).astimezone(ZoneInfo("Asia/Taipei")).date().isoformat() for r in events[left]])
    performance = {}
    for left, right in (("M0_P0", "M1_P0"), ("M0_P1", "M1_P1"), ("M0_P0", "M0_P1"), ("M1_P0", "M1_P1")):
        a, b = arms[left].get("account"), arms[right].get("account")
        key = right + "_minus_" + left
        if a is None or b is None:
            performance[key] = dict(status="NOT_ASSESSABLE", reason="SOURCE_BOUND_AFTER_COST_ACCOUNT_PATHS_REQUIRED")
            continue
        if a["initial_equity"] != b["initial_equity"]:
            raise ValueError("FR2_INITIAL_ACCOUNT_EQUITY_MISMATCH")
        def statistic(groups):
            x = np.prod([a["daily_gross"][i] for i in groups])
            y = np.prod([b["daily_gross"][i] for i in groups])
            return float(y - x)
        ci = _bootstrap(list(range(len(sessions))), statistic, tests=16)
        difference, dd = b["return_net"] - a["return_net"], b["max_drawdown"] - a["max_drawdown"]
        performance[key] = dict(net_return_difference=difference, max_drawdown_worsening=dd,
            confidence_interval=ci, status="NOT_ASSESSABLE" if ci is None else
            "PASS" if difference > 0 and ci["lower"] > 0 and dd <= POLICY["maximum_drawdown_worsening"] else "FAIL")
    checks = [c["status"] for f in forecast.values() for c in f["criteria"].values()]
    checks.extend(p["status"] for p in performance.values())
    status = "NOT_ASSESSABLE" if reasons or "NOT_ASSESSABLE" in checks else "PASS" if checks and all(c == "PASS" for c in checks) else "FAIL"
    return deepcopy(dict(schema=REPORT_SCHEMA, status=status, input_id=record["sha256"], acceptance_policy=POLICY,
        conditions=conditions, test_sessions=sessions,
        paired_forecast_event_count=len(next(iter(events.values()))) if events else 0,
        arms=arms, forecast_comparisons=forecast, after_cost_comparisons=performance,
        reason_codes=sorted(set(reasons)), source_authenticity="INPUT_HASHES_NOT_SOURCE_CERTIFICATION",
        interpretation="NO_BUY_COUNT_OR_PROFIT_GUARANTEE", research_only=True,
        approved_for_live=False, execution_action="NO_ACTION", order_authority=False))

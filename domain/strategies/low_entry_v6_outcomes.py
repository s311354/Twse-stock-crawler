"""Cost-aware, purged holdout diagnostics. Never grants entry or Kelly authority.

The caller supplies closed trade episodes, not future highs or assumed fills.
Provenance is declared, not independently verified by this pure research tool.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import sqrt

from domain.strategies.low_entry_v6_quality import number as _native_number
from domain.strategies.low_entry_v6_signals import SETUPS

SCHEMA = "les-v6-outcome-audit/v1"
INPUT_SCHEMA = "les-v6-closed-outcomes/v1"
POLICY_VERSION = "2026-10-02-purged-closed-outcomes-v1"
MIN_TRAINING = 30
MIN_VALIDATION = 20
VERSION_KEYS = ("model", "parameters", "code", "execution_policy")
SOURCE_KINDS = {"EXECUTED_TRADES", "SIMULATED_TRADES", "SYNTHETIC_TEST_ONLY"}


def number(value: object) -> float | None:
    try:
        return _native_number(value)
    except OverflowError:
        return None


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("OUTCOME_TIMESTAMP_REQUIRES_TIMEZONE")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("OUTCOME_TIMESTAMP_REQUIRES_TIMEZONE")
    return parsed.astimezone(timezone.utc)


def wilson_interval(wins: int, count: int) -> list[float] | None:
    """Two-sided 95% Wilson interval under an independent Bernoulli assumption.

    Trading outcomes may be dependent; this is not a dependence-robust bound.
    NIST: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
    """
    if type(wins) is not int or type(count) is not int or not 0 <= wins <= count:
        raise ValueError("INVALID_BINOMIAL_COUNTS")
    if count == 0:
        return None
    p, z = wins / count, 1.959963984540054
    divisor = 1 + z * z / count
    center = (p + z * z / (2 * count)) / divisor
    radius = z * sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / divisor
    return [max(0., center - radius), min(1., center + radius)]


def _versions(value: object) -> dict:
    if not isinstance(value, dict) or any(not isinstance(value.get(k), str) or not value[k]
                                          for k in VERSION_KEYS):
        raise ValueError("OUTCOME_REQUIRES_PINNED_VERSIONS")
    return {k: value[k] for k in VERSION_KEYS}


def _cohort(row: dict) -> tuple:
    # Fixed before outcomes: no best-bin search, score -> probability conversion,
    # or pooling the non-equivalent scales of the three native setups.
    score_bin = min(9, int(row["formal_score"] // 10)) * 10
    return (row["market"], row["sector"], row["market_regime"], row["setup"],
            row["entry_decision"], score_bin)


def _cohort_fields(key: tuple) -> dict:
    return dict(market=key[0], sector=key[1], market_regime=key[2], setup=key[3],
                entry_decision=key[4], score_band=[key[5], key[5] + 10], upper_inclusive=key[5] == 90)


def _observation(row: dict, versions: dict, as_of: datetime, split: datetime,
                 max_sessions: int) -> tuple[dict | None, str | None]:
    try:
        signal, entered, closed, available = (_timestamp(row[k]) for k in
                                              ("signal_at", "entry_at", "exit_at", "available_at"))
    except (ValueError, KeyError, TypeError):
        return None, "INVALID_OR_MISSING_OUTCOME_TIMESTAMPS"
    if available > as_of or closed > as_of:
        return None, "OUTCOME_UNAVAILABLE_AT_CUTOFF"
    if not signal < entered <= closed <= available:
        return None, "INVALID_OUTCOME_TIME_ORDER"
    if _versions(row.get("versions")) != versions:
        raise ValueError("MIXED_OUTCOME_VERSIONS")
    for key in ("episode_id", "symbol", "source_id", "source_sha256", "evidence_id",
                "research_result_sha256", "account_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            return None, "MISSING_OUTCOME_PROVENANCE"
    for key in ("source_sha256", "research_result_sha256"):
        if len(row[key]) != 64 or any(c not in "0123456789abcdef" for c in row[key]):
            return None, "INVALID_OUTCOME_PROVENANCE_HASH"
    if (row.get("market") not in {"TWSE", "TPEX"} or row.get("setup") not in SETUPS or
            not isinstance(row.get("sector"), str) or not row["sector"] or
            row.get("market_regime") not in {"RISK_ON", "NEUTRAL", "RISK_OFF"}):
        return None, "UNSUPPORTED_OUTCOME_COHORT"
    if row.get("entry_decision") not in {"BUY", "BUY_SMALL"}:
        return None, "NOT_A_FORMAL_ENTRY_EPISODE"
    if (row.get("cash_unit") != "TWD" or row.get("price_unit") != "TWD_PER_SHARE" or
            row.get("quantity_unit") != "SHARES"):
        return None, "INCOMPATIBLE_OR_MISSING_OUTCOME_UNITS"
    if row.get("fully_closed") is not True or row.get("costs_complete") is not True:
        return None, "INCOMPLETE_EPISODE_OR_COSTS"
    sessions, quantity = row.get("holding_sessions"), row.get("quantity")
    if type(sessions) is not int or not 1 <= sessions <= max_sessions:
        return None, "HOLDING_POLICY_MISMATCH"
    if (type(quantity) is not int or not 1 <= quantity <= 2 ** 53 - 1 or
            type(row.get("sold_quantity")) is not int or row["sold_quantity"] != quantity):
        return None, "INCOMPLETE_POSITION_QUANTITIES"
    score, coverage = number(row.get("formal_score")), number(row.get("score_coverage"))
    if score is None or not 0 <= score <= 100 or coverage != 100:
        return None, "INCOMPLETE_OR_INVALID_FORMAL_SCORE"
    cash = [number(row.get(k)) for k in ("entry_notional", "entry_cost", "exit_notional",
                                        "exit_cost", "net_cash_dividends")]
    if any(v is None or v < 0 for v in cash) or cash[0] <= 0 or cash[2] <= 0:
        return None, "INVALID_OR_MISSING_CASH_FLOWS"
    entry, buy_cost, exit_value, sell_cost, dividends = cash
    invested, proceeds = number(entry + buy_cost), number(exit_value - sell_cost + dividends)
    if invested is None or proceeds is None or exit_value < sell_cost:
        return None, "INVALID_NET_PROCEEDS"
    net_return = number(proceeds / invested - 1)
    if net_return is None:
        return None, "INVALID_NET_PROCEEDS"
    # Purge outcomes whose label was not yet available at the holdout boundary.
    if signal < split <= available:
        return None, "PURGED_LABEL_CROSSES_VALIDATION_BOUNDARY"
    partition = "training" if signal < split else "validation"
    fill_price = number(entry / quantity)
    if fill_price is None or fill_price <= 0:
        return None, "INVALID_NOTIONAL_QUANTITY_PRICE"
    low = number(row.get("post_fill_min_price"))
    mae = (min(0., low / fill_price - 1) if low is not None and low > 0 and
           row.get("excursion_basis") == "AFTER_FILL_UNTIL_FULL_EXIT" else None)
    return dict(episode_id=row["episode_id"], symbol=row["symbol"], market=row["market"],
                sector=row["sector"], market_regime=row["market_regime"],
                setup=row["setup"], entry_decision=row["entry_decision"], formal_score=score,
                partition=partition, signal_at=signal.isoformat(), entry_at=entered.isoformat(),
                exit_at=closed.isoformat(), available_at=available.isoformat(),
                holding_sessions=sessions, quantity=quantity, outcome_sha256=_digest(row),
                net_return=net_return, success=net_return > 0, mae_fraction=mae,
                source_id=row["source_id"], source_sha256=row["source_sha256"],
                evidence_id=row["evidence_id"], research_result_sha256=row["research_result_sha256"],
                account_id=row["account_id"]), None


def _stats(rows: list[dict], adverse_limit: float) -> dict:
    count = len(rows)
    wins = sum(row["success"] for row in rows)
    measured = [row for row in rows if row["mae_fraction"] is not None]
    captures = sum(row["success"] and row["mae_fraction"] >= -adverse_limit for row in measured)
    returns = [row["net_return"] for row in rows]
    return dict(sample_count=count, wins=wins, observed_win_rate=wins / count if count else None,
                wilson95_independence_assumption=wilson_interval(wins, count),
                mean_net_return=sum(value / count for value in returns) if count else None,
                worst_net_return=min(returns) if returns else None,
                low_capture_sample_count=len(measured), low_capture_successes=captures,
                low_capture_rate=captures / len(measured) if measured else None,
                low_capture_missing_count=count - len(measured),
                low_capture_wilson95_independence_assumption=wilson_interval(captures, len(measured)))


def outcome_audit(payload: dict) -> dict:
    """Frozen pre-holdout cohorts, diagnostic predictions, and separate MAE quality.

    Minimum counts are versioned research sufficiency rules, not a proof of
    profitability. Even adequate samples remain non-executable declared evidence.
    """
    if not isinstance(payload, dict) or payload.get("schema") != INPUT_SCHEMA:
        raise ValueError("INVALID_OUTCOME_PACKET_SCHEMA")
    versions = _versions(payload.get("versions"))
    if payload.get("source_kind") not in SOURCE_KINDS:
        raise ValueError("OUTCOME_SOURCE_KIND_REQUIRED")
    if payload.get("inclusion_policy") != "ALL_CLOSED_EPISODES_NO_WINNER_FILTER":
        raise ValueError("OUTCOME_INCLUSION_POLICY_REQUIRED")
    definition = payload.get("definition", {})
    if (not isinstance(definition, dict) or definition.get("success") != "POSITIVE_NET_CASH_RETURN" or
            definition.get("position_policy") != "ONE_ENTRY_FULL_EXIT_NO_ADDS"):
        raise ValueError("EXPLICIT_OUTCOME_DEFINITION_REQUIRED")
    max_sessions = definition.get("max_holding_sessions")
    adverse_limit = number(definition.get("max_adverse_fraction"))
    if type(max_sessions) is not int or max_sessions < 1 or adverse_limit is None or not 0 < adverse_limit < 1:
        raise ValueError("EXPLICIT_HORIZON_AND_ADVERSE_LIMIT_REQUIRED")
    as_of, split = _timestamp(payload["as_of"]), _timestamp(payload["validation_start"])
    if split > as_of:
        raise ValueError("VALIDATION_BOUNDARY_AFTER_CUTOFF")
    if not isinstance(payload.get("outcomes"), list):
        raise ValueError("EXPLICIT_OUTCOMES_REQUIRED")
    rows, excluded, identities, opportunities = [], [], set(), set()
    for index, raw in enumerate(payload["outcomes"]):
        if not isinstance(raw, dict):
            raise ValueError("OUTCOME_ROW_REQUIRES_OBJECT")
        row, reason = _observation(raw, versions, as_of, split, max_sessions)
        if reason:
            excluded.append(dict(index=index, episode_id=raw.get("episode_id") if isinstance(raw.get("episode_id"), str) else None,
                                 reason=reason))
            continue
        opportunity = row["market"], row["symbol"], row["signal_at"]
        if row["episode_id"] in identities or opportunity in opportunities:
            raise ValueError("DUPLICATE_OUTCOME_OR_CROSS_ACCOUNT_OPPORTUNITY")
        identities.add(row["episode_id"])
        opportunities.add(opportunity)
        rows.append(row)
    rows.sort(key=lambda r: (r["signal_at"], r["market"], r["symbol"], r["episode_id"]))
    by_symbol = defaultdict(list)
    for row in rows:
        by_symbol[(row["market"], row["symbol"])].append(row)
    for episodes in by_symbol.values():
        by_entry = sorted(episodes, key=lambda r: r["entry_at"])
        if any(_timestamp(right["entry_at"]) <= _timestamp(left["exit_at"])
               for left, right in zip(by_entry, by_entry[1:])):
            raise ValueError("OVERLAPPING_OUTCOME_EPISODES")
    groups = defaultdict(lambda: {"training": [], "validation": []})
    for row in rows:
        groups[_cohort(row)][row["partition"]].append(row)
    cohorts, predictions = [], []
    training = [row for row in rows if row["partition"] == "training"]
    baseline = sum(r["success"] for r in training) / len(training) if training else None
    for key, group in sorted(groups.items()):
        train, validation = group["training"], group["validation"]
        train_stats, validation_stats = _stats(train, adverse_limit), _stats(validation, adverse_limit)
        reasons = []
        if len(train) < MIN_TRAINING:
            reasons.append("INSUFFICIENT_TRAINING_OUTCOMES")
        if train and len({r["success"] for r in train}) < 2:
            reasons.append("SINGLE_CLASS_TRAINING_OUTCOMES")
        estimate = train_stats["observed_win_rate"] if not reasons else None
        validation_reasons = (["INSUFFICIENT_HOLDOUT_OUTCOMES"]
                              if len(validation) < MIN_VALIDATION else [])
        low_reasons = list(reasons)
        if train_stats["low_capture_sample_count"] < MIN_TRAINING:
            low_reasons.append("INSUFFICIENT_POST_FILL_EXCURSIONS")
        if train_stats["low_capture_missing_count"]:
            low_reasons.append("INCOMPLETE_POST_FILL_COVERAGE")
        low_validation_reasons = list(validation_reasons)
        if validation_stats["low_capture_missing_count"]:
            low_validation_reasons.append("INCOMPLETE_POST_FILL_COVERAGE")
        if len({r["success"] and r["mae_fraction"] >= -adverse_limit
                for r in train if r["mae_fraction"] is not None}) == 1:
            low_reasons.append("SINGLE_CLASS_LOW_CAPTURE_OUTCOMES")
        group_predictions = []
        for row in validation:
            group_predictions.append(dict(episode_id=row["episode_id"], **_cohort_fields(key),
                                    prediction_cutoff=split.isoformat(),
                                    estimated_win_rate=estimate, realized_success=row["success"],
                                    brier_loss=(estimate - row["success"]) ** 2 if estimate is not None else None,
                                    baseline_brier_loss=(baseline - row["success"]) ** 2 if estimate is not None else None))
        predictions.extend(group_predictions)
        losses = [p for p in group_predictions if p["brier_loss"] is not None]
        cohorts.append(dict(**_cohort_fields(key), training=train_stats, validation=validation_stats,
                            forecast_status="UNKNOWN" if reasons else "RESEARCH_ESTIMATE_NOT_EXECUTABLE",
                            reasons=reasons, estimated_win_rate=estimate,
                            holdout_status="NOT_ASSESSABLE" if reasons or validation_reasons else "DIAGNOSTICS_AVAILABLE_NOT_VALIDATED",
                            holdout_reasons=validation_reasons,
                            estimated_win_rate_interval=train_stats["wilson95_independence_assumption"] if estimate is not None else None,
                            estimated_low_capture_rate=None if low_reasons else train_stats["low_capture_rate"],
                            low_capture_status="UNKNOWN" if low_reasons else "RESEARCH_ESTIMATE_NOT_EXECUTABLE",
                            low_capture_reasons=low_reasons,
                            low_capture_holdout_status="NOT_ASSESSABLE" if low_reasons or low_validation_reasons else "DIAGNOSTICS_AVAILABLE_NOT_VALIDATED",
                            low_capture_holdout_reasons=low_validation_reasons,
                            holdout_brier=sum(p["brier_loss"] for p in losses) / len(losses) if losses else None,
                            holdout_baseline_brier=sum(p["baseline_brier_loss"] for p in losses) / len(losses) if losses else None))
    predicted = [p for p in predictions if p["estimated_win_rate"] is not None]
    selected_hash = _digest(rows)
    study_id = _digest(dict(policy_version=POLICY_VERSION, versions=versions,
                            source_kind=payload["source_kind"], as_of=as_of.isoformat(),
                            validation_start=split.isoformat(), definition=definition,
                            selected_outcomes_sha256=selected_hash))
    return dict(schema=SCHEMA, policy_version=POLICY_VERSION, study_id=study_id, versions=versions,
                source_kind=payload["source_kind"], source_posture="DECLARED_OUTCOMES_NOT_INDEPENDENTLY_VERIFIED",
                research_only=True, executable=False, kelly_calibration_eligible=False,
                bottom_guaranteed=False, performance_validated=False,
                forecast_status="RESEARCH_ESTIMATE_NOT_EXECUTABLE" if any(c["estimated_win_rate"] is not None for c in cohorts) else "UNKNOWN",
                as_of=as_of.isoformat(), validation_start=split.isoformat(), definition=deepcopy(definition),
                input_sha256=_digest(payload), selected_outcomes_sha256=selected_hash,
                sufficiency_policy=dict(min_training=MIN_TRAINING, min_validation=MIN_VALIDATION,
                                        score_bin_width=10, required_score_coverage=100),
                rows=rows, excluded=excluded, exclusion_counts=dict(sorted(Counter(r["reason"] for r in excluded).items())),
                cohorts=cohorts, predictions=predictions,
                validation_prediction_coverage=len(predicted) / len(predictions) if predictions else None,
                limitations=["CONDITIONAL_ON_DECLARED_FILLED_CLOSED_EPISODES_NOT_ENTIRE_UNIVERSE",
                             "PROVENANCE_HASHES_DO_NOT_VERIFY_SOURCE_OR_EXHAUSTIVE_INCLUSION",
                             "CROSS_SYMBOL_MARKET_DEPENDENCE_NOT_REMOVED_BY_WILSON_INTERVAL",
                             "REPEATED_HOLDOUT_TUNING_INVALIDATES_OUT_OF_SAMPLE_INTERPRETATION",
                             "PAST_WIN_RATE_AND_LOW_MAE_DO_NOT_GUARANTEE_FUTURE_BOTTOM_OR_CAGR",
                             "NATIVE_QUALITY_VETO_TRIGGER_RISK_ACCOUNT_AND_KELLY_UNCHANGED"])

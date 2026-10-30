"""Necessary-count preflight over verified FR3 native prepared shards.

This is not training, an OOF prediction, Quality approval or a performance
claim. A row with known utility is only a possible weight-training row: past
calibration, VALID forecasts and all six factors may still exclude it. Hence
insufficient upper bounds can prevent a futile fit, while sufficient counts
can never establish READY_TO_FREEZE. No missing utility is replaced by zero.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from domain.les_v7.contracts import CONTEXTS, finite, identity, timestamp
from infrastructure.les_v7_fr3_training import REQUEST_SCHEMA, validate_fr3_training_configs
from infrastructure.les_v7_scope_training import (
    MAX_PREPARED_BYTES, MAX_PREPARED_DECODED_BYTES, MAX_PREPARED_ROWS,
    _packet_feature_columns, _relative, verify_sharded_training_dataset,
    prepared_decoded_limit,
)
from infrastructure.research_archive import write_once
from infrastructure.storage.les_v7_prepared import inspect_prepared_packet
from infrastructure.storage.les_v7_training import MAX_CONFIG_BYTES, MAX_DATASET_BYTES, read_json_bounded


SCHEMA = "les-v7-fr3-prepared-capacity/v1"
SUFFICIENT = "SUFFICIENT_NECESSARY_COUNTS_NOT_FIT_OR_PERFORMANCE"
INSUFFICIENT = "INSUFFICIENT_NECESSARY_COUNTS_DO_NOT_FIT"


def _empty_counts():
    return dict(prepared_rows=0, utility_known_rows=0, utility_missing_rows=0,
        no_fill_rows=0, fit_mature_rows=0, fit_mature_utility_known_rows=0,
        oof_window_rows=0, oof_window_fit_mature_utility_upper_bound=0,
        holdout_mature_rows=0)


def _increment(counts, *, utility_known, no_fill, fit_mature, oof_window, holdout_mature):
    counts["prepared_rows"] += 1
    counts["utility_known_rows" if utility_known else "utility_missing_rows"] += 1
    counts["no_fill_rows"] += int(no_fill)
    counts["fit_mature_rows"] += int(fit_mature)
    counts["fit_mature_utility_known_rows"] += int(fit_mature and utility_known)
    counts["oof_window_rows"] += int(oof_window)
    counts["oof_window_fit_mature_utility_upper_bound"] += int(oof_window and fit_mature and utility_known)
    counts["holdout_mature_rows"] += int(holdout_mature)


def audit_fr3_prepared_capacity(wrapper_root, *, allow_synthetic=False):
    """Verify sources/config/provenance, then count one prepared shard at a time.

    Accepts a completed native *preparation*, without requiring an expensive
    training completion or a published model. The original native verifier
    remains responsible for source authenticity boundaries and row lineage.
    Neither thresholds, folds, dates, labels nor prepared bytes are modified.
    """
    if type(allow_synthetic) is not bool:
        raise ValueError("FR3_CAPACITY_SYNTHETIC_BOOLEAN_REQUIRED")
    root = Path(wrapper_root).expanduser().resolve()
    native_root = root / "native"
    request = read_json_bounded(root / "fr3-training-request.json", 2 * MAX_CONFIG_BYTES + 64 * 1024)
    body = request.get("payload", {})
    if request.get("sha256") != identity(body) or body.get("schema") != REQUEST_SCHEMA:
        raise ValueError("FR3_CAPACITY_REQUEST_HASH_OR_SCHEMA_MISMATCH")
    if body.get("dataset_kind") == "SYNTHETIC_TEST_ONLY" and not allow_synthetic:
        raise ValueError("FR3_CAPACITY_SYNTHETIC_REQUIRES_EXPLICIT_OPT_IN")
    build = read_json_bounded(native_root / "scope-build-config.json", MAX_CONFIG_BYTES)
    config = read_json_bounded(native_root / "scope-training-config.json", MAX_CONFIG_BYTES)
    components = validate_fr3_training_configs(build, config)
    if (body.get("build_config") != build or body.get("training_config") != config
            or body.get("components") != components):
        raise ValueError("FR3_CAPACITY_FROZEN_CONFIG_OR_COMPONENT_MISMATCH")
    record = read_json_bounded(native_root / "scope-training-dataset.json", MAX_DATASET_BYTES)
    manifest = verify_sharded_training_dataset(record, native_root, config=config)
    universe = manifest["universe"]
    symbols = [row[0] for row in universe]
    if (manifest.get("dataset_kind") != body.get("dataset_kind")
            or manifest.get("freeze_id") != body.get("freeze_id")
            or manifest.get("scope_id") != body.get("scope_id")
            or manifest.get("availability_policy") != body.get("availability_policy")
            or symbols != body.get("symbols")
            or Path(manifest["freeze_path"]).resolve() != Path(body["freeze_path"]).resolve()):
        raise ValueError("FR3_CAPACITY_FIXED_SCOPE_BINDING_MISMATCH")
    if manifest.get("dataset_kind") == "SYNTHETIC_TEST_ONLY" and not allow_synthetic:
        raise ValueError("FR3_CAPACITY_SYNTHETIC_REQUIRES_EXPLICIT_OPT_IN")
    fit, start, end, evaluation = (timestamp(config[key]) for key in
        ("fit_cutoff", "validation_start", "validation_end", "evaluation_cutoff"))
    folds = [(fold["fold_id"], timestamp(fold["prediction_start"]), timestamp(fold["prediction_end"]))
             for fold in config["folds"]]
    total = _empty_counts()
    contexts = {context: _empty_counts() for context in CONTEXTS}
    by_symbol = {symbol: dict(symbol=symbol, market=market, counts=_empty_counts(),
        contexts={context: _empty_counts() for context in CONTEXTS}) for symbol, market in universe}
    by_fold = {name: _empty_counts() for name, _, _ in folds}
    seen, coverage = set(), []
    encoded = decoded = 0
    for child in manifest["children"]:
        packet_read = inspect_prepared_packet(_relative(native_root, child["prepared_path"], "prepared"))
        encoded += packet_read.encoded_bytes
        decoded += packet_read.decoded_bytes
        packet = packet_read.value
        if (identity(packet) != child["prepared_sha256"]
                or packet.get("config_id") != identity(config)
                or packet.get("dataset_id") != child["dataset_id"]
                or len(packet["rows"]) != child["prepared_count"]
                or encoded > MAX_PREPARED_BYTES or decoded > prepared_decoded_limit(config,
                    legacy_limit=MAX_PREPARED_DECODED_BYTES)):
            raise ValueError("FR3_CAPACITY_PREPARED_CHANGED_OR_BUDGET_EXCEEDED")
        _packet_feature_columns(packet)  # No unpack/copy of feature graphs is needed for counts.
        for row in packet["rows"]:
            symbol, context = row["symbol"], row["context"]
            if (row["row_id"] in seen or symbol not in child["symbols"]
                    or symbol not in by_symbol or row["market"] != by_symbol[symbol]["market"]
                    or context not in contexts):
                raise ValueError("FR3_CAPACITY_DUPLICATE_OR_OUT_OF_SCOPE_ROW")
            seen.add(row["row_id"])
            if len(seen) > MAX_PREPARED_ROWS:
                raise ValueError("FR3_CAPACITY_PREPARED_ROW_BUDGET_EXCEEDED")
            cutoff, label_at = timestamp(row["cutoff"]), timestamp(row["label_available_at"])
            if not cutoff < label_at <= evaluation:
                raise ValueError("FR3_CAPACITY_FUTURE_OR_INVALID_LABEL_TIME")
            utility = row["utility"]
            known = utility is not None
            if known:
                finite(utility, "FR3_CAPACITY_KNOWN_UTILITY")
            windows = [name for name, first, last in folds if first <= cutoff <= last]
            options = dict(utility_known=known, no_fill=row["labels"]["fill"] == 0,
                fit_mature=label_at <= fit, oof_window=bool(windows),
                holdout_mature=start <= cutoff <= end and label_at <= evaluation)
            for counts in (total, contexts[context], by_symbol[symbol]["counts"],
                           by_symbol[symbol]["contexts"][context]):
                _increment(counts, **options)
            for name in windows:
                _increment(by_fold[name], **options)
        for stock in child["coverage"]:
            symbol = stock["symbol"]
            if symbol not in by_symbol or "native_coverage" in by_symbol[symbol]:
                raise ValueError("FR3_CAPACITY_DUPLICATE_OR_OUT_OF_SCOPE_COVERAGE")
            by_symbol[symbol]["native_coverage"] = deepcopy(stock)
            coverage.append(symbol)
        del packet, packet_read
    if (sorted(coverage) != sorted(symbols)
            or total["prepared_rows"] != manifest["counts"]["prepared_rows"]
            or any(by_symbol[symbol]["counts"]["prepared_rows"] !=
                   by_symbol[symbol]["native_coverage"]["prepared_rows"] for symbol in symbols)):
        raise ValueError("FR3_CAPACITY_FULL_SCOPE_OR_COUNT_MISMATCH")
    upper = total["oof_window_fit_mature_utility_upper_bound"]
    minima = dict(holdout=config["min_holdout_rows"], weight_rows=config["min_weight_rows"],
                  each_context=config["min_context_weight_rows"])
    reasons = []
    if total["holdout_mature_rows"] < minima["holdout"]:
        reasons.append("INSUFFICIENT_MATURE_HOLDOUT_NECESSARY_COUNT")
    if upper < minima["weight_rows"]:
        reasons.append("INSUFFICIENT_OOF_WEIGHT_UTILITY_UPPER_BOUND")
    for context in CONTEXTS:
        if contexts[context]["oof_window_fit_mature_utility_upper_bound"] < minima["each_context"]:
            reasons.append("INSUFFICIENT_CONTEXT_WEIGHT_UTILITY_UPPER_BOUND:" + context)
    result = dict(schema=SCHEMA, status=INSUFFICIENT if reasons else SUFFICIENT,
        request_id=request["sha256"], dataset_id=record["sha256"],
        training_config_id=identity(config), components=components,
        dataset_kind=manifest["dataset_kind"], universe=deepcopy(universe), universe_id=identity(universe),
        fixed_universe_count=len(symbols), zero_prepared_symbols=[s for s in symbols if not by_symbol[s]["counts"]["prepared_rows"]],
        counts=total, context_counts=contexts, fold_counts=by_fold,
        stocks=[by_symbol[s] for s in symbols], minima=minima,
        frozen_dates={key: config[key] for key in ("fit_cutoff", "validation_start", "validation_end", "evaluation_cutoff")},
        frozen_folds=deepcopy(config["folds"]),
        count_capacity_allows_fit=not reasons, fit_allowed=False if reasons else None,
        fit_permission_semantics="FALSE_BLOCKS_FIT_NULL_REQUIRES_NATIVE_READINESS_AND_AUTHORIZATION",
        necessary_counts_only=True, ready_to_freeze=False, model_fit_executed=False,
        forecast_or_calibration_executed=False, reason_codes=reasons,
        remaining_fit_requirements=["PAST_ONLY_OOF_HEAD_AND_CALIBRATION_READINESS", "VALID_PREDICTION_SUPPORT",
            "BOTH_CLASSES_FOR_REQUIRED_BINARY_HEADS_AND_CALIBRATORS", "COMPLETE_SIX_FACTOR_ROWS",
            "CONVERGED_ESTIMATED_CONTEXT_WEIGHTS", "NATIVE_HOLDOUT_ACCEPTANCE"],
        native_sources_verified=True,
        safety_evidence_status="NOT_ASSESSED_BY_CAPACITY_AUDIT_NOT_QUALITY_OR_RESTRICTION_PASS",
        no_fill_semantics="VALID_UNFILLED_EVENT_NOT_A_BAD_BAR_OR_ZERO_UTILITY_SUBSTITUTE",
        research_only=True, approved_for_live=False, execution_action="NO_ACTION",
        source_authenticity="NATIVE_ARCHIVE_CONTENT_AND_LINEAGE_NOT_INDEPENDENT_SOURCE_CERTIFICATION")
    return dict(result, capacity_id=identity(result))


def retain_fr3_capacity(report, output_root):
    """Content-addressed no-clobber publication; completion is always last."""
    body = {key: value for key, value in report.items() if key != "capacity_id"}
    sufficient = report.get("status") == SUFFICIENT
    if (report.get("schema") != SCHEMA or report.get("capacity_id") != identity(body)
            or report.get("status") not in {SUFFICIENT, INSUFFICIENT}
            or report.get("necessary_counts_only") is not True
            or report.get("model_fit_executed") is not False
            or report.get("forecast_or_calibration_executed") is not False
            or report.get("ready_to_freeze") is not False
            or report.get("research_only") is not True or report.get("approved_for_live") is not False
            or report.get("execution_action") != "NO_ACTION"
            or report.get("count_capacity_allows_fit") is not sufficient
            or sufficient and report.get("fit_allowed") is not None
            or not sufficient and report.get("fit_allowed") is not False):
        raise ValueError("FR3_CAPACITY_REPORT_HASH_OR_CONTRACT_INVALID")
    folder = Path(output_root) / report["capacity_id"]
    write_once(folder / "capacity.json", report)
    completion = dict(schema="les-v7-fr3-capacity-completion/v1", capacity_id=report["capacity_id"],
        status=report["status"], fit_allowed=report["fit_allowed"],
        necessary_counts_only=True, model_fit_executed=False, research_only=True,
        approved_for_live=False, execution_action="NO_ACTION")
    write_once(folder / "completion.json", completion)
    return folder, completion

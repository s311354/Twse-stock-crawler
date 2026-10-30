"""Bounded FR3 source/coverage and native comparison adapters.

This module is not a second model, Quality engine or account simulator.  A
changed limit plan changes execution labels: FR2-vs-FR3 is therefore a version
comparison, not the existing same-plan 2x2 experiment.  The same-plan FR3 M0/M1
comparison reuses native forecast/cost arithmetic.  Unknown evidence and absent
executed account paths remain NOT_ASSESSABLE, never a zero-return backtest.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from copy import deepcopy
from datetime import date, datetime, timezone
from math import isfinite
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from domain.les_v7.contracts import identity, timestamp
from infrastructure.les_v7_dataset_scope import _load_bound_scope_shard, _verify_freeze
from infrastructure.les_v7_training_source import FINAL_REFERENCE, _calendar
from infrastructure.research_documents import EvidenceDocuments

SCHEMA = "les-v7-fr3-research-comparison/v1"
AUDIT_SCHEMA = "les-v7-fr3-scope-source-audit/v1"
MAX_REPORTS = 100_000
TAIPEI = ZoneInfo("Asia/Taipei")


def fr3_reference_versions(reference_policy):
    """Exact native plan/path/label identity, never a filename or old default."""
    from domain.les_v7.fr3_plan import require_policy
    reference = require_policy(reference_policy)
    path = reference["cost_path_policy"]
    label = path.get("label_version")
    if not label or label != reference["structural_rules"].get("label_version"):
        raise ValueError("FR3_COMPARISON_PLAN_PATH_LABEL_BINDING_MISMATCH")
    return dict(reference_policy_id=identity(reference), reference_plan_version=reference["version"],
        path_version=path["version"], label_version=label)


def _seal(payload):
    return dict(payload=payload, sha256=identity(payload))


def _fixed_universe(values):
    supplied = list(values)
    if (not supplied or len(supplied) > 2048 or supplied != sorted(set(supplied))
            or any(not isinstance(value, str) or not value or value != value.strip() for value in supplied)):
        raise ValueError("FR3_FIXED_ORDERED_UNIVERSE_REQUIRED")
    return supplied


def inventory_local_safety_sources(universe, *, mops_cache, restriction_cache, cutoffs):
    """Read database headers only; distinguish absent-in-shard from absent-local.

    No raw financial or evidence JSON is expanded.  One read-only transaction
    per database retains consistent metadata.  Counts establish neither verified
    original publication times, version comparability nor native gate PASS.
    A record acquired later cannot become eligible by its fiscal period alone.
    """
    symbols = _fixed_universe(universe)
    supplied = [timestamp(value) for value in cutoffs]
    if not supplied or supplied != sorted(set(supplied)):
        raise ValueError("FR3_ORDERED_EXPLICIT_SAFETY_CUTOFFS_REQUIRED")
    marks = ",".join("?" for _ in symbols)
    mops, restriction = Path(mops_cache).resolve(), Path(restriction_cache).resolve()
    result = dict(schema="les-v7-fr3-local-safety-inventory/v1", universe=symbols,
        universe_id=identity(symbols), financial_database_present=mops.is_file(),
        restriction_database_present=restriction.is_file(), financial_cutoffs=[],
        original_documents_read=0, source_metadata_only=True,
        native_gate_verdicts_emitted=0, verified_original_publication_claim=False,
        financial_original_integrity="NOT_REVERIFIED_BY_METADATA_INVENTORY",
        research_only=True, execution_action="NO_ACTION", approved_for_live=False)
    if mops.is_file():
        with closing(sqlite3.connect(mops.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("BEGIN")
            cursor = connection.execute("SELECT symbol,available_at_epoch,fetched_at,availability_source "
                "FROM filings WHERE symbol IN (" + marks + ")", symbols)
            records = cursor.fetchmany(50_001)
            if len(records) > 50_000:
                raise ValueError("FR3_SAFETY_METADATA_RECORD_BUDGET_EXCEEDED")
        source_counts = Counter(row[3] for row in records)
        result.update(financial_record_count=len(records), financial_symbols=len({row[0] for row in records}),
            financial_availability_sources=dict(source_counts))
        for at in supplied:
            eligible, invalid = [], 0
            for symbol, available, fetched, source in records:
                try:
                    observed = timestamp(fetched)
                    if type(available) not in (int, float) or not isfinite(available):
                        raise ValueError("MISSING_AVAILABLE_AT_EPOCH")
                    if available <= at.timestamp() and observed <= at:
                        eligible.append(symbol)
                except (ValueError, TypeError, OverflowError):
                    invalid += 1
            present = set(eligible)
            result["financial_cutoffs"].append(dict(cutoff=at.isoformat(),
                observed_available_record_count=len(eligible), symbols_with_some_records=len(present),
                no_observed_available_records=sorted(set(symbols) - present), invalid_time_records=invalid,
                semantics="BOTH_STORED_AVAILABILITY_AND_REAL_FIRST_OBSERVATION_PRECEDE_CUTOFF_NOT_FULL_QUALITY_COVERAGE"))
    else:
        result.update(financial_record_count=None, financial_symbols=None, financial_availability_sources=None)
    if restriction.is_file():
        with closing(sqlite3.connect(restriction.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("BEGIN")
            records = connection.execute("SELECT market,COUNT(*),COUNT(DISTINCT symbol),"
                "MIN(available_at_ns),MAX(available_at_ns) FROM restriction_observations "
                "WHERE symbol IN (" + marks + ") GROUP BY market", symbols).fetchall()
            for at in supplied:
                delta = at - datetime(1970, 1, 1, tzinfo=timezone.utc)
                cutoff_ns = (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000
                count = connection.execute("SELECT COUNT(*) FROM restriction_observations "
                    "WHERE symbol IN (" + marks + ") AND available_at_ns<=?",
                    [*symbols, cutoff_ns]).fetchone()[0]
                result.setdefault("restriction_cutoffs", []).append(dict(cutoff=at.isoformat(),
                    observed_record_count=count, native_status="NOT_EVALUATED"))
        result["restriction_metadata"] = [dict(market=market, records=count, symbols=covered,
            first_observed_at=datetime.fromtimestamp(first / 1e9, timezone.utc).isoformat(),
            last_observed_at=datetime.fromtimestamp(last / 1e9, timezone.utc).isoformat())
            for market, count, covered, first, last in records]
    else:
        result.update(restriction_metadata=None, restriction_cutoffs=None)
    return dict(result, inventory_id=identity(result))


def audit_fr3_scope_sources(freeze_path, *, evaluation_cutoff, inspected_sessions=(),
                          allow_synthetic=False):
    """Fully verify each raw shard once, retaining every fixed-scope security.

    Presence is not cutoff eligibility or source certification.  Financial and
    restriction gates are deliberately not emitted by an inventory.  The native
    research engine must select/verify those operands at each requested cutoff.
    Final official calendars retain their explicitly retrospective convention.
    """
    path = Path(freeze_path).resolve()
    root = path.parent.parent.parent
    record = _verify_freeze(path, root, verify_sources=False)
    body = record["body"]
    if body["dataset_kind"] == "SYNTHETIC_TEST_ONLY" and not allow_synthetic:
        raise ValueError("FR3_SYNTHETIC_SOURCE_REQUIRES_EXPLICIT_OPT_IN")
    universe = _fixed_universe(body["universe"])
    evaluated = timestamp(evaluation_cutoff)
    inspected = list(inspected_sessions)
    if inspected != sorted(set(inspected)):
        raise ValueError("FR3_UNIQUE_ORDERED_INSPECTED_SESSIONS_REQUIRED")
    for value in inspected:
        date.fromisoformat(value)
    rows, calendar, calendar_audit = [], None, None
    source_ids = []
    for chunk in body["chunks"]:
        source_path, evidence_id, source = _load_bound_scope_shard(body, root, chunk)
        documents = EvidenceDocuments.for_evidence(source_path, evidence_id)
        documents.verify(dict(calendar=source["calendar"],
            actions=source.get("corporate_actions_final"), market=source.get("market_supplement_final")))
        source_ids.append(evidence_id)
        if calendar is None:
            first, last = date.fromisoformat(body["history_start"]), date.fromisoformat(body["history_end"])
            if body.get("reference_evidence_policy") == FINAL_REFERENCE:
                from infrastructure.les_v7_calendar_final import select_final_calendar
                calendar, calendar_audit = select_final_calendar(source["calendar"], evaluated,
                    documents=documents, start=first, end=last, evidence_id=evidence_id)
            else:
                calendar, calendar_audit = _calendar(source, evaluated, documents, first, last, evidence_id)
        for symbol in chunk["symbols"]:
            filings = source.get("filings", {}).get(symbol, ())
            restriction_count = sum(item.get("symbol") == symbol for item in source.get("restrictions", ()))
            reasons = []
            if not filings:
                reasons.append("HISTORICAL_FINANCIAL_ORIGINALS_NOT_IN_SOURCE_SHARD")
            if not restriction_count and source.get("restriction_ledger") is None:
                reasons.append("HISTORICAL_RESTRICTION_EVIDENCE_NOT_IN_SOURCE_SHARD")
            if not body["original_observation_receipts_available"]:
                reasons.append("ORIGINAL_OPEN_QUOTE_OBSERVATION_PROOF_NOT_RETAINED")
            rows.append(dict(symbol=symbol, market=body["market"],
                security_type=source["security_classifications"][symbol]["security_type"],
                source_evidence_id=evidence_id, financial_record_count=len(filings),
                restriction_record_count=restriction_count,
                restriction_ledger_present=source.get("restriction_ledger") is not None,
                kelly_calibration_record_count=len(source.get("kelly_calibrations", ())),
                native_gate_status="NOT_EVALUATED", source_presence_not_eligibility=True,
                reason_codes=reasons))
        del source, documents
    if [row["symbol"] for row in rows] != universe:
        raise ValueError("FR3_SCOPE_AUDIT_COVERAGE_MISMATCH")
    # Recheck the manifest after reading shards; no changed source identity may
    # hide behind a previously verified first read.
    if _verify_freeze(path, root, verify_sources=False) != record:
        raise ValueError("FR3_SCOPE_AUDIT_FREEZE_CHANGED")
    sessions = [day for day in calendar["sessions"]
        if timestamp(day + "T13:30:00+08:00") <= evaluated]
    matured = sessions[:-20] if len(sessions) > 20 else []
    after_inspected = [day for day in matured if not inspected or day > inspected[-1]]
    report = dict(schema=AUDIT_SCHEMA, freeze_id=record["freeze_id"], scope_id=body["scope_id"],
        dataset_kind=body["dataset_kind"], universe=universe, universe_id=identity(universe),
        universe_policy=body["universe_record"]["selection_policy"],
        universe_snapshot_day=body["universe_record"]["source_day"],
        history_start=body["history_start"], history_end=body["history_end"],
        evaluation_cutoff=evaluated.isoformat(), source_evidence_ids=source_ids,
        sources_verified=True, historical_information_set_certified=False,
        calendar_policy=body.get("reference_evidence_policy", "CUTOFF_VERIFIED_OFFICIAL_CALENDAR"),
        calendar_evidence_ids=calendar["evidence_ids"], calendar_audit=calendar_audit,
        official_sessions=sessions, h20_matured_sessions=matured,
        last_h20_matured_session=matured[-1] if matured else None,
        previously_inspected_sessions=inspected,
        matured_sessions_after_last_inspected=after_inspected,
        fresh_confirmatory_status="NOT_ASSESSABLE",
        fresh_confirmatory_reason="SEPARATE_PREDECLARED_UNSEEN_PROTOCOL_REQUIRED",
        stocks=rows, counts=dict(fixed_universe=len(universe), raw_shards=len(source_ids),
            financial_records=sum(row["financial_record_count"] for row in rows),
            restriction_records=sum(row["restriction_record_count"] for row in rows),
            symbols_without_financial_records=sum(not row["financial_record_count"] for row in rows),
            symbols_without_restriction_records_or_ledger=sum(not row["restriction_record_count"]
                and not row["restriction_ledger_present"] for row in rows)),
        status="SOURCE_INVENTORY_COMPLETED_NOT_NATIVE_ADMISSION_OR_ACCOUNT_PERFORMANCE",
        research_only=True, approved_for_live=False, execution_action="NO_ACTION", order_authority=False)
    return dict(report, audit_id=identity(report))


def compare_fr3_predictions(rows, config, baseline, fractal):
    """Recompute native five-head M0/M1 predictions on identical FR3 labels.

    Callers must supply newly rebuilt FR3 rows and fitted FR3 counterparts; old
    FR2 plan labels cannot be renamed into the new execution coordinate system.
    This retains the native statistical acceptance, including insufficient-CI
    statuses, and never infers account or PAE returns from one-share outcomes.
    """
    from domain.les_v7.fr3_plan import require_policy
    from infrastructure.les_v7_fr2_expansion import compare_prepared
    reference = require_policy(config["reference_policy"])
    version = reference["version"]
    if (not rows or any(row["plan"].policy_version != version for row in rows)
            or baseline.get("reference_policy") != reference or fractal.get("reference_policy") != reference):
        raise ValueError("FR3_NEW_PLAN_LABELS_AND_MATCHED_MODEL_PROFILES_REQUIRED")
    result = compare_prepared(rows, config, baseline, fractal)
    result.update(schema="les-v7-fr3-paired-forecast-comparison/v1",
        reference_policy_id=identity(reference),
        after_cost_account_status="NOT_ASSESSABLE",
        after_cost_account_reason="SOURCE_BOUND_EXECUTED_ACCOUNTS_REQUIRED",
        historical_deployment_claim=False, research_only=True, approved_for_live=False)
    from domain.les_v7.fr3_plan import PLAN_VERSION
    if version != PLAN_VERSION:
        result.update(fr3_reference_versions(reference),
            model_pair_scope="FR3_L2_M0_M1_SAME_NEW_PLAN_LABELS_OWN_PAST_CALIBRATION_AND_WEIGHTS")
    return result


def retain_fr3_source_audit(report, output_root):
    """Retain source inventory separately from predictive/economic conclusions."""
    from infrastructure.research_archive import write_once
    body = {key: value for key, value in report.items() if key != "audit_id"}
    if report.get("schema") != AUDIT_SCHEMA or report.get("audit_id") != identity(body):
        raise ValueError("FR3_SOURCE_AUDIT_HASH_OR_SCHEMA_INVALID")
    folder = Path(output_root) / report["audit_id"]
    write_once(folder / "source-audit.json", report)
    completion = dict(schema="les-v7-fr3-source-audit-completion/v1", audit_id=report["audit_id"],
        status="SOURCE_INVENTORY_COMPLETED_NOT_RESEARCH_OR_PERFORMANCE_ACCEPTANCE",
        research_only=True, approved_for_live=False, execution_action="NO_ACTION")
    write_once(folder / "completion.json", completion)
    return folder, completion


def fr3_reference_cost_diagnostic(root, dataset, config, validation):
    """Validate original native fee arithmetic; do not turn labels into returns."""
    from domain.les_v7.fr3_plan import require_policy
    from infrastructure.les_v7_fr2_archive import _reference_costs
    reference = require_policy(config["reference_policy"])
    if any(row["plan"].policy_version != reference["version"] for row in validation):
        raise ValueError("FR3_REFERENCE_COST_REQUIRES_NEW_PLAN_LABELS")
    summary, rows = _reference_costs(Path(root).resolve(), dataset, config, validation)
    summary.update(reference_policy_id=identity(reference), reference_plan_version=reference["version"],
        scope="NATIVE_FR3_REFERENCE_PATHS_NOT_ALLOCATION_OR_ACCOUNT_PERFORMANCE")
    return summary, rows


def _coverage(dataset):
    universe = _fixed_universe([row[0] for row in dataset["universe"]])
    rows = {}
    for child in dataset["children"]:
        for row in child["coverage"]:
            symbol = row["symbol"]
            if symbol in rows or symbol not in universe:
                raise ValueError("FR3_TRAINING_COVERAGE_DUPLICATE_OR_OUT_OF_SCOPE")
            rows[symbol] = deepcopy(row)
    if set(rows) != set(universe):
        raise ValueError("FR3_TRAINING_COVERAGE_INCOMPLETE")
    return rows


def _reports(values, universe, *, kind, validator=None):
    """Do not count an unverified decision string as a native research result."""
    from domain.les_v7.report import validate_report
    result, observed = {}, set()
    supplied = list(values)
    if len(supplied) > MAX_REPORTS:
        raise ValueError("FR3_BOUNDED_RESEARCH_REPORTS_REQUIRED")
    for item in supplied:
        if kind == "FR3":
            if item.get("schema") != "les-v7-fr3-report/v1":
                raise ValueError("FR3_NATIVE_REPORT_SCHEMA_REQUIRED")
            body = {key: value for key, value in item.items() if key != "report_id"}
            if item.get("report_id") != identity(body):
                raise ValueError("FR3_NATIVE_REPORT_HASH_MISMATCH")
            row = dict(symbol=item["symbol"], market=item["market"], cutoff=timestamp(item["cutoff"]).isoformat(),
                evidence_id=item["evidence_id"], report_id=item["report_id"], model_id=item["model_id"],
                decision=item["decision"]["strategy_decision"], score=deepcopy(item["score"]),
                plan_id=identity(item["plan"]) if item.get("plan") is not None else None)
            if validator is not None:
                # The public FR3 validator re-executes against the caller's
                # separately bound request/model; a hash check alone is not it.
                validator(item)
            row["verification"] = "NATIVE_REEXECUTION_VERIFIED" if validator else "UNVERIFIED_REPORT_HASH_ONLY"
        else:
            validate_report(item)
            if validator is not None:
                validator(item)
            event, strategy = item["envelope"]["event"], item["envelope"]["strategy"]
            row = dict(symbol=event["security_id"], market=item["diagnostics"]["gate_bindings"]["SAFETY"]["market"],
                cutoff=timestamp(event["cutoff_at"]).isoformat(), evidence_id=event["evidence_id"],
                report_id=identity(item), model_id=item["envelope"]["model"].get("bundle_sha256"),
                decision=strategy["strategy_decision"], score=deepcopy(item["envelope"]["score"]),
                plan_id=event.get("plan_id"), verification="NATIVE_REPORT_VALIDATED")
        key = row["symbol"], row["market"], row["cutoff"]
        if row["symbol"] not in universe or key in observed:
            raise ValueError("FR3_REPORT_DUPLICATE_OR_OUT_OF_SCOPE")
        observed.add(key)
        result.setdefault(row["symbol"], []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: (row["cutoff"], row["market"]))
    return result


def _compare_verified_inputs(original, old_dataset, outcome, new_dataset, *, source_audit,
                             old_reports=(), new_reports=(), old_report_validator=None,
                             new_report_validator=None, forecast_comparison=None,
                             reference_costs=None, after_cost_comparison=None,
                             old_reference=None, new_reference=None):
    """Internal aggregation after the filesystem verifiers, also fixture-testable."""
    universe = _fixed_universe([row[0] for row in old_dataset["universe"]])
    if (new_dataset["universe"] != old_dataset["universe"] or outcome["symbols"] != universe
            or source_audit.get("universe") != universe
            or source_audit.get("audit_id") != identity({k: v for k, v in source_audit.items() if k != "audit_id"})):
        raise ValueError("FR3_COMPLETE_IDENTICAL_FIXED_UNIVERSE_REQUIRED")
    from domain.les_v7.fr3_plan import PLAN_VERSION, is_fr3_reference
    new_versions = fr3_reference_versions(new_reference) if new_reference is not None else None
    l2 = new_versions is not None and new_versions["reference_plan_version"] != PLAN_VERSION
    old_fr3 = l2 and is_fr3_reference(old_reference)
    old_label, new_label = ("FR3_R1" if old_fr3 else "FR2", "FR3_L2") if l2 else ("FR2", "FR3")
    if l2:
        if not isinstance(old_reference, dict) or old_reference.get("version") == new_reference["version"]:
            raise ValueError("FR3_L2_DISTINCT_PARENT_REFERENCE_VERSION_REQUIRED")
        old_reports, new_reports = list(old_reports), list(new_reports)
        for values, model_id, reference_version in ((old_reports, original["sha256"], old_reference["version"]),
                (new_reports, outcome["native_outcome"].get("bundle_sha256"), new_reference["version"])):
            if values and model_id is None:
                raise ValueError("FR3_L2_REPORT_MODEL_ID_MISSING")
            for item in values:
                if item.get("model_id", item.get("envelope", {}).get("model", {}).get("bundle_sha256")) != model_id:
                    raise ValueError("FR3_L2_REPORT_MODEL_ID_MISMATCH")
                if item.get("schema") == "les-v7-fr3-report/v1":
                    plan = item.get("plan")
                    bound = item.get("components", {}).get("entry", {}).get("policy", {}).get("reference_plan_version")
                    if ((plan is not None and plan.get("policy_version") != reference_version)
                            or reference_version == new_reference["version"] and bound != reference_version):
                        raise ValueError("FR3_L2_REPORT_REFERENCE_VERSION_MISMATCH")
    old_coverage, new_coverage = _coverage(old_dataset), _coverage(new_dataset)
    old = _reports(old_reports, universe, kind="FR3" if old_fr3 else "FR2", validator=old_report_validator)
    new = _reports(new_reports, universe, kind="FR3", validator=new_report_validator)
    rows, counters = [], {old_label: Counter(), new_label: Counter()}
    for symbol in universe:
        left, right = old.get(symbol, []), new.get(symbol, [])
        for label, reports in ((old_label, left), (new_label, right)):
            counters[label].update(row["decision"] for row in reports
                if not row["verification"].startswith("UNVERIFIED"))
        exact = {(row["market"], row["cutoff"], row["evidence_id"]): row for row in left}
        paired = []
        for row in right:
            match = exact.get((row["market"], row["cutoff"], row["evidence_id"]))
            if match is not None:
                paired.append(dict(cutoff=row["cutoff"], old_report_id=match["report_id"],
                    new_report_id=row["report_id"], same_plan=(match["plan_id"] == row["plan_id"]
                        and (not l2 or row["plan_id"] is not None)),
                    decision_comparison="VERSIONED_POLICY_COMPARISON_NOT_SAME_PLAN_ABLATION",
                    verification="UNVERIFIED" if any(item["verification"].startswith("UNVERIFIED")
                        for item in (match, row)) else "VERIFIED_REPORTS"))
        rows.append(dict(symbol=symbol, market=old_coverage[symbol]["market"],
            old=dict(coverage=old_coverage[symbol], reports=left,
                decision_status="NOT_EVALUATED" if not left else
                    "UNVERIFIED" if old_fr3 and any(row["verification"].startswith("UNVERIFIED") for row in left)
                    else "NATIVE_REEXECUTION_VERIFIED" if old_fr3 else "NATIVE_REPORT_VALIDATED"),
            new=dict(coverage=new_coverage[symbol], reports=right,
                decision_status="NOT_EVALUATED" if not right else
                    "UNVERIFIED" if any(row["verification"].startswith("UNVERIFIED") for row in right) else "NATIVE_REEXECUTION_VERIFIED"),
            matched_cutoff_evidence_reports=paired))
    reasons = []
    if any(not row["old"]["reports"] for row in rows):
        reasons.append("MISSING_" + old_label + "_NATIVE_RESEARCH_REPORTS_FOR_FIXED_SCOPE")
    if old_fr3 and any(row["old"]["decision_status"] != "NATIVE_REEXECUTION_VERIFIED" for row in rows):
        reasons.append("MISSING_OR_UNVERIFIED_FR3_R1_NATIVE_RESEARCH_REPORTS_FOR_FIXED_SCOPE")
    if any(row["new"]["decision_status"] != "NATIVE_REEXECUTION_VERIFIED" for row in rows):
        reasons.append("MISSING_OR_UNVERIFIED_" + new_label + "_NATIVE_RESEARCH_REPORTS_FOR_FIXED_SCOPE")
    if source_audit["counts"]["symbols_without_financial_records"]:
        reasons.append("HISTORICAL_QUALITY_FINANCIAL_ORIGINALS_NOT_IN_FIXED_SOURCE")
    if source_audit["counts"]["symbols_without_restriction_records_or_ledger"]:
        reasons.append("HISTORICAL_TRADING_RESTRICTION_EVIDENCE_NOT_IN_FIXED_SOURCE")
    if source_audit.get("fresh_confirmatory_status") != "PASS":
        reasons.append("FRESH_CONFIRMATORY_HOLDOUT_NOT_ESTABLISHED")
    if outcome["status"] != "READY_TO_FREEZE":
        reasons.append("FR3_NATIVE_TRAINING_NOT_READY")
    if forecast_comparison is None:
        reasons.append("FR3_MATCHED_M0_M1_PREDICTIONS_NOT_RECOMPUTED")
    if after_cost_comparison is None:
        reasons.append("FOUR_SOURCE_BOUND_EXECUTED_AFTER_COST_ACCOUNT_PATHS_REQUIRED")
    result = dict(schema=SCHEMA, status="NOT_ASSESSABLE" if reasons else
            "PASS" if after_cost_comparison.get("status") == "PASS" else after_cost_comparison["status"],
        comparison_kind="FR2_VS_FR3_VERSIONED_PLAN_AND_POLICY_NOT_FRACTAL_ONLY_ABLATION",
        source_bindings=dict(original_bundle_id=original["sha256"], fr3_request_id=outcome["request_id"],
            fr3_native_job_id=outcome["native_outcome"]["native_job_id"],
            fr3_training_dataset_id=outcome["native_outcome"]["dataset_id"],
            fr3_bundle_id=outcome["native_outcome"].get("bundle_sha256"),
            source_audit_id=source_audit["audit_id"], original_bundle_modified=False),
        universe=universe, universe_id=identity(universe), fixed_universe_count=len(universe), stocks=rows,
        verified_decision_counts={key: dict(value) if value else None for key, value in counters.items()},
        zero_count_is_not_missing_result=True,
        native_training_status=outcome["status"],
        same_plan_fr3_forecast_comparison=deepcopy(forecast_comparison),
        reference_cost_diagnostic=deepcopy(reference_costs),
        reference_costs_are_not_account_returns=True,
        after_cost_comparison=deepcopy(after_cost_comparison) if after_cost_comparison else
            dict(status="NOT_ASSESSABLE", reason="FOUR_SOURCE_BOUND_EXECUTED_AFTER_COST_ACCOUNT_PATHS_REQUIRED"),
        reason_codes=reasons, historical_information_set_certified=False,
        source_authenticity="CONTENT_INTEGRITY_AND_NATIVE_COMPUTATION_NOT_EXTERNAL_SOURCE_CERTIFICATION",
        research_only=True, approved_for_live=False, execution_action="NO_ACTION", order_authority=False)
    if l2:
        result.update(comparison_kind=old_label + "_VS_FR3_L2_VERSIONED_PLAN_PATH_LABELS_NOT_FRACTAL_ONLY_ABLATION",
            model_versions={old_label: dict(reference_policy_id=identity(old_reference),
                reference_plan_version=old_reference["version"],
                path_version=old_reference["cost_path_policy"]["version"],
                label_version=old_reference["cost_path_policy"].get("label_version")), new_label: new_versions},
            policy_version_changes_are_not_same_label_ablation=True)
    return result


def compare_fr3_four_arms(record, *, reference_policy_id):
    """Reuse the fixed acceptance kernel on one common *new* FR3 plan only.

    P0/P1 here denote admission variants of the same newly trained M0/M1 pair,
    not the old FR2 predictor.  Cost/account source authenticity still belongs
    to the native producer, exactly as in the existing public comparison port.
    """
    from domain.les_v7.fr2_comparison import SCHEMA as KERNEL_SCHEMA, compare
    if (not isinstance(record, dict) or record.get("sha256") != identity(record.get("payload"))
            or record["payload"].get("schema") != "les-v7-fr3-comparison-input/v1"):
        raise ValueError("FR3_FOUR_ARM_INPUT_HASH_OR_SCHEMA_INVALID")
    body = deepcopy(record["payload"])
    if (body["conditions"].get("reference_policy_id") != reference_policy_id
            or any(model["payload"].get("reference_policy_id") != reference_policy_id
                   for model in body["models"].values())):
        raise ValueError("FR3_FOUR_ARMS_REQUIRE_ONE_NEW_REFERENCE_POLICY")
    body["schema"] = KERNEL_SCHEMA
    result = compare(_seal(body))
    result.update(schema="les-v7-fr3-four-arm-comparison/v1",
        original_input_id=record["sha256"], acceptance_kernel="FR_ACCEPT_R2_UNCHANGED",
        model_pair_scope="NEW_FR3_M0_M1_SAME_PLAN_LABELS_NOT_OLD_FR2_PREDICTOR",
        historical_deployment_claim=False)
    return result


def _bind_source_audit_to_dataset(source_audit, dataset, training_config):
    """Bind the existing v1 inventory to this exact verified native input.

    A common universe does not identify the same source history. Existing v1
    audits already retain these fields; missing fields or other schemas are
    rejected rather than inferred from dates, model names or today's cache.
    This binding does not turn inventory presence into cutoff eligibility.
    """
    if (not isinstance(source_audit, dict) or source_audit.get("schema") != AUDIT_SCHEMA
            or source_audit.get("audit_id") != identity({key: value for key, value in source_audit.items()
                                                       if key != "audit_id"})):
        raise ValueError("FR3_SOURCE_AUDIT_HASH_OR_SCHEMA_INVALID")
    try:
        matches = (source_audit.get("sources_verified") is True
            and source_audit.get("freeze_id") == dataset["freeze_id"]
            and source_audit.get("scope_id") == dataset["scope_id"]
            and source_audit.get("dataset_kind") == dataset["dataset_kind"]
            and source_audit.get("source_evidence_ids") == dataset["source_evidence_ids"]
            and timestamp(source_audit["evaluation_cutoff"]) == timestamp(training_config["evaluation_cutoff"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("FR3_SOURCE_AUDIT_NATIVE_BINDING_MISMATCH") from error
    if not matches:
        raise ValueError("FR3_SOURCE_AUDIT_NATIVE_BINDING_MISMATCH")


def compare_fr3_research(original_archive, fr3_output_root, *, source_audit,
                         old_reports=(), new_reports=(), old_report_validator=None,
                         new_report_validator=None, forecast_comparison=None,
                         reference_costs=None, after_cost_input=None):
    """Verify original/new native archives, then retain all fixed-scope coverage.

    An outcome with no published bundle is valid NOT_READY evidence, not an
    excuse to reuse old coefficients.  Native reports are optional; absence is
    explicitly NOT_EVALUATED and is never counted as zero BUY signals.
    """
    from infrastructure.les_v7_fr3_training import REQUEST_SCHEMA, verify_fr3_scope_training
    from infrastructure.storage.les_v7_training import LesV7TrainingRepository, MAX_DATASET_BYTES, read_json_bounded
    original_path = Path(original_archive).resolve()
    if original_path.name != "frozen_bundle.json":
        raise ValueError("FR3_ORIGINAL_FROZEN_ARCHIVE_REQUIRED")
    original_receipt = LesV7TrainingRepository.verify(original_path.parent)
    original = read_json_bounded(original_path, MAX_DATASET_BYTES)
    old_record = read_json_bounded(original_path.parent / "dataset.json", MAX_DATASET_BYTES)
    outcome = verify_fr3_scope_training(fr3_output_root)
    new_record = read_json_bounded(Path(outcome["native_root"]) / "scope-training-dataset.json", MAX_DATASET_BYTES)
    if (original.get("sha256") != identity(original.get("payload"))
            or original["sha256"] != original_receipt["bundle_sha256"]
            or old_record.get("sha256") != identity(old_record.get("payload"))
            or old_record["sha256"] != original_receipt["dataset_id"]
            or new_record.get("sha256") != identity(new_record.get("payload"))
            or new_record["sha256"] != outcome["native_outcome"]["dataset_id"]):
        raise ValueError("FR3_COMPARISON_NATIVE_ARCHIVE_BINDING_MISMATCH")
    request = read_json_bounded(Path(fr3_output_root) / "fr3-training-request.json", MAX_DATASET_BYTES)
    request_body = request.get("payload", {})
    config = request_body.get("training_config") if isinstance(request_body, dict) else None
    if (not isinstance(request_body, dict) or request_body.get("schema") != REQUEST_SCHEMA
            or request.get("sha256") != identity(request_body)
            or request["sha256"] != outcome["request_id"] or not isinstance(config, dict)
            or identity(config) != new_record["payload"].get("training_config_id")
            or identity(config) != outcome["native_outcome"].get("config_id")):
        raise ValueError("FR3_COMPARISON_TRAINING_REQUEST_MISMATCH")
    _bind_source_audit_to_dataset(source_audit, new_record["payload"], config)
    costs = None
    if after_cost_input is not None:
        reference_id = identity(config["reference_policy"])
        costs = compare_fr3_four_arms(after_cost_input, reference_policy_id=reference_id)
    return _compare_verified_inputs(original, old_record["payload"], outcome, new_record["payload"],
        source_audit=source_audit, old_reports=old_reports, new_reports=new_reports,
        old_report_validator=old_report_validator, new_report_validator=new_report_validator,
        forecast_comparison=forecast_comparison, reference_costs=reference_costs,
        after_cost_comparison=costs, old_reference=original["payload"].get("reference_policy"),
        new_reference=config.get("reference_policy"))


def retain_fr3_comparison(report, output_root):
    """Write-once content-addressed comparison, with completion published last."""
    from infrastructure.research_archive import write_once
    if report.get("schema") != SCHEMA or report.get("research_only") is not True:
        raise ValueError("FR3_RESEARCH_COMPARISON_REQUIRED")
    body = deepcopy(report)
    comparison_id = identity(body)
    folder = Path(output_root) / comparison_id
    write_once(folder / "comparison.json", body)
    completion = dict(schema="les-v7-fr3-comparison-completion/v1", comparison_id=comparison_id,
        status=body["status"], research_only=True, approved_for_live=False, execution_action="NO_ACTION")
    write_once(folder / "completion.json", completion)
    return folder, completion

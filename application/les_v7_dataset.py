"""Fixed-universe reference data construction, not account/PAE simulation."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta

import pandas as pd

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.reference_path import evaluate_reference_path, path_window, validate_reference_path_policy
from domain.les_v7.training import (DATASET_SCHEMA, prepare_events, prepare_reference,
                                  reference_config_id, validate_reference_config)
from domain.les_v7.reference_memo import _bind_reference_history
from domain.security_identity import resolve_security_identity
from domain.les_v7.fr3_plan import is_fr3_reference

BUILDER_VERSION = "LES_V7_REFERENCE_DATASET_BUILDER_R1"


def validate_build_config(config):
    if config.get("schema") != "les-v7-dataset-build-config/v1":
        raise ValueError("DATASET_BUILD_CONFIG_SCHEMA_REQUIRED")
    start, end = date.fromisoformat(config["event_start"]), date.fromisoformat(config["event_end"])
    if start > end or (end - start).days >= 3660:
        raise ValueError("BOUNDED_DATASET_EVENT_RANGE_REQUIRED")
    if timestamp(config["evaluation_cutoff"]) <= timestamp(config["event_end"] + "T13:30:00+08:00"):
        raise ValueError("DATASET_EVALUATION_MUST_FOLLOW_EVENT_RANGE")
    if config.get("event_sampling") != "ALL_FIXED_UNIVERSE_OFFICIAL_SESSIONS":
        raise ValueError("NO_WINNER_OR_BUY_SIGNAL_EVENT_FILTER_ALLOWED")
    if type(config["max_events"]) is not int or not 1 <= config["max_events"] <= 4096:
        raise ValueError("BOUNDED_DATASET_EVENT_COUNT_REQUIRED")
    validate_reference_config(config["seed"])
    policy = validate_reference_path_policy(config["path_policy"])
    if config.get("reference_evidence_policy") not in {None, "OFFICIAL_FINAL_REFERENCE_RECONSTRUCTION_R1"}:
        raise ValueError("INVALID_REFERENCE_EVIDENCE_POLICY")
    if ((config.get("reference_evidence_policy") == "OFFICIAL_FINAL_REFERENCE_RECONSTRUCTION_R1")
            != (policy["coordinate_policy"] == "OFFICIAL_FINAL_ACTIONS_RECONSTRUCTION_R1")):
        raise ValueError("FINAL_REFERENCE_COORDINATE_POLICY_MISMATCH")
    if config["seed"]["reference_policy"].get("cost_path_policy") != policy:
        raise ValueError("REFERENCE_PLAN_AND_EXECUTION_COST_POLICY_MISMATCH")
    coverages = config.get("coordinate_coverages", [])
    if not isinstance(coverages, list) or len(coverages) > 4096:
        raise ValueError("BOUNDED_COORDINATE_COVERAGES_REQUIRED")
    identity(config)


def build_training_dataset(source_record, config):
    """Retain all candidate diagnostics; only evidence-ready events get labels.

    Observed bar dates on an UNKNOWN calendar are diagnostic candidates, NOT
    declarations that those dates were officially OPEN. A partial dataset is
    valid to retain, but never means the trainer should relax its requirements.
    """
    validate_build_config(config)
    source = source_record["payload"]
    if source.get("schema") != "les-v7-training-source/v1" or identity(source) != source_record["sha256"]:
        raise ValueError("TRAINING_SOURCE_PACKET_HASH_OR_SCHEMA_MISMATCH")
    if source.get("dataset_kind") not in {"RESEARCH", "SYNTHETIC_TEST_ONLY"}:
        raise ValueError("EXPLICIT_DATASET_SOURCE_KIND_REQUIRED")
    if timestamp(source["lineage"]["evaluation_cutoff"]) != timestamp(config["evaluation_cutoff"]):
        raise ValueError("TRAINING_SOURCE_EVALUATION_CUTOFF_MISMATCH")
    if (source.get("price_unit"), source.get("volume_unit"), source.get("amount_unit")) != ("TWD", "SHARES", "TWD"):
        raise ValueError("TRAINING_SOURCE_UNITS_MISMATCH")
    histories = sorted(source["histories"], key=lambda h: (h["symbol"], h["market"]))
    if sorted(source["universe"]) != [[h["symbol"], h["market"]] for h in histories]:
        raise ValueError("FIXED_DATASET_UNIVERSE_MISMATCH")
    policy, seed = config["path_policy"], config["seed"]
    if source.get("reference_evidence_policy") != config.get("reference_evidence_policy"):
        raise ValueError("SOURCE_AND_BUILD_REFERENCE_EVIDENCE_POLICY_MISMATCH")
    coverages = [*config.get("coordinate_coverages", []), *source.get("coordinate_coverages", [])]
    calendars = source["calendars"]
    candidate_days = sorted({b["Date"] for h in histories for b in h["bars"]
        if config["event_start"] <= b["Date"] <= config["event_end"]} | {
        d for c in calendars.values() for d, status in c["days"].items()
        if status == "OPEN" and config["event_start"] <= d <= config["event_end"]})
    if len(candidate_days) * len(histories) > config["max_events"]:
        raise ValueError("DATASET_EVENT_COUNT_EXCEEDS_DECLARED_BOUND")
    events, diagnostics = [], []
    for history in histories:
        symbol, market = history["symbol"], history["market"]
        if market != policy["market"] or market not in calendars:
            raise ValueError("DATASET_PATH_POLICY_OR_CALENDAR_MARKET_MISMATCH")
        calendar = calendars[market]
        frame = pd.DataFrame(history["bars"])
        reference = _bind_reference_history(history['bars'])
        for day in candidate_days:
            cutoff = day + "T13:30:00+08:00"
            ids = [*source["source_evidence_ids"], source_record["sha256"]]
            fr3 = is_fr3_reference(seed["reference_policy"])
            options = dict(session_evidence=calendar) if fr3 else {}
            prepared = (reference(symbol, market, cutoff, seed, ids, **options) if reference is not None else
                        prepare_reference(symbol, market, frame, cutoff, seed, ids, **options))
            plan = prepared["plan"]
            diagnostic = dict(symbol=symbol, market=market, cutoff=cutoff,
                row_id=identity(dict(symbol=symbol, market=market, cutoff=timestamp(cutoff).isoformat())),
                evidence_id=prepared["evidence_id"], context=prepared["context"],
                feature_audit=prepared["feature_audit"], plan=asdict(plan) if plan else None,
                reasons=[], status="DATA_GAP")
            diagnostics.append(diagnostic)
            if (not resolve_security_identity(symbol).low_entry_model_supported
                    or history.get("security_type", "COMMON") != "COMMON"):
                diagnostic["reasons"].append("UNSUPPORTED_REFERENCE_INSTRUMENT")
            if prepared["feature_audit"].get("cutoff_session") != day:
                diagnostic["reasons"].append("NO_COMPLETE_BAR_FOR_EVENT_SESSION")
            if prepared["feature_gap"]:
                diagnostic["reasons"].append(prepared["feature_gap"])
            if fr3 and plan is not None:
                from domain.les_v7.fr3_plan import validate_band
                risk_check = validate_band(plan, seed["reference_policy"])
                diagnostic["fr3_risk_check"] = risk_check
                if risk_check["state"] != "PASS":
                    diagnostic["reasons"].append("FR3_FIXED_PLAN_RISK_NOT_PASS")
            for name in seed["feature_names"]:
                if (name in prepared["feature_audit"].get("fractal_history", {})
                        and prepared["features"].get(name) is None):
                    diagnostic["reasons"].append("FRACTAL_WARMUP_INSUFFICIENT:" + name)
            if calendar["days"].get(day) != "OPEN":
                diagnostic["reasons"].append("CALENDAR_SESSION_UNKNOWN_OR_CLOSED")
            if diagnostic["reasons"]:
                continue
            sessions = calendar["sessions"]
            start = sessions.index(day)
            expected = sessions[start:start + 21]
            if len(expected) != 21:
                diagnostic["reasons"].append("RIGHT_CENSORED_OFFICIAL_CALENDAR")
                continue
            first, last = date.fromisoformat(day), date.fromisoformat(expected[-1])
            window_dates = [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]
            if any(calendar["days"].get(d) not in {"OPEN", "CLOSED"} for d in window_dates):
                diagnostic["reasons"].append("CALENDAR_WINDOW_CONTAINS_UNKNOWN_DATES")
                continue
            if any(not calendar.get("availability_by_date", {}).get(d) for d in window_dates):
                diagnostic["reasons"].append("CALENDAR_WINDOW_PUBLICATION_UNVERIFIED")
                continue
            matched_coverages = [c for c in coverages
                if c.get("coordinate_id") == history["coordinate_id"] and c.get("market") == market
                and c.get("start_session", "~") <= day and c.get("end_session", "") >= expected[-1]]
            if len(matched_coverages) > 1 and len({identity(c) for c in matched_coverages}) > 1:
                diagnostic["reasons"].append("CONFLICTING_COORDINATE_COVERAGES")
                continue
            coverage = matched_coverages[0] if matched_coverages else None
            window = path_window(history, expected)
            if fr3:
                from domain.les_v7.fr3_plan import bind_entry_route
                path_options = dict(entry_route=bind_entry_route(plan, seed["reference_policy"]))
            else:
                path_options = {}
            outcome = evaluate_reference_path(plan, window, policy, expected_sessions=expected,
                cutoff=cutoff, coordinate_id=history["coordinate_id"], source_evidence_ids=ids,
                corporate_action_coverage=coverage, **path_options)
            diagnostic.update(status=outcome["status"], path=outcome["diagnostic"],
                              reasons=outcome["diagnostic"]["reason_codes"])
            if outcome["execution"] is None:
                continue
            event = dict(symbol=symbol, market=market, cutoff=cutoff, evidence_ids=ids,
                execution=outcome["execution"], execution_sha256=identity(outcome["execution"]),
                simulation_reference=dict(path_policy_id=identity(policy),
                    coverage_sha256=identity(coverage), raw_window_sha256=identity(window)))
            events.append(event)
    exported_calendars = {market: deepcopy(cal) for market, cal in calendars.items() if cal.get("available_at")}
    if source["dataset_kind"] == "SYNTHETIC_TEST_ONLY":
        for cal in exported_calendars.values():
            cal["source_archive_calendar_source"] = cal["source"]
            cal["source"] = "SYNTHETIC_TEST_ONLY"
    payload = dict(schema=DATASET_SCHEMA, dataset_kind=source["dataset_kind"],
        price_unit="TWD", volume_unit="SHARES", amount_unit="TWD", universe=deepcopy(source["universe"]),
        universe_policy=source["universe_policy"], source_evidence_ids=deepcopy(source["source_evidence_ids"]),
        histories=deepcopy(histories), calendars=exported_calendars, events=events,
        builder_manifest=dict(version=BUILDER_VERSION, source_packet_id=source_record["sha256"],
            build_config_id=identity(config), reference_config_id=reference_config_id(seed),
            path_policy=deepcopy(policy), coordinate_coverages=deepcopy(coverages),
            availability_policy=source["availability_policy"],
            historical_information_set_certified=source["historical_information_set_certified"],
            source_authenticity="NOT_INDEPENDENTLY_CERTIFIED", model_derived_inputs=False,
            event_sampling=config["event_sampling"]))
    record = dict(payload=payload, sha256=identity(payload))
    selection_config = dict(seed, evaluation_cutoff=config["evaluation_cutoff"])
    if config.get("reference_evidence_policy"):
        payload["reference_evidence_policy"] = config["reference_evidence_policy"]
        record["sha256"] = identity(payload)
        selection_config["reference_evidence_policy"] = config["reference_evidence_policy"]
    rows, label_audit = prepare_events(record, selection_config)
    reasons = Counter(reason for d in diagnostics for reason in d["reasons"])
    reasons.update(d["excluded_reason"] for d in label_audit if d.get("excluded_reason"))
    result = dict(schema="les-v7-dataset-build-result/v1", dataset_id=record["sha256"],
        status="EVIDENCE_READY" if len(rows) == len(diagnostics) and rows else "PARTIAL",
        counts=dict(candidate_events=len(diagnostics), path_events=len(events), eligible_events=len(rows),
                    labels=dict(Counter(r["labels"]["fill"] for r in rows))),
        diagnostics=diagnostics, label_diagnostics=label_audit, reason_counts=dict(sorted(reasons.items())),
        source_diagnostics=deepcopy(source["diagnostics"]), research_only=True, approved_for_live=False)
    return dict(dataset=record, result=result)

"""Bounded ELEC-wide reference-data orchestration over one frozen cache scope.

This is an orchestration layer around the existing source adapter and builder,
not a second training/replay engine. No financial refresh, HTTP or account port.
"""
from collections import Counter
from contextlib import closing
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from application.les_v7_dataset import validate_build_config
from application.replay_contracts import digest
from application.research_evidence import RAW_EVIDENCE_SCHEMA
from domain.les_v7.contracts import canonical, identity, timestamp
from domain.security_identity import resolve_security_identity
from infrastructure.les_v7_training_backfill import _cached_market_records, MAX_CACHE_DAY_BYTES
from infrastructure.les_v7_training_source import FINAL_CLOSE, OBSERVED_ONLY, FINAL_VINTAGE, MAX_BYTES, MAX_DAYS, MAX_SYMBOLS, load_training_source
from infrastructure.replay_evidence import load_evidence, read_only
from infrastructure.research_archive import create_dataset, utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.les_v7_dataset import LesV7DatasetRepository, _lock
from infrastructure.storage.les_v7_training import read_json_bounded, MAX_DATASET_BYTES

SCOPE_SCHEMA = "les-v7-elec-scope-freeze/v1"
REQUEST_SCHEMA = "les-v7-elec-scope-request/v1"
COMPLETION_SCHEMA = "les-v7-elec-scope-completion/v1"
MAX_UNIVERSE = 2048
SCOPE_FIELDS = ("universe_record", "universe", "calendar_evidence_id", "calendar_sha256", "market", "sector",
                "history_start", "history_end", "batch_size", "dataset_kind")


def _read(path):
    return read_json_bounded(Path(path), MAX_DATASET_BYTES)


def _universe_entries(rows):
    entries = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) < 16:
            raise ValueError("ELEC_UNIVERSE_ROW_SCHEMA_CHANGED")
        stock = resolve_security_identity(row[0], row[1])
        entries.append(dict(index=index, symbol=stock.market_symbol, name=stock.stock_name,
            security_type=stock.security_type.value, identity_source=stock.mapping_source,
            source_row_sha256=digest(row)))
    return entries


def _universe(connection, day):
    header = connection.execute("""SELECT length(CAST(payload_json AS BLOB)) FROM market_history_cache
        WHERE trade_date=? AND dataset='STOCK_TABLE' AND market_type='13' AND schema_version=1""",
        (day.strftime("%Y%m%d"),)).fetchone()
    if header is None or header[0] > MAX_CACHE_DAY_BYTES:
        raise ValueError("ELEC_UNIVERSE_SOURCE_MISSING_OR_TOO_LARGE")
    serialized, = connection.execute("""SELECT payload_json FROM market_history_cache
        WHERE trade_date=? AND dataset='STOCK_TABLE' AND market_type='13' AND schema_version=1""",
        (day.strftime("%Y%m%d"),)).fetchone()
    rows = json.loads(serialized)
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_UNIVERSE:
        raise ValueError("BOUNDED_ELEC_UNIVERSE_REQUIRED")
    entries = _universe_entries(rows)
    symbols = [row["symbol"] for row in entries]
    if len(symbols) != len(set(symbols)):
        raise ValueError("ELEC_UNIVERSE_DUPLICATE_SYMBOL")
    return dict(source_day=day.isoformat(), market="TWSE", market_type="13", schema_version=1,
        raw_sha256=hashlib.sha256(serialized.encode()).hexdigest(), raw_payload_json=serialized,
        canonical_rows_sha256=digest(rows), rows=entries,
        selection_policy="ALL_ROWS_OF_FIXED_OFFICIAL_FINAL_ELEC_SNAPSHOT_NOT_CANDIDATE_PIT_MEMBERSHIP")


def freeze_elec_scope(calendar_evidence, *, market_cache, history_start, history_end,
                      universe_date, config, root, batch_size=32, allow_synthetic=False,
                      market_supplement=None):
    """Freeze all shards BEFORE research, sharing one read-only SQLite snapshot."""
    validate_build_config(config)
    if timestamp(config["evaluation_cutoff"]) > timestamp(utc_now()):
        raise ValueError("DATASET_EVALUATION_AFTER_BUILD_CLOCK")
    if (any(type(d) is not date for d in (history_start, history_end, universe_date))
            or not history_start <= universe_date <= history_end
            or (history_end - history_start).days + 1 > MAX_DAYS
            or history_end > timestamp(config["evaluation_cutoff"]).astimezone(ZoneInfo("Asia/Taipei")).date()
            or not history_start.isoformat() <= config["event_start"] <= config["event_end"] <= history_end.isoformat()
            or type(batch_size) is not int or not 1 <= batch_size <= MAX_SYMBOLS
            or config["path_policy"]["market"] != "TWSE"):
        raise ValueError("ELEC_SCOPE_RANGE_BATCH_OR_MARKET_INVALID")
    event_days = (date.fromisoformat(config["event_end"]) - date.fromisoformat(config["event_start"])).days + 1
    effective_size = min(batch_size, config["max_events"] // event_days)
    if effective_size < 1:
        raise ValueError("ELEC_SCOPE_EVENT_BUDGET_INSUFFICIENT")
    parent_id, parent = load_evidence(calendar_evidence, max_bytes=MAX_BYTES)
    if parent.get("schema") != RAW_EVIDENCE_SCHEMA or parent.get("calendar", {}).get("source") != "TWSE_OFFICIAL":
        raise ValueError("ELEC_SCOPE_RAW_OFFICIAL_CALENDAR_REQUIRED")
    synthetic = parent.get("purpose") == "SYNTHETIC_TEST_ONLY"
    if synthetic and not allow_synthetic:
        raise ValueError("SYNTHETIC_SCOPE_REQUIRES_EXPLICIT_OPT_IN")
    documents = EvidenceDocuments.for_evidence(calendar_evidence, parent_id)
    documents.verify(parent["calendar"])
    calendar = deepcopy(parent["calendar"])
    market_body, market_documents = None, None
    if market_supplement:
        from infrastructure.les_v7_market_final import load_final_market
        market_body, market_documents = load_final_market(market_supplement)
        if market_body["start"] < history_start.isoformat() or market_body["end"] > history_end.isoformat():
            raise ValueError("ELEC_SCOPE_MARKET_SUPPLEMENT_OUT_OF_RANGE")
    if config.get("reference_evidence_policy") and parent.get("reference_evidence_policy") != config["reference_evidence_policy"]:
        raise ValueError("ELEC_SCOPE_REFERENCE_REPAIR_POLICY_MISMATCH")
    calendar["days"] = {(history_start + timedelta(days=i)).isoformat():
        calendar.get("days", {}).get((history_start + timedelta(days=i)).isoformat(), "UNKNOWN")
        for i in range((history_end - history_start).days + 1)}
    root, descriptors = Path(root), []
    collected_at = utc_now()
    with closing(read_only(Path(market_cache))) as connection:
        connection.execute("BEGIN")
        universe_record = _universe(connection, universe_date)
        universe = sorted(row["symbol"] for row in universe_record["rows"])
        scope = dict(universe_record=universe_record, universe=universe,
            calendar_evidence_id=parent_id, calendar_sha256=digest(calendar), market="TWSE", sector="13",
            history_start=history_start.isoformat(), history_end=history_end.isoformat(),
            batch_size=effective_size, dataset_kind="SYNTHETIC_TEST_ONLY" if synthetic else "RESEARCH")
        if market_body is not None:
            scope["market_supplement_sha256"] = digest(market_body)
        if config.get("reference_evidence_policy"):
            scope["reference_evidence_policy"] = config["reference_evidence_policy"]
            scope["corporate_actions_packet_sha256"] = digest(parent.get("corporate_actions_final"))
        scope_id = identity(scope)
        classifications = {row["symbol"]: row for row in universe_record["rows"]}
        for index, offset in enumerate(range(0, len(universe), effective_size)):
            symbols = universe[offset:offset + effective_size]
            payload = dict(schema=RAW_EVIDENCE_SCHEMA, market="TWSE", sector="13", symbols=symbols,
                start=history_start.isoformat(), end=history_end.isoformat(), calendar=deepcopy(calendar),
                price_unit="TWD", volume_unit="SHARES", amount_unit="TWD", market_policy=FINAL_VINTAGE,
                universe_policy=universe_record["selection_policy"], raw_documents={}, breadth_sources={},
                filings={s: [] for s in symbols}, batches={s: [] for s in symbols}, restrictions=[],
                restriction_ledger=None, kelly_calibrations=[], security_classifications={s: classifications[s] for s in symbols},
                scope_binding=dict(scope_id=scope_id, chunk_index=index, complete_scope_size=len(universe),
                    calendar_source_evidence_id=parent_id, original_universe_day=universe_date.isoformat()),
                preparation_gaps=["MARKET_ONLY_REFERENCE_TRAINING_NO_FINANCIAL_VERSIONS_COLLECTED",
                    "HISTORICAL_RESTRICTIONS_NOT_COLLECTED_FOR_REFERENCE_DATASET",
                    "KELLY_CALIBRATIONS_NOT_COLLECTED_FOR_REFERENCE_DATASET",
                    "CORPORATE_ACTION_COVERAGE_NOT_COLLECTED"],
                purpose="SYNTHETIC_TEST_ONLY" if synthetic else "ELEC_MARKET_REFERENCE_RESEARCH_ONLY")
            payload["bars"] = _cached_market_records(Path(market_cache), payload, history_start, connection=connection)
            if config.get("reference_evidence_policy"):
                payload["reference_evidence_policy"] = config["reference_evidence_policy"]
                payload["corporate_actions_final"] = deepcopy(parent["corporate_actions_final"])
            if market_body is not None:
                from infrastructure.les_v7_market_final import final_market_records
                payload["bars"].extend(final_market_records(market_body, market_documents, symbols))
                payload["market_supplement_final"] = market_body
            if len(canonical(dict(evidence_id=identity(payload), payload=payload)).encode()) > MAX_BYTES:
                raise ValueError("ELEC_SCOPE_SHARD_SIZE_LIMIT")
            from infrastructure.les_v7_reference_repair import reference_document_reader
            reader = reference_document_reader(*[d for d in (documents, market_documents) if d is not None])
            evidence = create_dataset(payload, root / "evidence", collected_at=collected_at,
                purpose="ELEC_MARKET_REFERENCE_SCOPE", document_reader=reader)
            descriptors.append(dict(index=index, symbols=symbols, evidence_id=evidence.parent.name))
    documents.check()
    if load_evidence(calendar_evidence, max_bytes=MAX_BYTES)[0] != parent_id:
        raise ValueError("ELEC_SCOPE_CALENDAR_INPUT_CHANGED")
    body = dict(schema=SCOPE_SCHEMA, scope_id=scope_id, **scope, chunks=descriptors,
        historical_information_set_certified=False, original_transport_bytes_retained=False,
        original_observation_receipts_available=False, source_authenticity="NOT_INDEPENDENTLY_CERTIFIED")
    freeze_id = identity(body)
    path = root / "freezes" / freeze_id / "manifest.json"
    record = dict(freeze_id=freeze_id, body=body, created_at=utc_now())
    if path.exists():
        previous = _read(path)
        if previous.get("freeze_id") != freeze_id or previous.get("body") != body:
            raise ValueError("ELEC_SCOPE_IMMUTABLE_FREEZE_CONFLICT")
    else:
        write_once(path, record)
    _verify_freeze(path, root)
    return path


def _load_bound_scope_shard(body, root, chunk):
    """One shared shard/scope identity check, before any native source use."""
    index = chunk["index"]
    original = body["universe_record"]
    classifications = {row["symbol"]: row for row in original["rows"]}
    path = Path(root) / "evidence" / chunk["evidence_id"] / "evidence.json"
    eid, payload = load_evidence(path, max_bytes=MAX_BYTES)
    if (eid != chunk["evidence_id"] or payload["symbols"] != chunk["symbols"]
            or payload["scope_binding"]["scope_id"] != body["scope_id"]
            or payload["scope_binding"]["chunk_index"] != index
            or payload["scope_binding"]["complete_scope_size"] != len(body["universe"])
            or payload["scope_binding"]["calendar_source_evidence_id"] != body["calendar_evidence_id"]
            or payload["scope_binding"]["original_universe_day"] != original["source_day"]
            or payload["start"] != body["history_start"] or payload["end"] != body["history_end"]
            or payload["market"] != body["market"] or payload["sector"] != body["sector"]
            or payload["market_policy"] != FINAL_VINTAGE or digest(payload["calendar"]) != body["calendar_sha256"]
            or (payload["price_unit"], payload["volume_unit"], payload["amount_unit"]) != ("TWD", "SHARES", "TWD")
            or payload["security_classifications"] != {s: classifications[s] for s in chunk["symbols"]}
            or (payload.get("purpose") == "SYNTHETIC_TEST_ONLY") != (body["dataset_kind"] == "SYNTHETIC_TEST_ONLY")):
        raise ValueError("ELEC_SCOPE_SHARD_BINDING_MISMATCH")
    if "reference_evidence_policy" in body and (
            payload.get("reference_evidence_policy") != body["reference_evidence_policy"]
            or digest(payload.get("corporate_actions_final")) != body["corporate_actions_packet_sha256"]):
        raise ValueError("ELEC_SCOPE_FINAL_REFERENCE_BINDING_MISMATCH")
    if "market_supplement_sha256" in body and digest(payload.get("market_supplement_final")) != body["market_supplement_sha256"]:
        raise ValueError("ELEC_SCOPE_MARKET_SUPPLEMENT_BINDING_MISMATCH")
    return path, eid, payload


def _verify_freeze(path, root, *, verify_sources=True):
    record = _read(path)
    body = record["body"]
    if (body.get("schema") != SCOPE_SCHEMA or identity(body) != record.get("freeze_id")
            or Path(path).parent.name != record["freeze_id"]
            or body["universe"] != sorted(set(body["universe"]))
            or not 1 <= len(body["universe"]) <= MAX_UNIVERSE):
        raise ValueError("ELEC_SCOPE_FREEZE_IDENTITY_INVALID")
    extra_scope_keys = [key for key in ("market_supplement_sha256", "reference_evidence_policy", "corporate_actions_packet_sha256") if key in body]
    if (body["scope_id"] != identity({key: body[key] for key in (*SCOPE_FIELDS, *extra_scope_keys)})
            or body["market"] != "TWSE" or body["sector"] != "13"
            or type(body["batch_size"]) is not int or not 1 <= body["batch_size"] <= MAX_SYMBOLS
            or body["dataset_kind"] not in {"RESEARCH", "SYNTHETIC_TEST_ONLY"}
            or body["historical_information_set_certified"] is not False
            or body["original_observation_receipts_available"] is not False):
        raise ValueError("ELEC_SCOPE_POLICY_BINDING_MISMATCH")
    first, end = date.fromisoformat(body["history_start"]), date.fromisoformat(body["history_end"])
    if first > end or (end - first).days + 1 > MAX_DAYS:
        raise ValueError("ELEC_SCOPE_RANGE_INVALID")
    original = body["universe_record"]
    raw_rows = json.loads(original["raw_payload_json"])
    if (hashlib.sha256(original["raw_payload_json"].encode()).hexdigest() != original["raw_sha256"]
            or digest(raw_rows) != original["canonical_rows_sha256"]
            or sorted(str(row[0]) for row in raw_rows) != body["universe"]
            or original["rows"] != _universe_entries(raw_rows)
            or original["market"] != "TWSE" or original["market_type"] != "13" or original["schema_version"] != 1
            or original["source_day"] < body["history_start"] or original["source_day"] > body["history_end"]):
        raise ValueError("ELEC_SCOPE_UNIVERSE_SOURCE_BINDING_MISMATCH")
    all_symbols = []
    expected_chunks = [body["universe"][i:i + body["batch_size"]]
                       for i in range(0, len(body["universe"]), body["batch_size"])]
    if [c["symbols"] for c in body["chunks"]] != expected_chunks:
        raise ValueError("ELEC_SCOPE_CHUNK_PLAN_INVALID")
    for index, chunk in enumerate(body["chunks"]):
        if chunk["index"] != index or not 1 <= len(chunk["symbols"]) <= MAX_SYMBOLS:
            raise ValueError("ELEC_SCOPE_CHUNK_PLAN_INVALID")
        all_symbols.extend(chunk["symbols"])
        if not verify_sources:
            continue
        path, eid, payload = _load_bound_scope_shard(body, root, chunk)
        EvidenceDocuments.for_evidence(path, eid).verify(dict(calendar=payload["calendar"],
            actions=payload.get("corporate_actions_final"), market=payload.get("market_supplement_final")))
    if all_symbols != body["universe"]:
        raise ValueError("ELEC_SCOPE_COVERAGE_MISSING_DUPLICATED_OR_REORDERED")
    timestamp(record["created_at"])
    return record


class _PinnedGuard:
    def __init__(self, guard, objects):
        self.guard, self.root = guard, objects.root
        self.pinned = guard.retain(objects)

    def check(self):
        return self.guard.check()

    def retain(self, objects):
        if objects.root != self.root:
            raise ValueError("ELEC_SCOPE_OBJECT_ROOT_CHANGED")
        self.check()
        return dict(self.pinned)


def _coverage(source, result):
    rows = []
    for history in source["payload"]["histories"]:
        diagnostics = [d for d in result["diagnostics"] if d["symbol"] == history["symbol"]]
        rows.append(dict(symbol=history["symbol"], market=history["market"],
            security_type=history.get("security_type", "COMMON"), normalized_bar_count=len(history["bars"]),
            history_start=history["bars"][0]["Date"] if history["bars"] else None,
            history_end=history["bars"][-1]["Date"] if history["bars"] else None,
            candidate_events=len(diagnostics), reasons=dict(Counter(reason for d in diagnostics for reason in d["reasons"])),
            cutoff_history=[dict(cutoff=d["cutoff"], context=d["context"],
                fractal_history=d["feature_audit"].get("fractal_history", {}), reasons=d["reasons"])
                for d in diagnostics]))
    return rows


def _aggregate_summary(rows, counts, reasons):
    return dict(counts, reason_counts=dict(sorted(reasons.items())),
        symbols_with_513_closes=sum(bool(row["cutoff_history"]) and all(
            d["fractal_history"].get("dfa512", {}).get("available_close_count", 0) >= 513
            for d in row["cutoff_history"]) for row in rows),
        symbols_with_no_bars=sum(row["normalized_bar_count"] == 0 for row in rows),
        unsupported_instruments=[row["symbol"] for row in rows if row["security_type"] != "COMMON"])


def run_elec_scope(freeze_path, config, *, root, allow_final_daily_reconstruction=False,
                   allow_synthetic=False, guard_factory=ResearchVersionGuard,
                   repository_factory=LesV7DatasetRepository, clock=utc_now, progress=None, after_batch=None):
    validate_build_config(config)
    if timestamp(config["evaluation_cutoff"]) > timestamp(clock()):
        raise ValueError("DATASET_EVALUATION_AFTER_BUILD_CLOCK")
    root = Path(root)
    record = _verify_freeze(freeze_path, root)
    body = record["body"]
    if body["dataset_kind"] == "SYNTHETIC_TEST_ONLY" and not allow_synthetic:
        raise ValueError("SYNTHETIC_SCOPE_REQUIRES_EXPLICIT_OPT_IN")
    days = (date.fromisoformat(config["event_end"]) - date.fromisoformat(config["event_start"])).days + 1
    if (days * body["batch_size"] > config["max_events"]
            or not body["history_start"] <= config["event_start"] <= config["event_end"] <= body["history_end"]):
        raise ValueError("ELEC_SCOPE_CONFIG_WINDOW_OR_BUDGET_MISMATCH")
    guard = _PinnedGuard(guard_factory(), ResearchObjects(root / "builds" / "objects"))
    request = dict(schema=REQUEST_SCHEMA, freeze_id=record["freeze_id"], config_id=identity(config),
        availability_policy=FINAL_CLOSE if allow_final_daily_reconstruction else OBSERVED_ONLY,
        pinned_objects=guard.pinned, dataset_kind=body["dataset_kind"], research_only=True, approved_for_live=False)
    if config.get("reference_evidence_policy"):
        if not allow_final_daily_reconstruction or body.get("reference_evidence_policy") != config["reference_evidence_policy"]:
            raise ValueError("ELEC_SCOPE_FINAL_REFERENCE_EXPLICIT_POLICY_REQUIRED")
        request["reference_evidence_policy"] = config["reference_evidence_policy"]
    job_id = identity(request)
    folder = root / "jobs" / job_id
    with _lock(folder):
        guard.check()
        if (folder / "completion.json").exists():
            return verify_elec_scope(folder, root=root)
        write_once(folder / "request.json", request)
        write_once(folder / "config.json", config)
        write_once(folder / "freeze_reference.json", dict(freeze_id=record["freeze_id"]))
        repository = repository_factory(root / "builds", guard_factory=lambda: guard, clock=clock)
        completed, rows, counts, reasons = [], [], Counter(), Counter()
        for chunk in body["chunks"]:
            checkpoint = folder / "batches" / (str(chunk["index"]) + ".json")
            if checkpoint.exists():
                saved = _read(checkpoint)
                receipt = LesV7DatasetRepository.verify(root / "builds" / saved["job_id"])
                if (saved["index"] != chunk["index"] or saved["symbols"] != chunk["symbols"]
                        or saved["evidence_id"] != chunk["evidence_id"]
                        or receipt["config_id"] != request["config_id"] or receipt["pinned_objects"] != guard.pinned
                        or identity(_read(Path(receipt["output"]) / "completion.json")) != saved["completion_sha256"]):
                    raise ValueError("ELEC_SCOPE_BATCH_CHECKPOINT_MISMATCH")
                source = _read(Path(receipt["output"]) / "source.json")
            else:
                evidence = root / "evidence" / chunk["evidence_id"] / "evidence.json"
                extra = {"reference_evidence_policy": request["reference_evidence_policy"]} if request.get("reference_evidence_policy") else {}
                source = load_training_source(evidence, evaluation_cutoff=config["evaluation_cutoff"],
                    availability_policy=request["availability_policy"], **extra)
                receipt = repository.run(source, config, allow_synthetic=allow_synthetic)
            if (source["payload"]["source_evidence_ids"] != [chunk["evidence_id"]]
                    or source["payload"]["availability_policy"] != request["availability_policy"]
                    or timestamp(source["payload"]["lineage"]["evaluation_cutoff"]) != timestamp(config["evaluation_cutoff"])):
                raise ValueError("ELEC_SCOPE_SELECTED_SOURCE_BINDING_MISMATCH")
            result = _read(Path(receipt["output"]) / "build_result.json")
            rows.extend(_coverage(source, result))
            for key in ("candidate_events", "path_events", "eligible_events"):
                counts[key] += result["counts"][key]
            reasons.update(result["reason_counts"])
            completed.append(dict(index=chunk["index"], symbols=chunk["symbols"], evidence_id=chunk["evidence_id"],
                job_id=receipt["job_id"], dataset_id=receipt["dataset_id"], completion_sha256=identity(
                    _read(Path(receipt["output"]) / "completion.json"))))
            write_once(checkpoint, completed[-1])
            if progress:
                progress(chunk["index"] + 1, len(body["chunks"]), receipt)
            if after_batch:
                after_batch(chunk["index"], receipt)
            del source, result
        if sorted(row["symbol"] for row in rows) != body["universe"]:
            raise ValueError("ELEC_SCOPE_RESULT_SYMBOL_COVERAGE_MISMATCH")
        write_once(folder / "coverage.json", dict(schema="les-v7-elec-scope-coverage/v1", freeze_id=record["freeze_id"], rows=rows))
        summary = _aggregate_summary(rows, counts, reasons)
        receipt = dict(schema=COMPLETION_SCHEMA, job_id=job_id, freeze_id=record["freeze_id"],
            request_sha256=identity(request), config_id=identity(config), pinned_objects=guard.pinned,
            dataset_kind=body["dataset_kind"], symbol_count=len(rows), batch_count=len(completed), batches=completed, summary=summary,
            status="EVIDENCE_READY" if counts["eligible_events"] == counts["candidate_events"] and counts["eligible_events"] else "PARTIAL",
            coverage_sha256=identity(_read(folder / "coverage.json")), completed_at=clock(),
            research_only=True, approved_for_live=False, execution_action="NO_ACTION")
        guard.check()
        write_once(folder / "completion.json", receipt)
        return verify_elec_scope(folder, root=root)


def verify_elec_scope(jobfolder, *, root=None):
    folder = Path(jobfolder)
    root = Path(root) if root else folder.parent.parent
    receipt, request, config = (_read(folder / name) for name in ("completion.json", "request.json", "config.json"))
    if (receipt.get("schema") != COMPLETION_SCHEMA or request.get("schema") != REQUEST_SCHEMA
            or identity(request) != folder.name or receipt["job_id"] != folder.name
            or receipt["request_sha256"] != identity(request) or identity(config) != request["config_id"]
            or receipt["config_id"] != request["config_id"] or receipt["pinned_objects"] != request["pinned_objects"]
            or receipt["execution_action"] != "NO_ACTION" or receipt["approved_for_live"] is not False
            or receipt["research_only"] is not True or request["research_only"] is not True
            or request["approved_for_live"] is not False or receipt["dataset_kind"] != request["dataset_kind"]
            or _read(folder / "freeze_reference.json") != dict(freeze_id=request["freeze_id"])):
        raise ValueError("ELEC_SCOPE_COMPLETION_BINDING_MISMATCH")
    record = _verify_freeze(root / "freezes" / request["freeze_id"] / "manifest.json", root)
    coverage = _read(folder / "coverage.json")
    if (coverage.get("schema") != "les-v7-elec-scope-coverage/v1"
            or record["body"]["dataset_kind"] != request["dataset_kind"]
            or receipt["freeze_id"] != request["freeze_id"] or coverage["freeze_id"] != request["freeze_id"]
            or identity(coverage) != receipt["coverage_sha256"]
            or sorted(row["symbol"] for row in coverage["rows"]) != record["body"]["universe"]
            or receipt["symbol_count"] != len(coverage["rows"]) or len(receipt["batches"]) != len(record["body"]["chunks"])):
        raise ValueError("ELEC_SCOPE_COVERAGE_BINDING_MISMATCH")
    total = Counter()
    expected_rows, reason_totals = [], Counter()
    for child, chunk in zip(receipt["batches"], record["body"]["chunks"]):
        completed = LesV7DatasetRepository.verify(root / "builds" / child["job_id"])
        if (child["index"] != chunk["index"] or child["symbols"] != chunk["symbols"]
                or child["evidence_id"] != chunk["evidence_id"] or child["dataset_id"] != completed["dataset_id"]
                or identity(_read(Path(completed["output"]) / "completion.json")) != child["completion_sha256"]
                or completed["config_id"] != request["config_id"] or completed["pinned_objects"] != request["pinned_objects"]):
            raise ValueError("ELEC_SCOPE_BATCH_RESULT_BINDING_MISMATCH")
        if _read(folder / "batches" / (str(child["index"]) + ".json")) != child:
            raise ValueError("ELEC_SCOPE_BATCH_CHECKPOINT_MISMATCH")
        source = _read(Path(completed["output"]) / "source.json")
        result = _read(Path(completed["output"]) / "build_result.json")
        if (source["payload"]["source_evidence_ids"] != [chunk["evidence_id"]]
                or source["payload"]["availability_policy"] != request["availability_policy"]
                or timestamp(source["payload"]["lineage"]["evaluation_cutoff"]) != timestamp(config["evaluation_cutoff"])):
            raise ValueError("ELEC_SCOPE_SELECTED_SOURCE_BINDING_MISMATCH")
        expected_rows.extend(_coverage(source, result))
        reason_totals.update(result["reason_counts"])
        for key in ("candidate_events", "path_events", "eligible_events"):
            total[key] += completed["counts"][key]
    expected_status = "EVIDENCE_READY" if total["eligible_events"] == total["candidate_events"] and total["eligible_events"] else "PARTIAL"
    if receipt["status"] != expected_status or receipt["batch_count"] != len(receipt["batches"]) or any(
            receipt["summary"].get(k) != v for k, v in total.items()):
        raise ValueError("ELEC_SCOPE_AGGREGATE_COUNTS_MISMATCH")
    if coverage["rows"] != expected_rows or receipt["summary"] != _aggregate_summary(expected_rows, total, reason_totals):
        raise ValueError("ELEC_SCOPE_NATIVE_COVERAGE_CONTENT_MISMATCH")
    timestamp(receipt["completed_at"])
    return dict(receipt, output=str(folder), coverage=str(folder / "coverage.json"))

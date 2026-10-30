"""Prepend verified official FINAL history without replacing frozen old bars.

This is a source assembly adapter over the native scope/document ports, not a
second normalization or research engine. It never queries a mutable market
cache, downloads, fits, changes a label, or grants trading permission.
"""
from copy import deepcopy
from contextlib import contextmanager
from datetime import date, timedelta
import json
from pathlib import Path
import sqlite3
import tempfile

from application.les_v7_dataset import validate_build_config
from application.replay_contracts import digest
from application.research_evidence import verified_document
from domain.les_v7.contracts import timestamp
from domain.les_v7.fr3_plan import is_fr3_reference
from infrastructure.les_v7_actions_final import load_actions_final
from infrastructure.les_v7_calendar_final import load_final_calendar, select_final_calendar
from infrastructure.les_v7_dataset_scope import (
    SCOPE_FIELDS, SCOPE_SCHEMA, _load_bound_scope_shard, _verify_freeze,
)
from infrastructure.les_v7_market_final import load_final_market, parse_final_market
from infrastructure.les_v7_reference_repair import repair_reference_evidence, reference_document_reader
from infrastructure.les_v7_training_prepare import _repartition_shards
from infrastructure.les_v7_training_source import FINAL_REFERENCE, MAX_BYTES, MAX_DAYS
from infrastructure.replay_evidence import load_evidence
from infrastructure.research_archive import create_dataset, utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments
from infrastructure.research_objects import ResearchVersionGuard
from infrastructure.storage.les_v7_dataset import _lock
from infrastructure.storage.les_v7_training import MAX_CONFIG_BYTES, read_json_bounded

SCHEMA = "les-v7-fr3-source-prefix/v1"
MAX_PREFIX_INDEX_BYTES = 512 * 1024 * 1024
MAX_PREFIX_SELECTED_ROWS = 1_000_000


class _PrefixIndex:
    def __init__(self, connection):
        self.connection = connection

    def records(self, symbols):
        if not 1 <= len(symbols) <= 32 or len(set(symbols)) != len(symbols):
            raise ValueError("FR3_PREFIX_BOUNDED_SYMBOL_BATCH_REQUIRED")
        placeholders = ",".join("?" for _ in symbols)
        rows = self.connection.execute("""SELECT b.day,b.taiex,r.payload
            FROM benchmark AS b LEFT JOIN stock_rows AS r
              ON r.day=b.day AND r.symbol IN (""" + placeholders + """)
            ORDER BY b.sequence,r.sequence""", tuple(symbols))
        output, current, stocks, benchmark = [], None, [], None
        def append():
            output.extend([
                dict(date=current, dataset="STOCK_TABLE", market_type="ALLBUT0999", source="TWSE", data=stocks),
                dict(date=current, dataset="TAIEX", market_type="IND", source="TWSE_TAIEX", data=benchmark)])
        for day, taiex, payload in rows:
            if current is not None and current != day:
                append()
                stocks = []
            current, benchmark = day, taiex
            if payload is not None:
                stocks.append(json.loads(payload))
        if current is not None:
            append()
        return output


@contextmanager
def _prefix_index(market, documents, universe):
    """One native raw-day parse, bounded local SSD index, no trusted disk cache.

    The SQLite file is new and ephemeral, never a copied/live project database.
    Source bytes are reverified here; batches preserve original row/day order,
    empty pre-listing days and exact benchmark string conversion. All state is
    discarded on success or exception, not reused across datasets/cutoffs.
    """
    wanted = set(universe)
    with tempfile.TemporaryDirectory(prefix="fr3-source-prefix-") as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "prefix.sqlite3"))
        try:
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            connection.execute("PRAGMA max_page_count=" + str(MAX_PREFIX_INDEX_BYTES // page_size))
            connection.execute("CREATE TABLE benchmark(day TEXT PRIMARY KEY,sequence INTEGER,taiex TEXT)")
            connection.execute("CREATE TABLE stock_rows(day TEXT,symbol TEXT,sequence INTEGER,payload TEXT,PRIMARY KEY(day,symbol))")
            count = 0
            for sequence, entry in enumerate(market["documents"]):
                raw = verified_document(entry["document"], document_reader=documents.read)
                rows, benchmark = parse_final_market(raw, entry["date"])
                day = entry["date"].replace("-", "")
                connection.execute("INSERT INTO benchmark VALUES(?,?,?)", (day, sequence, str(benchmark)))
                values = [(day, str(row[0]), index, json.dumps(row, ensure_ascii=False, allow_nan=False))
                    for index, row in enumerate(rows) if str(row[0]) in wanted]
                count += len(values)
                if count > MAX_PREFIX_SELECTED_ROWS:
                    raise ValueError("FR3_PREFIX_INDEX_SELECTED_ROW_LIMIT")
                connection.executemany("INSERT INTO stock_rows VALUES(?,?,?,?)", values)
            connection.commit()
            documents.check()
            yield _PrefixIndex(connection)
            documents.check()
        except sqlite3.Error as error:
            raise ValueError("FR3_PREFIX_INDEX_INTEGRITY_OR_SIZE_LIMIT:" + str(error)) from error
        finally:
            connection.close()


def retain_corrected_fr3_calendar(calendar_path, output_root):
    """Bind corrected parser semantics to a NEW identity, retaining original bytes.

    No price-data absence is used to classify a day. Selection must re-read the
    same official annual descriptions with the date-bound parser. Earlier
    receipts and any mistaken derived OPEN counts remain untouched for audit.
    """
    from domain.trading_calendar import ANNUAL_PARSER_VERSION
    calendar, documents = load_final_calendar(calendar_path)
    root = Path(output_root).resolve()
    source = Path(calendar_path).resolve()
    if root == source.parent or root.is_relative_to(source.parent):
        raise ValueError("FR3_CALENDAR_REPAIR_MUST_PRESERVE_ORIGINAL")
    store = EvidenceDocuments(root / "objects")
    corrected = deepcopy(calendar)
    corrected.update(annual_parser_version=ANNUAL_PARSER_VERSION,
        calendar_semantics_policy="SOURCE_DATE_BOUND_TRADING_VS_SETTLEMENT_NOT_MISSING_PRICE_INFERENCE",
        parent_calendar_id=digest(calendar))
    corrected = store.externalize(corrected, document_reader=documents.read)
    identifier = digest(corrected)
    path = root / "collections" / identifier / "calendar.json"
    write_once(path, dict(calendar_id=identifier, payload=corrected))
    loaded, retained = load_final_calendar(path)
    if loaded != corrected:
        raise ValueError("FR3_CALENDAR_REPAIR_CONTENT_MISMATCH")
    documents.check()
    retained.check()
    write_once(path.parent / "completion.json", dict(schema="les-v7-fr3-calendar-semantics-repair/v1",
        calendar_id=identifier, parent_calendar_id=digest(calendar),
        annual_parser_version=ANNUAL_PARSER_VERSION, official_source_bytes_changed=False,
        historical_information_set_certified=False, research_only=True,
        approved_for_live=False, execution_action="NO_ACTION"))
    return path


def freeze_fr3_expansion_settings(parent_training_root, calendar_path, *, history_start,
                                  through, output_root):
    """Verify the native parent and official calendar before outcome-blind planning."""
    from application.les_v7_fr3_expansion import plan_fr3_expansion, verify_fr3_expansion_plan
    from infrastructure.les_v7_fr3_training import verify_fr3_scope_training
    parent_root = Path(parent_training_root).resolve()
    output = Path(output_root).resolve()
    if output.is_relative_to(parent_root):
        raise ValueError("FR3_EXPANSION_SETTINGS_MUST_PRESERVE_PARENT")
    native = verify_fr3_scope_training(parent_root)
    request = read_json_bounded(parent_root / "fr3-training-request.json", 2 * MAX_CONFIG_BYTES + 64 * 1024)
    body = request["payload"]
    if request["sha256"] != digest(body) or native["request_id"] != request["sha256"]:
        raise ValueError("FR3_EXPANSION_PARENT_REQUEST_MISMATCH")
    cutoff = through + "T13:30:00+08:00"
    if timestamp(cutoff) > timestamp(utc_now()):
        raise ValueError("FR3_EXPANSION_REAL_CUTOFF_REQUIRED")
    calendar, documents = load_final_calendar(calendar_path)
    selected, diagnostics = select_final_calendar(calendar, cutoff, documents=documents,
        start=history_start, end=through, evidence_id=digest(calendar))
    if diagnostics["unknown_dates"] or calendar.get("collection_errors"):
        raise ValueError("FR3_EXPANSION_COMPLETE_VERIFIED_CALENDAR_REQUIRED")
    plan = plan_fr3_expansion(body["build_config"], body["training_config"],
        sessions=selected["sessions"], history_start=history_start, through=through,
        universe_id=digest([[s, "TWSE"] for s in body["symbols"]]), calendar_id=digest(calendar))
    verify_fr3_expansion_plan(plan)
    folder = output / plan["plan_id"]
    with _lock(folder):
        write_once(folder / "plan.json", plan)
        for name in ("build_config", "training_config"):
            write_once(folder / (name.replace("_", "-") + ".json"), plan["payload"][name])
        documents.check()
        receipt = dict(schema="les-v7-fr3-expansion-settings-completion/v1", plan_id=plan["plan_id"],
            parent_request_id=request["sha256"], calendar_id=digest(calendar),
            calendar_file=str(Path(calendar_path).resolve()),
            native_sources_verified=True, calendar_sources_verified=True,
            status="CALENDAR_PLAN_ONLY_NOT_TRAINING_READY", model_fitted=False,
            historical_information_set_certified=False, research_only=True,
            approved_for_live=False, execution_action="NO_ACTION")
        completion = folder / "completion.json"
        if not completion.exists():
            write_once(completion, dict(receipt, created_at=utc_now()))
        else:
            previous = read_json_bounded(completion, MAX_CONFIG_BYTES)
            if {k: v for k, v in previous.items() if k != "created_at"} != receipt:
                raise ValueError("FR3_EXPANSION_SETTINGS_COMPLETION_CONFLICT")
        return dict(receipt, directory=str(folder))


def _prefix_coverage(parent, calendar, market, actions, selected, diagnostics, build):
    """All required days must be accounted for; missing bars never imply CLOSED."""
    first, previous = date.fromisoformat(market["start"]), date.fromisoformat(parent["history_start"])
    end = date.fromisoformat(parent["history_end"])
    if (first >= previous or market["end"] != (previous - timedelta(days=1)).isoformat()
            or (end - first).days + 1 > MAX_DAYS
            or timestamp(build["evaluation_cutoff"]).date() < end
            or not first.isoformat() <= build["event_start"] <= build["event_end"] <= end.isoformat()):
        raise ValueError("FR3_PREFIX_RANGE_OR_BUILD_INVALID")
    required = {day for day in selected["sessions"] if first.isoformat() <= day < previous.isoformat()}
    captured = [row["date"] for row in market["documents"]]
    if (diagnostics["unknown_dates"] or any(s == "UNKNOWN" for s in selected["days"].values())
            or market["calendar_sha256"] != digest(calendar)
            or market["status"] != "COMPLETE" or market.get("gaps")
            or len(captured) != len(set(captured)) or set(captured) != required):
        raise ValueError("FR3_PREFIX_COMPLETE_CALENDAR_AND_MARKET_REQUIRED")
    if actions["start"] > first.isoformat() or actions["end"] < end.isoformat():
        raise ValueError("FR3_PREFIX_COMPLETE_ACTION_RANGE_REQUIRED")
    sessions = selected["sessions"]
    if len(sessions) <= 20 or build["event_end"] > sessions[-21]:
        raise ValueError("FR3_PREFIX_MATURE_H20_EVENTS_REQUIRED")
    return first.isoformat(), len(required)


def prepend_fr3_scope(seed_scope, calendar_path, market_path, actions_path, *, build_config,
                      output_root, progress=None, guard_factory=ResearchVersionGuard):
    """Create a NEW immutable exact-universe scope from an earlier complete prefix.

    Original records, metadata and source references are preserved verbatim.
    Only strictly earlier market days are added, so this cannot overwrite an
    original day's values or silently choose between conflicting records. Full
    native source verification precedes publication of the discoverable scope.
    """
    build = deepcopy(build_config)
    validate_build_config(build)
    if not is_fr3_reference(build["seed"]["reference_policy"]):
        raise ValueError("FR3_PREFIX_EXPLICIT_FR3_CONFIG_REQUIRED")
    guard = guard_factory()
    guard.check()
    seed = Path(seed_scope).resolve()
    seed_root = seed.parent.parent.parent
    output = Path(output_root).resolve()
    if output.is_relative_to(seed_root):
        raise ValueError("FR3_PREFIX_OUTPUT_MUST_PRESERVE_SEED")
    original = _verify_freeze(seed, seed_root, verify_sources=True)
    parent = original["body"]
    if (parent["dataset_kind"] != "RESEARCH"
            or parent.get("reference_evidence_policy") != FINAL_REFERENCE):
        raise ValueError("FR3_PREFIX_REAL_FINAL_REFERENCE_SCOPE_REQUIRED")
    calendar, calendar_docs = load_final_calendar(calendar_path)
    market, market_docs = load_final_market(market_path)
    actions = load_actions_final(actions_path)
    action_docs = EvidenceDocuments(Path(actions_path).resolve().parent.parent / "objects")
    selected, diagnostics = select_final_calendar(calendar, build["evaluation_cutoff"],
        documents=calendar_docs, start=market["start"], end=parent["history_end"],
        evidence_id=digest(calendar))
    first, prefix_sessions = _prefix_coverage(parent, calendar, market, actions, selected, diagnostics, build)
    event_days = (date.fromisoformat(build["event_end"]) - date.fromisoformat(build["event_start"])).days + 1
    batch_size = min(parent["batch_size"], build["max_events"] // event_days)
    if batch_size < 1:
        raise ValueError("FR3_PREFIX_EVENT_SHARD_BUDGET_INSUFFICIENT")
    request = dict(schema=SCHEMA, parent_freeze_id=original["freeze_id"],
        calendar_id=digest(calendar), market_id=digest(market), actions_id=digest(actions),
        build_config_id=digest(build), history_start=first, history_end=parent["history_end"],
        universe_sha256=digest(parent["universe"]), batch_size=batch_size)
    root = output / digest(request)
    with _lock(root), _prefix_index(market, market_docs, parent["universe"]) as market_index:
        write_once(root / "request.json", request)
        first_path, _, _ = _load_bound_scope_shard(parent, seed_root, parent["chunks"][0])
        repaired = repair_reference_evidence(first_path, calendar_path=calendar_path,
            actions_path=actions_path, store=root / "reference")
        calendar_source, _ = load_evidence(repaired, max_bytes=MAX_BYTES)
        calendar["days"] = selected["days"]
        scope = {key: deepcopy(parent[key]) for key in SCOPE_FIELDS}
        scope.update(history_start=first, batch_size=batch_size,
            calendar_evidence_id=calendar_source, calendar_sha256=digest(calendar),
            market_supplement_sha256=digest(market), reference_evidence_policy=FINAL_REFERENCE,
            corporate_actions_packet_sha256=digest(actions))
        scope_id, chunks = digest(scope), []
        for index, symbols, payload, parents, stores in _repartition_shards(parent, seed_root, batch_size):
            guard.check()
            if progress:
                progress(dict(event="fr3_prefix_shard", index=index, symbol_count=len(symbols)))
            payload["original_calendar_before_fr3_expansion"] = payload["calendar"]
            payload["original_actions_before_fr3_expansion"] = payload.get("corporate_actions_final")
            if "market_supplement_final" in payload:
                payload["original_market_supplement_before_fr3_expansion"] = payload["market_supplement_final"]
            added = market_index.records(symbols)
            # The COMPLETE prefix contract already bound every original query.
            # Defend the strict disjoint-date invariant at the normalized port too.
            if any(not first.replace("-", "") <= row["date"] < parent["history_start"].replace("-", "")
                   for row in added):
                raise ValueError("FR3_PREFIX_RECORD_OVERLAPS_ORIGINAL_HISTORY")
            payload.update(start=first, calendar=deepcopy(calendar),
                corporate_actions_final=deepcopy(actions), market_supplement_final=deepcopy(market))
            payload["bars"].extend(added)
            payload["fr3_prefix_lineage"] = dict(request, parent_shards=parents,
                original_observation_receipts_added=False, historical_information_set_certified=False)
            payload["scope_binding"] = dict(scope_id=scope_id, chunk_index=index,
                complete_scope_size=len(parent["universe"]), calendar_source_evidence_id=calendar_source,
                original_universe_day=parent["universe_record"]["source_day"])
            child = create_dataset(payload, root / "scope" / "evidence",
                parent=Path(parents[0]["evidence_path"]), purpose="FR3_OFFICIAL_FINAL_HISTORY_PREFIX_NOT_PIT",
                document_reader=reference_document_reader(*stores, calendar_docs, action_docs, market_docs))
            chunks.append(dict(index=index, symbols=symbols, evidence_id=child.parent.name))
            for documents in stores:
                documents.check()
            del payload, added
        body = dict(schema=SCOPE_SCHEMA, **scope, scope_id=scope_id, chunks=chunks,
            historical_information_set_certified=False, original_transport_bytes_retained=False,
            original_observation_receipts_available=False, source_authenticity="NOT_INDEPENDENTLY_CERTIFIED")
        freeze_id = digest(body)
        manifest = root / "scope" / "freezes" / freeze_id / "manifest.json"
        candidate = manifest.parent / ".verification.json"
        if not candidate.exists():
            write_once(candidate, dict(freeze_id=freeze_id, body=body, created_at=utc_now()))
        checked = _verify_freeze(candidate, root / "scope", verify_sources=True)
        if checked["body"] != body or _verify_freeze(seed, seed_root, verify_sources=True) != original:
            raise ValueError("FR3_PREFIX_SOURCE_CHANGED_DURING_ASSEMBLY")
        for documents in (calendar_docs, action_docs, market_docs):
            documents.check()
        guard.check()
        write_once(manifest, checked)
        receipt = dict(request, schema="les-v7-fr3-source-prefix-completion/v1",
            dataset_id=freeze_id, scope_file=str(manifest), prefix_open_sessions=prefix_sessions,
            symbol_count=len(parent["universe"]), old_source_preserved=True,
            mutable_market_cache_read=False, model_fitted=False, historical_information_set_certified=False,
            research_only=True, approved_for_live=False, execution_action="NO_ACTION")
        guard.check()
        write_once(root / "completion.json", receipt)
        return receipt

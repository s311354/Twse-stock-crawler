"""Paired native R23/R24 research over exact saved originals, one stock at a time.

No quotes/fundamentals refresh, mail, fills, orders or account/PAE writes. Optional
current restriction/calendar observations remain separately identified and are
never passed off as evidence available at the saved historical source cutoff.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from application.daily_performance import DailyPerformance, measure_stage
from application.evidence_context import ReadOnlyEvidence
from application.les_v7 import prepare_quality_context
from application.low_entry_v7_daily import finalize_daily_report
from application.replay_contracts import digest
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.native_evidence import produce_native_receipts
from domain.les_v7.policy_r24 import AVAILABLE_AT, BASELINE_VERSION
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy
from domain.les_v7.source_projection import join_official_pe
from domain.strategy import get_strategy
from examples import compare_les_r14 as sources
from examples.compare_les_r13 import file_hash
from examples.compare_les_r17 import differences
from infrastructure.daily_work import open_daily_work
from infrastructure.research_archive import write_once, write_bytes_once
from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
from infrastructure.storage.report_evidence import ReportEvidenceReader, ReportEvidenceWriter
from infrastructure.storage.valuation_source_verification import (
    valuation_history_verifier_scope, verify_report_valuation_sources,
)

PREVIOUS = "V7_RULE_BASED_BASELINE_R23"
SCHEMA = "les-r23-r24-fixed-source-native-comparison/v1"
OVERRIDE_SCHEMA = "les-r24-current-source-overrides/v1"


def load_overrides(path, *, cutoff, symbols):
    """Verify a separately retained observation packet; native validators recur.

    Its digest identifies this capture; it is not a signature or certification
    of a historical information set. Transport originals may additionally be
    bound by the producer through documents[path,sha256].
    """
    if path is None:
        return None
    value = json.loads(Path(path).read_text())
    body = {k: v for k, v in value.items() if k != "content_sha256"}
    if (value.get("schema") != OVERRIDE_SCHEMA or digest(body) != value.get("content_sha256")
            or timestamp(value["captured_at"]) > timestamp(cutoff)
            or not isinstance(value.get("by_symbol"), dict)
            or not set(value["by_symbol"]) <= set(symbols)):
        raise ValueError("R24_CURRENT_SOURCE_OVERRIDE_INVALID")
    for symbol, node in value["by_symbol"].items():
        if not isinstance(node, dict) or set(node) - {"restriction_evidence", "calendar_evidence"}:
            raise ValueError("R24_CURRENT_SOURCE_OVERRIDE_FIELDS_INVALID:" + symbol)
        for key, evidence in node.items():
            if not isinstance(evidence, dict):
                raise ValueError("R24_CURRENT_SOURCE_OVERRIDE_OBJECT_REQUIRED:" + key)
            for name in ("available_at", "observed_at", "source_available_at"):
                if evidence.get(name) is not None and timestamp(evidence[name]) > timestamp(cutoff):
                    raise ValueError("R24_CURRENT_SOURCE_OVERRIDE_FUTURE:" + key)
    if any("restriction_evidence" in node for node in value["by_symbol"].values()):
        verify_restriction_originals(value, root=Path(path).resolve().parent, cutoff=cutoff)
    return value


def retained_path(root, relative):
    path = Path(relative)
    target = (root / path).resolve()
    if path.is_absolute() or not target.is_relative_to(root.resolve()):
        raise ValueError("R24_CURRENT_SOURCE_ORIGINAL_PATH_ESCAPE")
    return target


def verify_restriction_originals(value, *, root, cutoff):
    """Reproject all retained official responses with the existing collector."""
    from domain.trading_restrictions import RESTRICTION_ENDPOINTS
    from infrastructure.crawler.trading_restrictions import TradingRestrictionClient
    meta = value.get("metadata", {})
    if (meta.get("schema") != "les-r24-retained-restriction-originals/v1"
            or meta.get("market") != "TWSE"
            or meta.get("projection_policy") != "EXISTING_TRADING_RESTRICTION_CLIENT_AVAILABLE_AT_CAPTURE_COMPLETION"
            or meta.get("completed_at") != value["captured_at"]
            or not isinstance(meta.get("symbols"), list)
            or len(meta["symbols"]) != len(set(meta["symbols"]))
            or set(meta["symbols"]) != set(value["by_symbol"])):
        raise ValueError("R24_CURRENT_RESTRICTION_CAPTURE_CONTRACT_INVALID")
    start, end = timestamp(meta["started_at"]), timestamp(meta["completed_at"])
    day = start.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat()
    if (not start <= end <= timestamp(cutoff) or meta.get("session_date") != day
            or end.astimezone(ZoneInfo("Asia/Taipei")).date().isoformat() != day):
        raise ValueError("R24_CURRENT_RESTRICTION_CAPTURE_CLOCK_INVALID")
    documents = meta.get("documents", ())
    expected = {"https://openapi.twse.com.tw/v1/" + part for part in RESTRICTION_ENDPOINTS}
    if (len(documents) != len(expected) or {doc.get("url") for doc in documents} != expected):
        raise ValueError("R24_CURRENT_RESTRICTION_ORIGINAL_SCOPE_INVALID")
    payloads = {}
    for doc in documents:
        raw = retained_path(root, doc["path"]).read_bytes()
        if (sha256(raw).hexdigest() != doc["content_sha256"] or doc.get("status_code") != 200
                or not start <= timestamp(doc["fetched_at"]) <= end):
            raise ValueError("R24_CURRENT_SOURCE_ORIGINAL_HASH_OR_TIME_MISMATCH")
        payload = json.loads(raw)
        if digest(payload) != doc["payload_sha256"]:
            raise ValueError("R24_CURRENT_SOURCE_PAYLOAD_HASH_MISMATCH")
        payloads[doc["url"]] = payload

    class RetainedClient:
        def get_json(self, url, *, context):
            return SimpleNamespace(payload=payloads[url])

    rebuilt = TradingRestrictionClient(RetainedClient()).load(
        meta["symbols"], session_date=day, observed_at=start)
    for symbol, evidence in rebuilt.items():
        evidence["available_at"] = meta["completed_at"]
        if value["by_symbol"][symbol].get("restriction_evidence") != evidence:
            raise ValueError("R24_CURRENT_RESTRICTION_PROJECTION_MISMATCH:" + symbol)


def enrich_filings(saved, filings, *, companion_loader, cutoff, source_symbol=None):
    """R7 adds a proof to the exact restored vector, never reads latest values."""
    binding = saved["diagnostics"]["evidence_binding"]
    if identity(filings) != binding["financials_hash"]:
        raise ValueError("SAVED_FINANCIAL_VECTOR_HASH_REQUIRED")
    original = saved["diagnostics"]["native_evidence"]["source_packet"]
    source_cutoff = original["cutoff"]
    if any(timestamp(row["available_at"]) > timestamp(source_cutoff) for row in filings):
        raise ValueError("R24_SAVED_FINANCIAL_VECTOR_POST_SNAPSHOT")
    market = binding["market"]
    companions = companion_loader(source_symbol or saved["symbol"], as_of=source_cutoff,
        records=filings, market=market, audit_repair_r7=True)
    context = dict(symbol=saved["symbol"], market=market, as_of=cutoff, filings=filings,
        model_bundle=None, rule_baseline=True, rule_baseline_policy_version=BASELINE_VERSION,
        quality_policy=baseline_policy(market, version=BASELINE_VERSION)["quality_policy"])
    prepare_quality_context(context, financial_audits_r7=companions.get("financial_audits_r7", {}))
    new = list(context["filings"])
    # Only the new proof may differ. No changed CFO, tax, EPS, period, scope,
    # revision, source availability, old audit or cash source can ride along.
    stripped = [{k: v for k, v in row.items() if k != "audit_evidence_r7"} for row in new]
    if identity(stripped) != identity(filings):
        raise ValueError("R24_ORIGINAL_FINANCIAL_OPERAND_CHANGED")
    return new, dict(original_financials_sha256=identity(filings),
        r7_financials_sha256=identity(new), selected_source_cutoff=source_cutoff,
        r7_companions_sha256=digest(companions.get("financial_audits_r7", {})),
        added_audit_proof_count=sum("audit_evidence_r7" in row for row in new),
        original_values_and_times_unchanged=True)


def _run(saved, filings, packet, *, cutoff, version):
    """Public native strategy + daily finalization, with explicit actual hashes."""
    binding = saved["diagnostics"]["evidence_binding"]
    bars = packet["market_rows"]
    if identity(bars) != binding["bars_hash"]:
        raise ValueError("SAVED_BAR_VECTOR_HASH_REQUIRED")
    context = dict(symbol=saved["symbol"], market=binding["market"], as_of=cutoff,
        filings=filings, model_bundle=None, quality_policy=None, receipts={},
        external_evidence_ids=binding["external_evidence_ids"], rule_baseline=True,
        rule_baseline_policy_version=version, native_evidence_source=packet)
    with measure_stage(version.rsplit("_", 1)[-1].lower() + "_native"):
        report = get_strategy("low_entry_score_v7").evaluate(pd.DataFrame(bars),
            context=ReadOnlyEvidence(context), receipt_producer=produce_native_receipts)
        report = finalize_daily_report(report,
            account=sources._account(saved.get("pae_authorization", {}).get("account"), cutoff),
            daily_input_errors=saved["diagnostics"].get("daily_input_errors", ()))
        validate_report(report)
    actual = report["diagnostics"]["evidence_binding"]
    for name in ("symbol", "market", "bars_hash", "external_evidence_ids"):
        if actual[name] != binding[name]:
            raise ValueError("R24_PAIRED_RAW_INPUT_CHANGED:" + name)
    if (actual["financials_hash"] != identity(filings)
            or timestamp(actual["cutoff"]) != timestamp(cutoff)
            or report["pae_authorization"]["live_order_authorized"] is not False
            or report["pae_authorization"]["account"] != sources._account(
                saved["pae_authorization"]["account"], cutoff)):
        raise ValueError("R24_PAIRED_BINDING_OR_ACCOUNT_MISMATCH")
    return report


def compare_report(saved, filings, new_filings, *, cutoff, overrides=None):
    cutoff = timestamp(cutoff).isoformat()
    if timestamp(cutoff) < timestamp(AVAILABLE_AT):
        raise ValueError("R24_BEFORE_RELEASE")
    validate_report(saved)
    if identity(filings) != saved["diagnostics"]["evidence_binding"]["financials_hash"]:
        raise ValueError("SAVED_FINANCIAL_VECTOR_HASH_REQUIRED")
    if identity([{k: v for k, v in row.items() if k != "audit_evidence_r7"}
                 for row in new_filings]) != identity(filings):
        raise ValueError("R24_ORIGINAL_FINANCIAL_OPERAND_CHANGED")
    original = saved["diagnostics"]["native_evidence"]["source_packet"]
    visibility = sources.assert_snapshot_visibility(original)
    packet = sources.reselect_packet(original, evaluation_cutoff=cutoff)
    if overrides is not None:
        packet.update(deepcopy(overrides.get("by_symbol", {}).get(saved["symbol"], {})))
    packet["market_rows"], packet["source_join_diagnostics"] = join_official_pe(
        packet["quote_market_rows"], packet["official_pe"], symbol=saved["symbol"],
        market=saved["diagnostics"]["evidence_binding"]["market"], cutoff=cutoff,
        source_policy=baseline_policy(version=PREVIOUS)["source_policy"])
    old = _run(saved, filings, packet, cutoff=cutoff, version=PREVIOUS)
    new = _run(saved, new_filings, packet, cutoff=cutoff, version=BASELINE_VERSION)
    if (old["pae_authorization"]["account_sha256"] != new["pae_authorization"]["account_sha256"]
            or old["diagnostics"]["native_evidence"]["source_packet"] !=
               new["diagnostics"]["native_evidence"]["source_packet"]):
        raise ValueError("R24_PAIRED_ACCOUNT_OR_SOURCE_SELECTION_CHANGED")
    return old, new, dict(snapshot_visibility=visibility,
        original_source_packet_sha256=digest(original), selected_source_packet_sha256=digest(packet),
        source_override_sha256=overrides["content_sha256"] if overrides else None,
        original_financials_sha256=identity(filings), r7_financials_sha256=identity(new_filings),
        same_bars_account_source_packet=True,
        business=list(differences(sources.business(old), sources.business(new))),
        complete=list(differences(old, new)))


def _summary(rows):
    result = dict(total=len(rows), states=dict(Counter(r["state"] for r in rows)))
    for name in ("r23", "r24"):
        usable = [row[name] for row in rows if name in row]
        result[name] = dict(decisions=dict(Counter(r["strategy"]["strategy_decision"] for r in usable)),
            quality=dict(Counter(r["quality"]["status"] for r in usable)),
            final_zero_count=sum(r["score"]["final_score"] == 0 for r in usable),
            final_missing_count=sum(r["score"]["final_score"] is None for r in usable))
    return result


def select_shard(symbols, *, index, count):
    if (type(count) is not int or type(index) is not int or not 1 <= count <= 16
            or not 0 <= index < count or count > len(symbols)):
        raise ValueError("R24_SHARD_RANGE_INVALID")
    return [symbol for number, symbol in enumerate(symbols) if number % count == index]


def merge_shards(directory, *, comparison_root):
    """Verify compact indexes/one row at a time, never decode aggregate trees."""
    directory, comparison_root = Path(directory).resolve(), Path(comparison_root).resolve()
    original = json.loads((comparison_root / "comparison-index.json").read_text())
    original_done = json.loads((comparison_root / "complete-index.json").read_text())
    if digest(original) != original_done["index_sha256"]:
        raise ValueError("SOURCE_INDEX_HASH_MISMATCH")
    original_symbols = [r["symbol"] for r in original["records"]]
    expected_source = {r["symbol"]: r["row_sha256"] for r in original["records"]}
    first = json.loads((directory / "shard-00/job.json").read_text())
    common = ("schema", "source_index_sha256", "versions", "cutoff", "full_symbols", "shard_count",
              "source_snapshot_policy", "source_override_sha256", "account_policy", "program_sha256",
              "order_authority", "mail_enabled", "account_or_pae_writes")
    if (first["source_index_sha256"] != digest(original) or not first["full_symbols"]
            or len(first["full_symbols"]) != len(set(first["full_symbols"]))
            or not set(first["full_symbols"]) <= set(original_symbols)):
        raise ValueError("R24_MERGE_ORIGINAL_SCOPE_MISMATCH")
    records, children, rows = {}, [], []
    for number in range(first["shard_count"]):
        child = directory / f"shard-{number:02d}"
        job = json.loads((child / "job.json").read_text())
        index = json.loads((child / "comparison-index.json").read_text())
        receipt = json.loads((child / "complete-index.json").read_text())
        if (any(job.get(k) != first[k] for k in common) or job.get("shard_index") != number
                or job["symbols"] != select_shard(first["full_symbols"], index=number, count=first["shard_count"])
                or digest(index) != receipt["index_sha256"] or index.get("job_sha256") != digest(job)
                or file_hash(child / "programs" / Path(__file__).name) != first["program_sha256"]
                or file_hash(child / "report_evidence.sqlite3") != receipt.get("report_store_sha256")
                or len(index["records"]) != len(job["symbols"])):
            raise ValueError("R24_SHARD_IDENTITY_OR_COMPLETION_MISMATCH")
        child_rows = []
        for ordinal, entry in enumerate(index["records"]):
            path = retained_path(child, entry["row_path"])
            row = json.loads(path.read_text())
            symbol = row["symbol"]
            if (digest(row) != entry["row_sha256"] or symbol != job["symbols"][ordinal]
                    or symbol != entry["symbol"] or symbol in records
                    or row["state"] != entry["state"]
                    or row["source_row_sha256"] != expected_source[symbol]):
                raise ValueError("R24_SHARD_ROW_BINDING_MISMATCH")
            child_rows.append({k: row[k] for k in ("symbol", "state", "r23", "r24") if k in row})
            records[symbol] = dict(entry, row_path=str(path.relative_to(directory)), shard=number)
        if _summary(child_rows) != receipt["summary"] or receipt["summary"] != index["summary"]:
            raise ValueError("R24_SHARD_SUMMARY_MISMATCH")
        rows.extend(child_rows)
        children.append(dict(directory=child.name, complete_sha256=file_hash(child / "complete-index.json"),
            index_sha256=digest(index), report_store_sha256=receipt["report_store_sha256"]))
    if set(records) != set(first["full_symbols"]):
        raise ValueError("R24_MERGE_SCOPE_MISMATCH")
    result = dict(schema=SCHEMA, source_index_sha256=digest(original), cutoff=first["cutoff"],
        source_override_sha256=first["source_override_sha256"], versions=first["versions"],
        fixed_symbols=first["full_symbols"], full_original_universe_processed=first["full_symbols"] == original_symbols,
        records=[records[s] for s in first["full_symbols"]], summary=_summary(rows), children=children,
        full_reports_and_business_differences_retained_in_child_stores=True,
        score_policy_changed=False, oos_or_performance_claim=False, simulated_fills=0,
        order_authority=False, account_or_pae_writes=False)
    write_once(directory / "comparison-index.json", result)
    write_once(directory / "complete-index.json", dict(index_sha256=digest(result), summary=result["summary"]))
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return 0 if result["summary"]["states"].get("SOURCE_OR_RECOMPUTATION_GAP", 0) == 0 else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--fundamental-db", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluation-cutoff")
    parser.add_argument("--source-overrides", type=Path)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-shards", type=Path)
    args = parser.parse_args(argv)
    if args.merge_shards is not None:
        return merge_shards(args.merge_shards, comparison_root=args.comparison_root)
    if args.fundamental_db is None or args.output_dir is None or args.evaluation_cutoff is None:
        parser.error("--fundamental-db, --output-dir and --evaluation-cutoff are required for execution")
    cutoff = timestamp(args.evaluation_cutoff).isoformat()
    if timestamp(cutoff) < timestamp(AVAILABLE_AT):
        raise ValueError("R24_BEFORE_RELEASE")
    root = args.comparison_root.resolve()
    index = json.loads((root / "comparison-index.json").read_text())
    complete = json.loads((root / "complete-index.json").read_text())
    if digest(index) != complete["index_sha256"]:
        raise ValueError("SOURCE_INDEX_HASH_MISMATCH")
    entries = {row["symbol"]: row for row in index["records"]}
    symbols = args.symbols or list(entries)
    if (len(entries) != len(index["records"]) or len(symbols) != len(set(symbols))
            or not symbols or not set(symbols) <= set(entries)):
        raise ValueError("SOURCE_SCOPE_MISMATCH")
    full_symbols = list(symbols)
    symbols = select_shard(full_symbols, index=args.shard_index, count=args.shard_count)
    overrides = load_overrides(args.source_overrides, cutoff=cutoff, symbols=entries)
    # New directory is deliberately required. Old results/checkpoints are never
    # overwritten or silently resumed after a program or source change.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    performance = DailyPerformance()
    guard = ResearchVersionGuard()
    runtime = guard.retain(ResearchObjects(args.output_dir / "runtime/objects"))
    job = dict(schema=SCHEMA, source_index_sha256=digest(index), versions=asdict(guard.versions),
        runtime=runtime, cutoff=cutoff, symbols=symbols, full_symbols=full_symbols,
        shard_count=args.shard_count, shard_index=args.shard_index, program_sha256=file_hash(Path(__file__)),
        source_snapshot_policy="EXACT_SAVED_ORIGINALS",
        source_override_sha256=overrides["content_sha256"] if overrides else None,
        account_policy="EXPLICIT_SAVED_VALUES_COUNTERFACTUAL_NOT_ACTUAL_ACCOUNT_UPDATE",
        order_authority=False, mail_enabled=False, account_or_pae_writes=False)
    write_once(args.output_dir / "job.json", job)
    write_bytes_once(args.output_dir / "programs" / Path(__file__).name, Path(__file__).read_bytes())
    if overrides is not None:
        write_once(args.output_dir / "source-overrides.json", overrides)
        for doc in overrides.get("metadata", {}).get("documents", ()):
            raw = retained_path(args.source_overrides.resolve().parent, doc["path"]).read_bytes()
            write_bytes_once(args.output_dir / doc["path"], raw)
    rows = []
    with ExitStack() as stack:
        stack.enter_context(performance.activate())
        stack.enter_context(open_daily_work(max_bytes=64 * 1024 * 1024))
        verifier = stack.enter_context(valuation_history_verifier_scope())
        restorer = sources.HashMatchedFilings(args.fundamental_db, symbols)
        stack.callback(restorer.close)
        writer = stack.enter_context(ReportEvidenceWriter(args.output_dir / "report_evidence.sqlite3"))
        readers = {}
        for number, symbol in enumerate(symbols):
            guard.check()
            if file_hash(Path(__file__)) != job["program_sha256"]:
                raise ValueError("R24_COMPARISON_PROGRAM_CHANGED_DURING_JOB")
            began = perf_counter()
            entry = entries[symbol]
            record = dict(symbol=symbol, source_row_sha256=entry["row_sha256"],
                state="SOURCE_OR_RECOMPUTATION_GAP")
            try:
                path = (root / entry["row_path"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("SOURCE_ROW_PATH_ESCAPE")
                with measure_stage("saved_report_read_verify"):
                    row = json.loads(path.read_text())
                    if digest(row) != entry["row_sha256"]:
                        raise ValueError("SOURCE_ROW_HASH_MISMATCH")
                    if path.parent.parent not in readers:
                        readers[path.parent.parent] = stack.enter_context(ReportEvidenceReader(path.parent.parent))
                    saved = readers[path.parent.parent].resolve(row["r20_report"])
                    verify_report_valuation_sources(saved, verifier=verifier)
                    if saved["symbol"] != symbol:
                        raise ValueError("SOURCE_ROW_SYMBOL_MISMATCH")
                with measure_stage("exact_financial_restore"):
                    filings, restoration = restorer.restore(saved)
                with measure_stage("r7_source_enrichment"):
                    new_filings, enriched = enrich_filings(saved, filings,
                        companion_loader=restorer.service.load_v7_financial_sources, cutoff=cutoff,
                        source_symbol=restoration.get("restored_source_symbol", symbol))
                old, new, diff = compare_report(saved, filings, new_filings, cutoff=cutoff, overrides=overrides)
                with measure_stage("result_retention"):
                    refs = dict(r23=writer.retain(old), r24=writer.retain(new))
                record.update(state="FULL_NATIVE_RECOMPUTED", reports=refs, restoration=restoration,
                    enrichment=enriched, signal_date=saved["signal_date"], differences=diff,
                    r23=old["envelope"], r24=new["envelope"],
                    quality_reasons={v: r["diagnostics"]["quality"].get("reason_codes", [])
                                     for v, r in (("r23", old), ("r24", new))},
                    gate_states={v: {g["gate_id"]: g["state"] for g in r["diagnostics"]["gate_receipts"]}
                                 for v, r in (("r23", old), ("r24", new))})
                del saved, filings, new_filings, old, new, diff, row
            except (OSError, KeyError, TypeError, ValueError) as error:
                record["error"] = str(error)
            record["duration_seconds"] = perf_counter() - began
            guard.check()
            if file_hash(Path(__file__)) != job["program_sha256"]:
                raise ValueError("R24_COMPARISON_PROGRAM_CHANGED_DURING_JOB")
            write_once(args.output_dir / "rows" / f"{number:04d}.json", record)
            rows.append(dict(symbol=symbol, state=record["state"], row_path=f"rows/{number:04d}.json",
                row_sha256=digest(record), **{k: record[k] for k in ("r23", "r24") if k in record}))
            print(json.dumps(dict(completed=number + 1, total=len(symbols), symbol=symbol,
                state=record["state"], seconds=record["duration_seconds"])), flush=True)
    guard.check()
    summary = _summary(rows)
    result = dict(schema=SCHEMA, job_sha256=digest(job), summary=summary,
        records=[{k: r[k] for k in ("symbol", "state", "row_path", "row_sha256")} for r in rows],
        total_wall_seconds=perf_counter() - started, performance=performance.snapshot(),
        score_policy_changed=False, oos_or_performance_claim=False, simulated_fills=0,
        order_authority=False, account_or_pae_writes=False)
    write_once(args.output_dir / "comparison-index.json", result)
    write_once(args.output_dir / "complete-index.json", dict(index_sha256=digest(result), summary=summary,
        report_store_sha256=file_hash(args.output_dir / "report_evidence.sqlite3")))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["states"].get("SOURCE_OR_RECOMPUTATION_GAP", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

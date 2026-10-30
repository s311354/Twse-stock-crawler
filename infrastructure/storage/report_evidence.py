"""Daily-report compatibility facade over the shared immutable evidence graph."""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from application.daily_performance import count_work, measure_stage
from application.replay_contracts import canonical_json, digest
from infrastructure.storage.evidence_graph import (
    EvidenceGraphReader, EvidenceGraphWriter, MAX_NODE_BYTES, MAX_REPORT_BYTES, object_id,
)
from infrastructure.storage.report_bundle_stream import iter_bundle_records

SCHEMA = "daily-report-reference/v1"
STORE_NAME = "report_evidence.sqlite3"


@dataclass
class ReportBundleContext:
    """One stream's envelope and current bucket; never retains report contents.

    Metadata may follow the arrays. Only complete=True after EOF certifies the
    whole envelope. Counts distinguish referenced verification from legacy parse.
    """
    metadata: dict[str, str | None] = field(default_factory=dict)
    bucket: str | None = None
    referenced: bool | None = None
    records: int = 0
    referenced_records: int = 0
    complete: bool = False
    _started: bool = field(default=False, repr=False)


def _object_id(value):
    return object_id(value, "REPORT_EVIDENCE")


class ReportEvidenceWriter(EvidenceGraphWriter):
    error_prefix = "REPORT_EVIDENCE"
    counter_prefix = "report_evidence"

    def retain(self, report):
        with measure_stage("report_evidence_write"):
            with measure_stage("report_evidence_encode"):
                root = self._encode(report)
            with measure_stage("report_evidence_value_hash"):
                sha = digest(report)
            ref = dict(schema=SCHEMA, store=self.path.name, root=root,
                       report_sha256=sha, symbol=report["symbol"],
                       signal_date=report["signal_date"], model_version=report["model_version"],
                       parameter_version=report["parameter_version"])
            count_work("reports_retained")
            return ref


class ReportEvidenceReader(EvidenceGraphReader):
    error_prefix = "REPORT_EVIDENCE"
    counter_prefix = "report_evidence"

    def __init__(self, directory, *, cache_bytes=16 * 1024 * 1024,
                 max_value_bytes=MAX_REPORT_BYTES, batch_nodes=128, batch_bytes=4 * 1024 * 1024):
        if type(max_value_bytes) is not int or not 1 <= max_value_bytes <= MAX_REPORT_BYTES:
            raise ValueError("INVALID_REPORT_EVIDENCE_MAX_VALUE_BYTES")
        super().__init__(directory, cache_bytes=cache_bytes, max_value_bytes=max_value_bytes,
                         batch_nodes=batch_nodes, batch_bytes=batch_bytes)

    def resolve(self, value):
        if not isinstance(value, dict):
            raise ValueError("INVALID_REPORT_EVIDENCE_REFERENCE")
        ref = value.get("_report_reference", value)
        if not isinstance(ref, dict):
            raise ValueError("INVALID_REPORT_EVIDENCE_REFERENCE")
        if ref.get("schema") != SCHEMA:
            return value
        store = ref.get("store")
        self.check_store(store)
        with measure_stage("report_evidence_read"):
            report = self._decode(ref["root"], store, [0])
            if not isinstance(report, dict) or digest(report) != _object_id(ref["report_sha256"]):
                raise ValueError("REPORT_EVIDENCE_REPORT_HASH_MISMATCH")
            if any(key not in report or key not in ref or type(report[key]) is not type(ref[key]) or report[key] != ref[key]
                   for key in ("symbol", "signal_date", "model_version", "parameter_version")):
                raise ValueError("REPORT_EVIDENCE_BINDING_MISMATCH")
            if "_report_reference" in value and canonical_json(compact_report(report, ref)) != canonical_json(value):
                raise ValueError("REPORT_EVIDENCE_SUMMARY_MISMATCH")
            return report

    def iter_reports(
        self, values: Iterable[dict[str, Any]], *, require_reference: bool = True,
        context: ReportBundleContext | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield detached reports only after the same complete resolve validation.

        Legacy embedded JSON remains readable, but has no independent stored
        digest. Consumers requiring that guarantee must set require_reference.
        """
        for value in values:
            ref = value.get("_report_reference", value) if isinstance(value, dict) else None
            referenced = isinstance(ref, dict) and ref.get("schema") == SCHEMA
            if not referenced and (require_reference or isinstance(value, dict) and "_report_reference" in value):
                raise ValueError("REPORT_EVIDENCE_REFERENCE_REQUIRED")
            report = self.resolve(value)
            if not referenced:
                # Preserve resolve()'s legacy API without sharing mutable stream output.
                size = 0
                try:
                    for part in json.JSONEncoder(ensure_ascii=False).iterencode(report):
                        size += len(part.encode())
                        if size > self.max_value_bytes:
                            raise ValueError("REPORT_EVIDENCE_EXPANSION_TOO_LARGE")
                    report = deepcopy(report)
                except RecursionError as error:
                    raise ValueError("INVALID_REPORT_EVIDENCE_NODE") from error
            if context is not None:
                context.referenced = referenced
                context.records += 1
                context.referenced_records += int(referenced)
            yield report
            del report, value, ref


def iter_report_bundle(
    path: str | Path, *, buckets: tuple[str, ...] = ("reports", "stale_reports"),
    require_reference: bool = True, cache_bytes: int = 16 * 1024 * 1024,
    batch_nodes: int = 128, batch_bytes: int = 4 * 1024 * 1024,
    max_value_bytes: int = MAX_REPORT_BYTES, max_record_bytes: int = MAX_REPORT_BYTES,
    chunk_bytes: int = 64 * 1024,
    context: ReportBundleContext | None = None,
    valuation_history_verifier_scope=None,
) -> Iterator[dict[str, Any]]:
    """Public read-only stream for CLI/BankEmulator, in bundle file order.

    Exhaust to validate the entire envelope; earlier verified reports are not
    revoked if a later record is invalid. Close the iterator when stopping early.
    No report collection is retained by this iterator or by its node cache.
    An optional fresh context exposes bucket identity and small envelope fields
    without requiring a second read or retaining compact report arrays.
    """
    if context is not None:
        if not isinstance(context, ReportBundleContext) or context._started:
            raise ValueError("INVALID_REPORT_BUNDLE_CONTEXT")
        context._started = True
        context.metadata.clear()
        context.bucket = context.referenced = None
        context.records = context.referenced_records = 0
        context.complete = False
    path = Path(path)
    from infrastructure.storage.valuation_source_verification import (
        valuation_history_verifier_scope as default_verifier_scope, verify_report_valuation_sources,
    )
    scope = valuation_history_verifier_scope or default_verifier_scope
    with ReportEvidenceReader(path.parent, cache_bytes=cache_bytes, batch_nodes=batch_nodes,
                              batch_bytes=batch_bytes, max_value_bytes=max_value_bytes) as reader, scope() as verifier:
        records = iter_bundle_records(path, buckets=buckets, chunk_bytes=chunk_bytes,
                                      max_record_bytes=max_record_bytes, context=context)
        try:
            for report in reader.iter_reports(records, require_reference=require_reference, context=context):
                verify_report_valuation_sources(report, verifier=verifier)
                yield report
                del report
            if context is not None:
                context.complete = True
        finally:
            records.close()


def compact_report(report, ref):
    """Display/business summary only. Not an executable native report until resolved."""
    value = dict(report, _report_reference=ref)
    quality = dict(report.get("quality", {}))
    quality["metrics"] = {k: v for k, v in quality.get("metrics", {}).items()
                          if k not in {"quality_field_checks", "quality_ttm_inputs", "quality_reported_inputs"}}
    value["quality"] = quality
    if "research" in report:
        research = dict(report["research"])
        research["verification"] = {k: v for k, v in research.get("verification", {}).items()
                                    if k not in {"field_checks", "field_gaps"}}
        # before_refresh may contain the same complete verification ledger.
        research["before_refresh"] = {k: v for k, v in research.get("before_refresh", {}).items()
                                      if k not in {"field_checks", "field_gaps"}}
        value["research"] = research
    return value


def load_report_bundle(path):
    from infrastructure.storage.valuation_source_verification import (
        valuation_history_verifier_scope, verify_report_valuation_sources,
    )
    path = Path(path)
    with measure_stage("report_bundle_read"):
        payload = json.loads(path.read_bytes())
        with ReportEvidenceReader(path.parent) as reader, valuation_history_verifier_scope() as verifier:
            for bucket in ("reports", "stale_reports"):
                if bucket in payload:
                    resolved = []
                    for value in payload[bucket]:
                        report = reader.resolve(value)
                        verify_report_valuation_sources(report, verifier=verifier)
                        resolved.append(report)
                    payload[bucket] = resolved
        return payload


def resolve_csv_report(cell, directory):
    from infrastructure.storage.valuation_source_verification import verify_report_valuation_sources
    with ReportEvidenceReader(directory) as reader:
        report = reader.resolve(json.loads(cell))
        verify_report_valuation_sources(report)
        return report

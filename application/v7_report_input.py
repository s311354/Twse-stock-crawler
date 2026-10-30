"""Detached native report inputs with job-local successful validation reuse.

Successful validation tokens bind exact immutable JSON bytes, never a mutable
report, source-file verdict or account calculation. A separate compact frozen
projection supports rechecking caller fields without retaining the report graph.
The public domain validator remains unconditional for callers that own mutable report objects.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from application.daily_performance import count_work, measure_stage
from application.daily_work import current_work


VALIDATION_CACHE_VERSION = "native-v7-json-validation/v1"


def read_v7_report_json(encoded: str) -> dict:
    """Parse independently and fully validate an exact report once per job.

    Changed bytes, another work version, eviction, a disabled cache or no work
    context all invoke the original validator. Failures are never stored. Each
    return is freshly decoded, so downstream mutations cannot poison a hit.
    External retained-source checks deliberately remain at their I/O boundary.
    """
    if type(encoded) is not str:
        raise TypeError("V7_REPORT_JSON_STRING_REQUIRED")
    with measure_stage("v7_report_decode"):
        report = json.loads(encoded)
        count_work("v7_report_json_decodes")

    def validate():
        from domain.les_v7.report import validate_report
        with measure_stage("v7_report_full_validation"):
            count_work("v7_report_full_validations")
            validate_report(report)
        return True

    work = current_work()
    if work is None or not work.enabled:
        validate()
    else:
        with measure_stage("v7_report_validation_key"):
            raw = encoded.encode("utf-8")
            key = VALIDATION_CACHE_VERSION, hashlib.sha256(raw).hexdigest(), len(raw)
            count_work("v7_report_validation_input_bytes", len(raw))
        # A small proof key, not the JSON/body graph, occupies the independently
        # bounded token LRU. The existing version guard remains independent.
        work.validate_once("v7_report_validation", key, validate)
    return report


@dataclass(frozen=True, slots=True)
class V7ReportBinding:
    """Compact immutable operands for rechecking each mutable decision/account.

    Not a model decision cache: every caller still compares its actual fields,
    timestamp and current account. Full reports remain detached mutable reads.
    """
    symbol: str
    cutoff_at: str
    signal_date: str
    close: float
    fields: tuple
    authorized: bool
    account_value: float | None
    account_cash: float | None
    account_positions: tuple


def v7_report_binding(encoded: str) -> V7ReportBinding:
    """Avoid repeat multi-MB decode/encode for exact validated canonical bytes.

    Reuse requires BOTH a bounded immutable header and a still-live successful
    validation token for these exact bytes, validator policy and work version.
    Eviction, exceptions, disabled caches and changed bytes fully revalidate.
    Source-file checks and caller-specific account checks remain unconditional.
    """
    if type(encoded) is not str:
        raise TypeError("V7_REPORT_JSON_STRING_REQUIRED")
    with measure_stage("v7_report_binding_key"):
        raw = encoded.encode("utf-8")
        key = VALIDATION_CACHE_VERSION, hashlib.sha256(raw).hexdigest(), len(raw)

    def build():
        from application.low_entry_v7_daily import _project_validated_report
        report = read_v7_report_json(encoded)
        with measure_stage("v7_report_binding_build"):
            # Preserve the old StrategyDecisionSchema's exact canonical-string
            # check; alternate whitespace/key order is not silently accepted.
            canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
            if canonical != encoded:
                raise ValueError("v7 projection mismatch:LES v7 Report")
            projection = _project_validated_report(report, include_report=False)
            if any(type(v) not in (str, int, float, bool, type(None)) for v in projection.values()):
                raise ValueError("V7_BINDING_NON_SCALAR_PROJECTION")
            authorization = report.get("pae_authorization") or {}
            account = authorization.get("account") or {}
            result = V7ReportBinding(report["symbol"], report["envelope"]["event"]["cutoff_at"],
                report["signal_date"], report["diagnostics"]["features"]["close"],
                tuple(projection.items()), authorization.get("authorized") is True,
                account.get("value"), account.get("available_cash"),
                tuple((p["symbol"], p["shares"]) for p in account.get("positions", ())))
            count_work("v7_report_binding_builds")
            return result

    work = current_work()
    if work is None or not work.enabled:
        return build()
    # A header cannot survive invalidation of its validation proof. Building
    # first when that token is absent preserves the existing failure lifecycle.
    fresh = None if work.validation_cached("v7_report_validation", key) else build()
    return work.memo_immutable("v7_report_binding", key, lambda: fresh if fresh is not None else build(),
        size=lambda value: 512 + sum(len(k.encode()) + len(str(v).encode()) + 64 for k,v in value.fields)
                           + sum(len(str(p).encode()) + 64 for p in value.account_positions))

"""Read-only, bounded R13 valuation arithmetic comparison of retained reports.

This deliberately is NOT a second LES engine or source re-verification path.
The existing graph reader verifies the hashes of traversed section nodes only;
it does not expand/verify the full report hash or financial evidence graph.
Stored inputs are reused without asserting their source authenticity or PIT
availability. The output is not a new score, decision, allocation or fill.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.replay_contracts import digest
from infrastructure.research_archive import write_bytes_once, write_once
from infrastructure.storage.evidence_graph import object_id
from infrastructure.storage.report_bundle_stream import iter_bundle_records
from infrastructure.storage.report_evidence import ReportBundleContext, ReportEvidenceReader, SCHEMA

SCHEMA_VERSION = "les-r13-retained-valuation-comparison/v1"
SUPPLEMENTAL_FIELDS = frozenset({"entry_quality_required", "input_policy", "input_policy_id", "quality_dependency"})
MISSING = object()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _section(reader, ref, path, *, default=MISSING):
    """Use the existing verified node/decode implementations, never custom I/O.

    No public section API exists. This small compatibility adapter understands
    the current lossless dict descriptors and refuses malformed/duplicate keys.
    It intentionally does not call resolve(), which expands the financial graph.
    """
    descriptor = ref["root"]
    for name in path:
        if not isinstance(descriptor, list) or len(descriptor) != 2:
            raise ValueError("INVALID_COMPARISON_SECTION_DESCRIPTOR")
        kind, value = descriptor
        node = reader._node(ref["store"], value) if kind == "ref" else value if kind == "inline" else None
        if not isinstance(node, list) or len(node) != 2 or node[0] != "dict" or not isinstance(node[1], list):
            raise ValueError("INVALID_COMPARISON_SECTION_NODE")
        children = {}
        for pair in node[1]:
            if (not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str)
                    or pair[0] in children):
                raise ValueError("INVALID_COMPARISON_SECTION_KEY")
            children[pair[0]] = pair[1]
        if name not in children:
            if default is MISSING:
                raise ValueError("COMPARISON_REQUIRED_SECTION_MISSING:" + ".".join(path))
            return default
        descriptor = children[name]
    return reader._decode(descriptor, ref["store"], [0])


def _old_arithmetic(diagnostic):
    from domain.les_v7.valuation_price import POLICY_VERSION, evaluate_price_valuation
    from domain.les_v7.valuation_recovery_r10 import POLICY_VERSION as R10_VERSION, evaluate_recovery_price_valuation
    if diagnostic.get("policy_version") not in {POLICY_VERSION, R10_VERSION}:
        raise ValueError("UNSUPPORTED_RETAINED_VALUATION_POLICY")
    result = evaluate_price_valuation(**diagnostic["inputs"])
    if diagnostic["policy_version"] == R10_VERSION:
        result = evaluate_recovery_price_valuation(legacy_result=result,
            comparables=diagnostic.get("comparables", {}), recovery=diagnostic.get("recovery"))
    if (set(diagnostic) - SUPPLEMENTAL_FIELDS != set(result)
            or any(digest(diagnostic[key]) != digest(value) for key, value in result.items())):
        raise ValueError("RETAINED_VALUATION_ARITHMETIC_MISMATCH")
    return result


def _summary(diagnostic, grade):
    checks = diagnostic.get("gate_checks") or {}
    return dict(policy_version=diagnostic.get("policy_version"),
        policy_id=diagnostic.get("policy_id"), policy_sha256=digest(diagnostic.get("policy")),
        method=diagnostic.get("selected_method", diagnostic.get("policy_version")),
        status=diagnostic.get("status"), grade=grade, reasons=diagnostic.get("reasons", []),
        soft_checks={key: checks.get(key) for key in
                     ("BASE_PRICE_CAGR_GTE_REQUIRED_RETURN", "BASE_FAIR_PRICE_GT_CURRENT_PRICE")},
        bear_check=checks.get("BEAR_LOSS_LTE_MAXIMUM"),
        bear_hard_check=checks.get("BEAR_LOSS_LTE_HARD_MAXIMUM", checks.get("BEAR_LOSS_LTE_MAXIMUM")),
        bear_shortfall_band=diagnostic.get("bear_shortfall_band"),
        discounted_bear_value_shortfall=diagnostic.get("discounted_bear_value_shortfall",
                                                       diagnostic.get("bear_permanent_loss")),
        anchor=diagnostic.get("anchor"), scenarios=diagnostic.get("scenarios"))


def compare_sections(ref, metadata, score, strategy, receipt):
    """Compare deterministic valuation arithmetic only; every row is retained."""
    from domain.les_v7.entry_policy_r9 import valuation_grade
    diagnostic = receipt.get("valuation_diagnostics", {}) if isinstance(receipt, dict) else {}
    price = diagnostic.get("inputs", {}).get("current_price") if isinstance(diagnostic, dict) else None
    old_grade = valuation_grade(receipt, current_price=price) if isinstance(receipt, dict) else None
    record = dict(symbol=metadata["symbol"], signal_date=metadata["signal_date"],
        model_version=metadata["model_version"], parameter_version=metadata["parameter_version"],
        cutoff_at=metadata.get("cutoff_at"), persisted_strategy_decision=strategy.get("strategy_decision"),
        persisted_final_score=score.get("final_score"), new_final_score=None,
        new_strategy_decision=None, comparison_state="INPUTS_UNAVAILABLE", error=None,
        old=_summary(diagnostic, old_grade), new=None, valuation_deduction_difference=None,
        source=dict(report_reference=ref, valuation_receipt_sha256=digest(receipt),
            diagnostics_sha256=digest(diagnostic), score_sha256=digest(score), strategy_sha256=digest(strategy),
            valuation_policy_id=diagnostic.get("policy_id"),
            valuation_input_policy_id=diagnostic.get("input_policy_id"),
            comparables_sha256=digest(diagnostic.get("comparables")),
            recovery_sha256=digest(diagnostic.get("recovery")),
            recovery_source_id=(diagnostic.get("recovery") or {}).get("source_id"),
            recovery_proof_ids=(diagnostic.get("recovery") or {}).get("proof_ids", {}),
            input_hash=receipt.get("input_hash") if isinstance(receipt, dict) else None,
            input_ids=receipt.get("input_ids", []) if isinstance(receipt, dict) else [],
            proof_record_ids=receipt.get("proof_record_ids", []) if isinstance(receipt, dict) else [],
            source_available_at=receipt.get("source_available_at") if isinstance(receipt, dict) else None))
    if not isinstance(diagnostic, dict) or not isinstance(diagnostic.get("inputs"), dict):
        return record
    try:
        old = _old_arithmetic(diagnostic)
        if receipt.get("state") != old["status"]:
            raise ValueError("RETAINED_VALUATION_RECEIPT_STATE_MISMATCH")
        from domain.les_v7.valuation_relaxed_r13 import evaluate_annual_price_valuation, evaluate_relaxed_price_valuation
        annual = evaluate_annual_price_valuation(**old["inputs"])
        new = evaluate_relaxed_price_valuation(legacy_result=annual,
            comparables=old.get("comparables", {}), recovery=old.get("recovery"))
        new_receipt = dict(gate_id="VALUATION", state=new["status"], valuation_diagnostics=new)
        new_grade = valuation_grade(new_receipt, current_price=price)
        record.update(comparison_state="RECOMPUTED_ARITHMETIC", new=_summary(new, new_grade),
            valuation_deduction_difference=(new_grade["points"] - old_grade["points"]))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        record.update(comparison_state="REJECTED_STORED_ARITHMETIC", error=str(error))
    return record


def compare_bundle(path):
    """Stream every referenced row; verify all traversed nodes and bindings."""
    path = Path(path)
    before = file_hash(path)
    context = ReportBundleContext()
    records = []
    with ReportEvidenceReader(path.parent, cache_bytes=8 * 1024 * 1024) as reader:
        for value in iter_bundle_records(path, buckets=("reports", "stale_reports"), context=context):
            ref = value.get("_report_reference", value) if isinstance(value, dict) else None
            if not isinstance(ref, dict) or ref.get("schema") != SCHEMA:
                raise ValueError("COMPARISON_REQUIRES_CONTENT_ADDRESSED_REPORT_REFERENCE")
            object_id(ref.get("report_sha256"), "REPORT_EVIDENCE")
            reader.check_store(ref.get("store"))
            meta = {key: _section(reader, ref, (key,)) for key in
                    ("symbol", "signal_date", "model_version", "parameter_version")}
            if any(type(meta[key]) is not type(ref.get(key)) or meta[key] != ref.get(key) for key in meta):
                raise ValueError("COMPARISON_REPORT_METADATA_BINDING_MISMATCH")
            meta["cutoff_at"] = _section(reader, ref, ("envelope", "event", "cutoff_at"), default=None)
            score = _section(reader, ref, ("envelope", "score"), default={})
            strategy = _section(reader, ref, ("envelope", "strategy"), default={})
            receipt = _section(reader, ref, ("diagnostics", "native_evidence", "produced_receipts", "VALUATION"), default={})
            if (not isinstance(score, dict) or not isinstance(strategy, dict)
                    or receipt is not None and not isinstance(receipt, dict)):
                raise ValueError("COMPARISON_SELECTED_SECTION_TYPE_MISMATCH")
            records.append(compare_sections(ref, meta, score, strategy, receipt))
    if file_hash(path) != before:
        raise ValueError("COMPARISON_INPUT_CHANGED_DURING_READ")
    result = dict(schema=SCHEMA_VERSION, source_bundle_sha256=before, source_bundle_name=path.name,
        section_integrity="TRAVERSED_CONTENT_ADDRESSED_NODES_AND_METADATA_BINDING_ONLY",
        full_report_hash_verified=False, financial_originals_reverified=False,
        source_authenticity_certified=False, historical_information_set_certified=False,
        historical_deployment_claim=False, full_research_rerun=False,
        new_buy_count=None, new_buy_small_count=None,
        notes=["R13 is retrospective arithmetic, not a policy available at the saved cutoff.",
               "Valuation deduction difference is not a recalculated LES Final or signal.",
               "Quality, plan, native vetoes, restrictions, sizing and accounts were not recomputed.",
               "Bear value shortfall is a scenario metric, not a predicted drawdown or loss probability."],
        summary=dict(report_count=len(records),
            persisted_strategy_decisions=dict(sorted(Counter(r["persisted_strategy_decision"] or "MISSING" for r in records).items())),
            comparison_states=dict(sorted(Counter(r["comparison_state"] for r in records).items())),
            old_valuation_grades=dict(sorted(Counter((r["old"]["grade"] or {}).get("state", "UNKNOWN") for r in records).items())),
            new_valuation_grades=dict(sorted(Counter((r["new"]["grade"] if r["new"] else {}).get("state", "NOT_RECOMPUTED") for r in records).items()))),
        records=records)
    return dict(result, comparison_id=digest(result))


def markdown(result):
    summary = result["summary"]
    return ("# R13 retained-input valuation arithmetic comparison\n\n"
        f"Bundle SHA256: `{result['source_bundle_sha256']}`\n\n"
        f"Reports: {summary['report_count']}\n\n"
        f"Stored decisions: `{json.dumps(summary['persisted_strategy_decisions'], ensure_ascii=False)}`\n\n"
        f"Arithmetic comparison: `{json.dumps(summary['comparison_states'], ensure_ascii=False)}`\n\n"
        f"Old valuation grades: `{json.dumps(summary['old_valuation_grades'], ensure_ascii=False)}`\n\n"
        f"R13 valuation grades: `{json.dumps(summary['new_valuation_grades'], ensure_ascii=False)}`\n\n"
        "No new LES score, BUY count, allocation or fill is asserted. Only selected graph nodes were "
        "hash-verified; complete financial originals and the full report hash were not revalidated. "
        "R13 is applied retrospectively, not claimed to have been deployed at the saved cutoff.\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    # An existing directory may be an interrupted/idempotent publication. Every
    # artifact remains write-once and a different input must fail, never replace.
    result = compare_bundle(args.report_bundle)
    write_once(args.output_dir / "comparison.json", result)
    write_bytes_once(args.output_dir / "comparison.md", markdown(result).encode("utf-8"))
    write_once(args.output_dir / "complete.json", dict(comparison_id=result["comparison_id"],
        comparison_sha256=file_hash(args.output_dir / "comparison.json"),
        markdown_sha256=file_hash(args.output_dir / "comparison.md")))
    print(json.dumps(dict(comparison_id=result["comparison_id"], output=str(args.output_dir),
                         summary=result["summary"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

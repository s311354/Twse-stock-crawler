"""Pinned-original parser comparison; no refresh, Quality verdict or orders."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.replay_contracts import digest
from domain.fundamental_derivation import FiledFact, FiledFinancials, _AUDIT_CONCEPTS
from domain.les_v7.contracts import timestamp
from domain.les_v7.financial_audit_r7 import audit_proof_r7, policy
from domain.les_v7.financial_sources_r5 import audit_proof_r5
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl_evidence
from infrastructure.research_archive import write_once, write_bytes_once


def compare_pinned(connection, raw_root, candidate, *, cutoff, market):
    if market not in {"TWSE", "TPEX"}:
        raise ValueError("EXPLICIT_MARKET_REQUIRED")
    required = {"symbol", "fiscal_year", "quarter", "content_sha256"}
    if not isinstance(candidate, dict) or set(candidate) != required:
        raise ValueError("EXPLICIT_FILING_VERSION_REQUIRED")
    values = (candidate["symbol"], candidate["fiscal_year"], candidate["quarter"], candidate["content_sha256"])
    rows = connection.execute("SELECT * FROM filings WHERE symbol=? AND fiscal_year=? AND quarter=? AND content_sha256=?",
        values).fetchall()
    if len(rows) != 1:
        raise ValueError("PINNED_ORIGINAL_MISSING_OR_CONFLICTING")
    row = dict(rows[0])
    raw_path = (raw_root.parent / row["raw_path"]).resolve()
    if not raw_path.is_relative_to(raw_root.resolve()):
        raise ValueError("RAW_ORIGINAL_OUTSIDE_REPOSITORY")
    content = raw_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != candidate["content_sha256"]:
        raise ValueError("RAW_ORIGINAL_HASH_MISMATCH")
    facts, decoding = parse_ixbrl_evidence(content)
    # The whole document is parsed, but this audit uses only exactly typed
    # undimensioned audit facts. No financial number or revision is repaired.
    audit_facts = tuple(FiledFact(f.concept, f.numeric_value, f.text_value, f.start_date,
        f.end_date, f.instant, dimensions=f.dimensions, unit=f.unit, context_id=f.context_id)
        for f in facts if f.concept in _AUDIT_CONCEPTS)
    filing = FiledFinancials(row["fiscal_year"], row["quarter"], row["available_at"], row["revision_number"],
        audit_facts, availability_source=row["availability_source"], filing_id=row["id"],
        content_sha256=row["content_sha256"], source_url=row["source_url"], period_end=row["period_end"],
        statement_scope=row["statement_scope"], audit_status=row["audit_status"])
    old = audit_proof_r5(filing, symbol=row["symbol"], market=market, original_verified=True)
    new = audit_proof_r7(filing, symbol=row["symbol"], cutoff=cutoff, market=market, original_verified=True)
    return dict(symbol=row["symbol"], market=market,
        market_binding="EXPLICIT_RESEARCH_SCOPE_NOT_HISTORICAL_MEMBERSHIP_PROOF",
        original=dict(candidate, source_url=row["source_url"],
        available_at=row["available_at"], fetched_at=row["fetched_at"], revision_number=row["revision_number"],
        scope=row["statement_scope"], raw_path=row["raw_path"], parsing_decoding=decoding),
        old=old, candidate=new, input_facts_sha256=digest([asdict(f) for f in audit_facts]),
        quality_gate_recomputed=False, entry_authority=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fundamental-db", required=True, type=Path)
    p.add_argument("--filings-file", required=True, type=Path)
    p.add_argument("--market", required=True, choices=("TWSE", "TPEX"))
    p.add_argument("--cutoff", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args(argv)
    cutoff = timestamp(args.cutoff).isoformat()
    selections = json.loads(args.filings_file.read_bytes())
    if not isinstance(selections, list) or not selections or len({digest(c) for c in selections}) != len(selections):
        raise ValueError("FIXED_FILING_SCOPE_REQUIRED")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="financial-parser-r7-audit-request/v1", filings=selections, cutoff=cutoff, market=args.market,
        scope="PINNED_ORIGINALS_NOT_LATEST_QUERY", policy=policy(), full_quality_or_model_execution=False)
    write_once(args.output_dir / "request.json", manifest)
    program_root = Path(__file__).resolve().parents[1]
    programs = {}
    for relative in ("examples/audit_financial_parser_r7.py", "domain/les_v7/financial_audit_r7.py",
        "domain/les_v7/financial_sources_r5.py", "domain/fundamental_derivation.py",
        "domain/text_normalization.py"):
        content = (program_root / relative).read_bytes()
        programs[relative] = hashlib.sha256(content).hexdigest()
        write_bytes_once(args.output_dir / "programs" / relative, content)
    records = []
    db = args.fundamental_db.resolve()
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=1")
        for candidate in selections:
            result = compare_pinned(connection, db.parent / "raw", candidate, cutoff=cutoff, market=args.market)
            record_path = "records/" + result["symbol"] + "-" + candidate["content_sha256"][:12] + ".json"
            write_once(args.output_dir / record_path, result)
            records.append(dict(symbol=result["symbol"], path=record_path, result_sha256=digest(result),
                source_sha256=candidate["content_sha256"], old_status=result["old"]["status"],
                candidate_status=result["candidate"]["status"], old_opinion=result["old"]["audit"]["audit_opinion"],
                candidate_opinion=result["candidate"]["audit"]["audit_opinion"],
                requires_review=(result["candidate"]["status"] != "VERIFIED_REPORTED" or
                    result["candidate"]["audit"]["audit_review_required"] or
                    result["candidate"]["audit"]["audit_opinion"] == "UNKNOWN"),
                binding_reasons=result["candidate"]["period_binding"]["reasons"]))
    summary = dict(schema="financial-parser-r7-audit-result/v1", request_sha256=digest(manifest),
        programs=programs, records=records, formal_policy_changed=False, actual_quality_pass_count=None,
        buy_count=None, historical_deployment_claim=False, orders=False)
    write_once(args.output_dir / "comparison.json", summary)
    write_once(args.output_dir / "complete.json", dict(comparison_sha256=digest(summary), count=len(records)))
    print(json.dumps(records, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

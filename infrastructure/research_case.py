"""Read existing run reports serially; never open the BankEmulator ledger writable."""
import hashlib
import json
from pathlib import Path


def audit_existing_case(root: Path, symbols: tuple[str, ...]) -> dict:
    def read(path):
        body = path.read_bytes()
        return json.loads(body), hashlib.sha256(body).hexdigest()
    manifest, manifest_sha = read(root / "manifest.json")
    days = []
    for path in sorted((root / "reports").glob("*.json")):
        if path.name.endswith(".receipt.json"): continue
        report, sha = read(path)
        records = []
        for decision in report.get("decisions", []):
            symbol = decision["symbol"]
            if symbol not in symbols: continue
            diag = report["diagnostics"]
            les = decision.get("les_report", {})
            records.append(dict(symbol=symbol, raw=decision["raw"],
                quality=les.get("quality"), liquidity=les.get("liquidity", {}),
                restriction=diag["restrictions"].get(symbol), kelly=diag["kelly"].get(symbol),
                financial=diag["financials"].get(symbol), missing_bars=diag["missing_bars"].get(symbol),
                hard_gates=decision["hard_gates"], excluded_evidence=diag["excluded_evidence"].get(symbol),
                selected_filings=report["evidence"]["filings"].get(symbol),
                research_entry_eligible=decision["research_entry_eligible"]))
        days.append(dict(path=str(path.resolve()), file_sha256=sha, as_of=report["as_of"],
            history_start=report["history_start"], history_end=report["history_end"],
            account=report["account"], records=records))
    return dict(schema="existing-case-audit/v1", status="AUDITED_NOT_REEXECUTED",
        source_run=str(root.resolve()), manifest_sha256=manifest_sha,
        case={key: manifest.get(key) for key in ("body", "dates", "historyStart", "firstSignal", "versions", "evidenceId", "universePolicy")},
        original_universe_count=len(manifest["symbols"]), audited_symbols=list(symbols), days=days,
        interpretation="Report provenance can locate gaps; codes alone do not prove publication, closure, or source completeness")

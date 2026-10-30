"""Synthetic exact-version/raw-byte audit boundary; no real Quality PASS."""
import hashlib
from html import escape
import json
from pathlib import Path
import sqlite3

import pytest

from application.replay_contracts import digest
from examples.audit_financial_parser_r7 import compare_pinned, main
from tests.fundamentals.test_v7_audit_cash_r5 import QUALIFIED_BODY


def synthetic_repository(tmp_path):
    root = tmp_path / "mops"
    raw = root / "raw"
    raw.mkdir(parents=True)
    content = ('<html><xbrli:context id="SIGNED_CURRENT"><xbrli:period>'
        '<xbrli:instant>2026-08-10</xbrli:instant></xbrli:period></xbrli:context>'
        '<ix:nonnumeric name="tifrs-ar:QualifiedOpinionAbstract" contextref="SIGNED_CURRENT">Y</ix:nonnumeric>'
        '<ix:nonnumeric name="tifrs-ar:AccountantsReportBody" contextref="SIGNED_CURRENT">' +
        escape(QUALIFIED_BODY) + '</ix:nonnumeric></html>').encode()
    sha = hashlib.sha256(content).hexdigest()
    (raw / "synthetic.html").write_bytes(content)
    db = root / "mops.sqlite3"
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    c.execute('CREATE TABLE filings(id,symbol,fiscal_year,quarter,content_sha256,raw_path,available_at,'
        'fetched_at,revision_number,source_url,period_end,statement_scope,audit_status,availability_source)')
    c.execute('INSERT INTO filings VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (1, "9999", 2026, 2, sha,
        "raw/synthetic.html", "2026-08-14T00:00:00Z", "2026-08-14T00:00:00Z", 1,
        "https://mops.twse.com.tw/SYNTHETIC_ONLY", "2026-06-30", "consolidated", "reviewed", "SYNTHETIC_ONLY"))
    c.commit()
    return c, raw, dict(symbol="9999", fiscal_year=2026, quarter=2, content_sha256=sha)


def compare(c, raw, candidate, **kwargs):
    return compare_pinned(c, raw, candidate, cutoff="2026-10-28T12:00:00Z", market="TWSE", **kwargs)


def test_exact_version_original_is_verified_without_mutating_database_or_granting_quality(tmp_path):
    c, raw, selected = synthetic_repository(tmp_path)
    before = [tuple(r) for r in c.execute('SELECT * FROM filings')]
    r = compare(c, raw, selected)
    assert before == [tuple(row) for row in c.execute('SELECT * FROM filings')]
    assert r["candidate"]["status"] == "VERIFIED_REPORTED"
    assert r["candidate"]["original_available_at"] == "2026-08-14T00:00:00Z"
    assert not r["quality_gate_recomputed"] and not r["entry_authority"]
    c.close()


@pytest.mark.parametrize("damage", ["missing", "duplicate", "raw_hash", "outside_path"])
def test_missing_or_conflicting_source_cannot_be_replaced_with_latest_or_other_file(tmp_path, damage):
    c, raw, selected = synthetic_repository(tmp_path)
    if damage == "missing": selected["content_sha256"] = "f" * 64
    elif damage == "duplicate": c.execute('INSERT INTO filings SELECT * FROM filings')
    elif damage == "raw_hash": (raw / "synthetic.html").write_bytes(b"SYNTHETIC_TAMPER")
    else: c.execute('UPDATE filings SET raw_path=?', ("../outside.html",))
    with pytest.raises(ValueError, match="MISSING_OR_CONFLICTING|HASH_MISMATCH|OUTSIDE_REPOSITORY"):
        compare(c, raw, selected)
    c.close()


def test_public_manual_cli_retains_complete_hash_bound_result_and_requires_new_output(tmp_path):
    c, raw, selected = synthetic_repository(tmp_path)
    c.close()
    source = tmp_path / "scope.json"
    source.write_text(json.dumps([selected]))
    output = tmp_path / "result"
    argv = ["--fundamental-db", str(raw.parent / "mops.sqlite3"), "--filings-file", str(source),
        "--market", "TWSE", "--cutoff", "2026-10-28T12:00:00Z", "--output-dir", str(output)]
    assert main(argv) == 0
    summary = json.loads((output / "comparison.json").read_bytes())
    done = json.loads((output / "complete.json").read_bytes())
    assert digest(summary) == done["comparison_sha256"]
    assert summary["actual_quality_pass_count"] is None and summary["buy_count"] is None
    for row in summary["records"]:
        assert digest(json.loads((output / row["path"]).read_bytes())) == row["result_sha256"]
    for file, sha in summary["programs"].items():
        assert hashlib.sha256((output / "programs" / file).read_bytes()).hexdigest() == sha
    with pytest.raises(FileExistsError): main(argv)

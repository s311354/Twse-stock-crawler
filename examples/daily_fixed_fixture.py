"""Freeze raw, cutoff-bound daily inputs for an offline whole-universe benchmark.

Saved scores, Quality verdicts, features and allocations are deliberately not
copied. Original path strings inside hash-bound monthly selections are logical
aliases only: the retained source loader resolves them into portable fixture
files without changing their evidence identities. No network or live state is
opened. A completed manifest is written last; an incomplete directory is never
an executable fixture.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from application.replay_contracts import digest
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.allocation import account_positions
from infrastructure.fundamentals.mops_repository import MopsRepository
from infrastructure.research_archive import write_bytes_once, write_once
from infrastructure.storage.report_evidence import ReportBundleContext, iter_report_bundle


SCHEMA = "daily-fixed-fixture/v1"
STOCK_SCHEMA = "daily-fixed-stock/v1"
MAX_SYMBOLS = 436
MAX_ORIGINAL_BYTES = 16 * 1024 * 1024
MAX_TOTAL_RAW_BYTES = 6 * 1024 * 1024 * 1024
_SAFE_SYMBOL = re.compile(r"[0-9A-Z]{4,12}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while body := stream.read(1024 * 1024):
            value.update(body)
    return value.hexdigest()


def _relative(root, text):
    relative = Path(text)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("FIXED_FIXTURE_RELATIVE_PATH_REQUIRED")
    lexical = root.resolve() / relative
    current = root.resolve()
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("FIXED_FIXTURE_SYMLINK_NOT_ALLOWED")
    target = lexical.resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("FIXED_FIXTURE_PATH_OUTSIDE_ROOT")
    return target


def _manifest_id(value):
    return digest({key: child for key, child in value.items() if key not in {"input_id", "created_at"}})


def _file_ref(root, path):
    return dict(path=path.relative_to(root).as_posix(), sha256=file_hash(path), size_bytes=path.stat().st_size)


def _account(value, cutoff):
    if (not isinstance(value, dict) or any(value.get(key) is None for key in
            ("value", "available_cash", "positions", "intended_weight", "capacity_weight", "as_of"))
            or account_positions(value["positions"]) is None):
        raise ValueError("FIXED_FIXTURE_EXPLICIT_SAVED_ACCOUNT_REQUIRED")
    # No default empty account and no silently changed historical account clock.
    if timestamp(value["as_of"]) != timestamp(cutoff):
        raise ValueError("FIXED_FIXTURE_ACCOUNT_SOURCE_CUTOFF_MISMATCH")
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _issuer(packet, symbol):
    rows = [*packet.get("filings", ()), *packet.get("valuation_filings", ())]
    if any(not isinstance(row, dict) or not isinstance(row.get("symbol"), str)
           or not _SAFE_SYMBOL.fullmatch(row["symbol"]) for row in rows):
        raise ValueError("FIXED_FIXTURE_FILING_ISSUER_UNVERIFIED")
    values = {row["symbol"] for row in rows}
    if len(values) > 1:
        raise ValueError("FIXED_FIXTURE_CONFLICTING_DECLARED_ISSUERS")
    return next(iter(values), symbol)


def _name(report, packet, symbol):
    # A missing display label never establishes or changes security identity.
    for value in (report.get("name"), report.get("stock_name"), packet.get("name")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return symbol


def _monthly_selections(value):
    if isinstance(value, dict):
        if value.get("schema") == "les-v7-monthly-valuation-selection/v1":
            yield value
        else:
            for child in value.values():
                yield from _monthly_selections(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _monthly_selections(child)


def _copy_body(source, target, expected, *, counters, maximum, missing=None):
    if not _SHA.fullmatch(str(expected)):
        raise ValueError("FIXED_FIXTURE_INVALID_CONTENT_HASH")
    if not source.is_file():
        if missing is not None:
            missing.append(dict(path=target.as_posix(), sha256=expected, reason="LOCAL_ORIGINAL_MISSING_NOT_REPAIRED"))
            return False
        raise ValueError("FIXED_FIXTURE_REQUIRED_SOURCE_MISSING")
    size = source.stat().st_size
    if size > MAX_ORIGINAL_BYTES:
        raise ValueError("FIXED_FIXTURE_ORIGINAL_CAPACITY_EXCEEDED")
    body = source.read_bytes()
    if len(body) != size or hashlib.sha256(body).hexdigest() != expected:
        raise ValueError("FIXED_FIXTURE_SOURCE_CONTENT_HASH_MISMATCH")
    if target.exists():
        if target.stat().st_size != size or file_hash(target) != expected:
            raise ValueError("FIXED_FIXTURE_EXISTING_OBJECT_CONFLICT")
        return True
    if counters["raw_bytes"] + size > maximum:
        raise ValueError("FIXED_FIXTURE_TOTAL_RAW_CAPACITY_EXCEEDED")
    write_bytes_once(target, body)
    counters["raw_bytes"] += size
    return True


def _copy_monthly(packet, destination, aliases, counters, maximum):
    from infrastructure.crawler.valuation_history import load_valuation_history
    for selection in _monthly_selections(packet):
        text = selection.get("manifest")
        if not text or text in aliases:
            continue
        source = Path(text).resolve()
        capture = load_valuation_history(source, source.parent.parent)
        if capture["evidence_id"] != selection.get("capture_evidence_id"):
            raise ValueError("FIXED_FIXTURE_MONTHLY_CAPTURE_BINDING_MISMATCH")
        capture_root = destination / "valuation-history" / capture["evidence_id"]
        target = capture_root / "manifests" / source.name
        manifest_body = source.read_bytes()
        write_bytes_once(target, manifest_body)
        for document in json.loads(manifest_body)["payload"]["documents"].values():
            reference = document["body_ref"]
            relative = reference["path"]
            _copy_body(_relative(source.parent.parent, relative), _relative(capture_root, relative),
                       reference["sha256"], counters=counters, maximum=maximum)
        # Revalidate the portable copy; the raw selection path remains unchanged.
        load_valuation_history(target, capture_root)
        aliases[text] = target.relative_to(destination).as_posix()
        # The existing verifier normalizes its loader argument with resolve().
        # Both keys are logical lookup tokens; neither is read by the worker.
        aliases[str(source)] = aliases[text]


def _table_columns(connection, table):
    return [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]


def _insert(connection, table, columns, row):
    connection.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                       [row[column] for column in columns])


def selected_periods(rows, period_count=14):
    """Native companion dependency window, not fourteen contiguous quarters.

    Match MopsFundamentalService.load_v7_financial_sources: nine calendar
    quarters ending at latest, plus five visible annual Q4 original periods.
    Smaller explicit smoke limits remain a latest-N-period bounded fixture.
    """
    visible = sorted({(row["fiscal_year"], row["quarter"]) for row in rows})
    if not visible:
        return []
    if period_count != 14:
        return visible[-period_count:]
    year, quarter = visible[-1]
    latest = year * 4 + quarter - 1
    recent = {(value // 4, value % 4 + 1) for value in range(latest - 8, latest + 1)}
    annuals = [period for period in visible if period[1] == 4][-5:]
    return sorted(recent | set(annuals))


def freeze_fixture(bundle, database, destination, *, expected_bundle_sha256,
                   symbols=None, evaluation_cutoff, period_count=14,
                   max_total_raw_bytes=MAX_TOTAL_RAW_BYTES):
    """Copy only raw source operands, using one read-only SQLite transaction.

    Per instrument all eligible versions in the native nine-quarter plus five
    annual dependency window are retained (at most fourteen periods). Shared
    declared issuers use their latest source clock
    for copying; the runner must still select at each individual source cutoff.
    No snapshot_json/annual_json or prior strategy output is retained.
    """
    bundle, database, destination = Path(bundle).resolve(), Path(database).resolve(), Path(destination).resolve()
    evaluation = timestamp(evaluation_cutoff).isoformat()
    if destination.exists():
        raise ValueError("FIXED_FIXTURE_DESTINATION_ALREADY_EXISTS")
    if not 1 <= period_count <= 14 or type(period_count) is not int:
        raise ValueError("FIXED_FIXTURE_PERIOD_LIMIT_INVALID")
    if type(max_total_raw_bytes) is not int or max_total_raw_bytes < 1:
        raise ValueError("FIXED_FIXTURE_RAW_LIMIT_INVALID")
    if file_hash(bundle) != expected_bundle_sha256:
        raise ValueError("FIXED_FIXTURE_SOURCE_BUNDLE_HASH_MISMATCH")
    requested = None if symbols is None else list(symbols)
    if requested is not None and (not requested or len(requested) > MAX_SYMBOLS
            or len(set(requested)) != len(requested) or any(not _SAFE_SYMBOL.fullmatch(str(s)) for s in requested)):
        raise ValueError("FIXED_FIXTURE_SYMBOL_SCOPE_INVALID")
    destination.mkdir(parents=True)
    stocks, gaps, aliases, counters = [], [], {}, dict(raw_bytes=0, filing_rows=0, fact_rows=0, batch_rows=0)
    seen = set()
    context = ReportBundleContext()
    from examples.compare_les_r14 import assert_snapshot_visibility
    with closing(iter_report_bundle(bundle, context=context)) as reports:
        for report in reports:
            symbol = report["symbol"]
            if not _SAFE_SYMBOL.fullmatch(symbol) or symbol in seen:
                raise ValueError("FIXED_FIXTURE_SOURCE_UNIVERSE_CONFLICT")
            seen.add(symbol)
            if len(seen) > MAX_SYMBOLS:
                raise ValueError("FIXED_FIXTURE_UNIVERSE_CAPACITY_EXCEEDED")
            if requested is not None and symbol not in requested:
                continue
            diagnostics = report["diagnostics"]
            packet = diagnostics["native_evidence"]["source_packet"]
            binding = diagnostics["evidence_binding"]
            source_cutoff = timestamp(packet["cutoff"]).isoformat()
            if (timestamp(evaluation) < timestamp(source_cutoff) or packet["symbol"] != symbol
                    or packet["market"] != binding["market"]
                    or timestamp(binding["cutoff"]) != timestamp(source_cutoff)
                    or identity(packet["market_rows"]) != binding["bars_hash"]):
                raise ValueError("FIXED_FIXTURE_SOURCE_BINDING_OR_CUTOFF_INVALID")
            assert_snapshot_visibility(packet)
            account = _account(report.get("pae_authorization", {}).get("account"), source_cutoff)
            if report["pae_authorization"].get("account_sha256") != identity(account):
                raise ValueError("FIXED_FIXTURE_ACCOUNT_HASH_MISMATCH")
            _copy_monthly(packet, destination, aliases, counters, max_total_raw_bytes)
            issuer = _issuer(packet, symbol)
            shard = dict(schema=STOCK_SCHEMA, symbol=symbol, name=_name(report, packet, symbol),
                market=packet["market"], source_issuer=issuer, source_cutoff=source_cutoff,
                evaluation_cutoff=evaluation, account=account, source_packet=packet)
            shard_path = destination / "stocks" / (symbol + ".json")
            write_once(shard_path, shard)
            stocks.append(dict(symbol=symbol, name=shard["name"], market=packet["market"], source_issuer=issuer,
                source_cutoff=source_cutoff, source_packet_sha256=identity(packet),
                source_packet_ref=_file_ref(destination, shard_path), account=account,
                account_sha256=identity(account)))
    if not context.complete or not stocks or requested is not None and set(requested) != {s["symbol"] for s in stocks}:
        raise ValueError("FIXED_FIXTURE_REQUESTED_SCOPE_NOT_COMPLETE")
    if file_hash(bundle) != expected_bundle_sha256:
        raise ValueError("FIXED_FIXTURE_SOURCE_BUNDLE_CHANGED")
    selected = {}
    for stock in stocks:
        issuer = stock["source_issuer"]
        previous = selected.get(issuer)
        if previous is None or timestamp(previous) < timestamp(stock["source_cutoff"]):
            selected[issuer] = stock["source_cutoff"]
    repository = MopsRepository(destination / "mops", connect_attempts=1)
    repository._ensure_schema()
    periods_by_issuer = {}
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source:
        source.execute("PRAGMA query_only=1")
        source.row_factory = sqlite3.Row
        source.execute("BEGIN")
        columns = {table: _table_columns(source, table) for table in ("filings", "facts", "financial_batch_snapshots")}
        with repository.unit_of_work() as unit:
            target = unit.connection
            for table, actual in columns.items():
                # A migrated legacy table can append available_at_epoch at a
                # different physical ordinal; explicit column names preserve
                # identical values without requiring a needless migration.
                if set(actual) != set(_table_columns(target, table)):
                    raise ValueError("FIXED_FIXTURE_SQLITE_SCHEMA_MISMATCH:" + table)
            for issuer, cutoff in sorted(selected.items()):
                rows = [dict(row) for row in source.execute(
                    "SELECT * FROM filings WHERE symbol=? AND available_at_epoch<=? "
                    "ORDER BY fiscal_year,quarter,revision_number,id", (issuer, int(timestamp(cutoff).timestamp())))
                    if timestamp(row["available_at"]) <= timestamp(cutoff)]
                periods = selected_periods(rows, period_count)
                periods_by_issuer[issuer] = [list(period) for period in periods]
                if not rows:
                    gaps.append(dict(symbol=issuer, reason="NO_CUTOFF_VALID_LOCAL_FILING_METADATA"))
                for row in rows:
                    if (row["fiscal_year"], row["quarter"]) not in periods:
                        continue
                    raw_path = Path(row["raw_path"])
                    original = _relative(database.parent, raw_path)
                    if not original.is_relative_to((database.parent / "raw").resolve()):
                        raise ValueError("FIXED_FIXTURE_ORIGINAL_OUTSIDE_RAW_ROOT")
                    target_raw = _relative(repository.root, raw_path)
                    missing = []
                    _copy_body(original, target_raw, row["content_sha256"], counters=counters,
                               maximum=max_total_raw_bytes, missing=missing)
                    gaps.extend(dict(symbol=issuer, fiscal_year=row["fiscal_year"], quarter=row["quarter"],
                        content_sha256=row["content_sha256"], reason=item["reason"],
                        path=target_raw.relative_to(destination).as_posix()) for item in missing)
                    _insert(target, "filings", columns["filings"], row)
                    counters["filing_rows"] += 1
                    for fact in source.execute("SELECT * FROM facts WHERE filing_id=? ORDER BY id", (row["id"],)):
                        _insert(target, "facts", columns["facts"], dict(fact)); counters["fact_rows"] += 1
                    signed_root = database.parent / "signed-audit"
                    for path in sorted((signed_root / "bindings" / row["content_sha256"]).glob("*.json")):
                        body = path.read_bytes()
                        proof = json.loads(body)
                        if timestamp(proof["available_at"]) > timestamp(cutoff):
                            continue
                        if (proof.get("source_id") != path.stem
                                or identity({k: v for k, v in proof.items() if k != "source_id"}) != path.stem
                                or proof.get("filing_sha256") != row["content_sha256"]):
                            raise ValueError("FIXED_FIXTURE_SIGNED_BINDING_INVALID")
                        write_bytes_once(repository.root / "signed-audit" / path.relative_to(signed_root), body)
                        ref = proof["body_ref"]
                        signed_missing = []
                        pdf = repository.root / "signed-audit" / "objects" / ref["sha256"]
                        _copy_body(signed_root / "objects" / ref["sha256"], pdf, ref["sha256"],
                            counters=counters, maximum=max_total_raw_bytes, missing=signed_missing)
                        gaps.extend(dict(symbol=issuer, reason="SIGNED_PDF_MISSING_NOT_REPAIRED",
                            content_sha256=ref["sha256"], path=pdf.relative_to(destination).as_posix()) for _ in signed_missing)
                for row in source.execute(
                        "SELECT * FROM financial_batch_snapshots WHERE symbol=? AND observed_at_epoch<=? ORDER BY id",
                        (issuer, int(timestamp(cutoff).timestamp()))):
                    row = dict(row)
                    if timestamp(row["observed_at"]) <= timestamp(cutoff):
                        _insert(target, "financial_batch_snapshots", columns["financial_batch_snapshots"], row)
                        counters["batch_rows"] += 1
    for stock in stocks:
        stock["periods"] = periods_by_issuer[stock["source_issuer"]]
    database_copy = repository.database_path
    with closing(sqlite3.connect(database_copy.as_uri() + "?mode=ro", uri=True)) as connection:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("FIXED_FIXTURE_SQLITE_INTEGRITY_FAILED")
    accounts = {s["account_sha256"] for s in stocks}
    manifest = dict(schema=SCHEMA, status="COMPLETE", created_at=datetime.now(timezone.utc).isoformat(),
        evidence="REAL_RETAINED_SOURCE_OPERANDS_NOT_OLD_MODEL_OUTPUTS", source_bundle_sha256=expected_bundle_sha256,
        source_schema="daily-reports/v2", symbols=[s["symbol"] for s in stocks], stocks=stocks,
        source_cutoffs=sorted({s["source_cutoff"] for s in stocks}), evaluation_cutoff=evaluation,
        account=stocks[0]["account"] if len(accounts) == 1 else None,
        account_sha256=stocks[0]["account_sha256"] if len(accounts) == 1 else None,
        pae_state=dict(policy="ISOLATED_EXPLICIT_SAVED_ACCOUNT_NO_FORMAL_PERSISTENCE",
                       prestate="EMPTY", saved_positions_required=True, live_order_authority=False),
        mops=_file_ref(destination, database_copy), valuation_aliases=aliases, gaps=gaps,
        counts=counters, period_limit=period_count,
        period_selection_policy=("NATIVE_NINE_CALENDAR_QUARTERS_PLUS_FIVE_VISIBLE_ANNUAL_Q4_UNION"
                                 if period_count == 14 else "EXPLICIT_BOUNDED_LATEST_VISIBLE_PERIODS_SMOKE_ONLY"),
        files=[_file_ref(destination, path) for path in sorted(destination.rglob("*")) if path.is_file()])
    manifest["input_id"] = _manifest_id(manifest)
    write_once(destination / "manifest.json", manifest)
    return manifest


def validate_fixture(root):
    """Verify complete immutable bytes once at worker start; no DDL or writes."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_bytes())
    if (manifest.get("schema") != SCHEMA or manifest.get("status") != "COMPLETE"
            or manifest.get("input_id") != _manifest_id(manifest)):
        raise ValueError("FIXED_FIXTURE_MANIFEST_INVALID")
    if (not 1 <= len(manifest.get("symbols", ())) <= MAX_SYMBOLS
            or manifest["symbols"] != [row["symbol"] for row in manifest["stocks"]]
            or len(set(manifest["symbols"])) != len(manifest["symbols"])):
        raise ValueError("FIXED_FIXTURE_UNIVERSE_INVALID")
    seen = set()
    for reference in manifest["files"]:
        path = _relative(root, reference["path"])
        if reference["path"] in seen or path.is_symlink():
            raise ValueError("FIXED_FIXTURE_FILE_REFERENCE_INVALID")
        seen.add(reference["path"])
        if not path.is_file() or path.stat().st_size != reference["size_bytes"] or file_hash(path) != reference["sha256"]:
            raise ValueError("FIXED_FIXTURE_FILE_CONTENT_MISMATCH:" + reference["path"])
    inventory = {value["path"]: value for value in manifest["files"]}
    if (inventory.get(manifest["mops"]["path"]) != manifest["mops"]
            or any(inventory.get(s["source_packet_ref"]["path"]) != s["source_packet_ref"]
                   for s in manifest["stocks"])):
        raise ValueError("FIXED_FIXTURE_REQUIRED_FILE_NOT_IN_INVENTORY")
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("FIXED_FIXTURE_SYMLINK_NOT_ALLOWED")
        if path.is_file() and path != root / "manifest.json":
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != seen:
        raise ValueError("FIXED_FIXTURE_UNLISTED_OR_MISSING_FILES")
    return manifest


def load_symbol(root, manifest, entry):
    """Detached raw shard. Each use rechecks bytes and semantic bindings."""
    root = Path(root).resolve()
    if entry not in manifest["stocks"]:
        raise ValueError("FIXED_FIXTURE_STOCK_NOT_IN_MANIFEST")
    reference = entry["source_packet_ref"]
    path = _relative(root, reference["path"])
    if not path.is_file() or path.stat().st_size != reference["size_bytes"] or file_hash(path) != reference["sha256"]:
        raise ValueError("FIXED_FIXTURE_STOCK_SHARD_CHANGED")
    value = json.loads(path.read_bytes())
    if (value.get("schema") != STOCK_SCHEMA or value.get("symbol") != entry["symbol"]
            or identity(value["source_packet"]) != entry["source_packet_sha256"]
            or identity(value["account"]) != entry["account_sha256"]
            or value["source_cutoff"] != entry["source_cutoff"]
            or value["evaluation_cutoff"] != manifest["evaluation_cutoff"]):
        raise ValueError("FIXED_FIXTURE_STOCK_BINDING_INVALID")
    return value


def valuation_capture_loader(root, manifest):
    """Resolve hash-bound old path aliases without consulting those old paths."""
    from infrastructure.crawler.valuation_history import load_valuation_history
    root = Path(root).resolve()
    def load(path):
        text = str(path)
        relative = manifest["valuation_aliases"].get(text)
        if relative is None:
            raise ValueError("FIXED_FIXTURE_MONTHLY_ALIAS_NOT_FROZEN")
        retained = _relative(root, relative)
        return load_valuation_history(retained, retained.parent.parent)
    return load

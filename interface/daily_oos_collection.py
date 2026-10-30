"""Collect immutable, no-mail daily V7 source reports for forward OOS research.

This is a source-capture phase only. It does not simulate BankEmulator fills,
calibrated slippage, company actions, or an evolving trading account.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

from application.replay_contracts import digest
from infrastructure.research_archive import write_once
from infrastructure.replay_runtime import current_versions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = PROJECT_ROOT / "data"
DEFAULT_COLLECTION_ID = "les-r18-r19-oos-forward-2026"
TAIPEI = ZoneInfo("Asia/Taipei")
COLLECTION_SCHEMA = "les-v7-daily-oos-source-collection/v1"
IDENTITY_SCHEMA = "les-v7-daily-oos-collection-identity/v1"
R19_VERSION = "V7_RULE_BASED_BASELINE_R19"
OPEN_CAPTURE_TIME = dt.time(15, 45)
CAPITAL_NTD = 1_000_000.0
EXPECTED_STOCKLIST_SIZE = 436


class CollectionError(RuntimeError):
    """Fail-closed daily collection error."""


def validate_collection_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) is None:
        raise CollectionError("INVALID_COLLECTION_ID")
    return value


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _stocklist_identity(path: Path) -> dict:
    raw = path.read_bytes()
    tokens = [item.strip() for item in raw.decode("utf-8-sig").split(";") if item.strip()]
    if (len(tokens) != EXPECTED_STOCKLIST_SIZE or tokens != [str(index) for index in range(EXPECTED_STOCKLIST_SIZE)]):
        raise CollectionError("ELEC_STOCKLIST_0_435_REQUIRED")
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                selector_count=len(tokens), selector_policy="stocklist_elec_indices_0_through_435")


def collection_identity(collection_id: str, *, stocklist_path: Path | None = None) -> dict:
    """Pin effective source-code/runtime and research inputs once per series."""
    collection_id = validate_collection_id(collection_id)
    stocklist = _stocklist_identity(stocklist_path or PROJECT_ROOT / "stocklist_elec")
    from domain.les_v7.policy_r18 import AVAILABLE_AT as R18_RELEASE
    from domain.les_v7.policy_r19 import AVAILABLE_AT as R19_RELEASE_AT
    from domain.les_v7.policy_r19 import DAILY_ACTIVATED_AT as R19_ACTIVATED_AT
    from config.settings import AppSettings
    root = SOURCE_DATA / "les-oos-collections" / collection_id
    settings = AppSettings.from_env(environ=_child_environment(root / "preflight-only", root),
        account_equity=CAPITAL_NTD, mops_history_years=5, validate_paths=False)
    if not settings.les_v7_rule_baseline or settings.les_v7_bundle_json is not None:
        raise CollectionError("NATIVE_R19_RULE_BASELINE_REQUIRED")
    effective = {key: value for key, value in asdict(settings).items() if not key.startswith("smtp_")}
    effective = json.loads(json.dumps(effective, default=str))
    versions = asdict(current_versions())
    return dict(schema=IDENTITY_SCHEMA, collection_id=collection_id,
        model="V7_RULE_BASELINE", baseline_version=R19_VERSION,
        comparison_targets=["R18", "R19"], source_policy="CURRENT_NATIVE_REPORT_WITH_FULL_EVIDENCE_REFERENCES",
        policy_release=dict(r18=R18_RELEASE, r19=R19_RELEASE_AT, r19_daily_activation=R19_ACTIVATED_AT),
        market="TWSE", sector="ELEC", stocklist=stocklist, stock_count=EXPECTED_STOCKLIST_SIZE,
        versions=versions, effective_settings=effective,
        account_inputs=dict(equity=CAPITAL_NTD, cash=CAPITAL_NTD, holdings=[]),
        email_enabled=False, broker_orders_authorized=False, execution_action="NO_ACTION",
        cost_scope=dict(fees="NATIVE_R19_DECLARED_FEE_ASSUMPTION", slippage="ZERO_ASSUMPTION_NOT_CALIBRATED",
            corporate_actions="NOT_CAPTURED_BY_THIS_SOURCE_COLLECTION", full_cost_oos_claim=False),
        bank_fill_observation=dict(status="NOT_CAPTURED", owner="BankEmulator"),
        identity_sha256=None)


def _sealed_identity(identity: dict) -> dict:
    value = dict(identity)
    value["identity_sha256"] = digest({key: item for key, item in value.items() if key != "identity_sha256"})
    return value


def _validate_identity_record(path: Path, expected: dict) -> None:
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise CollectionError("COLLECTION_IDENTITY_UNREADABLE") from error
    if existing != _sealed_identity(expected):
        raise CollectionError("COLLECTION_IDENTITY_CHANGED_START_NEW_COLLECTION_ID")


def _verified_receipt(session_dir: Path) -> dict | None:
    receipt_path = session_dir / "complete.json"
    if not receipt_path.exists():
        return None
    try:
        if session_dir.is_symlink() or any(path.is_symlink() for path in session_dir.rglob("*")):
            raise ValueError("symlink artifact")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != COLLECTION_SCHEMA or receipt.get("status") not in {"SOURCE_CAPTURED", "PARTIAL_SOURCE_CAPTURED"}:
            raise ValueError("receipt contract")
        if not isinstance(receipt.get("files"), dict) or not receipt["files"]:
            raise ValueError("artifact inventory required")
        for relative, expected in receipt["files"].items():
            path = (session_dir / relative).resolve()
            if session_dir.resolve() not in path.parents or file_sha256(path) != expected:
                raise ValueError("file digest")
        actual = {str(path.relative_to(session_dir)) for path in session_dir.rglob("*")
                  if path.is_file() and path.name != "complete.json"}
        if actual != set(receipt["files"]):
            raise ValueError("artifact inventory changed")
        identity = json.loads((session_dir.parent.parent / "identity.json").read_text(encoding="utf-8"))
        if identity != _sealed_identity(identity) or receipt.get("identity_sha256") != identity.get("identity_sha256"):
            raise ValueError("identity binding")
        manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
        capture_id = digest({key: value for key, value in manifest.items() if key != "capture_id"})
        if (manifest.get("capture_id") != capture_id or receipt.get("capture_id") != capture_id
                or receipt.get("session_date") != session_dir.name
                or manifest.get("session_date") != session_dir.name
                or manifest.get("identity_sha256") != identity["identity_sha256"]):
            raise ValueError("capture binding")
        return receipt
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CollectionError("EXISTING_SESSION_RECEIPT_INVALID_NO_OVERWRITE") from error


def _daily_command(data_dir: Path, *, python: str) -> list[str]:
    return [python, str(PROJECT_ROOT / "main.py"), "-o", "shirong", "-t", "ELEC",
        "-e", "700", "-b", "0", "--account-equity", str(int(CAPITAL_NTD)),
        "--pae-current-cash", str(int(CAPITAL_NTD)), "--mops-history-years", "5",
        "--no-mail", str(PROJECT_ROOT / "stocklist_elec")]


def _child_environment(data_dir: Path, collection_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "TWSE_DATA_DIR": str(data_dir),
        "TWSE_PROJECT_ROOT": str(PROJECT_ROOT),
        "TWSE_MARKET_HISTORY_CACHE_PATH": str(SOURCE_DATA / "market_history.sqlite3"),
        "MOPS_CACHE_DIR": str(SOURCE_DATA / "mops"),
        "TWSE_RUN_LOCK_PATH": str(SOURCE_DATA / "stockdataanalysis.lock"),
        "TWSE_PAE_PERSISTENCE_PATH": str(collection_root / "pae" / "r19" / "portfolio.sqlite3"),
        "TWSE_PAE_HOLDINGS_JSON": "[]",
        "TWSE_PAE_CURRENT_CASH": str(int(CAPITAL_NTD)),
        "TWSE_ACCOUNT_EQUITY": str(int(CAPITAL_NTD)),
        "TWSE_DAILY_ENTRY_MODEL": "v7",
        "TWSE_LES_V7_RULE_BASELINE": "true",
        "TWSE_LES_V7_SOURCE_REPAIR": "true",
        "TWSE_LES_V7_BUNDLE": "",
        "TWSE_LES_V7_STRATEGY_POLICY": "",
        "TWSE_LES_V7_DIAGNOSTICS_ONLY": "false",
        "TWSE_INDUSTRY_BENCHMARK_FILE": "",
        "TWSE_DAILY_STRATEGY_WORKERS": "1",
        "TWSE_DAILY_EXECUTOR": "serial",
        "TWSE_RUNTIME_MODE": "release",
        "TWSE_SMTP_EMAIL": "",
        "TWSE_SMTP_PASSWORD": "",
        "TWSE_SMTP_RECEIVERS": "",
        "TWSE_SMTP_CC": "",
        "TWSE_SMTP_HOST": "",
    })
    return env


def _validate_native_outputs(data_dir: Path, *, signal_date: str) -> dict:
    from domain.les_v7.report import validate_report
    from infrastructure.storage.daily_candidate_archive import build_candidate, load_snapshot
    from infrastructure.storage.report_evidence import ReportEvidenceReader

    report_path = data_dir / "shirong_les_v7_reports.json"
    store_path = data_dir / "report_evidence.sqlite3"
    candidate_paths = sorted((data_dir / "shirong_les_v7_candidates").glob("*.json"))
    if not report_path.is_file() or not store_path.is_file() or len(candidate_paths) != 1:
        raise CollectionError("NATIVE_DAILY_ARTIFACTS_MISSING_OR_AMBIGUOUS")
    try:
        bundle = json.loads(report_path.read_text(encoding="utf-8"))
        if (bundle.get("model") != "low_entry_score_v7" or bundle.get("storage_schema") != "daily-reports/v2"
                or bundle.get("research_only") is not True or bundle.get("execution_enabled") is not False):
            raise ValueError("report envelope")
        references = bundle.get("reports")
        if not isinstance(references, list) or not references:
            raise ValueError("empty reports")
        snapshot = load_snapshot(candidate_paths[0])
        if (snapshot.get("candidate_count") != len(snapshot.get("candidates", ()))
                or not any(item.get("signal_date") == signal_date for item in snapshot.get("candidates", ()))):
            raise ValueError("daily candidate snapshot")
        candidates = {item["symbol"]: item for item in snapshot["candidates"]}
        if len(candidates) != len(snapshot["candidates"]) or len(candidates) != len(references):
            raise ValueError("candidate/report scope mismatch")
        seen = set()
        statuses, cutoffs = {}, set()
        with ReportEvidenceReader(data_dir) as reader:
            for reference in references:
                report = reader.resolve(reference)
                validate_report(report)
                symbol = report.get("symbol")
                if not isinstance(symbol, str) or symbol in seen:
                    raise ValueError("duplicate or missing symbol")
                seen.add(symbol)
                policy = (report.get("diagnostics", {}).get("rule_baseline") or {}).get("policy", {})
                baseline = policy.get("policy_version")
                if baseline != R19_VERSION:
                    raise ValueError("non-R19 or mixed baseline")
                cutoff = dt.datetime.fromisoformat(report["envelope"]["event"]["cutoff_at"].replace("Z", "+00:00"))
                if (cutoff.tzinfo is None or cutoff.astimezone(TAIPEI).date().isoformat() != signal_date
                        or cutoff.astimezone(TAIPEI).time().replace(tzinfo=None) < dt.time(13, 30)
                        or cutoff > dt.datetime.now(dt.timezone.utc)):
                    raise ValueError("native evaluation cutoff is not the collected session")
                cutoffs.add(cutoff.isoformat())
                if report.get("signal_date") != signal_date:
                    statuses[symbol] = "STALE_OR_MISSING_CURRENT_SESSION_REPORT"
                else:
                    statuses[symbol] = "CURRENT_SESSION_REPORT"
                source = (report.get("diagnostics", {}).get("native_evidence") or {}).get("source_packet")
                if not isinstance(source, dict):
                    raise ValueError("raw source packet missing")
                if report.get("pae_authorization", {}).get("live_order_authorized") is not False:
                    raise ValueError("order authority must remain false")
                candidate = candidates.get(symbol)
                if candidate is None or build_candidate(report, reference, candidate["erem"]) != candidate:
                    raise ValueError("candidate/native report content mismatch")
        if not seen or not any(status == "CURRENT_SESSION_REPORT" for status in statuses.values()):
            raise ValueError("no current-session symbols")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        if isinstance(error, CollectionError):
            raise
        raise CollectionError("NATIVE_REPORT_VALIDATION_FAILED:" + str(error)) from error
    current = sorted(symbol for symbol, status in statuses.items() if status == "CURRENT_SESSION_REPORT")
    return dict(report_bundle=report_path.name, report_bundle_sha256=file_sha256(report_path),
        report_store=store_path.name, report_store_sha256=file_sha256(store_path),
        candidate_snapshot=str(candidate_paths[0].relative_to(data_dir)),
        candidate_snapshot_sha256=file_sha256(candidate_paths[0]),
        candidate_snapshot_id=snapshot["snapshot_id"], report_count=len(references),
        distinct_symbols=len(seen), current_session_symbols=current,
        selected_symbols=sorted(seen), evaluation_cutoffs=sorted(cutoffs),
        stale_or_missing_symbols=sorted(set(seen) - set(current)), signal_date=signal_date,
        scope_status="COMPLETE" if len(seen) == EXPECTED_STOCKLIST_SIZE and len(current) == EXPECTED_STOCKLIST_SIZE else "PARTIAL",
        r19_baseline=R19_VERSION, r18_counterfactual="NOT_RECOMPUTED_FROM_THIS_CAPTURE",
        all_native_source_packets_present=True, full_cost_oos_claim=False)


def _write_capture_manifest(stage: Path, *, identity: dict, session_date: str,
                            captured_at: str, outputs: dict, calendar_evidence: dict | None = None,
                            execution_objects: dict | None = None) -> dict:
    data_dir = stage / "crawler-data"
    files = {}
    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(stage))
            files[relative] = file_sha256(path)
    manifest = dict(schema=COLLECTION_SCHEMA, status=("SOURCE_CAPTURED" if outputs["scope_status"] == "COMPLETE"
            else "PARTIAL_SOURCE_CAPTURED"), collection_id=identity["collection_id"],
        session_date=session_date, market_cutoff_policy="COMPLETED_DAILY_CLOSE_AT_13_30_TAIPEI",
        evaluation_cutoff_policy="NATIVE_COLLECTION_COMPLETE_WALL_CLOCK_NOT_13_30_REPLAY",
        calendar_evidence=calendar_evidence,
        execution_objects=execution_objects,
        collection_started_at=captured_at, collection_completed_at=dt.datetime.now(TAIPEI).isoformat(),
        identity_sha256=identity["identity_sha256"], versions=identity["versions"],
        account_inputs=identity["account_inputs"], outputs=outputs,
        paired_policy_status=dict(r18="AWAITING_SAME_SOURCE_RECOMPUTATION", r19="NATIVE_CAPTURED"),
        bank_status=dict(fill_observations="NOT_CAPTURED", owner="BankEmulator", account_ledger="NOT_SIMULATED"),
        cost_status=dict(fees="DECLARED_NATIVE_POLICY_ONLY", slippage="ZERO_ASSUMPTION_NOT_CALIBRATED",
            corporate_actions="NOT_CAPTURED", full_cost_oos_eligibility=False),
        execution=dict(order_authority=False, live_orders=False, email=False, execution_action="NO_ACTION"),
        source_data=dict(mops_database=str(SOURCE_DATA / "mops" / "mops.sqlite3"),
            mops_database_identity="MUTABLE_CACHE_NOT_AN_IMMUTABLE_SQLITE_SNAPSHOT",
            self_contained_raw_financial_database=False),
        files=files)
    manifest["capture_id"] = digest(manifest)
    return manifest


def _verify_complete_collection(root: Path, session_date: str) -> dict:
    session = root / "sessions" / session_date
    receipt = _verified_receipt(session)
    if receipt is None:
        raise CollectionError("EXISTING_SESSION_INCOMPLETE_NO_OVERWRITE")
    return receipt


def collect_daily(*, collection_id: str = DEFAULT_COLLECTION_ID, now: dt.datetime | None = None,
                  calendar=None, runner=subprocess.run, python: str | None = None,
                  stocklist_path: Path | None = None) -> dict:
    """Capture one completed official session; same-day retry is immutable/no-op."""
    collection_id = validate_collection_id(collection_id)
    supplied_now = now or dt.datetime.now(TAIPEI)
    if supplied_now.tzinfo is None or supplied_now.utcoffset() is None:
        raise CollectionError("COLLECTION_CLOCK_MUST_BE_TIMEZONE_AWARE")
    local_now = supplied_now.astimezone(TAIPEI)
    if local_now.timetz().replace(tzinfo=None) < OPEN_CAPTURE_TIME:
        raise CollectionError("COLLECTION_BEFORE_POST_CLOSE_WINDOW")
    day = local_now.date()
    from infrastructure.calendar.twse_calendar import TwseCalendar
    official = calendar or TwseCalendar()
    try:
        is_open = official.is_trading_day(day)
    except Exception as error:
        raise CollectionError("OFFICIAL_TWSE_CALENDAR_UNKNOWN") from error
    if not is_open:
        return dict(status="SKIPPED_CLOSED", session_date=day.isoformat(), collection_id=collection_id)
    calendar_evidence = dict(status="OPEN", applicable_date=day.isoformat(), checked_at=local_now.isoformat(),
        annual=official.annual_evidence() if hasattr(official, "annual_evidence") else {},
        emergency=official.emergency_evidence() if hasattr(official, "emergency_evidence") else None)

    root = SOURCE_DATA / "les-oos-collections" / collection_id
    if root.is_symlink() or root.parent.is_symlink():
        raise CollectionError("COLLECTION_ROOT_SYMLINK_FORBIDDEN")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".collector.lock"
    if lock_path.is_symlink():
        raise CollectionError("COLLECTION_LOCK_SYMLINK_FORBIDDEN")
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CollectionError("DAILY_COLLECTION_ALREADY_RUNNING") from error
        identity = _sealed_identity(collection_identity(collection_id, stocklist_path=stocklist_path))
        identity_path = root / "identity.json"
        if identity_path.is_symlink():
            raise CollectionError("COLLECTION_IDENTITY_SYMLINK_FORBIDDEN")
        for path in (root / "objects", root / "source-index", root / "runtime.json",
                     root / "pae", root / "pae" / "r19", root / "pae" / "r19" / "portfolio.sqlite3"):
            if path.is_symlink():
                raise CollectionError("COLLECTION_STATE_OR_OBJECT_SYMLINK_FORBIDDEN")
        if identity_path.exists():
            _validate_identity_record(identity_path, identity)
        else:
            unexpected = [path for path in root.iterdir() if path.name != ".collector.lock"]
            if unexpected:
                raise CollectionError("UNBOUND_COLLECTION_ROOT_REQUIRES_REVIEW")
            write_once(identity_path, identity)
        # Reuse the existing immutable code/environment archive implementation;
        # a source hash alone cannot recover dirty, later-modified source files.
        from infrastructure.research_objects import ResearchObjects, ResearchVersionGuard
        guard = ResearchVersionGuard()
        if asdict(guard.versions) != identity["versions"]:
            raise CollectionError("COLLECTION_VERSION_CHANGED_DURING_PREFLIGHT")
        objects = ResearchObjects(root / "objects")
        execution_objects = guard.retain(objects)
        write_once(root / "runtime.json", execution_objects)
        session_path = root / "sessions" / day.isoformat()
        sessions_root = session_path.parent
        if sessions_root.is_symlink():
            raise CollectionError("COLLECTION_SESSIONS_SYMLINK_FORBIDDEN")
        attempts = tuple(sessions_root.glob(f".{day.isoformat()}.attempt-*")) if sessions_root.exists() else ()
        if attempts:
            raise CollectionError("INCOMPLETE_SAME_DAY_ATTEMPT_REQUIRES_REVIEW")
        if sessions_root.exists() and any(sessions_root.glob(".*.attempt-*")):
            raise CollectionError("EARLIER_INCOMPLETE_ATTEMPT_REQUIRES_REVIEW")
        if session_path.is_symlink():
            raise CollectionError("SESSION_PATH_SYMLINK_FORBIDDEN")
        if session_path.exists():
            receipt = _verify_complete_collection(root, day.isoformat())
            return dict(status="ALREADY_CAPTURED", session_date=day.isoformat(),
                        collection_id=collection_id, capture_id=receipt["capture_id"])
        sessions_root.mkdir(parents=True, exist_ok=True)
        if any(not (path / "complete.json").is_file() for path in sessions_root.iterdir()
               if path.is_dir() and not path.name.startswith(".")):
            raise CollectionError("EARLIER_INCOMPLETE_SESSION_REQUIRES_REVIEW")
        completed_days = sorted(path.name for path in sessions_root.iterdir() if path.is_dir() and not path.name.startswith("."))
        if completed_days and completed_days[-1] > day.isoformat():
            raise CollectionError("OUT_OF_ORDER_COLLECTION_FORBIDDEN")
        stage = Path(tempfile.mkdtemp(prefix=f".{day.isoformat()}.attempt-", dir=session_path.parent))
        data_dir = stage / "crawler-data"
        data_dir.mkdir()
        captured_at = local_now.isoformat()
        child_python = python or str(PROJECT_ROOT / ".venv-py315" / "bin" / "python")
        if not Path(child_python).is_file():
            child_python = sys.executable
        environment = _child_environment(data_dir, root)
        command = _daily_command(data_dir, python=child_python)
        completed = runner(command, cwd=PROJECT_ROOT, env=environment, capture_output=True,
                           text=True, check=False)
        (stage / "crawler.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
        (stage / "crawler.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            raise CollectionError(f"CRAWLER_DAILY_RUN_FAILED:{completed.returncode}")
        guard.check()
        outputs = _validate_native_outputs(data_dir, signal_date=day.isoformat())
        if outputs["r19_baseline"] != R19_VERSION:
            raise CollectionError("R19_BASELINE_NOT_ACTIVE_FOR_SESSION")
        # Index selectors are not a frozen security universe. Bind their actual
        # resolved symbols on the first full-scope capture; never silently rotate.
        universe_path = root / "universe.json"
        if universe_path.is_symlink():
            raise CollectionError("COLLECTION_UNIVERSE_SYMLINK_FORBIDDEN")
        universe = dict(schema="les-oos-resolved-universe/v1", identity_sha256=identity["identity_sha256"],
                        symbols=outputs["selected_symbols"])
        if universe_path.exists():
            if json.loads(universe_path.read_text(encoding="utf-8")) != universe:
                raise CollectionError("RESOLVED_ELEC_UNIVERSE_CHANGED_REQUIRES_REVIEW")
        elif len(universe["symbols"]) == EXPECTED_STOCKLIST_SIZE:
            write_once(universe_path, universe)
        outputs["frozen_universe_sha256"] = file_sha256(universe_path) if universe_path.exists() else None
        outputs["universe_status"] = "FROZEN" if universe_path.exists() else "PARTIAL_NOT_YET_FROZEN"
        # Check again after the child so a changed deployment cannot be sealed
        # as one reproducible collection version.
        _validate_identity_record(identity_path, collection_identity(collection_id, stocklist_path=stocklist_path))
        manifest = _write_capture_manifest(stage, identity=identity, session_date=day.isoformat(),
            captured_at=captured_at, outputs=outputs, calendar_evidence=calendar_evidence,
            execution_objects=execution_objects)
        write_once(stage / "manifest.json", manifest)
        receipt = dict(schema=COLLECTION_SCHEMA, status=manifest["status"],
            collection_id=collection_id, session_date=day.isoformat(), capture_id=manifest["capture_id"],
            identity_sha256=identity["identity_sha256"], files={
                str(path.relative_to(stage)): file_sha256(path) for path in sorted(stage.rglob("*"))
                if path.is_file() and path.name != "complete.json"})
        write_once(stage / "complete.json", receipt)
        os.replace(stage, session_path)
        directory_fd = os.open(session_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return dict(status=manifest["status"], collection_id=collection_id,
            session_date=day.isoformat(), capture_id=manifest["capture_id"],
            scope_status=outputs["scope_status"], report_count=outputs["report_count"],
            current_session_count=len(outputs["current_session_symbols"]))


def _check_summary(collection_id: str) -> dict:
    identity = _sealed_identity(collection_identity(collection_id))
    return dict(status="CONFIG_OK_NO_NETWORK_OR_CAPTURE", collection_id=collection_id,
        research_root=str(SOURCE_DATA / "les-oos-collections" / collection_id),
        run_lock=str(SOURCE_DATA / "stockdataanalysis.lock"),
        identity_sha256=identity["identity_sha256"], account_inputs=identity["account_inputs"],
        email_enabled=False, formal_account_path_used=False,
        fill_observation="NOT_CAPTURED", full_cost_oos_claim=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--check", action="store_true", help="Validate settings and identity without calendar/network/write.")
    args = parser.parse_args(argv)
    try:
        value = _check_summary(validate_collection_id(args.collection_id)) if args.check else collect_daily(collection_id=args.collection_id)
    except (CollectionError, OSError, ValueError) as error:
        print(json.dumps(dict(status="FAILED", reason=str(error)), ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

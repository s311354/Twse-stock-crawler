"""Bounded SYNTHETIC before/after verification, not a market/replay AB engine.

Use --mode before once to freeze baseline-created fixtures and full results;
then --mode compare --baseline-report ... after source editing has stopped.
Each round is a fresh child under this exact interpreter. No crawler CLI,
network, SMTP, official database, account or report archive is invoked.
"""
from __future__ import annotations

import argparse
import datetime as dt
from enum import Enum
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import time


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def source_identity(root):
    files = [root / "main.py", root / "requirements.txt", root / "pyproject.toml"]
    for folder in ("application", "domain", "infrastructure", "interface", "config", "twse"):
        files.extend(sorted((root / folder).rglob("*.py")))
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    return dict(content_sha256=sha(hashes), files=hashes)


def plain_input(value):
    from collections.abc import Mapping
    from dataclasses import asdict, is_dataclass
    if is_dataclass(value): return plain_input(asdict(value))
    if isinstance(value, Mapping): return {str(k): plain_input(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain_input(v) for v in value]
    if isinstance(value, (dt.date, dt.datetime)): return value.isoformat()
    if hasattr(value, "item"): return plain_input(value.item())
    return value


def canonical(value):
    """Type-sensitive complete business tree; no field exclusion/tolerance."""
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass
    import numpy as np
    import pandas as pd
    if value is pd.NA: return ["pd.NA"]
    if value is pd.NaT: return ["pd.NaT"]
    if isinstance(value, Enum):
        return ["enum", type(value).__module__ + "." + type(value).__qualname__, canonical(value.value)]
    if is_dataclass(value):
        return ["dataclass", type(value).__module__ + "." + type(value).__qualname__,
                [[field.name, canonical(getattr(value, field.name))] for field in fields(value)]]
    if isinstance(value, pd.DataFrame):
        return ["DataFrame", canonical(value.index), canonical(value.columns),
                [str(dtype) for dtype in value.dtypes],
                [[canonical(value.iloc[i, j]) for j in range(len(value.columns))] for i in range(len(value))]]
    if isinstance(value, pd.Index):
        return ["Index", type(value).__qualname__, str(value.dtype), canonical(value.names),
                [canonical(item) for item in value.tolist()]]
    if isinstance(value, pd.Timestamp): return ["Timestamp", value.isoformat(), str(value.tz)]
    if isinstance(value, np.generic):
        return ["numpy", value.dtype.str, value.tobytes().hex()]
    if isinstance(value, Mapping):
        items = [[canonical(k), canonical(v)] for k, v in value.items()]
        return ["mapping", sorted(items, key=lambda pair: encoded(pair[0]))]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [canonical(item) for item in value]]
    if value is None: return ["None"]
    if type(value) is bool: return ["bool", value]
    if type(value) is int: return ["int", str(value)]
    if type(value) is float: return ["float64", struct.pack(">d", value).hex()]
    if isinstance(value, str): return ["str", value]
    if isinstance(value, (dt.datetime, dt.date)): return [type(value).__name__, value.isoformat()]
    raise TypeError("Unrecognized business result type: " + str(type(value)))


def make_fixtures():
    """Freeze shared inputs using the baseline's explicitly synthetic fixtures."""
    from copy import deepcopy
    from dataclasses import replace
    from application.daily_stock import PER_STOCK_SEQUENCES, PER_STOCK_MAPPINGS
    from application.low_entry_v7_daily import daily_report
    from application.market_history import MarketHistoryAssembler
    from tests.strategies.test_daily_v6_pipeline import daily_market
    from tests.strategies.test_daily_v7_pipeline import executable_synthetic_market
    from tests.strategies.test_low_entry_score_v7 import receipt
    from tests.strategies.test_expected_return_exit_v3 import make_history

    def two_stocks(market):
        first = market.stock_numbers[0]
        second = "SYNTHETIC_PEER" if market.entry_model == "v7" else "2317"
        changes = {key: [deepcopy(getattr(market, key)[0]), deepcopy(getattr(market, key)[0])]
                   for key in PER_STOCK_SEQUENCES if key not in {"stock_numbers", "stock_names"}}
        for key in PER_STOCK_MAPPINGS:
            values = getattr(market, key)
            if first in values:
                changes[key] = {first: deepcopy(values[first]), second: deepcopy(values[first])}
        for row in changes.get("fundamental_snapshots", {}).get(second, {}).get("les_v6_filings", []):
            row["symbol"] = second
            for field in row.get("field_evidence", {}).values(): field["symbol"] = second
        account = dict(value=1_000_000., available_cash=1_000_000., intended_weight=.2,
                       capacity_weight=.2, positions=[])
        return replace(market, stock_count=2, stock_numbers=[first, second],
            stock_names=["SYNTHETIC_BASE", "SYNTHETIC_PEER"], les_v6_portfolio=account, **changes)

    def v7_receipts(market):
        snapshots = deepcopy(market.fundamental_snapshots)
        for index, symbol in enumerate(market.stock_numbers):
            gates = {}
            snapshots[symbol]["les_v7_context"] = {"receipts": gates}
            for name in ("SAFETY", "DATA_INTEGRITY", "TRADING_STATUS", "VALUATION", "STRESS_RISK"):
                candidate = replace(market, fundamental_snapshots=snapshots)
                report = daily_report(MarketHistoryAssembler().build(candidate, index))
                gates[name] = receipt(name, report["diagnostics"]["gate_bindings"][name])
        return replace(market, fundamental_snapshots=snapshots)

    cases = []
    def add(name, market, *, held=False):
        symbol = market.stock_numbers[0]
        positions = []
        cash = 1_000_000.
        if held:
            cash -= market.closes[0][-1] / 1000 * 100
            positions = [dict(symbol=symbol, shares=100., average_cost=95., entry_les=80.,
                              entry_decision="BUY", entry_date="2026-09-08", sector="24")]
            market = replace(market, entry_les_by_stock={symbol: 80.},
                entry_dates_by_stock={symbol: "2026-09-08"}, entry_decisions_by_stock={symbol: "BUY"})
        account = dict(account_equity=1_000_000., current_cash=cash, positions=positions, default_sector="ELEC")
        input_account = dict(value=1_000_000., available_cash=cash, intended_weight=.2, capacity_weight=.2,
                             positions=[dict(symbol=p["symbol"], shares=p["shares"]) for p in positions])
        market = replace(market, les_v6_portfolio=input_account)
        cases.append(dict(name=name, market=plain_input(market), account=account,
            persistence=[dict(incumbent_stock_id=symbol, challenger_stock_id=market.stock_numbers[1],
                              recent_passes=[True, False])] if held else []))

    v6 = two_stocks(daily_market())
    add("v6_evidenced_candidate", v6)
    add("v6_missing_financials", replace(v6, fundamental_snapshots={}))
    future = deepcopy(v6.fundamental_snapshots)
    for snapshot in future.values():
        row = deepcopy(snapshot["les_v6_filings"][-1])
        row["available_at"], row["revision_id"] = "2026-12-01T10:00:00+08:00", "SYNTHETIC_FUTURE_REVISION"
        row["revenue"] *= 10
        snapshot["les_v6_filings"].append(row)
    add("v6_future_revision", replace(v6, fundamental_snapshots=future))
    add("v6_account_and_persistence", v6, held=True)
    v7 = v7_receipts(two_stocks(executable_synthetic_market()))
    add("v7_evidenced_candidate", v7)
    missing = deepcopy(v7.fundamental_snapshots)
    for snapshot in missing.values(): snapshot.pop("les_v7_context", None)
    add("v7_missing_model", replace(v7, les_v7_bundle_json="", fundamental_snapshots=missing))
    future = deepcopy(v7.fundamental_snapshots)
    for snapshot in future.values():
        row = deepcopy(snapshot["les_v6_filings"][-1])
        row["available_at"] = "2026-12-01T10:00:00+08:00"
        snapshot["les_v6_filings"].append(row)
    add("v7_future_filing", v7_receipts(replace(v7, fundamental_snapshots=future)))
    erem = make_history(True)
    erem_input = {key: [None if item is None or str(item) in {"<NA>", "nan"} else plain_input(item)
                       for item in erem[key].tolist()] for key in erem}
    return dict(schema="python315-lifetime-fixtures/v1", data_kind="SYNTHETIC_ONLY",
                daily_cases=cases, erem=dict(columns=erem_input, entry_decisions=["BUY", "BUY_SMALL", "EARLY_BUY"]))


def worker(root, action):
    started = time.perf_counter()
    sys.path.insert(0, str(root))
    before = source_identity(root)
    if action == "fixtures":
        payload = make_fixtures()
        if source_identity(root) != before: raise ValueError("BENCHMARK_SOURCE_CHANGED")
        return payload
    payload = json.loads(sys.stdin.read())
    if payload.get("data_kind") != "SYNTHETIC_ONLY": raise ValueError("SYNTHETIC_ONLY_REQUIRED")
    if not 1 <= len(payload.get("daily_cases", ())) <= 12:
        raise ValueError("BOUNDED_DAILY_CASES_REQUIRED")
    for case in payload["daily_cases"]:
        market = case["market"]
        if not 2 <= market["stock_count"] <= 8 or any(len(values) > 700 for values in market["dates"]):
            raise ValueError("BOUNDED_SYMBOLS_AND_HISTORY_REQUIRED")
    imported = time.perf_counter()
    import numpy as np
    import pandas as pd
    from application.analysis_dataset import AnalysisDatasetBuilder
    from application.daily_performance import DailyPerformance, measure_stage
    from application.daily_work import DailyWorkContext
    from application.market_history import MarketHistoryInput
    from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
    from application.portfolio_contracts import PortfolioRuntimeContext
    from domain.models import PortfolioPosition
    from domain.strategies.expected_return_exit_v3 import ExpectedReturnExitV3Strategy
    from domain.strategies.portfolio_allocation_v2 import PersistenceEvidenceV2
    import_seconds = time.perf_counter() - imported

    class MemoryPersistence:
        def __init__(self, seed):
            self.seed = tuple(PersistenceEvidenceV2(**dict(item, recent_passes=tuple(item["recent_passes"]))) for item in seed)
            self.events = []
        def record_session_v2(self, trade_date, observations):
            self.events.append(dict(operation="record", trade_date=trade_date, observations=tuple(observations)))
        def load_recent_v2(self, incumbents, challengers, *, through_date, limit_per_pair):
            self.events.append(dict(operation="load", incumbents=tuple(incumbents), challengers=tuple(challengers),
                                    through_date=through_date, limit_per_pair=limit_per_pair))
            return self.seed

    stats_before = gc.get_stats()
    collections, collecting = [], {}
    def observe(phase, info):
        generation = info["generation"]
        if phase == "start": collecting[generation] = time.perf_counter()
        else:
            collections.append(dict(generation=generation, collected=info["collected"], uncollectable=info["uncollectable"],
                seconds=time.perf_counter() - collecting.pop(generation, time.perf_counter())))
    gc.callbacks.append(observe)
    probe, results, cases = DailyPerformance(), {}, []
    work = DailyWorkContext("SYNTHETIC_LIFETIME_BENCHMARK", max_bytes=16 * 1024 * 1024)
    try:
        with work.activate(), probe.activate(), measure_stage("benchmark_business_total"):
            for case in payload["daily_cases"]:
                case_started = time.perf_counter()
                market = MarketHistoryInput(**case["market"])
                persistence = MemoryPersistence(case["persistence"])
                account = dict(case["account"])
                account["positions"] = tuple(PortfolioPosition(**dict(position,
                    entry_date=dt.date.fromisoformat(position["entry_date"]))) for position in account["positions"])
                context = PortfolioRuntimeContext(**account)
                with measure_stage("daily_native_and_erem"):
                    try:
                        decisions = AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]] * market.stock_count)
                    except Exception as error:
                        raise RuntimeError("SYNTHETIC_CASE_FAILED:" + case["name"]) from error
                with measure_stage("pae_in_memory"):
                    allocated = PortfolioAllocationV2Pipeline(persistence=persistence,
                        portfolio_id="SYNTHETIC_LIFETIME_BENCHMARK").enrich(decisions, context)
                with measure_stage("business_canonicalization"):
                    results[case["name"]] = canonical(dict(native=decisions, allocated=allocated,
                        account=context, persistence_before=persistence.seed, persistence_events=persistence.events))
                cases.append(dict(name=case["name"], seconds=time.perf_counter() - case_started,
                    native_signals=[item.value("低點決策") for item in decisions],
                    quality_status=[item.value("Quality Status") for item in decisions],
                    pae_modes=[item.value("PAE Portfolio Mode") for item in allocated]))
            with measure_stage("erem_standalone"):
                history = pd.DataFrame(payload["erem"]["columns"])
                for decision in payload["erem"]["entry_decisions"]:
                    history["Entry Decision"] = decision
                    output = ExpectedReturnExitV3Strategy().run(history)
                    results["erem_" + decision] = canonical(output)
            cache_stats = work.stats
    finally:
        work.close()
        gc.callbacks.remove(observe)
    validating = time.perf_counter()
    if source_identity(root) != before: raise ValueError("BENCHMARK_SOURCE_CHANGED")
    validation_seconds = time.perf_counter() - validating
    business_bytes = encoded(results)
    return dict(input_sha256=sha(payload), source_identity=before,
        environment=dict(python=platform.python_version(), executable=str(Path(sys.executable).resolve()),
            numpy=np.__version__, pandas=pd.__version__, platform=platform.platform()),
        business=results, business_sha256=hashlib.sha256(business_bytes).hexdigest(), business_bytes=len(business_bytes),
        measurements=dict(child_wall_seconds=time.perf_counter() - started, model_import_seconds=import_seconds,
            final_source_verification_seconds=validation_seconds, stages=probe.snapshot(), cases=cases,
            cache_stats=cache_stats, gc=dict(before=stats_before, after=gc.get_stats(), collections=collections,
                observation_policy="OBSERVE_ONLY_NO_COLLECT_DISABLE_OR_THRESHOLD_CHANGE")))


def child(root, action, payload=None):
    started = time.perf_counter()
    env = dict(os.environ, PYTHONHASHSEED="0", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", action,
        "--source-root", str(root)], cwd=root, env=env, input=encoded(payload).decode() if payload else "",
        capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError("BENCHMARK_CHILD_FAILED:" + completed.stderr[-16000:])
    wall = time.perf_counter() - started
    if len(completed.stdout.encode()) > 128 * 1024 * 1024: raise ValueError("BOUNDED_OUTPUT_CAPACITY_EXCEEDED")
    result = json.loads(completed.stdout)
    if action == "run":
        result["measurements"].update(parent_spawn_through_exit_seconds=wall,
            child_stdout_bytes=len(completed.stdout.encode()), child_stderr_bytes=len(completed.stderr.encode()),
            parent_wall_scope="INCLUDES_STARTUP_IMPORT_INPUT_MODEL_GC_HASH_OUTPUT_CAPTURE_AND_EXIT")
    return result


def differences(left, right, path="$", result=None):
    result = [] if result is None else result
    if type(left) is not type(right): result.append(dict(path=path, before=left, after=right))
    elif isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                result.append(dict(path=path + "." + key, before=left.get(key), after=right.get(key)))
            else: differences(left[key], right[key], path + "." + key, result)
    elif isinstance(left, list):
        if len(left) != len(right): result.append(dict(path=path, before_length=len(left), after_length=len(right)))
        for index, (a, b) in enumerate(zip(left, right)): differences(a, b, path + "[" + str(index) + "]", result)
    elif left != right: result.append(dict(path=path, before=left, after=right))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("before", "compare"))
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--current-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("fixtures", "run"), help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.source_root.resolve(), args.worker), ensure_ascii=False, allow_nan=False))
        return
    if args.mode is None or args.output is None: parser.error("Provide --mode and a new --output path")
    if not 3 <= args.rounds <= 5: parser.error("Use 3-5 rounds")
    if args.output.exists(): parser.error("Output is immutable; choose a new path")
    harness_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.mode == "before":
        if args.baseline_root is None: parser.error("before requires --baseline-root")
        root = args.baseline_root.resolve()
        payload = child(root, "fixtures")
        runs = [child(root, "run", payload) for _ in range(args.rounds)]
        business = runs[0].pop("business")
        for run in runs[1:]:
            if differences(business, run.pop("business")): raise ValueError("BASELINE_NONDETERMINISTIC_RESULTS")
        report = dict(schema="python315-lifetime-benchmark/v1", status="BEFORE_FROZEN_PENDING_AFTER",
            harness_sha256=harness_sha256,
            data_kind="SYNTHETIC_ONLY", fixtures=payload, input_sha256=sha(payload), baseline_root=str(root),
            baseline_business=business, baseline_runs=runs, comparison_policy="COMPLETE_TYPED_BUSINESS_TREE_NO_EXCLUDED_FIELDS",
            scope="DAILY_V6_V7_EREM_IN_MEMORY_PAE_NOT_FULL_ANALYSIS_NO_NETWORK_MAIL_OR_FORMAL_STATE")
    else:
        if args.baseline_report is None: parser.error("compare requires --baseline-report")
        before_bytes = args.baseline_report.read_bytes()
        before = json.loads(before_bytes)
        if before["harness_sha256"] != harness_sha256: raise ValueError("BENCHMARK_HARNESS_VERSION_MISMATCH")
        payload = before["fixtures"]
        if sha(payload) != before["input_sha256"]: raise ValueError("INPUT_FIXTURE_HASH_MISMATCH")
        if before["data_kind"] != "SYNTHETIC_ONLY": raise ValueError("SYNTHETIC_ONLY_REQUIRED")
        runs = [child(args.current_root.resolve(), "run", payload) for _ in range(args.rounds)]
        business = runs[0].pop("business")
        for run in runs[1:]:
            if differences(business, run.pop("business")): raise ValueError("AFTER_NONDETERMINISTIC_RESULTS")
        diff = differences(before["baseline_business"], business)
        environments = [run["environment"] for run in before["baseline_runs"] + runs]
        if any(value != environments[0] for value in environments): raise ValueError("ENVIRONMENT_MISMATCH")
        report = dict(before, status="COMPLETE" if not diff else "BUSINESS_PARITY_FAILED",
            baseline_artifact_sha256=hashlib.sha256(before_bytes).hexdigest(), after_business=business, after_runs=runs,
            complete_business_parity=not diff, field_differences=diff,
            allowed_execution_differences=["source_identity", "elapsed timings", "GC observations", "RSS", "cache counters"],
            filesystem_policy="NO_FILESYSTEM_SPEEDUP_CLAIM_SAME_INTERPRETER_AND_FIXTURE_CONTENT_DIFFERENT_SOURCE_ROOTS")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output.resolve()), status=report["status"], input_sha256=report["input_sha256"],
        business_parity=report.get("complete_business_parity"), rounds=args.rounds), sort_keys=True))
    if report["status"] == "BUSINESS_PARITY_FAILED": raise SystemExit(1)


if __name__ == "__main__": main()

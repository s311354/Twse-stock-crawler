"""Synthetic lossless-storage and preprocessing tests; no live requests."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import sqlite3
import weakref

import pandas as pd
import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_performance import DailyPerformance
from application.daily_work import DailyWorkContext
from infrastructure.storage.csv_repository import CsvRepository
from infrastructure.storage.report_evidence import (ReportEvidenceReader, ReportEvidenceWriter,
    STORE_NAME, compact_report, load_report_bundle, resolve_csv_report)
from infrastructure.storage.research_cache import SQLiteRawParseCache
from tests.strategies.test_daily_v6_pipeline import build, daily_market


def test_native_report_is_saved_once_and_csv_roundtrip_is_exact(tmp_path):
    builder, decisions = build()
    native = json.loads(decisions[0].value("LES v6 Report"))
    dataset = builder.project_decisions(decisions)
    before = dataset.copy(deep=True)
    repository = CsvRepository(tmp_path)
    repository.write_analysis_dataset("elec", dataset)
    pd.testing.assert_frame_equal(before, dataset, check_exact=True)
    saved = pd.read_csv(tmp_path / "elec_analysis_dataset.csv", dtype={"證券代號": str})
    cell = saved.iloc[0]["LES v6 Report"]
    assert len(cell) < 1000 and "quality_field_checks" not in cell
    assert resolve_csv_report(cell, tmp_path) == native
    assert load_report_bundle(tmp_path / "elec_les_v6_reports.json")["reports"] == [native]
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        count = connection.execute("SELECT COUNT(*) FROM report_objects").fetchone()[0]
    repository.write_analysis_dataset("another", dataset)
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        assert connection.execute("SELECT COUNT(*) FROM report_objects").fetchone()[0] == count
    with sqlite3.connect(tmp_path / "les_v6_leadership_history.sqlite3") as connection:
        evidence = json.loads(connection.execute("SELECT evidence_json FROM observations LIMIT 1").fetchone()[0])
        assert set(evidence) == {"_report_reference"}
    from infrastructure.storage.leadership_watch_repository import LeadershipWatchRepository
    timeline = LeadershipWatchRepository(tmp_path / "les_v6_leadership_history.sqlite3").timeline("elec", native["symbol"])
    assert timeline[0]["quality"] == native["quality"]


@pytest.mark.parametrize("change", ["summary", "binding", "hash", "missing", "corrupt"])
def test_evidence_references_fail_closed(change, tmp_path):
    _, decisions = build()
    native = json.loads(decisions[0].value("LES v6 Report"))
    with ReportEvidenceWriter(tmp_path / STORE_NAME) as writer:
        ref = writer.retain(native)
    summary = compact_report(native, deepcopy(ref))
    if change == "summary":
        summary["decision"] = dict(summary["decision"], score=1000)
    elif change == "binding":
        summary["_report_reference"]["symbol"] = "3016"
    elif change == "hash":
        summary["_report_reference"]["report_sha256"] = "0" * 64
    else:
        with sqlite3.connect(tmp_path / STORE_NAME) as connection:
            if change == "missing":
                connection.execute("DELETE FROM report_objects WHERE sha256=?", (ref["root"][1],))
            else:
                connection.execute("UPDATE report_objects SET body=? WHERE sha256=?", (b"corrupt", ref["root"][1]))
    with ReportEvidenceReader(tmp_path) as reader, pytest.raises(ValueError, match="REPORT_EVIDENCE"):
        reader.resolve(summary)


def test_legacy_and_restored_evidence_are_isolated(tmp_path):
    original = dict(symbol="TEST", signal_date="2026-09-18", model_version="6.0.0", parameter_version="synthetic",
                    evidence=[dict(source="SYNTHETIC_ONLY", proof="abcd" * 10000)] * 3)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(dict(reports=[original])))
    assert load_report_bundle(path)["reports"] == [original]
    with ReportEvidenceWriter(tmp_path / STORE_NAME) as writer:
        ref = writer.retain(original)
    with ReportEvidenceReader(tmp_path) as reader:
        result = reader.resolve(ref)
        result["evidence"][0]["proof"] = "mutated"
        assert result["evidence"][1]["proof"] == original["evidence"][1]["proof"]
        assert reader.resolve(ref) == original


def test_evidence_writer_batches_lookups_and_rolls_back_flushed_nodes(tmp_path):
    report = dict(symbol="TEST", signal_date="2026-09-18", model_version="6.0.0", parameter_version="synthetic",
                  evidence=[{"proof": "SYNTHETIC-" + str(i) + "x" * 600} for i in range(700)])
    statements = []
    path = tmp_path / STORE_NAME
    with ReportEvidenceWriter(path, batch_nodes=256) as writer:
        writer.connection.set_trace_callback(statements.append)
        assert writer.connection.execute("PRAGMA cache_size").fetchone()[0] == -32768
        ref = writer.retain(report)
        assert len(writer._pending) < 256
        assert writer._pending_bytes < 8 * 1024 * 1024
    lookups = [s for s in statements if "CROSS JOIN report_objects AS stored" in s]
    assert 1 < len(lookups) <= 4  # Not one disk lookup per ~700 nodes.
    with ReportEvidenceReader(tmp_path) as reader:
        assert reader.resolve(ref) == report
    with sqlite3.connect(path) as connection:
        before = connection.execute("SELECT COUNT(*) FROM report_objects").fetchone()[0]
    changed = dict(report, evidence=[{"proof": "different-SYNTHETIC-" + str(i) + "y" * 600} for i in range(700)])
    with pytest.raises(RuntimeError, match="interrupted"), ReportEvidenceWriter(path, batch_nodes=256) as writer:
        writer.retain(changed)  # Includes flushed batches inside the transaction.
        raise RuntimeError("interrupted")
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM report_objects").fetchone()[0] == before
    with ReportEvidenceReader(tmp_path) as reader:
        assert reader.resolve(ref) == report


def test_csv_publication_does_not_retain_all_expanded_native_reports(tmp_path, monkeypatch):
    from infrastructure.storage import csv_repository
    builder, decisions = build()
    base = builder.project_decisions(decisions)
    native = json.loads(decisions[0].value("LES v6 Report"))
    rows = []
    for symbol in ("1001", "1002", "1003"):
        row = base.copy()
        row["證券代號"] = symbol
        row["LES v6 Report"] = json.dumps(dict(native, symbol=symbol))
        rows.append(row)
    dataset = pd.concat(rows, ignore_index=True)
    references = []
    class TrackedReport(dict):
        pass
    original_loads = json.loads
    original_retain = ReportEvidenceWriter.retain
    def tracked_loads(*args, **kwargs):
        value = original_loads(*args, **kwargs)
        if isinstance(value, dict) and value.get("model_version") == "6.0.0":
            value = TrackedReport(value)
            references.append(weakref.ref(value))
        return value
    def retain(writer, report):
        assert sum(ref() is not None for ref in references) == 1
        return original_retain(writer, report)
    monkeypatch.setattr(csv_repository.json, "loads", tracked_loads)
    monkeypatch.setattr(ReportEvidenceWriter, "retain", retain)
    CsvRepository(tmp_path).write_analysis_dataset("synthetic", dataset)
    assert not any(ref() is not None for ref in references)


def test_publication_parses_each_native_row_once_and_preserves_duplicates(tmp_path):
    builder,decisions=build()
    base=builder.project_decisions(decisions)
    native=json.loads(decisions[0].value('LES v6 Report'))
    duplicate=base.copy()
    duplicate['LES v6 Report']=json.dumps(native,sort_keys=True,indent=2)
    dataset=pd.concat([base,duplicate],ignore_index=True)
    probe=DailyPerformance()
    with probe.activate():CsvRepository(tmp_path).write_analysis_dataset('synthetic',dataset)
    assert probe.counters['report_native_parses']==2
    assert load_report_bundle(tmp_path/'synthetic_les_v6_reports.json')['reports']==[native]
    conflicting=deepcopy(native);conflicting['decision']['score']+=1
    duplicate['LES v6 Report']=json.dumps(conflicting)
    with pytest.raises(ValueError,match='CONFLICTING_V6_REPORTS'):
        CsvRepository(tmp_path).write_analysis_dataset('conflict',pd.concat([base,duplicate],ignore_index=True))
    assert not (tmp_path/'conflict_analysis_dataset.csv').exists()


def test_late_batch_validation_rolls_back_already_flushed_report_nodes(tmp_path):
    builder,decisions=build();base=builder.project_decisions(decisions)
    native=json.loads(decisions[0].value('LES v6 Report'))
    native['evidence']=[{'proof':'SYNTHETIC-'+str(i)+'x'*600} for i in range(700)]
    rows=[]
    for symbol,policy in [('1001',2),('1002',3)]:
        report=deepcopy(native);report['symbol']=symbol;report['parameters']['decision_policy_version']=policy
        row=base.copy();row['證券代號']=symbol;row['LES v6 Report']=json.dumps(report);rows.append(row)
    with pytest.raises(ValueError,match='MIXED_DAILY_V6_SCORE_POLICIES'):
        CsvRepository(tmp_path).write_analysis_dataset('invalid',pd.concat(rows,ignore_index=True))
    with sqlite3.connect(tmp_path/STORE_NAME) as connection:
        assert connection.execute('SELECT COUNT(*) FROM report_objects').fetchone()[0]==0
    assert not (tmp_path/'invalid_analysis_dataset.csv').exists()
    assert not (tmp_path/'invalid_les_v6_reports.json').exists()


def test_cached_and_uncached_native_results_and_account_changes_are_identical():
    market = daily_market()
    counter = DailyPerformance()
    variants = [market, replace(market, les_v6_portfolio=dict(market.les_v6_portfolio, available_cash=1))]
    expected = [AnalysisDatasetBuilder().build_decisions(m, [[1, 1, 1, 1, .01]]) for m in variants]
    work = DailyWorkContext("synthetic-test-only")
    with work.activate(), counter.activate():
        actual = [AnalysisDatasetBuilder().build_decisions(m, [[1, 1, 1, 1, .01]]) for m in variants]
    assert actual == expected
    assert counter.counters["feature_pair_builds"] == 1
    assert counter.counters["feature_pair_cache_hits"] == 1
    assert counter.counters["quality_diagnostics_cache_hits"] == 1
    assert counter.calls["quality_gate"] == 2  # Actual gate is NOT memoized.
    assert actual[0][0].value("建議股數") != actual[1][0].value("建議股數")
    work.close()
    with pytest.raises(ValueError, match="CLOSED"), work.activate():
        pass


@pytest.mark.parametrize("case", ["complete", "missing_cfo", "conflicting_proof", "future_revision"])
@pytest.mark.parametrize("enabled", [False, True])
def test_source_bound_readonly_diagnostics_cross_json_cache_boundary(case, enabled, monkeypatch):
    from application import low_entry_v6_daily
    from tests.strategies.test_low_entry_v6_comparative_ttm import originals

    # These synthetic iXBRL records include nested, source-bound field proofs;
    # unlike the simple normalized fixture, EvidenceContext freezes those too.
    records, _ = originals()
    if case == "missing_cfo":
        records[-1]["cfo"] = None
    elif case == "conflicting_proof":
        records[-1]["field_evidence"]["cfo"]["revision_id"] = "different-original"
    elif case == "future_revision":
        future = deepcopy(records[-1])
        future.update(available_at="2026-09-12T15:00:00+08:00", revision_id="future", cfo=-10**12)
        records.append(future)
    market = replace(daily_market(), fundamental_snapshots={
        "2330": {"sector": "24", "les_v6_filings": records}})
    before = deepcopy(records)
    expected = AnalysisDatasetBuilder(optimized_daily=False).build_decisions(market, [[1, 1, 1, 1, .01]])
    calls = []
    original_verify = low_entry_v6_daily.verification_diagnostics
    def counted(*args, **kwargs):
        calls.append(1)
        return original_verify(*args, **kwargs)
    monkeypatch.setattr(low_entry_v6_daily, "verification_diagnostics", counted)
    work = DailyWorkContext("synthetic-source-bound-proofs", enabled=enabled)
    probe = DailyPerformance()
    with work.activate(), probe.activate():
        actual = [AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]]) for _ in range(2)]
    assert actual == [expected, expected]  # Full typed/native fields, not just score.
    assert records == before
    assert len(calls) == (1 if enabled else 2)
    assert probe.counters["quality_diagnostics_cache_hits"] == (1 if enabled else 0)
    report = json.loads(actual[0][0].value("LES v6 Report"))
    checks = report["research"]["verification"]["field_checks"]
    assert any(check.get("content_sha256") and check.get("evidence") for check in checks)
    json.dumps(report, allow_nan=False)
    if case in {"missing_cfo", "conflicting_proof"}:
        assert report["quality"]["status"] == "DATA_INSUFFICIENT"
        assert report["decision"]["signal"] not in {"BUY", "BUY_SMALL"}
    work.close()


def test_work_cache_is_bounded_copy_on_read_and_version_guarded():
    calls = []
    work = DailyWorkContext("synthetic", max_bytes=2048, check_version=lambda: calls.append("check"))
    with work.activate():
        value = work.memo_json("proof", "cutoff1", lambda: {"value": [1]})
        value["value"][0] = 99
        assert work.memo_json("proof", "cutoff1", lambda: pytest.fail("recomputed")) == {"value": [1]}
        assert work.memo_json("proof", "cutoff2", lambda: {"value": [2]}) == {"value": [2]}
        for key in range(50):
            work.memo_immutable("synthetic", key, lambda: (key,), lambda _: 600)
        assert work.stats["immutable_bytes"] <= 1024
        assert work.stats["json"]["bytes"] <= 1024
    assert calls == ["check", "check"]


def test_original_parse_survives_lru_eviction_and_restart_but_not_version_or_hash_change(tmp_path):
    from infrastructure.fundamentals.mops_queries import _raw_normalized_facts
    from tests.fundamentals.test_ixbrl_units import document
    content = document()
    path = tmp_path / "SYNTHETIC.xml"
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    stat = path.stat()
    signature = stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size
    probe = DailyPerformance()
    results = []
    with probe.activate():
        for version in ("parser1", "parser1", "parser2"):
            _raw_normalized_facts.cache_clear()
            with SQLiteRawParseCache(tmp_path / "preprocess.sqlite3") as disk:
                with DailyWorkContext("synthetic-code", parse_memo=disk).activate():
                    results.append(_raw_normalized_facts(path, sha, signature, version))
                    assert _raw_normalized_facts(path, "0" * 64, signature, version) is None
    assert results[0] == results[1] == results[2]
    assert probe.counters["raw_unit_parse_calls"] == 2
    assert probe.counters["raw_parse_disk_hits"] == 1
    assert probe.counters["raw_unit_hash_rejections"] == 3


def test_metrics_receipt_is_saved_with_logging_disabled(tmp_path):
    from types import SimpleNamespace
    from application.contracts import StockAnalysisRequest
    from infrastructure.daily_performance import retain_daily_performance
    request = StockAnalysisRequest(["2330"], "ELEC", "synthetic", 700)
    context = SimpleNamespace(resource=SimpleNamespace(csv_repository=CsvRepository(tmp_path)), market_date="2026-09-18")
    for status in ("FAILED", "SUCCESS"):
        path = retain_daily_performance(request, context, dict(status=status, stages={"les": {"seconds": 1., "calls": 1}}), run_id="synthetic")
        assert json.loads(path.read_text())["status"] == status


def test_global_parse_lru_never_retains_native_work_versions(tmp_path):
    from infrastructure.fundamentals.mops_queries import _raw_normalized_facts
    from tests.fundamentals.test_ixbrl_units import document
    content = document()
    path = tmp_path / "SYNTHETIC-version.xml"
    path.write_bytes(content)
    stat = path.stat()
    signature = stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size
    sha = hashlib.sha256(content).hexdigest()
    probe = DailyPerformance()
    _raw_normalized_facts.cache_clear()
    with probe.activate():
        for version in ("code-A", "code-A", "code-B"):
            work = DailyWorkContext(version)
            try:
                with work.activate():
                    _raw_normalized_facts(path, sha, signature, "same-parser-label")
                    _raw_normalized_facts(path, sha, signature, "same-parser-label")
            finally:
                work.close()
    # Each work re-verifies its source; only its own retry is a memory hit.
    assert probe.counters["raw_unit_parse_calls"] == 3
    assert _raw_normalized_facts.cache_info().currsize == 0


def test_disk_cache_capacity_evicts_only_repairable_parses(tmp_path):
    with SQLiteRawParseCache(tmp_path / "synthetic-cache.sqlite3", max_bytes=30) as disk:
        for i in range(10):
            assert disk.memo("ixbrl", ("synthetic", str(i)), lambda: [{"value": "test"}]) == [{"value": "test"}]
        assert disk.bytes <= 30
        assert disk.connection.execute("SELECT COUNT(*) FROM research_parses").fetchone()[0] == 1


@pytest.mark.parametrize("change", ["cutoff", "proof", "units", "config"])
def test_preprocessing_key_invalidates_needed_inputs(change):
    from domain.features import SharedFeaturePipeline, SharedFeatureConfig
    from application.strategy_execution import StrategyExecutionService
    from tests.test_daily_computation_performance import watched_histories
    history = watched_histories()[0]
    probe = DailyPerformance()
    work = DailyWorkContext("synthetic")
    with probe.activate(), work.activate():
        executor = StrategyExecutionService()
        executor.prepare([history], include_normalized=True)
        if change == "cutoff":
            history["Date"] = pd.to_datetime(history["Date"]) + pd.Timedelta(days=1)
        elif change == "units":
            history["Close"] /= 1000.
        elif change == "config":
            executor = StrategyExecutionService(feature_pipeline_factory=lambda: SharedFeaturePipeline(SharedFeatureConfig(ema_long_window=120)))
        else:
            # Financial proofs are deliberately not a feature dependency.
            history.attrs["fundamental_snapshot"] = {"proof": "changed"}
        executor.prepare([history], include_normalized=True)
    assert probe.counters["feature_pair_builds"] == (1 if change == "proof" else 2)

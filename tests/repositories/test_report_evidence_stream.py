"""Synthetic-only bounded reads and full-integrity per-stock delivery."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import weakref
import zlib

import pytest

from application.replay_contracts import canonical_json
from infrastructure.storage import report_evidence
from infrastructure.storage.evidence_graph import MAX_COMPRESSED_NODE_BYTES, MAX_REPORT_BYTES
from infrastructure.storage.report_bundle_stream import iter_bundle_records
from infrastructure.storage.report_evidence import (
    ReportEvidenceReader, ReportEvidenceWriter, STORE_NAME, compact_report,
    iter_report_bundle, load_report_bundle,
)


def native(symbol="TEST", count=30):
    return dict(
        symbol=symbol, signal_date="2026-09-18", model_version="6.0.0",
        parameter_version="SYNTHETIC_ONLY", decision=dict(signal="WATCH", score=61.25),
        quality=dict(metrics=dict(quality_field_checks=[
            dict(id=i, source="SYNTHETIC_ONLY", proof=f"proof-{i}-" + "證據\\\"" * 200)
            for i in range(count)], retained_metric=42)),
        research=dict(verification=dict(field_checks=[dict(source="SYNTHETIC_ONLY", proof="x" * 600)])),
    )


def bundle(tmp_path, reports, *, stale=(), legacy=False):
    with ReportEvidenceWriter(tmp_path / STORE_NAME) as writer:
        summaries = [compact_report(r, writer.retain(r)) for r in reports]
        stale_summaries = [compact_report(r, writer.retain(r)) for r in stale]
    payload = dict(metadata=dict(notes="合成測試；非歷史證明", samples=[True, None, 1.23e-5]),
                   reports=reports if legacy else summaries, stale_reports=stale_summaries)
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path, payload


def test_batched_queries_restore_exact_report_with_bounded_cache(tmp_path):
    path, payload = bundle(tmp_path, [native(count=700)])
    counts = []
    for batch_nodes in (1, 128):
        with ReportEvidenceReader(tmp_path, batch_nodes=batch_nodes) as reader:
            reader.check_store(STORE_NAME)
            statements = []
            reader._connections[STORE_NAME].set_trace_callback(statements.append)
            assert reader.resolve(payload["reports"][0]) == native(count=700)
            queries = [s for s in statements if s.startswith("SELECT") and "report_objects" in s]
            counts.append(len(queries))
            assert reader._bytes <= reader.cache_bytes
            assert all(type(value) is bytes for value in reader._cache.values())
            assert all(s.count("','") < 128 for s in queries)
    assert counts[0] > 700
    assert counts[1] <= 20 and counts[1] * 10 < counts[0]
    assert next(iter_report_bundle(path, require_reference=True)) == native(count=700)


@pytest.mark.parametrize("cache_bytes,batch_bytes", [(0, 1024), (1024, 512), (32 * 1024, 4096)])
def test_small_or_disabled_cache_never_changes_full_report(cache_bytes, batch_bytes, tmp_path):
    report = native(count=80)
    path, _ = bundle(tmp_path, [report, dict(report, symbol="SECOND")])
    assert list(iter_report_bundle(path, cache_bytes=cache_bytes, batch_bytes=batch_bytes)) == [
        report, dict(report, symbol="SECOND")]


@pytest.mark.parametrize("field", ["symbol", "signal_date", "model_version", "parameter_version"])
def test_every_binding_is_checked_before_delivery(field, tmp_path):
    path, payload = bundle(tmp_path, [native("FIRST"), native("SECOND")])
    payload["reports"][1]["_report_reference"][field] = "tampered"
    path.write_text(json.dumps(payload))
    with closing(iter_report_bundle(path, require_reference=True)) as reports:
        assert next(reports) == native("FIRST")
        with pytest.raises(ValueError, match="BINDING_MISMATCH"):
            next(reports)


@pytest.mark.parametrize("change,error", [
    ("summary", "SUMMARY_MISMATCH"), ("hash", "REPORT_HASH_MISMATCH"),
    ("missing", "OBJECT_MISSING"), ("corrupt", "HASH_MISMATCH"),
    ("oversized", "NODE_TOO_LARGE"), ("expansion", "EXPANSION_TOO_LARGE"),
])
def test_no_delivery_before_all_node_report_and_summary_checks(change, error, tmp_path):
    path, payload = bundle(tmp_path, [native()])
    reference = payload["reports"][0]["_report_reference"]
    options = {}
    if change == "summary":
        payload["reports"][0]["decision"]["score"] = 100
    elif change == "hash":
        reference["report_sha256"] = "0" * 64
    elif change == "expansion":
        options["max_value_bytes"] = 1000
    else:
        with sqlite3.connect(tmp_path / STORE_NAME) as connection:
            sha = reference["root"][1]
            if change == "missing":
                connection.execute("DELETE FROM report_objects WHERE sha256=?", (sha,))
            else:
                body = b"invalid" if change == "corrupt" else b"x" * (MAX_COMPRESSED_NODE_BYTES + 1)
                connection.execute("UPDATE report_objects SET body=? WHERE sha256=?", (body, sha))
    path.write_text(json.dumps(payload))
    with closing(iter_report_bundle(path, **options)) as reports, pytest.raises(ValueError, match=error):
        next(reports)


def test_corrupt_prefetched_child_is_not_skipped(tmp_path):
    path, payload = bundle(tmp_path, [native(count=300)])
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        sha = next(sha for sha, body in connection.execute("SELECT sha256,body FROM report_objects")
                   if b'proof-150-' in zlib.decompress(body))
        connection.execute("UPDATE report_objects SET body=? WHERE sha256=?", (b"bad", sha))
    with closing(iter_report_bundle(path)) as reports, pytest.raises(ValueError, match="HASH_MISMATCH"):
        next(reports)


def test_streaming_legacy_stale_and_reference_parity(tmp_path):
    original = native()
    path, _ = bundle(tmp_path, [original], stale=[native("STALE")])
    expected = load_report_bundle(path)
    assert list(iter_report_bundle(path, chunk_bytes=3)) == expected["reports"] + expected["stale_reports"]
    assert list(iter_report_bundle(path, buckets=("stale_reports",), chunk_bytes=1)) == expected["stale_reports"]
    path, _ = bundle(tmp_path, [original], legacy=True)
    assert list(iter_report_bundle(path, require_reference=False)) == [original]
    with closing(iter_report_bundle(path, require_reference=True)) as reports:
        with pytest.raises(ValueError, match="REFERENCE_REQUIRED"):
            next(reports)
    with ReportEvidenceReader(tmp_path) as reader:
        assert reader.resolve(original) is original  # Existing legacy API preserved.
        returned = next(reader.iter_reports([original], require_reference=False))
        returned["decision"]["score"] = -100
        assert original["decision"]["score"] == 61.25


def test_stream_only_holds_current_report_and_closes_resources(tmp_path, monkeypatch):
    path, _ = bundle(tmp_path, [native(str(i)) for i in range(8)])
    refs = []
    readers = []
    original_resolve = ReportEvidenceReader.resolve
    original_exit = ReportEvidenceReader.__exit__
    class TrackedReport(dict):
        pass
    def resolve(reader, value):
        assert not any(ref() is not None for ref in refs)
        report = TrackedReport(original_resolve(reader, value))
        refs.append(weakref.ref(report))
        return report
    def finish(reader, *args):
        original_exit(reader, *args)
        readers.append(reader)
    monkeypatch.setattr(ReportEvidenceReader, "resolve", resolve)
    monkeypatch.setattr(ReportEvidenceReader, "__exit__", finish)
    reports = iter_report_bundle(path)
    for _ in range(8):
        report = next(reports)
        del report
    with pytest.raises(StopIteration):
        next(reports)
    assert not any(ref() is not None for ref in refs)
    assert not readers[-1]._connections and not readers[-1]._cache
    reports = iter_report_bundle(path)
    report = next(reports)
    del report
    reports.close()
    assert not any(ref() is not None for ref in refs)
    assert not readers[-1]._connections


def test_stream_uses_bounded_file_reads_and_preserves_sources(tmp_path, monkeypatch):
    path, _ = bundle(tmp_path, [native("FIRST"), native("SECOND")])
    originals = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path, tmp_path / STORE_NAME)}
    opened = []
    original_open = Path.open
    class BoundedFile:
        def __init__(self, handle):
            self.handle = handle
            self.calls = []
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def read(self, size):
            assert 0 < size <= 17
            self.calls.append(size)
            return self.handle.read(size)
    def open_file(file_path, *args, **kwargs):
        handle = original_open(file_path, *args, **kwargs)
        if file_path == path:
            file = BoundedFile(handle)
            opened.append(file)
            return file
        return handle
    monkeypatch.setattr(Path, "open", open_file)
    with closing(iter_report_bundle(path, chunk_bytes=17)) as reports:
        assert next(reports) == native("FIRST")
    assert opened[-1].handle.closed and opened[-1].calls
    monkeypatch.setattr(Path, "open", original_open)
    assert originals == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in originals}


def test_mutating_stream_output_does_not_modify_warm_shared_nodes(tmp_path):
    path, payload = bundle(tmp_path, [native(), native()])
    with ReportEvidenceReader(tmp_path) as reader:
        reports = reader.iter_reports(payload["reports"])
        first = next(reports)
        first["quality"]["metrics"]["quality_field_checks"][0]["proof"] = "modified"
        assert next(reports) == native()
        assert reader.resolve(payload["reports"][0]) == native()


def test_warm_cache_is_invalidated_on_external_commit(tmp_path):
    _, payload = bundle(tmp_path, [native(), native()])
    with ReportEvidenceReader(tmp_path) as reader:
        reports = reader.iter_reports(payload["reports"])
        assert next(reports) == native()
        with sqlite3.connect(tmp_path / STORE_NAME) as connection:
            connection.execute("UPDATE report_objects SET body=?", (b"corrupt",))
        with pytest.raises(ValueError, match="HASH_MISMATCH"):
            next(reports)


def test_portable_nodes_are_verified_every_time(tmp_path):
    path, payload = bundle(tmp_path, [native(), native()])
    directory = tmp_path / (STORE_NAME + ".nodes")
    directory.mkdir()
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        for sha, body in connection.execute("SELECT sha256,body FROM report_objects"):
            (directory / sha).write_bytes(zlib.decompress(body))
    (tmp_path / STORE_NAME).unlink()
    with closing(iter_report_bundle(path)) as reports:
        assert next(reports) == native()
        (directory / payload["reports"][0]["_report_reference"]["root"][1]).write_bytes(b"changed")
        with pytest.raises(ValueError, match="HASH_MISMATCH"):
            next(reports)


@pytest.mark.parametrize("chunk_bytes", [1, 2, 7, 65536])
def test_incremental_parser_handles_bom_cjk_escapes_and_token_boundaries(tmp_path, chunk_bytes):
    values = [dict(symbol="證據", text="a\\\"b\n非重要 子公司", number=-12345.67e-12,
                   flags=[None, True, False], nested=dict(items=[{}, []]))]
    path = tmp_path / "input.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(dict(metadata=[{}, []], reports=values), ensure_ascii=False).encode())
    assert list(iter_bundle_records(path, buckets=("reports",), chunk_bytes=chunk_bytes)) == values


@pytest.mark.parametrize("text,error", [
    ('{"reports":[{"symbol":"TEST"}', "TRUNCATED|INVALID"),
    ('{"reports":[{"broken":}]}', "INVALID"),
    ('{"reports":{}}', "INVALID"),
    ('{"reports":[],"reports":[]}', "KEY"),
    ('{"reports":[]} false', "TRAILING"),
    ('{"ignored":[1,],"reports":[]}', "INVALID"),
    ('{"ignored":{"key":true false},"reports":[]}', "INVALID"),
])
def test_invalid_json_never_silently_becomes_empty_data(text, error, tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError, match=error):
        list(iter_report_bundle(path, chunk_bytes=1, require_reference=False))


@pytest.mark.parametrize("options,error", [
    (dict(cache_bytes=-1), "CACHE_BYTES"), (dict(cache_bytes=65 * 1024 * 1024), "CACHE_BYTES"),
    (dict(batch_nodes=129), "BATCH_NODES"), (dict(batch_nodes=0), "BATCH_NODES"),
    (dict(batch_bytes=0), "BATCH_BYTES"), (dict(max_value_bytes=None), "MAX_VALUE_BYTES"),
    (dict(max_record_bytes=MAX_REPORT_BYTES + 1), "MAX_RECORD_BYTES"),
    (dict(chunk_bytes=0), "CHUNK_BYTES"), (dict(buckets=("arbitrary",)), "BUCKETS"),
])
def test_public_stream_limits_cannot_be_disabled(options, error, tmp_path):
    path, _ = bundle(tmp_path, [native()])
    with pytest.raises(ValueError, match=error):
        next(iter_report_bundle(path, **options))


def test_record_expansion_and_depth_limits_are_explicit(tmp_path):
    path = tmp_path / "large.json"
    path.write_text(json.dumps(dict(reports=[dict(proof="x" * 1000)])))
    with pytest.raises(ValueError, match="RECORD_TOO_LARGE"):
        next(iter_report_bundle(path, max_record_bytes=100, chunk_bytes=7))
    path.write_text('{"reports":[' + '[' * 130 + '0' + ']' * 130 + ']}')
    with pytest.raises(ValueError, match="TOO_DEEP"):
        next(iter_report_bundle(path, chunk_bytes=7))


def test_binding_fields_cannot_be_missing_from_both_report_and_reference(tmp_path):
    report = native()
    del report["model_version"]
    with ReportEvidenceWriter(tmp_path / STORE_NAME) as writer:
        root = writer._encode(report)
    reference = dict(schema=report_evidence.SCHEMA, store=STORE_NAME, root=root,
                     report_sha256=hashlib.sha256(canonical_json(report).encode()).hexdigest(),
                     symbol=report["symbol"], signal_date=report["signal_date"], parameter_version=report["parameter_version"])
    with ReportEvidenceReader(tmp_path) as reader, pytest.raises(ValueError, match="BINDING_MISMATCH"):
        next(reader.iter_reports([reference], require_reference=True))


def test_stream_matches_actual_native_pipeline_output(tmp_path):
    from tests.strategies.test_daily_v6_pipeline import build
    _, decisions = build()
    reports = [json.loads(d.value("LES v6 Report")) for d in decisions]
    path, _ = bundle(tmp_path, reports)
    assert list(iter_report_bundle(path, require_reference=True, batch_nodes=16)) == reports


def test_readonly_verification_entry_exhausts_bundle_without_model_execution(tmp_path):
    from examples.benchmark_daily_storage import verify_report_stream
    path, payload = bundle(tmp_path, [native("ONE"), native("TWO")], stale=[native("STALE")])
    result = verify_report_stream(path)
    assert result["reports_verified"] == 3
    assert result["sample_symbols"] == ["ONE", "TWO", "STALE"]
    assert result["integrity_policy"] == "REQUIRE_REFERENCES"
    payload["stale_reports"][0]["decision"]["score"] = -100
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SUMMARY_MISMATCH"):
        verify_report_stream(path)


def test_prefetch_bytes_and_open_connections_are_bounded(tmp_path, monkeypatch):
    _, payload = bundle(tmp_path, [native(count=300)])
    original_prefetch = ReportEvidenceReader._prefetch
    def bounded(reader, descriptors, store):
        before = set(reader._cache)
        original_prefetch(reader, descriptors, store)
        added = sum(len(value) for key, value in reader._cache.items() if key not in before)
        assert added <= min(reader.cache_bytes, reader.batch_bytes)
        assert reader._bytes <= reader.cache_bytes
    monkeypatch.setattr(ReportEvidenceReader, "_prefetch", bounded)
    with ReportEvidenceReader(tmp_path, cache_bytes=8 * 1024, batch_bytes=4096) as reader:
        assert reader.resolve(payload["reports"][0]) == native(count=300)
        for index in range(10):
            report = native(str(index), count=1)
            with ReportEvidenceWriter(tmp_path / f"store-{index}.sqlite3") as writer:
                reference = writer.retain(report)
            assert reader.resolve(reference) == report
            assert len(reader._connections) <= 8


def test_unicode_expansion_uses_bytes_not_code_points(tmp_path):
    report = dict(symbol="TEST", signal_date="2026-09-18", model_version="6.0.0",
                  parameter_version="SYNTHETIC_ONLY", text="證" * 3000)
    _, payload = bundle(tmp_path, [report])
    with ReportEvidenceReader(tmp_path, max_value_bytes=5000) as reader:
        with pytest.raises(ValueError, match="EXPANSION_TOO_LARGE"):
            reader.resolve(payload["reports"][0])


def test_unknown_reference_schema_does_not_fall_back_to_legacy_stream(tmp_path):
    _, payload = bundle(tmp_path, [native()])
    payload["reports"][0]["_report_reference"]["schema"] = "unknown-reference-version"
    with ReportEvidenceReader(tmp_path) as reader, pytest.raises(ValueError, match="REFERENCE_REQUIRED"):
        next(reader.iter_reports(payload["reports"]))


def test_summary_equality_does_not_coerce_boolean_to_numeric_score(tmp_path):
    report = native()
    report["decision"]["score"] = 1
    path, payload = bundle(tmp_path, [report])
    payload["reports"][0]["decision"]["score"] = True
    path.write_text(json.dumps(payload))
    with closing(iter_report_bundle(path)) as reports, pytest.raises(ValueError, match="SUMMARY_MISMATCH"):
        next(reports)


def test_binding_equality_does_not_coerce_types(tmp_path):
    report = native()
    report["parameter_version"] = 1
    _, payload = bundle(tmp_path, [report])
    payload["reports"][0]["_report_reference"]["parameter_version"] = True
    with ReportEvidenceReader(tmp_path) as reader, pytest.raises(ValueError, match="BINDING_MISMATCH"):
        next(reader.iter_reports(payload["reports"]))


def test_decompression_limit_is_enforced_before_delivery(tmp_path):
    from infrastructure.storage.evidence_graph import MAX_NODE_BYTES
    path, payload = bundle(tmp_path, [native()])
    encoded = canonical_json(["dict", [["text", ["value", "x" * MAX_NODE_BYTES]]]]).encode()
    sha = hashlib.sha256(encoded).hexdigest()
    with sqlite3.connect(tmp_path / STORE_NAME) as connection:
        connection.execute("INSERT INTO report_objects VALUES (?,?)", (sha, zlib.compress(encoded)))
    payload["reports"][0]["_report_reference"]["root"] = ["ref", sha]
    path.write_text(json.dumps(payload))
    with closing(iter_report_bundle(path)) as reports, pytest.raises(ValueError, match="HASH_MISMATCH"):
        next(reports)


def test_bounded_lookahead_coalesces_narrow_nested_proof_branches(tmp_path):
    def branch(prefix, depth):
        if not depth:
            return dict(proof=prefix + "SYNTHETIC_ONLY" * 80)
        return {str(i): branch(f"{prefix}-{i}", depth - 1) for i in range(3)}
    report = native(count=1)
    report["quality"]["metrics"]["quality_field_checks"] = branch("proof", 4)
    _, payload = bundle(tmp_path, [report])
    counts = []
    for size in (1, 128):
        with ReportEvidenceReader(tmp_path, batch_nodes=size) as reader:
            reader.check_store(STORE_NAME)
            statements = []
            reader._connections[STORE_NAME].set_trace_callback(statements.append)
            assert reader.resolve(payload["reports"][0]) == report
            counts.append(sum(s.startswith("SELECT") and "report_objects" in s for s in statements))
    assert counts[1] * 5 < counts[0]


def test_legacy_cli_summary_does_not_claim_independent_hash_verification(tmp_path):
    from examples.benchmark_daily_storage import verify_report_stream
    path, _ = bundle(tmp_path, [native()], legacy=True)
    result = verify_report_stream(path, require_reference=False)
    assert result["reports_processed"] == 1
    assert result["reports_verified"] is None
    assert result["status"] == "LEGACY_PARSE_COMPLETE_NOT_HASH_VERIFIED"


def test_public_context_keeps_bucket_and_late_metadata_without_report_contents(tmp_path):
    from infrastructure.storage.report_evidence import ReportBundleContext
    path, payload = bundle(tmp_path, [native()], stale=[native("STALE")])
    # Bucket order is arbitrary and envelope metadata may follow both arrays.
    path.write_text(json.dumps(dict(stale_reports=payload["stale_reports"], reports=payload["reports"],
                                    signal_date="2026-09-18", storage_schema="daily-reports/v2")))
    context = ReportBundleContext()
    with closing(iter_report_bundle(path, context=context, chunk_bytes=7)) as reports:
        report = next(reports)
        assert context.bucket == "stale_reports" and context.referenced is True
        assert not context.complete and context.metadata == {}
        del report
        report = next(reports)
        assert context.bucket == "reports" and context.records == context.referenced_records == 2
        del report
        assert not context.complete
        with pytest.raises(StopIteration):
            next(reports)
    assert context.complete and context.metadata == dict(signal_date="2026-09-18", storage_schema="daily-reports/v2")
    with pytest.raises(ValueError, match="CONTEXT"):
        next(iter_report_bundle(path, context=context))


@pytest.mark.parametrize("ending", [" false", ',"signal_date":{}'])
def test_context_never_marks_invalid_or_interrupted_envelope_complete(tmp_path, ending):
    from infrastructure.storage.report_evidence import ReportBundleContext
    path, _ = bundle(tmp_path, [native()])
    text = path.read_text()
    path.write_text(text + ending if ending.startswith(" ") else text[:-1] + ending + "}")
    context = ReportBundleContext()
    with closing(iter_report_bundle(path, context=context)) as reports:
        report = next(reports)
        del report
        with pytest.raises(ValueError):
            next(reports)
    assert context.records == 1 and not context.complete
    path, _ = bundle(tmp_path, [native()])
    context = ReportBundleContext()
    with closing(iter_report_bundle(path, context=context)) as reports:
        report = next(reports)
        del report
    assert not context.complete


def test_context_distinguishes_legacy_parse_from_hash_verification(tmp_path):
    from infrastructure.storage.report_evidence import ReportBundleContext
    path, _ = bundle(tmp_path, [native()], stale=[native("STALE")], legacy=True)
    context = ReportBundleContext()
    with closing(iter_report_bundle(path, context=context, require_reference=False)) as reports:
        for report in reports:
            del report
    assert context.complete and context.records == 2 and context.referenced_records == 1

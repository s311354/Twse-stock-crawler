"""Synthetic-only benchmark plumbing; never a market-performance assertion."""
from argparse import Namespace
from copy import deepcopy
import json

import pytest

from application.replay_contracts import digest
from examples.benchmark_daily_preprocessing import (
    PARITY_FIELDS, compare_parity, file_hash, freeze_fixture, synthetic_fixture, worker,
)
from infrastructure.fundamentals.mops_repository import MopsRepository


def arguments(tmp_path, *, mode="reference"):
    fixture = tmp_path / "fixture"
    synthetic_fixture(fixture, symbols=["SYNTHETIC"], cutoff="2026-10-16T23:00:00+08:00")
    return Namespace(fixture=fixture, storage=tmp_path / "stores", workspace=tmp_path / "workspace",
                     mode=mode, rounds=2)


def test_complete_quality_proofs_and_storage_match_with_restart_warm_cache(tmp_path):
    args = arguments(tmp_path)
    database = args.fixture / "mops.sqlite3"
    baseline = database.stat().st_mtime_ns, file_hash(database)
    reference = worker(args)
    args.mode = "optimized"
    optimized = worker(args)
    assert (database.stat().st_mtime_ns, file_hash(database)) == baseline
    compare_parity([reference, optimized])
    for row in reference["measurements"] + optimized["measurements"]:
        assert row["quality_gates"]["SYNTHETIC"]["status"] != "PASS"
    warm = optimized["measurements"][1]["counters"]
    assert warm.get("raw_parse_disk_hits", 0) > 0
    assert warm.get("financial_companion_disk_hits", 0) > 0
    assert warm.get("raw_unit_parse_calls", 0) == 0
    assert warm.get("financial_companion_builds", 0) == 0
    assert warm.get("evidence_nodes_written", 0) == 0
    assert reference["measurements"][1]["counters"]["raw_unit_parse_calls"] > 0


def test_compressed_and_uncompressed_cache_keep_complete_financial_and_proof_identity(tmp_path):
    args = arguments(tmp_path, mode="uncompressed")
    args.parse_cache_bytes = 4 * 1024 * 1024
    reference = worker(args)
    args.mode = "optimized"
    optimized = worker(args)
    compare_parity([reference, optimized])
    for result in (reference, optimized):
        warm = result["measurements"][1]
        assert warm["counters"].get("raw_unit_parse_calls", 0) == 0
        assert warm["quality_gates"]["SYNTHETIC"]["status"] != "PASS"
        assert warm["parse_cache"]["max_bytes"] == args.parse_cache_bytes


@pytest.mark.parametrize("field", PARITY_FIELDS)
def test_parity_does_not_ignore_any_business_or_proof_identity(field):
    values = {f: "a" for f in PARITY_FIELDS}
    original = dict(measurements=[values])
    changed = deepcopy(original)
    changed["measurements"][0][field] = "b"
    with pytest.raises(ValueError, match="FULL_PARITY_FAILED:" + field):
        compare_parity([original, changed])


def test_tampered_fixed_manifest_is_rejected_before_work(tmp_path):
    args = arguments(tmp_path)
    path = args.fixture / "fixture.json"
    body = json.loads(path.read_bytes())
    body["cutoff"] = "2027-01-01T23:00:00+08:00"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="FIXTURE_MANIFEST_CHANGED"):
        worker(args)


def test_freeze_uses_read_only_metadata_and_retains_real_missing_original_gap(tmp_path):
    original = tmp_path / "source"
    synthetic_fixture(original, symbols=["SYNTHETIC"], cutoff="2026-10-16T23:00:00+08:00")
    repository = MopsRepository(original)
    filing = repository.load_filings_many(["SYNTHETIC"], as_of="2026-10-16T23:00:00+08:00")["SYNTHETIC"][0]
    # Intentional fixture damage: no synthetic replacement may hide the gap.
    (original / filing.raw_path).unlink()
    database = original / "mops.sqlite3"
    stamp = database.stat().st_mtime_ns
    manifest = freeze_fixture(database, tmp_path / "frozen", symbols=["SYNTHETIC"],
                              cutoff="2026-10-16T23:00:00+08:00", period_count=1)
    assert database.stat().st_mtime_ns == stamp
    assert manifest["gaps"] == [dict(symbol="SYNTHETIC", content_sha256=filing.content_sha256,
                                     reason="LOCAL_ORIGINAL_MISSING_NOT_REPAIRED")]
    assert not manifest["inputs"][0]["original_present"]
    expected = manifest.pop("input_id")
    assert digest(manifest) == expected


def test_freeze_does_not_copy_financial_metadata_after_exact_cutoff(tmp_path):
    source = tmp_path / "source"
    synthetic_fixture(source, symbols=["SYNTHETIC"], cutoff="2026-10-16T23:00:00+08:00")
    with pytest.raises(ValueError, match="REQUIRES_1_TO_64"):
        freeze_fixture(source / "mops.sqlite3", tmp_path / "frozen", symbols=["SYNTHETIC"],
                       cutoff="2026-08-10T04:59:59.999999+00:00", period_count=1)

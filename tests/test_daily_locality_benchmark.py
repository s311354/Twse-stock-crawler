"""Synthetic bounded benchmark asserts equivalence/counters, not wall speed."""
from argparse import Namespace

from examples.benchmark_daily_locality import worker


def test_locality_benchmark_preserves_all_fields_and_nodes_on_retry():
    args = Namespace(mops_cache=None, symbols=["SYNTHETIC"], rounds=2,
                     cutoff="2026-10-07T05:30:00+00:00", mode="reference")
    reference = worker(args)
    args.mode = "optimized"
    optimized = worker(args)
    for field in ("full_business_sha256", "full_reports_sha256", "node_bytes_sha256", "references_sha256"):
        assert len({row[field] for output in (reference, optimized) for row in output["measurements"]}) == 1
    assert optimized["measurements"][1]["counters"].get("evidence_nodes_written", 0) == 0
    assert optimized["measurements"][1]["counters"]["evidence_batches"] < reference["measurements"][1]["counters"]["evidence_batches"]
    assert optimized["measurements"][1]["counters"]["evidence_lookup_keys"] == reference["measurements"][1]["counters"]["evidence_lookup_keys"]

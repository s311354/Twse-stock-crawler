"""Synthetic benchmark plumbing; no official evidence or performance claim."""
from argparse import Namespace

from examples.benchmark_daily_hotspots import worker


def test_benchmark_compares_complete_results_and_retries():
    args = Namespace(mops_cache=None, symbols=["SYNTHETIC"], rounds=2, mode="reference", cutoff="2026-10-07T05:30:00+00:00")
    reference = worker(args)
    args.mode = "optimized"
    result = worker(args)
    for field in ("full_business_sha256", "full_reports_sha256", "node_bytes_sha256", "retained_bytes"):
        assert len({row[field] for output in (reference, result) for row in output["measurements"]}) == 1
    assert result["measurements"][1]["counters"].get("evidence_nodes_written", 0) == 0
    assert result["measurements"][0]["counters"]["evidence_node_encodes"] < reference["measurements"][0]["counters"]["evidence_node_encodes"]

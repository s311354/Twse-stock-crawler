"""Native comparison contract, not a backtest or synthetic historical repair."""
import pytest

from examples.compare_les_r20 import compare_report
from tests.test_v7_r14_comparison import fixture


def test_common_sources_account_costs_and_complete_differences_are_retained():
    saved, filings, _ = fixture()
    old, new, row = compare_report(saved, filings, evaluation_cutoff="2026-10-28T02:30:00Z")
    assert row["state"] == "FULL_NATIVE_RECOMPUTED"
    assert row["account_sha256"] == old["pae_authorization"]["account_sha256"] == new["pae_authorization"]["account_sha256"]
    assert row["full_report_differences"]
    assert row["r20"]["gates"] == new["diagnostics"]["gate_receipts"]
    assert not new["pae_authorization"]["live_order_authorized"]
    with pytest.raises(ValueError, match="SAVED_FINANCIAL_VECTOR_HASH_REQUIRED"):
        compare_report(saved, [], evaluation_cutoff="2026-10-28T02:30:00Z")

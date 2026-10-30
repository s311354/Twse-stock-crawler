"""SYNTHETIC exact-source audit tests; never formal market evidence."""
from copy import deepcopy

import pytest

from application.replay_contracts import digest
from infrastructure.les_v7_fr2_evidence import Fr2HistoricalSources, retain_source_audit
from infrastructure.replay_evidence import save_evidence
from tests.test_filing_publication_repair import reviewed
from tests.test_research_archive import raw_fixture, add_raw


def sources(tmp_path, data=None):
    data = data or raw_fixture()
    if not data.get("filings"):
        add_raw(data, proof_present=False)
    path = tmp_path / "evidence.json"
    save_evidence(data, path)
    return Fr2HistoricalSources(path), data


def row(report, symbol="2330"):
    return next(stock for stock in report["stocks"] if stock["symbol"] == symbol)


def test_late_observation_cannot_backfill_or_clear_native_gates(tmp_path):
    source, _ = sources(tmp_path)
    report = source.select(symbols=["2330"], cutoff="2026-08-04T13:30:00+08:00")
    stock = row(report)
    assert stock["filings"][0]["status"] == "PUBLICATION_PROOF_REQUIRED"
    assert not stock["selected_filing_ids"]
    assert stock["historical_market_publication"]["status"] == "UNKNOWN"
    assert stock["native_quality_valuation"]["status"] == "NOT_EVALUATED"
    assert "NOT_NATIVE_GATE_CLEARANCE" in report["scope"]
    assert report["window"]["calendar_days"] == 700


def test_exact_original_proof_allows_late_collection_but_date_only_next_midnight(tmp_path):
    data = raw_fixture()
    item = add_raw(data, proof_present=False)
    item["publication_proof"] = reviewed(data, item, "2026-08-04")
    source, _ = sources(tmp_path, data)
    before = source.select(symbols=["2330"], cutoff="2026-08-04T13:30:00+08:00")
    after = source.select(symbols=["2330"], cutoff="2026-08-05T00:00:00+08:00")
    assert not row(before)["selected_filing_ids"]
    assert row(after)["selected_filing_ids"] == [item["content_sha256"]]
    assert row(after)["filings"][0]["fetched_at"].startswith("2026-09-20")
    assert before["verified_original_documents"] == after["verified_original_documents"] == 1


def test_future_revision_or_changed_proof_never_borrows_original_date(tmp_path):
    data = raw_fixture()
    item = add_raw(data, proof_present=False)
    item["publication_proof"] = reviewed(data, item, "2026-08-01")
    item["publication_proof"]["binding"]["content_sha256"] = "a" * 64
    source, _ = sources(tmp_path, data)
    report = source.select(symbols=["2330"], cutoff="2026-08-05T00:00:00+08:00")
    assert row(report)["filings"][0]["status"] == "SOURCE_OR_VERSION_UNVERIFIED"
    assert not row(report)["selected_filing_ids"]


def test_no_calendar_proof_is_unknown_and_no_quotes_do_not_imply_closed(tmp_path):
    data = raw_fixture()
    data["calendar"].pop("documents")
    source, _ = sources(tmp_path, data)
    report = source.select(symbols=["2330"], cutoff="2026-08-04T13:30:00+08:00")
    assert report["calendar"]["status"] == "UNKNOWN"
    assert report["calendar"]["day_status_counts"] == {"UNKNOWN": 700}
    assert row(report)["historical_market_publication"]["status"] == "UNKNOWN"


def test_empty_restrictions_are_not_clear_and_future_observation_is_excluded(tmp_path):
    source, _ = sources(tmp_path)
    report = source.select(symbols=["2330"], cutoff="2026-08-04T13:30:00+08:00")
    restriction = row(report)["trading_restrictions"]
    assert restriction["check"]["status"] == "UNKNOWN"
    assert restriction["excluded"] and restriction["excluded"][0]["reason"] == "AFTER_CUTOFF"


def test_fixed_scope_includes_missing_stock_and_no_shared_result_mutation(tmp_path):
    source, _ = sources(tmp_path)
    args = dict(symbols=["2330", "MISSING"], cutoff="2026-08-04T13:30:00+08:00")
    first = source.select(**args)
    pristine = deepcopy(first)
    assert row(first, "MISSING")["in_dataset_universe"] is False
    assert row(first, "MISSING")["observed_bar_day_count"] == 0
    first["stocks"][0]["selected_filing_ids"].append("MUTATED")
    assert source.select(**args) == pristine
    assert pristine["audit_id"] == digest({k: v for k, v in pristine.items() if k != "audit_id"})


def test_input_identity_change_and_ambiguous_scope_rejected(tmp_path):
    source, _ = sources(tmp_path)
    with pytest.raises(ValueError, match="FIXED_UNIVERSE"):
        source.select(symbols=["2330", "2330"], cutoff="2026-08-04T13:30:00+08:00")
    source.path.write_text(source.path.read_text() + " ")
    with pytest.raises(ValueError, match="DATASET_CHANGED"):
        source.select(symbols=["2330"], cutoff="2026-08-04T13:30:00+08:00")


def test_series_publication_idempotent_and_last_receipt_does_not_claim_clearance(tmp_path):
    source, _ = sources(tmp_path)
    args = dict(symbols=["2330"], cutoffs=["2026-08-04T13:30:00+08:00", "2026-08-05T13:30:00+08:00"],
        output_root=tmp_path / "audit")
    folder, report = retain_source_audit(source.path, **args)
    first = (folder / "completion.json").read_bytes()
    same, repeated = retain_source_audit(source.path, **args)
    assert same == folder and repeated == report
    assert (folder / "completion.json").read_bytes() == first
    assert report["status"] == "PARTIAL"
    assert len(list((folder / "cutoffs").glob("*.json"))) == 2


def test_warm_scope_cache_does_not_change_business_audit_identity(tmp_path):
    source, _ = sources(tmp_path)
    args = dict(symbols=["MISSING"], cutoff="2026-08-04T13:30:00+08:00")
    before = source.select(**args)
    source.select(symbols=["2330"], cutoff=args["cutoff"])
    assert source.select(**args) == before

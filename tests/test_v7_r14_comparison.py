"""SYNTHETIC ONLY: full native fixed-source comparison without live effects."""
from copy import deepcopy
from datetime import timedelta
import json
import sqlite3

import pytest

from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.policy_r14 import AVAILABLE_AT, BASELINE_VERSION
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import RELAXED_BASELINE_POLICY
from examples.compare_les_r14 import (
    HashMatchedFilings, _account, assert_snapshot_visibility, business, compare_report,
    local_report_copy, markdown, prepare_rows, reselect_packet,
    valuation_business,
)
from tests.test_v7_r13_pipeline import market, native


def fixture():
    value = market()
    report = native(value)
    snapshot = value.fundamental_snapshots["2330"]
    selected = prepare_rows(deepcopy(snapshot["les_v6_mops_filings"]), symbol="2330", market="TWSE",
        cutoff=report["envelope"]["event"]["cutoff_at"],
        quality_policy=report["diagnostics"]["rule_baseline"]["policy"]["quality_policy"],
        companions=snapshot.get("les_v7_context", {}))
    assert identity(selected) == report["diagnostics"]["evidence_binding"]["financials_hash"]
    return report, selected, snapshot


def at():
    return (timestamp(AVAILABLE_AT) + timedelta(seconds=1)).isoformat()


def test_full_native_comparison_recomputes_both_versions_and_retains_all_business_fields():
    saved, filings, _ = fixture()
    before = deepcopy(saved)
    old, new, result = compare_report(saved, filings, evaluation_cutoff=at())
    for report in old, new:
        validate_report(report)
        assert report["pae_authorization"]["live_order_authorized"] is False
    assert saved == before
    assert old["diagnostics"]["rule_baseline"]["policy"]["policy_version"] == RELAXED_BASELINE_POLICY
    assert new["diagnostics"]["rule_baseline"]["policy"]["policy_version"] == BASELINE_VERSION
    assert result["state"] == "FULL_NATIVE_RECOMPUTED"
    assert result["r13"] == business(old) and result["r14"] == business(new)
    assert result["account_inputs_equal"] is True
    assert old["diagnostics"]["quality"] == new["diagnostics"]["quality"]
    assert old["diagnostics"]["native_evidence"]["source_packet"] == new["diagnostics"]["native_evidence"]["source_packet"]
    assert result["source_snapshot_cutoff"] != result["evaluation_cutoff"]
    assert result["differences"]
    assert any("reference_plan" in row["path"] for row in result["differences"])


def test_financial_hash_mismatch_and_pre_release_cutoff_are_rejected():
    saved, filings, _ = fixture()
    with pytest.raises(ValueError, match="BEFORE_POLICY_RELEASE"):
        compare_report(saved, filings, evaluation_cutoff=(timestamp(AVAILABLE_AT) - timedelta(seconds=1)).isoformat())
    with pytest.raises(ValueError, match="FINANCIAL_VECTOR_HASH_REQUIRED"):
        compare_report(saved, filings[:-1], evaluation_cutoff=at())


def test_clock_reselection_preserves_source_originals_and_availability():
    saved, _, _ = fixture()
    packet = saved["diagnostics"]["native_evidence"]["source_packet"]
    before = deepcopy(packet)
    result = reselect_packet(packet, evaluation_cutoff=at())
    assert packet == before
    assert result["cutoff"] == at()
    assert result["market_rows"] == packet["market_rows"]
    assert result["valuation_filings"] == packet["valuation_filings"]
    assert result["restriction_evidence"] == packet["restriction_evidence"]
    assert result["security_evidence"]["master"] == packet["security_evidence"]["master"]
    assert result["security_evidence"]["selection_id"] != packet["security_evidence"]["selection_id"]
    old_issuer = packet["r10_valuation_context"]["issuer"]
    new_issuer = result["r10_valuation_context"]["issuer"]
    assert {k: v for k, v in old_issuer.items() if k != "security_evidence"} == {
        k: v for k, v in new_issuer.items() if k != "security_evidence"}
    with pytest.raises(ValueError, match="CANNOT_PRECEDE_SOURCE"):
        reselect_packet(packet, evaluation_cutoff="2020-01-01T00:00:00Z")


def test_explicit_saved_account_is_required_without_fabricated_empty_positions():
    with pytest.raises(ValueError, match="EXPLICIT_SAVED_ACCOUNT_REQUIRED"):
        _account({}, at())
    with pytest.raises(ValueError, match="EXPLICIT_SAVED_ACCOUNT_REQUIRED"):
        _account(dict(value=1_000_000., available_cash=1_000_000., intended_weight=.25,
                      capacity_weight=.25, as_of=at()), at())
    saved = dict(value=1_000_000., available_cash=900_000., intended_weight=.25,
                 capacity_weight=.25, positions=[], as_of="2026-10-14T05:30:00Z")
    current = _account(saved, at())
    assert current == dict(saved, as_of=at()) and saved["as_of"] != current["as_of"]


def test_saved_daily_input_errors_remain_blocking_after_full_recalculation():
    from application.low_entry_v7_daily import finalize_daily_report
    saved, filings, _ = fixture()
    source = deepcopy(saved)
    source.pop("pae_authorization")
    account = saved["pae_authorization"]["account"]
    source = finalize_daily_report(source, account=account, daily_input_errors=["STALE_MARKET_DATA"])
    old, new, result = compare_report(source, filings, evaluation_cutoff=at())
    for report in old, new:
        assert report["envelope"]["score"]["final_score"] is None
        assert "STALE_MARKET_DATA" in report["envelope"]["strategy"]["reason_codes"]
        assert report["pae_authorization"]["authorized"] is False
    assert result["r14"]["daily_input_errors"] == ["STALE_MARKET_DATA"]


def test_readonly_snapshot_candidates_must_match_hash_not_latest(tmp_path, monkeypatch):
    saved, expected, snapshot = fixture()
    original = deepcopy(snapshot["les_v6_mops_filings"])
    wrong = deepcopy(original)
    wrong[0]["unapproved_field"] = "SYNTHETIC_UNVERIFIED"
    path = tmp_path / "mops.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE fundamental_snapshots(symbol,filing_set_sha256,sector,created_at_epoch,snapshot_json)")
        for key, created, rows in (("old_exact", 1, original), ("new_wrong", 2, wrong)):
            connection.execute("INSERT INTO fundamental_snapshots VALUES(?,?,?,?,?)",
                ("2330", key, "24", created, json.dumps(dict(les_v6_mops_filings=rows))))
    from application.fundamental_service import MopsFundamentalService
    monkeypatch.setattr(MopsFundamentalService, "load_v7_financial_sources",
                        lambda self, *args, **kwargs: snapshot.get("les_v7_context", {}))
    restorer = HashMatchedFilings(path, ["2330"])
    try:
        selected, provenance = restorer.restore(saved)
        assert identity(selected) == identity(expected) and provenance["snapshot_key"] == "old_exact"
        assert restorer.counts["snapshot_candidates"] == 2
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            restorer.connection.execute("DELETE FROM fundamental_snapshots")
    finally:
        restorer.close()


def test_local_sqlite_backup_keeps_bundle_identity_and_cannot_overwrite(tmp_path):
    from examples.compare_les_r13 import file_hash
    source = tmp_path / "source"
    source.mkdir()
    bundle = source / "reports.json"
    bundle.write_text('{"reports":[],"synthetic_only":true}')
    with sqlite3.connect(source / "report_evidence.sqlite3") as connection:
        connection.execute("CREATE TABLE report_objects(sha256 TEXT PRIMARY KEY,body BLOB)")
        connection.execute("INSERT INTO report_objects VALUES('SYNTHETIC_ONLY',?)", (b"fixture",))
    copied, receipt_path = local_report_copy(bundle, tmp_path / "local")
    receipt = json.loads(receipt_path.read_text())
    assert copied.read_bytes() == bundle.read_bytes()
    assert receipt["bundle_sha256"] == file_hash(bundle)
    assert receipt["method"] == "SQLITE_BACKUP_READ_ONLY_SOURCE"
    assert receipt["original_store_before"] == receipt["original_store_after"]
    assert receipt["full_report_validation_still_required"] is True
    with pytest.raises(ValueError, match="REQUIRES_NEW_DIRECTORY"):
        local_report_copy(bundle, tmp_path / "local")


@pytest.mark.parametrize("location", ["valuation", "recovery", "peer_annual", "month_observation", "quote",
                                      "official_pe", "official_pe_proof"])
def test_later_evaluation_cannot_admit_future_operands_from_saved_packet(location):
    future = "2026-10-16T03:00:00Z"
    packet = dict(cutoff="2026-10-16T01:28:30Z", market_rows=[],
        valuation_filings=[], filings=[], market_evidence={},
        r10_valuation_context=dict(issuer=dict(annual_filings=[], market_rows=[], market_evidence={}), peers=[]))
    if location == "valuation": packet["valuation_filings"].append(dict(available_at=future))
    elif location == "recovery": packet["filings"].append(dict(available_at=future))
    elif location == "peer_annual": packet["r10_valuation_context"]["peers"].append(dict(annual_filings=[dict(available_at=future)]))
    elif location == "month_observation": packet["market_evidence"]["valuation_history_selection"] = dict(observed_at=future)
    elif location == "official_pe": packet["official_pe"] = [dict(source_available_at=future)]
    elif location == "official_pe_proof": packet["official_pe"] = [dict(selection_proofs=[dict(manifest=dict(available_at=future))])]
    else: packet["market_rows"].append(dict(Date="2026-10-16", **{"Available At": future}))
    with pytest.raises(ValueError, match="POST_CUTOFF_SOURCE"):
        assert_snapshot_visibility(packet)


def test_visibility_check_does_not_treat_later_policy_metadata_as_new_raw_data():
    packet = dict(cutoff="2026-10-16T01:28:30Z", market_rows=[], valuation_filings=[], filings=[],
                  policy=dict(available_at="2026-10-16T06:08:02Z"))
    assert assert_snapshot_visibility(packet)["post_cutoff_source_count"] == 0


def test_human_summary_keeps_gaps_and_distinguishes_decisions_from_fills():
    result = dict(evaluation_cutoff=at(), source_bundle_sha256="SYNTHETIC_ONLY",
        summary=dict(total=3, states=dict(FULL_NATIVE_RECOMPUTED=2, SOURCE_OR_RECOMPUTATION_GAP=1),
                     r13_decisions=dict(NO_BUY=2), r14_decisions=dict(NO_BUY=1, BUY_SMALL=1)))
    rendered = markdown(result)
    assert "缺口：1 檔" in rendered
    assert "| BUY_SMALL | 0 | 1 |" in rendered
    assert "不是歷史當時部署、獲利或成交證明" in rendered
    assert "未模擬 EREM、完整 PAE Persistence" in rendered


def hinted_restorer(tmp_path, monkeypatch, *, declared, snapshot_rows=None):
    """SYNTHETIC identity-selection test, not financial proof certification."""
    from application.fundamental_service import MopsFundamentalService
    import examples.compare_les_r14 as comparison
    rows = [dict(symbol="2330", market="TWSE", synthetic_only=True, value=123.)]
    path = tmp_path / "mops.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE fundamental_snapshots(symbol,filing_set_sha256,sector,snapshot_json,"
                           "PRIMARY KEY(symbol,filing_set_sha256,sector))")
        connection.execute("INSERT INTO fundamental_snapshots VALUES(?,?,?,?)",
                           ("2330", "synthetic-exact", "24", json.dumps(dict(
                               les_v6_mops_filings=rows if snapshot_rows is None else snapshot_rows))))
    calls = []
    def companions(self, source_symbol, **options):
        calls.append(("companions", source_symbol, options["as_of"]))
        return {}
    def prepare(records, *, symbol, market, cutoff, **options):
        calls.append(("prepare", symbol, cutoff))
        assert {row["symbol"] for row in records} == {symbol}
        return deepcopy(records)
    monkeypatch.setattr(MopsFundamentalService, "load_v7_financial_sources", companions)
    monkeypatch.setattr(comparison, "prepare_rows", prepare)
    saved = dict(symbol="2330A", diagnostics=dict(
        evidence_binding=dict(financials_hash=identity(rows), cutoff="2026-10-16T01:28:30Z", market="TWSE"),
        rule_baseline=dict(policy=dict(quality_policy="SYNTHETIC_ONLY")),
        native_evidence=dict(source_packet=dict(filings=declared,
            security_evidence=dict(status="OUT_OF_SCOPE", security_type="PREFERRED", issuer_symbol=None)))))
    return HashMatchedFilings(path, ["2330A"]), saved, rows, calls


def test_explicit_original_filing_symbol_hint_recovers_exact_vector_without_security_reassignment(tmp_path, monkeypatch):
    restorer, saved, rows, calls = hinted_restorer(tmp_path, monkeypatch, declared=[dict(symbol="2330")])
    before = deepcopy(saved)
    try:
        selected, provenance = restorer.restore(saved)
        assert selected == rows and identity(selected) == saved["diagnostics"]["evidence_binding"]["financials_hash"]
        assert calls == [("companions", "2330", "2026-10-16T01:28:30Z"),
                         ("prepare", "2330", "2026-10-16T01:28:30Z")]
        assert provenance["report_symbol"] == "2330A" and provenance["restored_source_symbol"] == "2330"
        assert provenance["source_identity_hint_basis"] == "SAVED_PACKET_EXPLICIT_FILING_SYMBOL"
        assert provenance["security_identity_not_reassigned"] is True
        assert saved == before and saved["diagnostics"]["native_evidence"]["source_packet"]["security_evidence"]["issuer_symbol"] is None
    finally:
        restorer.close()


def test_snapshot_lookup_does_not_infer_filing_identity_from_instrument_suffix(tmp_path, monkeypatch):
    restorer, saved, _, calls = hinted_restorer(tmp_path, monkeypatch, declared=[])
    try:
        with pytest.raises(ValueError, match="SAVED_FINANCIAL_VECTOR_NOT_RECOVERED"):
            restorer.restore(saved)
        assert calls == [] and "2330" not in restorer.loaded_symbols
    finally:
        restorer.close()


@pytest.mark.parametrize("declared,reason", [
    ([dict(symbol="2330"), dict(symbol="9999")], "FILING_IDENTITY_CONFLICT"),
    ([dict(synthetic_only=True)], "FILING_IDENTITY_UNVERIFIED"),
    ([dict(symbol=None)], "FILING_IDENTITY_UNVERIFIED"),
])
def test_conflicting_or_unknown_explicit_filing_hint_cannot_select_another_issuer(tmp_path, monkeypatch, declared, reason):
    restorer, saved, _, calls = hinted_restorer(tmp_path, monkeypatch, declared=declared)
    try:
        with pytest.raises(ValueError, match=reason):
            restorer.restore(saved)
        assert calls == []
    finally:
        restorer.close()


def test_explicit_filing_hint_cannot_accept_a_different_financial_vector(tmp_path, monkeypatch):
    wrong = [dict(symbol="2330", market="TWSE", synthetic_only=True, value=999.)]
    restorer, saved, _, _ = hinted_restorer(tmp_path, monkeypatch,
        declared=[dict(symbol="2330")], snapshot_rows=wrong)
    try:
        with pytest.raises(ValueError, match="FINANCIAL_VECTOR_HASH_MISMATCH"):
            restorer.restore(saved)
        assert restorer.counts["exact_matches"] == 0
    finally:
        restorer.close()


def test_valuation_and_recovery_filing_identity_hints_must_agree(tmp_path, monkeypatch):
    restorer, saved, _, calls = hinted_restorer(tmp_path, monkeypatch, declared=[dict(symbol="2330")])
    saved["diagnostics"]["native_evidence"]["source_packet"]["valuation_filings"] = [dict(symbol="9999")]
    try:
        with pytest.raises(ValueError, match="FILING_IDENTITY_CONFLICT"):
            restorer.restore(saved)
        assert calls == []
    finally:
        restorer.close()


def test_comparison_keeps_valuation_arithmetic_but_not_duplicate_raw_source_graphs():
    raw = dict(status="UNKNOWN", reasons=["SYNTHETIC_PRESENT_CONFLICT"], inputs=dict(current_price=100.),
        selected_method="SYNTHETIC_ANNUAL", scenarios=None, gate_checks=dict(SYNTHETIC_CHECK=False),
        comparables=dict(status="UNKNOWN", reasons=["SYNTHETIC_PRESENT_CONFLICT"],
            issuer_history=dict(status="INSUFFICIENT", count=1, minimum=24, median=10.,
                                sources=[dict(synthetic_raw_original="not repeated")]),
            peers=dict(status="INSUFFICIENT", count=0, minimum=10, median=None, sources=[])),
        recovery=dict(status="NO_RECOVERY", reasons=["SYNTHETIC_KNOWN_FAILED_CONDITION"],
                      values=dict(current_cfo=-1.), raw_original=dict(synthetic_only=True)))
    d = dict(native_evidence=dict(produced_receipts=dict(VALUATION=dict(valuation_diagnostics=raw))))
    result = valuation_business(d)
    assert result["inputs"] == raw["inputs"] and result["gate_checks"] == raw["gate_checks"]
    assert result["status"] == "UNKNOWN" and result["scenarios"] is None
    assert result["comparables"]["issuer_history"]["count"] == 1
    assert "sources" not in result["comparables"]["issuer_history"]
    assert result["recovery"]["values"]["current_cfo"] == -1.
    assert "raw_original" not in result["recovery"]
    assert valuation_business(dict(native_evidence=dict(produced_receipts=dict(VALUATION=None)))) is None

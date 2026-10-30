"""SYNTHETIC ONLY: retained source bindings, not official repaired EPS or BUYs."""
from contextlib import contextmanager
from copy import deepcopy
import json

import pytest

from application.analysis_dataset import AnalysisDatasetBuilder
from application.valuation_peers_r10 import _independent_prices
from domain.les_v7.contracts import identity
from infrastructure.crawler.valuation_history import (
    ValuationSelectionVerifier, project_valuation_selections, retain_valuation_history,
)
from infrastructure.storage.report_evidence import (
    ReportEvidenceWriter, STORE_NAME, iter_report_bundle,
    load_report_bundle, resolve_csv_report,
)
from infrastructure.storage.valuation_source_verification import verify_report_valuation_sources
from tests.test_v7_valuation_history import collect, database, OBSERVED


def retained(tmp_path):
    capture = collect(database(tmp_path / "synthetic.sqlite3"))
    root = tmp_path / "sources"
    manifest = retain_valuation_history(capture, root)
    return project_valuation_selections(capture, manifest=manifest)


def source_report(selection):
    rows = [*deepcopy(selection["rows"]), dict(Date="2026-07-14", Close=1234.5)]
    issuer = dict(symbol=selection["symbol"], market_rows=rows,
        market_evidence=dict(source_final=True, source_available_at=OBSERVED, valuation_history_selection=selection))
    return dict(symbol=selection["symbol"], signal_date="2026-07-14", model_version="7.0.0",
        parameter_version="SYNTHETIC_ONLY_SOURCE_BINDING",
        envelope=dict(event=dict(cutoff_at=OBSERVED)),
        diagnostics=dict(native_evidence=dict(source_packet=dict(
            r10_valuation_context=dict(cutoff=OBSERVED, issuer=issuer, peers=[])))))


def test_monthly_application_port_restores_original_before_using_prices(tmp_path):
    selected = retained(tmp_path)["2330"]
    rows = [dict(Date="2026-07-14", Close=1234.5)]
    source = dict(source_available_at=OBSERVED, source_final=True, source_record_ids=[identity(rows)])
    same, unknown = _independent_prices(selected, rows=rows, symbol="2330", cutoff=OBSERVED, source=source)
    assert same == rows and unknown["source_final"] is False
    assert unknown["valuation_history_verification_error"] == "ORIGINAL_SOURCE_VERIFIER_NOT_PROVIDED"
    with ValuationSelectionVerifier() as verifier:
        extended, bound = _independent_prices(selected, rows=rows, symbol="2330", cutoff=OBSERVED,
            source=source, verify_selection=verifier.verify)
        assert extended == [*selected["rows"], *rows]
        assert bound["source_final"] is True
        bad = deepcopy(selected); bad["rows"][0]["Close"] += 1.
        bad["selection_id"] = identity({k: v for k, v in bad.items() if k != "selection_id"})
        with pytest.raises(ValueError):
            _independent_prices(bad, rows=rows, symbol="2330", cutoff=OBSERVED,
                source=source, verify_selection=verifier.verify)


def test_public_source_validator_rejects_rehashed_projection_and_price_substitution(tmp_path):
    report = source_report(retained(tmp_path)["2330"])
    verify_report_valuation_sources(report)
    bad = deepcopy(report)
    block = bad["diagnostics"]["native_evidence"]["source_packet"]["r10_valuation_context"]["issuer"]
    selection = block["market_evidence"]["valuation_history_selection"]
    selection["rows"][0]["Close"] += 10.
    selection["selection_id"] = identity({k: v for k, v in selection.items() if k != "selection_id"})
    block["market_rows"][0]["Close"] = selection["rows"][0]["Close"]
    with pytest.raises(ValueError): verify_report_valuation_sources(bad)
    bad = deepcopy(report)
    block = bad["diagnostics"]["native_evidence"]["source_packet"]["r10_valuation_context"]["issuer"]
    block["market_rows"][0]["Close"] += 10.
    with pytest.raises(ValueError, match="PRICE_BINDING"):
        verify_report_valuation_sources(bad)


def test_public_bundle_reader_checks_named_monthly_original_not_only_report_hash(tmp_path):
    report = source_report(retained(tmp_path)["2330"])
    output = tmp_path / "reports"
    with ReportEvidenceWriter(output / STORE_NAME) as writer:
        ref = writer.retain(report)
    path = output / "bundle.json"
    path.write_text(json.dumps(dict(reports=[ref])), encoding="utf-8")
    assert list(iter_report_bundle(path)) == [report]
    assert load_report_bundle(path)["reports"] == [report]
    assert resolve_csv_report(json.dumps(ref), output) == report
    # Re-publish a completely hash-valid graph with a false original-source
    # claim: graph integrity alone must not make this report acceptable.
    bad = deepcopy(report)
    block = bad["diagnostics"]["native_evidence"]["source_packet"]["r10_valuation_context"]["issuer"]
    selection = block["market_evidence"]["valuation_history_selection"]
    selection["rows"][0]["Close"] += 1.
    selection["selection_id"] = identity({k: v for k, v in selection.items() if k != "selection_id"})
    with ReportEvidenceWriter(output / STORE_NAME) as writer:
        bad_ref = writer.retain(bad)
    path.write_text(json.dumps(dict(reports=[bad_ref])), encoding="utf-8")
    with pytest.raises(ValueError): list(iter_report_bundle(path))
    with pytest.raises(ValueError): load_report_bundle(path)
    with pytest.raises(ValueError): resolve_csv_report(json.dumps(bad_ref), output)


@pytest.mark.parametrize("damage", ["future_capture", "backdated_source", "context_cutoff"])
def test_public_boundaries_do_not_borrow_future_capture_or_backdate_observation(tmp_path, damage):
    report = source_report(retained(tmp_path)["2330"])
    context = report["diagnostics"]["native_evidence"]["source_packet"]["r10_valuation_context"]
    if damage == "future_capture":
        report["envelope"]["event"]["cutoff_at"] = context["cutoff"] = "2026-07-14T05:30:00Z"
    elif damage == "context_cutoff":
        context["cutoff"] = "2026-07-14T05:30:00Z"
    else:
        context["issuer"]["market_evidence"]["source_available_at"] = "2026-07-14T05:30:00Z"
    with pytest.raises(ValueError, match="CUTOFF_BINDING|CANNOT_BE_BACKDATED"):
        verify_report_valuation_sources(report)


def test_batch_source_verifier_is_closed_before_any_worker_or_after_capture_error(monkeypatch):
    seen = []
    def verify(value): return value
    @contextmanager
    def scope():
        seen.append("open")
        try: yield verify
        finally: seen.append("closed")
    def capture(market, *, verify_valuation_history):
        assert verify_valuation_history is verify
        assert seen == ["open"]
        raise ValueError("SYNTHETIC_CAPTURE_STOP")
    monkeypatch.setattr("application.valuation_peers_r10.capture_comparison_contexts", capture)
    builder = AnalysisDatasetBuilder(valuation_history_verifier_scope=scope)
    with pytest.raises(ValueError, match="SYNTHETIC_CAPTURE_STOP"):
        builder._daily_components(object())
    assert seen == ["open", "closed"]


def test_old_native_and_non_native_reports_need_no_external_monthly_files():
    for report in ({}, {"diagnostics": {"native_evidence": None}},
                   {"diagnostics": {"native_evidence": {"source_packet": {"r10_valuation_context": None}}}}):
        verify_report_valuation_sources(report)

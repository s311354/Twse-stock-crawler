"""SYNTHETIC ONLY: current-only R5 query/context, no real source or orders."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

from application.fundamental_service import MopsFundamentalService
from application.les_v7 import prepare_quality_context
from domain.les_v7.contracts import identity
from domain.les_v7.financial_sources import ixbrl_record_r2
from domain.les_v7.quality_policy import default_quality_policy, r5_quality_policy
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from tests.fundamentals.test_filing_refresh_planner import _candidate
from tests.fundamentals.test_ttm_audit import cache

CUTOFF = "2026-10-14T06:45:52.100000Z"


def read_packets(root, *, source_repair=True):
    repo = cache(root)
    headers = repo.load_filings("2330")
    periods = tuple({(h.fiscal_year, h.quarter) for h in headers})
    values = repo.load_v7_financial_sources("2330", CUTOFF, periods=periods, source_repair=source_repair)
    filings = repo.load_filed_financials("2330", CUTOFF)
    records = tuple(ixbrl_record_r2("2330", filing, original_verified=True) for filing in filings)
    return repo, headers, values, records


def test_query_rebuilds_only_latest_requested_cutoff_visible_period(tmp_path, monkeypatch):
    import domain.les_v7.financial_sources_r5 as module
    calls, original = [], module.repair_source_record_r5
    def counted(row, filing, **kwargs):
        calls.append((filing.fiscal_year, filing.quarter, filing.content_sha256))
        return original(row, filing, **kwargs)
    monkeypatch.setattr(module, "repair_source_record_r5", counted)
    repo, headers, values, records = read_packets(tmp_path / "mops")
    latest = max((h.fiscal_year, h.quarter) for h in headers)
    current = {h.content_sha256 for h in headers if (h.fiscal_year, h.quarter) == latest}
    assert set(values["financial_repairs_r5"]) == current
    assert {(year, quarter) for year, quarter, _ in calls} == {latest}
    assert len(values["financial_audits"]) > len(values["financial_repairs_r5"])
    assert set(values["financial_repairs_r5"]) <= {r["content_sha256"] for r in records}
    before = hashlib.sha256(repo.database_path.read_bytes()).hexdigest()
    legacy = repo.load_v7_financial_sources("2330", CUTOFF,
        periods=tuple({(h.fiscal_year, h.quarter) for h in headers}))
    assert "financial_repairs_r5" not in legacy
    assert values["financial_audits"] == legacy["financial_audits"]
    assert values["financial_supplements"] == legacy["financial_supplements"]
    assert before == hashlib.sha256(repo.database_path.read_bytes()).hexdigest()


def test_r5_query_excludes_future_subsecond_original_before_raw_parse(tmp_path):
    repo = cache(tmp_path / "mops")
    content = b"<html>SYNTHETIC ONLY future subsecond original</html>"
    header = repo.save_revision(replace(_candidate(2026, 3), period_end="2026-09-30"),
        content, parse_ixbrl(content), available_at="2026-10-14T06:45:52.900000Z",
        fetched_at="2026-10-14T06:45:52.900000Z")
    # Both values have the same integer index second. Full timestamp filtering
    # in the new port still excludes the later actual observation.
    values = repo.load_v7_financial_sources("2330", CUTOFF, periods=((2026, 2), (2026, 3)), source_repair=True)
    assert all(header.content_sha256 not in nodes for nodes in values.values())
    assert values["financial_repairs_r5"] == {}  # No older period substituted for requested Q3.


@contextmanager
def uow():
    yield object()


def annotated_packets(values, records, *, market="TPEX"):
    source = deepcopy(values)
    reader_calls = []
    def reader(*args, **kwargs):
        reader_calls.append((args, kwargs))
        return source
    service = object.__new__(MopsFundamentalService)
    service.repository = SimpleNamespace(load_v7_financial_sources=reader, unit_of_work=uow)
    result = service.load_v7_financial_sources("2330", as_of=CUTOFF, records=records,
                                               market=market, source_repair=True)
    return result, source, reader_calls


def test_tpex_market_enrichment_rebinds_audit_ids_without_mutating_repository_packet(tmp_path):
    _, _, values, records = read_packets(tmp_path / "mops")
    before = deepcopy(values)
    annotated, original, calls = annotated_packets(values, records)
    assert calls[0][1]["source_repair"] is True
    assert original == before
    for sha, repaired in annotated["financial_repairs_r5"].items():
        assert repaired["market"] == repaired["audit_evidence_r5"]["market"] == "TPEX"
        assert repaired["source_repair_r5"]["new_audit_proof_id"] == identity(repaired["audit_evidence_r5"])
        assert repaired["source_repair_r5"]["old_audit_proof_id"] == identity(annotated["financial_audits"][sha])
        assert values["financial_repairs_r5"][sha]["market"] == "TWSE"
    rows = tuple(dict(r, market="TPEX", audit_evidence=annotated["financial_audits"][r["content_sha256"]])
                 for r in records)
    context = dict(symbol="2330", market="TPEX", as_of=CUTOFF, quality_policy=r5_quality_policy(), filings=rows)
    prepare_quality_context(context, financial_audits=annotated["financial_audits"],
                            financial_repairs_r5=annotated["financial_repairs_r5"])
    latest = max(r["period_end"] for r in context["filings"])
    assert all(("audit_evidence_r5" in r) is (r["period_end"] == latest) for r in context["filings"])
    assert all("audit_evidence_r5" not in r for r in rows)
    assert original == before


def test_old_r4_context_does_not_apply_new_source_repair_packet(tmp_path):
    _, _, values, records = read_packets(tmp_path / "mops")
    annotated, _, _ = annotated_packets(values, records, market="TWSE")
    context = dict(symbol="2330", market="TWSE", as_of=CUTOFF,
                   quality_policy=default_quality_policy(), filings=records)
    prepare_quality_context(context, financial_audits=annotated["financial_audits"],
                            financial_repairs_r5=annotated["financial_repairs_r5"])
    assert context["quality_policy"] == default_quality_policy()
    assert context["filings"] == records
    assert all("audit_evidence_r5" not in r for r in context["filings"])


@pytest.mark.parametrize("damage", ["document", "time", "new_audit_id", "new_cash_id", "market"])
def test_context_rejects_same_key_with_mismatched_repair_provenance(tmp_path, damage):
    _, _, values, records = read_packets(tmp_path / "mops")
    annotated, _, _ = annotated_packets(values, records, market="TWSE")
    repaired = next(iter(annotated["financial_repairs_r5"].values()))
    if damage == "document": repaired["source_repair_r5"]["content_sha256"] = "f" * 64
    elif damage == "time": repaired["source_repair_r5"]["available_at"] = "2026-10-14T06:45:52.900000Z"
    elif damage == "new_audit_id": repaired["source_repair_r5"]["new_audit_proof_id"] = "f" * 64
    elif damage == "new_cash_id": repaired["source_repair_r5"]["new_cash_proof_id"] = "f" * 64
    else: repaired["audit_evidence_r5"]["market"] = "TPEX"
    context = dict(symbol="2330", market="TWSE", as_of=CUTOFF,
                   quality_policy=r5_quality_policy(), filings=records)
    with pytest.raises(ValueError, match="R5_FINANCIAL_REPAIR"):
        prepare_quality_context(context, financial_audits=annotated["financial_audits"],
                                financial_repairs_r5=annotated["financial_repairs_r5"])

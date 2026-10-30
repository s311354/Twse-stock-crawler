"""SYNTHETIC ONLY: additive R7 ports, cutoff isolation and legacy compatibility."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from application.daily_work import DailyWorkContext
from application.fundamental_service import MopsFundamentalService
from application.les_v7 import prepare_quality_context
from domain.fundamental_derivation import FundamentalBundle
from domain.les_v7.contracts import identity
from domain.les_v7.financial_audit_r7 import audit_source_r7
from domain.les_v7.financial_sources import ixbrl_record_r2
from domain.les_v7.quality_policy import native_financial_safety, r6_quality_policy, r7_quality_policy
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.runtime.fundamental_refresh import FundamentalRefreshCoordinator
from tests.fundamentals.test_filing_refresh_planner import _candidate
from tests.fundamentals.test_ttm_audit import cache
from tests.fundamentals.test_v7_audit_cash_r5 import original, QUALIFIED_BODY


CUTOFF = "2026-10-28T13:00:00Z"


def test_query_r7_is_opt_in_current_only_stable_and_read_only(tmp_path, monkeypatch):
    import domain.les_v7.financial_audit_r7 as module
    repo = cache(tmp_path / "mops")
    headers = repo.load_filings("2330")
    periods = tuple({(h.fiscal_year, h.quarter) for h in headers})
    latest = max(periods)
    current = {h.content_sha256 for h in headers if (h.fiscal_year, h.quarter) == latest}
    calls, build = [], module.audit_source_r7
    def counted(filing, **options):
        calls.append((filing.fiscal_year, filing.quarter))
        return build(filing, **options)
    monkeypatch.setattr(module, "audit_source_r7", counted)
    legacy = repo.load_v7_financial_sources("2330", CUTOFF, periods=periods)
    before = sha256(repo.database_path.read_bytes()).hexdigest()
    values = repo.load_v7_financial_sources("2330", CUTOFF, periods=periods, audit_repair_r7=True)
    assert set(values["financial_audits_r7"]) == current
    assert set(calls) == {latest}
    assert {k: v for k, v in values.items() if k != "financial_audits_r7"} == legacy
    assert "financial_audits_r7" not in legacy and "financial_repairs_r5" not in values
    later = repo.load_v7_financial_sources("2330", "2026-10-29T13:00:00Z", periods=periods,
        audit_repair_r7=True)
    assert later["financial_audits_r7"] == values["financial_audits_r7"]
    assert before == sha256(repo.database_path.read_bytes()).hexdigest()


def test_r7_query_excludes_future_subsecond_before_parsing_without_older_substitution(tmp_path):
    repo = cache(tmp_path / "mops")
    content = b"<html>SYNTHETIC ONLY future subsecond R7 original</html>"
    header = repo.save_revision(replace(_candidate(2026, 3), period_end="2026-09-30"),
        content, parse_ixbrl(content), available_at="2026-10-28T13:00:00.900000Z",
        fetched_at="2026-10-28T13:00:00.900000Z")
    values = repo.load_v7_financial_sources("2330", "2026-10-28T13:00:00.100000Z",
        periods=((2026, 2), (2026, 3)), audit_repair_r7=True)
    assert values["financial_audits_r7"] == {}
    assert all(header.content_sha256 not in nodes for nodes in values.values())


@pytest.mark.parametrize("source_repair", [False, True])
def test_real_service_and_facade_forward_r7_market_without_changing_original(tmp_path, source_repair):
    repo = cache(tmp_path / "mops")
    records = tuple(ixbrl_record_r2("2330", filing, market="TPEX", original_verified=True)
        for filing in repo.load_filed_financials("2330", CUTOFF))
    service = object.__new__(MopsFundamentalService)
    service.repository = repo
    before = deepcopy(records)
    values = service.load_v7_financial_sources("2330", as_of=CUTOFF, records=records,
        market="TPEX", source_repair=source_repair, audit_repair_r7=True)
    assert values["financial_audits_r7"]
    assert all(p["market"] == "TPEX" for p in values["financial_audits_r7"].values())
    assert records == before


@pytest.mark.parametrize("pending", [False, True])
def test_context_validates_same_original_and_keeps_genuine_pending_unknown(pending):
    row, filing = original("No independently interpretable report." if pending else
        QUALIFIED_BODY + "民國115年6月30日存貨成本如下。")
    proof = audit_source_r7(filing, symbol=row["symbol"], original_verified=True)
    before = deepcopy(row)
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF,
        filings=[row], quality_policy=r7_quality_policy())
    prepare_quality_context(context, financial_audits_r7={row["content_sha256"]: proof})
    actual = context["filings"][0]
    assert actual["audit_evidence_r7"] == proof and row == before
    safety = native_financial_safety(actual, symbol=row["symbol"], market="TWSE",
        cutoff=CUTOFF, policy=r7_quality_policy())
    assert safety["state"] == ("UNKNOWN" if pending else "PASS")
    old = dict(context, filings=[row], quality_policy=r6_quality_policy())
    prepare_quality_context(old, financial_audits_r7={row["content_sha256"]: proof})
    assert "audit_evidence_r7" not in old["filings"][0]


@pytest.mark.parametrize("with_pending_packet", [False, True])
def test_absent_audit_facts_or_packet_remain_unknown_without_aborting(with_pending_packet):
    _, filing = original()
    facts = tuple(f for f in filing.facts if not f.concept.startswith("tifrs-ar:"))
    filing = replace(filing, facts=facts, content_sha256=identity([str(f) for f in facts]))
    row = ixbrl_record_r2("SYNTHETIC_ONLY", filing, original_verified=True)
    proof = audit_source_r7(filing, symbol=row["symbol"], original_verified=True)
    assert proof["original_contexts"] == [] and proof["status"] == "PENDING_VERIFICATION"
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF,
        filings=[row], quality_policy=r7_quality_policy())
    prepare_quality_context(context,
        financial_audits_r7={row["content_sha256"]: proof} if with_pending_packet else {})
    safety = native_financial_safety(context["filings"][0], symbol=row["symbol"], market="TWSE",
        cutoff=CUTOFF, policy=r7_quality_policy())
    assert safety["state"] == "UNKNOWN"
    assert bool(context["filings"][0].get("audit_evidence_r7")) == with_pending_packet


@pytest.mark.parametrize("damage", ["market", "original_time", "classification", "source_context"])
def test_context_rejects_conflicting_or_rehashed_r7_packet(damage):
    row, filing = original()
    proof = audit_source_r7(filing, symbol=row["symbol"], original_verified=True)
    if damage == "market": proof["market"] = "TPEX"
    elif damage == "original_time": proof["available_at"] = "2026-08-01T00:00:00Z"
    elif damage == "classification":
        proof["audit"]["audit_opinion"] = "UNMODIFIED"
        proof["result_id"] = identity(proof["audit"])
    else: proof["original_contexts"][0]["context_id"] += "_DAMAGED"
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF,
        filings=[row], quality_policy=r7_quality_policy())
    with pytest.raises(ValueError, match="V7_R7_FINANCIAL_AUDIT_BINDING"):
        prepare_quality_context(context, financial_audits_r7={row["content_sha256"]: proof})


def test_signed_r6_precedence_does_not_hide_a_damaged_r7_packet():
    from domain.les_v7.signed_audit import audit_proof_r6
    from tests.fundamentals.test_signed_audit import fixture
    row, filing, packet, _ = fixture()
    signed = audit_proof_r6(filing, row, packet, symbol=row["symbol"], cutoff=CUTOFF)
    proof = audit_source_r7(filing, symbol=row["symbol"], original_verified=True)
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF,
        filings=[row], quality_policy=r7_quality_policy())
    prepare_quality_context(context, signed_financial_audits={row["content_sha256"]: signed},
        financial_audits_r7={row["content_sha256"]: proof})
    safety = native_financial_safety(context["filings"][0], symbol=row["symbol"], market="TWSE",
        cutoff=CUTOFF, policy=r7_quality_policy())
    assert safety["state"] == "PASS" and safety["signed_report_source_id"] == packet["source_id"]
    proof["original_r5_proof_id"] = "f" * 64
    with pytest.raises(ValueError, match="R7_AUDIT_ORIGINAL_REBUILD_MISMATCH"):
        prepare_quality_context(dict(context, filings=[row]),
            signed_financial_audits={row["content_sha256"]: signed},
            financial_audits_r7={row["content_sha256"]: proof})


def runtime_bundle(*, quality=None, captured=None, replay=None):
    row = dict(content_sha256="a" * 64, revision_id="a" * 64,
        available_at="2026-08-10T13:00:00Z", period_end="2026-06-30")
    bundle = FundamentalBundle(pd.DataFrame(), dict(les_v6_mops_filings=[row],
        les_v7_context={"quality_policy": quality} if quality else {}))
    settings = SimpleNamespace(daily_entry_model="v7", les_v7_bundle_json=captured,
        les_v7_rule_baseline=True, les_v7_source_repair=True)
    return bundle, SimpleNamespace(app_settings=settings, replay_as_of=replay)


def test_runtime_r7_opt_in_semantic_cache_identity_and_cutoff_selection(monkeypatch):
    import domain.les_v7.rule_baseline as baseline
    import infrastructure.fundamentals.parse_identity as parser
    monkeypatch.setattr(baseline, "current_source_baseline_version", lambda cutoff: baseline.AUDIT_SOURCE_BASELINE_POLICY)
    version = ["synthetic_parser_a"]
    monkeypatch.setattr(parser, "source_fingerprint", lambda dependencies: version[0])
    reader = Mock(return_value=dict(financial_audits={}, financial_supplements={}, financial_audits_r7={}))
    coordinator = FundamentalRefreshCoordinator(SimpleNamespace(load_v7_financial_sources=reader))
    bundle, crawler = runtime_bundle()
    work = DailyWorkContext("SYNTHETIC_ONLY")
    try:
        with work.activate():
            for _ in range(2):
                coordinator._attach_v7_sources(bundle, crawler=crawler, issuer="2330", cutoff=CUTOFF, market="TWSE")
            assert reader.call_count == 1 and reader.call_args.kwargs["audit_repair_r7"] is True
            version[0] = "synthetic_parser_b"
            coordinator._attach_v7_sources(bundle, crawler=crawler, issuer="2330", cutoff=CUTOFF, market="TWSE")
            assert reader.call_count == 2
            coordinator._attach_v7_sources(bundle, crawler=crawler, issuer="2330", cutoff="2026-10-29T13:00:00Z", market="TWSE")
            assert reader.call_count == 3
    finally:
        work.close()
    assert "financial_audits_r7" not in bundle.snapshot["les_v7_context"]


@pytest.mark.parametrize("mode", ["explicit_r6", "bundle_r6", "historical"])
def test_runtime_old_policy_and_replay_do_not_enable_r7(mode):
    quality = r6_quality_policy() if mode == "explicit_r6" else None
    captured = json.dumps({"payload": {"quality_policy": r6_quality_policy()}}) if mode == "bundle_r6" else None
    bundle, crawler = runtime_bundle(quality=quality, captured=captured,
        replay=CUTOFF if mode == "historical" else None)
    reader = Mock(return_value=dict(financial_audits={}, financial_supplements={}))
    result = FundamentalRefreshCoordinator(SimpleNamespace(load_v7_financial_sources=reader))._attach_v7_sources(
        bundle, crawler=crawler, issuer="2330", cutoff=CUTOFF, market="TWSE")
    assert "audit_repair_r7" not in reader.call_args.kwargs
    assert "financial_audits_r7" not in result.snapshot["les_v7_context"]


def test_daily_entry_routes_retained_companion_map_without_upgrading_legacy_policy(monkeypatch):
    import application.low_entry_v7_daily as module
    from application.market_history import MarketHistoryAssembler
    from tests.test_v7_source_repair_daily import repaired_market
    market = repaired_market()
    snapshots = deepcopy(market.fundamental_snapshots)
    route_only = {"SYNTHETIC_ROUTE_ONLY": {"schema": "SYNTHETIC_NOT_AN_AUDIT_PROOF"}}
    snapshots["2330"]["les_v7_context"] = dict(
        snapshots["2330"].get("les_v7_context", {}), financial_audits_r7=route_only)
    market = replace(market, fundamental_snapshots=snapshots)
    prepare, seen = module.prepare_quality_context, []
    def routed(context, **options):
        seen.append(deepcopy(options["financial_audits_r7"]))
        return prepare(context, **options)
    monkeypatch.setattr(module, "prepare_quality_context", routed)
    report = module.daily_report(MarketHistoryAssembler().build(market, 0))
    assert seen == [route_only]
    assert report["diagnostics"]["quality"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R5"

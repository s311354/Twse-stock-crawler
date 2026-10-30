"""SYNTHETIC ONLY: R7 daily deployment and public source-bound context."""
from copy import deepcopy

import pytest

from application.les_v7 import prepare_quality_context
from application.low_entry_v7_daily import daily_report
from application.market_history import MarketHistoryAssembler
from domain.les_v7.contracts import identity
from domain.les_v7.quality_policy import r5_quality_policy, r6_quality_policy
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import baseline_policy, SOURCE_BASELINE_POLICY, SIGNED_BASELINE_POLICY
from domain.les_v7.signed_audit import audit_proof_r6
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from infrastructure.crawler.security_master import select_security_master
from tests.test_security_master import packet as security_packet
from tests.test_v7_source_repair_daily import repaired_market
from tests.fundamentals.test_signed_audit import fixture, CUTOFF


def test_current_daily_selects_r7_but_retains_factor_valuation_and_missing_source_semantics():
    market = repaired_market()
    history = MarketHistoryAssembler().build(market, 0)
    before = deepcopy(history.attrs)
    old = daily_report(history)
    history.attrs[SIGNAL_DATE_ATTR] = CUTOFF
    history.attrs["les_v7_native_sources"] = dict(history.attrs["les_v7_native_sources"],
        security=select_security_master(security_packet(), "2330", CUTOFF))
    new = daily_report(history)
    validate_report(new)
    assert old["diagnostics"]["quality"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R5"
    assert new["diagnostics"]["quality"]["profile_id"] == "QG_V7_REPORTED_PERIOD_R6"
    assert new["diagnostics"]["rule_baseline"]["policy"]["policy_version"] == SIGNED_BASELINE_POLICY
    assert new["envelope"]["score"] == old["envelope"]["score"]
    assert new["envelope"]["strategy"]["strategy_decision"] == old["envelope"]["strategy"]["strategy_decision"]
    assert new["diagnostics"]["quality"]["quality_score"] == old["diagnostics"]["quality"]["quality_score"]
    assert new["pae_authorization"]["authorized"] is False  # Unchanged valuation FAIL, no fabricated trade.
    assert before["les_v7_native_sources"] != history.attrs["les_v7_native_sources"]


def test_explicit_new_quality_context_accepts_bound_signed_source_and_preserves_old_r5():
    row, filing, packet, _ = fixture()
    proof = audit_proof_r6(filing, row, packet, symbol=row["symbol"], cutoff=CUTOFF)
    original = deepcopy(row)
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=[row], quality_policy=r6_quality_policy())
    prepare_quality_context(context, signed_financial_audits={row["content_sha256"]: proof})
    assert context["filings"][0]["audit_evidence_r6"] == proof
    assert row == original
    old = dict(context, filings=[row], quality_policy=r5_quality_policy())
    prepare_quality_context(old, signed_financial_audits={row["content_sha256"]: proof})
    assert "audit_evidence_r6" not in old["filings"][0]
    damaged = deepcopy(proof)
    damaged["signed_source"]["financial_anchors"]["revenue"]["value"] += 1
    with pytest.raises(ValueError, match="SIGNED_REPORT_CONTENT_ID_INVALID"):
        prepare_quality_context(dict(context, filings=[row]), signed_financial_audits={row["content_sha256"]: damaged})


def test_previous_baseline_and_quality_ids_stay_frozen():
    # Bound to the retained pre-change R5 thirteen-stock audit, not a new golden.
    assert r5_quality_policy()["policy_sha256"] == "9c07717011c3ef9ba42f4005aadf87ae6c14005fa46e666891744a5140a12b39"
    previous = baseline_policy(version=SOURCE_BASELINE_POLICY)
    new = baseline_policy(version=SIGNED_BASELINE_POLICY)
    assert previous["quality_policy"] == r5_quality_policy()
    assert new["quality_policy"] == r6_quality_policy()
    for key in ("score_policy", "valuation_policy", "valuation_input_policy", "config", "context_policy"):
        if key in previous:
            assert new[key] == previous[key]
    assert identity(previous) != identity(new)


def test_actual_repository_facade_forwards_market_to_current_source_query(tmp_path):
    from tests.fundamentals.test_ttm_audit import cache
    from application.fundamental_service import MopsFundamentalService
    from domain.les_v7.financial_sources import ixbrl_record_r2
    repo = cache(tmp_path / "mops")
    records = tuple(ixbrl_record_r2("2330", filing, original_verified=True)
        for filing in repo.load_filed_financials("2330", CUTOFF))
    service = object.__new__(MopsFundamentalService)
    service.repository = repo  # Real facade, not a kwargs-accepting mock.
    values = service.load_v7_financial_sources("2330", as_of=CUTOFF, records=records,
        market="TPEX", source_repair=True)
    assert values["financial_repairs_r5"]
    assert all(p["market"] == "TPEX" for p in values["financial_repairs_r5"].values())

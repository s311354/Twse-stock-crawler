"""SYNTHETIC ONLY: declared rules/caps, no real-source or performance claim."""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
import json

import numpy as np
import pandas as pd
import pytest

from application.low_entry_v7_daily import daily_report, project_report
from application.market_history import MarketHistoryAssembler
from domain.les_v7.allocation import _authorize_validated, authorize_candidate
from domain.les_v7.contracts import GateReceipt, ReferencePlan, canonical, identity, timestamp
from domain.les_v7.entry_policy_r9 import (policy, quality_tier, valuation_grade, assess,
    R9PortfolioSecurityV2, pae_admission, cap_for)
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import (GRADED_BASELINE_POLICY, GRADED_AVAILABLE_AT, PENALTY_BASELINE_POLICY,
    baseline_policy, current_source_baseline_version, score_baseline, select_baseline_context)
from domain.les_v7.valuation_price import evaluate_price_valuation
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR
from domain.strategies.portfolio_allocation_v2_contracts import PortfolioSecurityV2, PortfolioAllocationRequestV2, PortfolioAllocationV2Config
from domain.strategies.portfolio_allocation_v2_evaluator import PortfolioSecurityEvaluatorV2
from domain.strategies.portfolio_allocation_v2_weight_allocator import PortfolioWeightAllocatorV2
from domain.strategies.portfolio_allocation_v2_utils import stable_dataclass_hash
from infrastructure.crawler.security_master import select_security_master
from tests.test_security_master import packet as security_packet
from tests.test_v7_source_repair_daily import repaired_market
from tests.strategies.test_v7_rule_baseline import unit_features


def graded_market(ratio=1.15, *, limited_quality=False):
    """Synthetic originals/quotes exercise real producers, never official repair.

    EPS contexts and original IDs are regenerated from declared synthetic facts;
    no prebound gate or modified verdict supplies permission to the daily path.
    """
    from domain.fundamental_derivation import FiledFact, FiledFinancials
    from domain.les_v7.financial_sources import ixbrl_record_r2
    from domain.les_v7.financial_sources_r5 import audit_proof_r5
    from domain.les_v7.valuation_earnings import valuation_earnings
    from tests.strategies.test_daily_v7_pipeline import executable_synthetic_market
    market = repaired_market()
    cutoff = (timestamp(GRADED_AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    snapshot = deepcopy(market.fundamental_snapshots)
    rows = snapshot["2330"]["les_v6_mops_filings"]
    if limited_quality:
        from tests.strategies.test_v7_reported_quality import source_records
        row = source_records(negative_profit=True)[-1]
        period = pd.Period(row["period_end"], freq="Q")
        facts = tuple(FiledFact(**dict(f, dimensions=tuple(tuple(d) for d in f.get("dimensions", ()))))
            for f in row["audit_evidence"]["contexts"])
        filing = FiledFinancials(period.year, period.quarter, row["available_at"], row["revision_number"], facts,
            content_sha256=row["content_sha256"], source_url=row["source_url"], period_end=row["period_end"], statement_scope=row["scope"])
        row["audit_evidence_r5"] = audit_proof_r5(filing, symbol="2330", original_verified=True)
        rows[-1] = row
    additions = {}
    price = market.closes[0][-1] / 1000.
    eps = price / (ratio * 20. * .9 * (.9 / 1.1) ** 5)
    for i, row in enumerate(rows):
        period = pd.Period(row["period_end"], freq="Q")
        if period.quarter != 4: continue
        contexts = list(row["audit_evidence"]["contexts"])
        for group in ("field_evidence", "quarter_field_evidence", "prior_ytd_field_evidence",
                      "comparative_ytd_field_evidence", "comparative_year_end_field_evidence"):
            for proof in row[group].values(): contexts.extend(proof.get("contexts", ()))
        unique = {identity(c): c for c in contexts}
        facts = tuple(FiledFact(c["concept"], c.get("numeric_value"), c.get("text_value", c.get("unresolved_text")),
            c.get("start_date"), c.get("end_date"), c.get("instant"), unit=c.get("unit"),
            context_id=c.get("context_id"), dimensions=tuple(tuple(d) for d in c.get("dimensions", ())))
            for c in unique.values())
        facts += tuple(FiledFact("ifrs-full:BasicEarningsLossPerShare", value, None,
            f"{year}-01-01", f"{year}-12-31", None, unit="iso4217:TWD/xbrli:shares",
            context_id="SYNTHETIC_ONLY_FULL_ANNUAL_EPS")
            for year, value in ((period.year, eps), (period.year - 1, eps / .8)))
        filing = FiledFinancials(period.year, 4, row["available_at"], row["revision_number"], facts,
            content_sha256=identity(dict(synthetic_only=True, facts=[str(f) for f in facts])),
            source_url=row["source_url"], period_end=row["period_end"], statement_scope=row["scope"],
            taxonomy_schema_refs=("https://mops.twse.com.tw/tifrs-ci-cr-2026-03-31.xsd",))
        rows[i] = ixbrl_record_r2("2330", filing, original_verified=True)
        rows[i]["audit_evidence_r5"] = audit_proof_r5(filing, symbol="2330", original_verified=True)
        additions[filing.content_sha256] = valuation_earnings(filing, symbol="2330",
            original_verified=True, as_of=cutoff)
    snapshot["2330"]["les_v6_filings"] = deepcopy(rows)
    snapshot["2330"]["les_v7_context"] = dict(valuation_earnings=additions)
    pe = deepcopy(market.pe_ratios); pe[0][-1] = 3.
    benchmark = deepcopy(market.taiex_closes)
    benchmark[0][-60:] = np.linspace(benchmark[0][-60], benchmark[0][-60] * .70, 60).tolist()
    sources = deepcopy(market.native_source_selections)
    sources["2330"]["security"] = select_security_master(security_packet(), "2330", cutoff)
    restrictions = deepcopy(market.trading_restrictions)
    restrictions["2330"]["available_at"] = cutoff  # Explicit synthetic fresh capture.
    return replace(market, lows=executable_synthetic_market().lows, pe_ratios=pe,
        taiex_closes=benchmark, fundamental_snapshots=snapshot, signal_dates={"2330": cutoff},
        trading_restrictions=restrictions, native_source_selections=sources,
        les_v6_portfolio=dict(account(), as_of=cutoff))


def quality(qs=80., **changes):
    return dict(profile_id="QG_V7_REPORTED_PERIOD_R6", status="PASS" if qs >= 65 else "WATCH_QUALITY",
        quality_score=qs, data_confidence=100., coverage_total=1., core_inputs_pass=True,
        core_reported_pass=True, mandatory_safety_pass=True, native_quality_complete=True,
        financial_safety=dict(state="PASS"), **changes)


def valuation(ratio=1.15, *, price=101.):
    # Exactly declared falling earnings, with positive reported premises. The
    # original arithmetic is used; no synthetic PASS flag replaces a scenario.
    eps = price / ratio / (20. * (.9 / 1.1) ** 5)
    d = evaluate_price_valuation(current_eps=eps, prior_eps=eps / .8,
        current_revenue=90., prior_revenue=100., current_margin=.2, prior_margin=.3,
        current_price=price, current_pe=1., recent_pe_median=20., eps_basis="DILUTED")
    return dict(gate_id="VALUATION", state=d["status"], valuation_diagnostics=d,
                provenance="SYNTHETIC_ONLY_NOT_SOURCE_CERTIFICATION")


def gates(value, **states):
    return [GateReceipt(g, "SYNTHETIC_ONLY", states.get(g, value["state"] if g == "VALUATION" else "PASS"),
        identity({}), GRADED_AVAILABLE_AT, (), ("SYNTHETIC_ONLY",), ("SYNTHETIC_ONLY",), GRADED_AVAILABLE_AT)
        for g in ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK")]


def outcome(qs=80., ratio=1.15, *, q=None, v=None, states=None, risks=(), missing_plan=False, f=None):
    f = f or unit_features(); v = v or valuation(ratio)
    p = None if missing_plan else ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL",
        101., 99., 109., 2., (101., 101.5), 1, .1, .1)
    prepared = dict(policy=baseline_policy(version=GRADED_BASELINE_POLICY), features=f,
        selection=select_baseline_context(f, policy_version=GRADED_BASELINE_POLICY).record())
    gs = gates(v, **(states or {}))
    native = dict(close=101., spread=.003, adv=100_000_000.)
    return score_baseline(prepared, plan=p, quality=q or quality(qs), gates=gs,
        native_features=native, risks=list(risks), standard=not missing_plan,
        starter_gate=not missing_plan, confirmed_gate=not missing_plan,
        valuation_receipt=v, entry_binding=dict(symbol="2330", cutoff=GRADED_AVAILABLE_AT))


@pytest.mark.parametrize("qs,tier,buy,small", [(75., "HIGH", .20, .05), (74.9999, "STANDARD", .10, .025),
    (65., "STANDARD", .10, .025), (64.9999, "LIMITED", 0., .02), (60., "LIMITED", 0., .02),
    (59.9999, None, 0., 0.)])
def test_quality_boundaries_have_final_caps_not_rescaled_scores(qs, tier, buy, small):
    q = quality(qs); before = deepcopy(q)
    result = quality_tier(q)
    assert (result["tier"], result["buy_cap"], result["small_cap"]) == (tier, buy, small)
    assert q == before


@pytest.mark.parametrize("key,value", [("status", "DATA_INSUFFICIENT"), ("status", "UNKNOWN"),
    ("data_confidence", 64.9999), ("coverage_total", .749999), ("core_inputs_pass", False),
    ("native_quality_complete", False), ("financial_safety", dict(state="UNKNOWN"))])
def test_unknown_missing_and_unverified_quality_cannot_become_a_low_tier(key, value):
    q = quality(); q[key] = value
    assert quality_tier(q)["state"] == "UNKNOWN"
    assert outcome(q=q)["decision"] == "DATA_INSUFFICIENT"


@pytest.mark.parametrize("ratio,band,points,multiplier,small", [(.9, "PASS", 0., 1., False),
    (1.10, "MILD", 5., .75, False), (1.100001, "MODERATE", 10., .5, False),
    (1.25, "MODERATE", 10., .5, False), (1.250001, "HIGH", 15., .25, True)])
def test_complete_verified_price_failure_uses_one_band(ratio, band, points, multiplier, small):
    v = valuation(ratio); before = deepcopy(v)
    g = valuation_grade(v, current_price=101.)
    assert (g["band"], g["points"], g["multiplier"], g["small_only"]) == (band, points, multiplier, small)
    assert v == before


def test_bear_loss_and_negative_premises_still_block_without_fabricating_scenarios():
    v = valuation(1.5)
    assert v["valuation_diagnostics"]["bear_permanent_loss"] > .35
    assert valuation_grade(v, current_price=101.)["state"] == "BLOCKED"
    assert outcome(v=v)["decision"] == "NO_BUY"
    assert valuation_grade(dict(gate_id="VALUATION", state="FAIL"), current_price=101.)["state"] == "BLOCKED"


@pytest.mark.parametrize("damage", ["price", "result", "state", "missing"])
def test_unverified_or_conflicting_valuation_remains_unknown(damage):
    v = valuation()
    if damage == "price": v["valuation_diagnostics"]["inputs"]["current_price"] += 1.
    elif damage == "result": v["valuation_diagnostics"]["scenarios"]["BASE"]["fair_price"] *= 2
    elif damage == "state": v["state"] = "PASS"
    else: v["valuation_diagnostics"].pop("inputs")
    assert valuation_grade(v, current_price=101.)["state"] in {"UNKNOWN", "BLOCKED"}
    assert outcome(v=v)["decision"] not in {"BUY", "BUY_SMALL"}


def test_score_quality_tier_and_known_warning_have_one_native_decision():
    r = outcome(qs=66.)
    assert r["score"]["raw_score"] == 94. and r["score"]["final_score"] == 84.
    assert r["penalties"] == {"VALUATION_PRICE_WARNING": 10.}
    assert r["decision"] == "BUY" and r["entry_admission"]["buy_total_cap"] == .05
    low = outcome(qs=60.)
    assert low["score"] == r["score"] and low["decision"] == "BUY_SMALL"
    assert low["entry_admission"]["small_total_cap"] == .01
    high = outcome(ratio=1.3)
    assert high["score"]["final_score"] == 79. and high["decision"] == "BUY_SMALL"


@pytest.mark.parametrize("gate", ["DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "STRESS_RISK"])
@pytest.mark.parametrize("state", ["FAIL", "UNKNOWN"])
def test_other_gate_failure_unknown_and_invalid_plan_cannot_be_compensated(gate, state):
    r = outcome(states={gate: state})
    assert r["score"]["final_score"] == 84. and r["decision"] not in {"BUY", "BUY_SMALL"}


def test_financial_reject_missing_stop_execution_failure_and_veto_remain_blocked():
    q = quality(); q["status"] = "REJECT"
    assert outcome(q=q)["decision"] == "NO_BUY"
    assert outcome(missing_plan=True)["decision"] not in {"BUY", "BUY_SMALL"}
    assert outcome(risks=["EXECUTION_FAIL"])["decision"] == "NO_BUY"
    assert outcome(risks=["SUPPORT_FAILURE"])["decision"] == "NO_BUY"
    # Only a validated economic valuation failure may be removed from vetoes.
    assert outcome(risks=["VALUATION_FAIL"])["decision"] == "BUY"


def boundary_report(signal="BUY_SMALL", qs=80., ratio=.9):
    r = outcome(qs=qs, ratio=ratio)
    from tests.test_v7_baseline_allocation import boundary_report as legacy
    report = legacy(signal)
    report["envelope"]["event"]["security_id"] = "2330"
    report["envelope"]["event"]["cutoff_at"] = GRADED_AVAILABLE_AT
    d = report["diagnostics"]
    d["rule_baseline"]["policy"] = baseline_policy(version=GRADED_BASELINE_POLICY)
    d["entry_admission"] = r["entry_admission"]
    d["gate_receipts"] = [asdict(g) for g in gates(valuation(ratio))]
    # Small loss makes the approved weight cap binding, not the risk budget.
    d["reference_plan"].update(entry=101., stop=100.99, band=[101., 101.], cost_up=0., cost_stop=0.)
    d["features"]["adv"] = 1_000_000_000.
    report["envelope"]["score"] = r["score"]
    return report


def account(**changes):
    return dict(value=1_000_000., available_cash=1_000_000., intended_weight=.25, capacity_weight=.25,
                positions=[], as_of=GRADED_AVAILABLE_AT, **changes)


def test_final_small_cap_is_applied_once_and_limits_do_not_change_score():
    r = boundary_report(); before = canonical(r["envelope"]["score"])
    quota = _authorize_validated(r, account())
    assert quota["authorized"] and quota["indicative_shares"] == 495
    assert quota["maximum_total_weight"] == .05
    assert quota["small_sizing"] == "DIRECT_FINAL_CAP_NO_REPEAT_MULTIPLIER"
    no_cash = account(); no_cash["available_cash"] = 0.
    assert not _authorize_validated(r, no_cash)["authorized"]
    assert canonical(r["envelope"]["score"]) == before


def test_existing_holdings_reduce_incremental_capacity_without_forced_exit():
    r = boundary_report()
    acc = account(); acc["positions"] = [dict(symbol="2330", shares=500)]
    quota = _authorize_validated(r, acc)
    assert quota["current_weight"] == .0505 and quota["incremental_weight_capacity"] == 0
    assert quota["indicative_shares"] == 0 and not quota["authorized"]
    assert r["envelope"]["strategy"]["strategy_decision"] == "BUY_SMALL"


def test_valuation_multiplier_restricts_total_cap_without_mutating_raw_fail():
    r = boundary_report(signal="BUY", qs=66., ratio=1.15)
    quota = _authorize_validated(r, account())
    assert quota["maximum_total_weight"] == .05 and quota["indicative_shares"] == 495
    assert next(g for g in r["diagnostics"]["gate_receipts"] if g["gate_id"] == "VALUATION")["state"] == "FAIL"


def policy_security(signal="BUY_SMALL", qs=60., ratio=1.15):
    a = outcome(qs=qs, ratio=ratio)["entry_admission"]
    return R9PortfolioSecurityV2(stock_id="2330", sector_id="ELEC", correlation_cluster_id="ELEC",
        quality_status=a["quality"]["native_status"], quality_score=qs, quality_confidence=100., data_confidence=100.,
        valuation_status="UNKNOWN", valuation_gate=a["valuation"]["native_state"], les_decision=signal,
        risk_gate="PASS", falling_knife_state="CLEAR", liquidity_status="PASS", executable_shares=10,
        les_position_cap=cap_for(a, signal), risk_budget_weight=1., liquidity_weight_cap=1.,
        entry_admission_json=canonical(a))


def test_pae_direct_selection_uses_admission_not_legacy_sidecar_gate_and_no_repeat_small_factor():
    s = policy_security(); config = PortfolioAllocationV2Config()
    assert PortfolioSecurityEvaluatorV2(config).direct_selection_executable(s)
    assert not PortfolioSecurityEvaluatorV2(config).evaluate(s, .1).executable_candidate  # Genuine missing CER remains.
    request = PortfolioAllocationRequestV2("SYNTHETIC_ONLY", timestamp(GRADED_AVAILABLE_AT), 1_000_000., 1_000_000., candidates=(s,))
    target, reasons = PortfolioWeightAllocatorV2(config).target_weight(request, s, .9, None)
    assert not reasons and target == .01  # Not .005: native SMALL cap already applied.


def test_old_pae_contract_serialization_and_small_multiplier_are_unchanged():
    s = PortfolioSecurityV2("2330", sector_id="ELEC", correlation_cluster_id="ELEC", les_decision="BUY_SMALL",
        les_position_cap=.02, risk_budget_weight=.02, liquidity_weight_cap=.02)
    assert "entry_admission_json" not in asdict(s) and pae_admission(s) is None
    import hashlib
    assert stable_dataclass_hash(s) == hashlib.sha256(json.dumps(asdict(s),sort_keys=True,default=str,separators=(",", ":")).encode()).hexdigest()
    request = PortfolioAllocationRequestV2("SYNTHETIC_ONLY", timestamp(GRADED_AVAILABLE_AT), 1_000_000., 1_000_000., candidates=(s,))
    assert PortfolioWeightAllocatorV2(PortfolioAllocationV2Config()).target_weight(request, s, .9, None)[0] == .01


def native_report():
    history = MarketHistoryAssembler().build(repaired_market(), 0)
    cutoff = (timestamp(GRADED_AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    history.attrs[SIGNAL_DATE_ATTR] = cutoff
    history.attrs["les_v7_native_sources"] = dict(history.attrs["les_v7_native_sources"],
        security=select_security_master(security_packet(), "2330", cutoff))
    return daily_report(history)


def test_native_daily_public_validation_projection_and_quota_use_r9_without_filling_gaps():
    report = native_report(); validate_report(report)
    assert report["diagnostics"]["rule_baseline"]["policy"]["policy_version"] == GRADED_BASELINE_POLICY
    assert report["diagnostics"]["entry_admission"]["valuation"]["state"] == "BLOCKED"
    assert report["pae_authorization"]["policy_version"] == "LES_V7_PAE_SIGNAL_AUTHORIZATION_R2"
    assert not report["pae_authorization"]["authorized"]
    row = project_report(report)
    assert row["低點分數"] == report["envelope"]["score"]["final_score"]
    assert row["建議股數"] == 0


def test_forged_admission_grade_cap_or_gate_cannot_validate_or_authorize():
    original = native_report()
    for key in ("state", "buy_total_cap", "quality", "valuation"):
        r = deepcopy(original)
        a = r["diagnostics"]["entry_admission"]
        if key == "quality": a[key]["score"] = 100.
        elif key == "valuation": a[key]["state"] = "ALLOWED"
        else: a[key] = "ALLOWED" if key == "state" else 1.
        a["assessment_id"] = identity({k:v for k,v in a.items() if k!="assessment_id"})
        with pytest.raises(ValueError): authorize_candidate(r, account())


def test_r8_policy_identity_is_retained_and_new_policy_cannot_borrow_old_cutoff():
    assert identity(baseline_policy(version=PENALTY_BASELINE_POLICY)) == "98fbc6d2887f01bbc3ee06c61696f3522ef553de65afc5f1def083f9e8a30c21"
    assert current_source_baseline_version(GRADED_AVAILABLE_AT) == GRADED_BASELINE_POLICY
    assert current_source_baseline_version((timestamp(GRADED_AVAILABLE_AT)-timedelta(seconds=1)).isoformat()) == PENALTY_BASELINE_POLICY


@pytest.mark.parametrize("ratio,signal,points,cap", [(.9, "BUY", 0., .20), (1.15, "BUY_SMALL", 10., .025)])
def test_real_native_producers_daily_projection_pae_and_renderer_are_connected(ratio, signal, points, cap):
    from application.analysis_dataset import AnalysisDatasetBuilder
    from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
    from application.portfolio_contracts import PortfolioRuntimeContext
    from infrastructure.report.html_renderer import RendererFactory
    market = graded_market(ratio); before = deepcopy(market.fundamental_snapshots)
    decision = AnalysisDatasetBuilder().build_decisions(market, [[1, 1, 1, 1, .01]])[0]
    r = json.loads(decision.value("LES v7 Report")); validate_report(r)
    assert r["envelope"]["strategy"]["strategy_decision"] == decision.value("低點決策") == signal
    assert r["diagnostics"]["entry_admission"]["valuation"]["points"] == points
    assert r["pae_authorization"]["authorized"] and r["pae_authorization"]["maximum_total_weight"] == cap
    assert r["pae_authorization"]["live_order_authorized"] is False
    assert decision.value("Position Fraction") <= cap and decision.value("建議股數") > 0
    assert r["envelope"]["score"]["final_score"] == r["envelope"]["score"]["raw_score"] - r["envelope"]["score"]["total_penalty"]
    context = PortfolioRuntimeContext(account_equity=1_000_000., current_cash=1_000_000., positions=(), default_sector="ELEC")
    security = PortfolioAllocationV2Pipeline()._security(decision, context, position=None)
    assert isinstance(security, R9PortfolioSecurityV2)
    assert security.valuation_gate == ("FAIL" if points else "PASS")
    assert PortfolioSecurityEvaluatorV2(PortfolioAllocationV2Config()).direct_selection_executable(security)
    assert security.les_position_cap == decision.value("Position Fraction")
    html = RendererFactory.get_renderer("daily_strategy_email").render((decision,))
    assert "品質級別 HIGH" in html and f"核准持倉總上限 {cap * 100:.2f}%" in html
    assert "限倉不回頭扣分" in html and "未使用機率預測模型" in html
    if points: assert "估值 MODERATE · 扣分 10.00" in html and "VALUATION: FAIL" in html
    assert market.fundamental_snapshots == before


def test_pae_rechecks_admission_and_total_limit_and_preserves_subtype_in_comparison():
    s = policy_security()
    s = replace(s, les_position_cap=1., valuation_status="COMPLETE", valuation_confidence=100.,
                bear_cagr=.1, base_cagr=.2, bull_cagr=.3)
    config = PortfolioAllocationV2Config(); evaluator = PortfolioSecurityEvaluatorV2(config)
    assert evaluator.evaluate(s, .1).executable_candidate
    allocator = PortfolioWeightAllocatorV2(config)
    request = PortfolioAllocationRequestV2("SYNTHETIC_ONLY", timestamp(GRADED_AVAILABLE_AT), 1_000_000., 1_000_000., candidates=(s,))
    assert allocator.target_weight(request, s, .9, None)[0] == .01
    copied = allocator.with_weight(s, .01)
    assert isinstance(copied, R9PortfolioSecurityV2) and copied.entry_admission_json == s.entry_admission_json
    a = json.loads(s.entry_admission_json)
    a.update(state="UNKNOWN", small_allowed=False, buy_allowed=False)
    a["assessment_id"] = identity({k:v for k,v in a.items() if k != "assessment_id"})
    blocked = replace(s, entry_admission_json=canonical(a))
    assert not evaluator.evaluate(blocked, .1).executable_candidate
    assert not evaluator.direct_selection_executable(blocked)


def test_r9_serial_shared_and_spawn_paths_recompute_identical_result():
    from application.analysis_dataset import AnalysisDatasetBuilder
    from infrastructure.daily_parallel import DailyProcessExecutor
    market = graded_market(); profits = [[1, 1, 1, 1, .01]]
    serial = AnalysisDatasetBuilder().build_decisions(market, profits)
    assert serial == AnalysisDatasetBuilder(optimized_daily=False).build_decisions(market, profits)
    assert serial == AnalysisDatasetBuilder(stock_executor=DailyProcessExecutor(2)).build_decisions(market, profits)


def test_source_verified_watch_quality_uses_limited_tier_without_promoting_native_status():
    from domain.les_v7.quality_reported import validate_reported_result
    report = daily_report(MarketHistoryAssembler().build(graded_market(.9, limited_quality=True), 0))
    q = report["diagnostics"]["quality"]; validate_reported_result(q); validate_report(report)
    assert 60 <= q["quality_score"] < 65 and q["status"] == "WATCH_QUALITY"
    assert q["financial_safety"]["state"] == "PASS" and q["native_quality_complete"]
    assert report["diagnostics"]["entry_admission"]["quality"]["tier"] == "LIMITED"
    assert report["diagnostics"]["readiness"]["quality"]["state"] == "READY_WITH_GRADED_LIMIT"
    assert report["envelope"]["strategy"]["strategy_decision"] == "BUY_SMALL"
    assert report["pae_authorization"]["authorized"] and report["pae_authorization"]["maximum_total_weight"] == .02
    assert project_report(report)["Quality Gate"] == "WATCH_QUALITY"


def test_new_total_cap_never_automatically_trims_existing_holding():
    from domain.strategies.portfolio_allocation_v2 import PortfolioAllocationEngineV2
    holding = replace(policy_security(), current_weight=.07, erem_decision="HOLD", erem_holding_confidence=80.)
    request = PortfolioAllocationRequestV2("SYNTHETIC_ONLY", timestamp(GRADED_AVAILABLE_AT), 1_000_000., 930_000., holdings=(holding,))
    result = PortfolioAllocationEngineV2().allocate(request)
    line = next(line for line in result.lines if line.stock_id == "2330")
    assert line.current_weight == line.target_weight == .07


def test_preflight_selects_new_policy_without_inventing_model_or_evidence():
    from application.les_v7_preflight import daily_v7_preflight
    from config.settings import AppSettings
    result = daily_v7_preflight(AppSettings(daily_entry_model="v7", les_v7_rule_baseline=True,
        les_v7_native_evidence=True, les_v7_source_repair=True), checked_at=GRADED_AVAILABLE_AT)
    assert result["state"] == "READY" and result["rule_baseline_policy"]["policy_version"] == GRADED_BASELINE_POLICY
    assert result["probability_defaults"] is None and not result["order_authority"]

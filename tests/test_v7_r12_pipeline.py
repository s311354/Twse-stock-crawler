"""SYNTHETIC ONLY: versioned close-confirmation integration, not actual BUYs."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json

import pandas as pd
import pytest

from application.low_entry_v7_daily import daily_report
from application.market_history import MarketHistoryAssembler
from application.valuation_peers_r10 import capture_comparison_contexts
from application.valuation_peers_r10 import _independent_prices
from domain.les_v7.contracts import FEATURE_VERSION, canonical, identity, timestamp
from domain.les_v7.decision import vetoes
from domain.les_v7.features import CLOSE_SUPPORT_FEATURE_VERSION, compute_features, select_bars
from domain.les_v7.native_evidence import _validated_features, CLOSE_SAFETY_PRODUCER_VERSION
from domain.les_v7.plan_r10 import make_plan, validate_band
from domain.les_v7.report import validate_report
from domain.les_v7.rule_baseline import (
    CLOSE_BASELINE_POLICY, CLOSE_AVAILABLE_AT, INDUSTRY_BASELINE_POLICY, RECOVERY_BASELINE_POLICY,
    baseline_policy, current_source_baseline_version, prepare_baseline, require_baseline_policy,
)
from tests.strategies.test_v7_close_support import structural_bars
from tests.test_v7_industry_pipeline import market as industry_market


def market():
    old = industry_market()
    at = (timestamp(CLOSE_AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    from infrastructure.crawler.security_master import select_security_master
    from tests.test_security_master import packet
    sources = deepcopy(old.native_source_selections)
    sources["2330"]["security"] = select_security_master(packet(), "2330", at)
    return replace(old, signal_dates={"2330": at}, native_source_selections=sources,
        les_v6_portfolio=dict(old.les_v6_portfolio, as_of=at))


def native(m=None):
    return daily_report(MarketHistoryAssembler().build(capture_comparison_contexts(m or market()), 0))


def test_old_hashes_and_release_vintage_remain_exact():
    assert identity(baseline_policy(version=INDUSTRY_BASELINE_POLICY)) == "3d77a52c75ae582b21a54052a300b7f5467fdacde527d630e143ad6e612dd2fc"
    assert identity(baseline_policy(version=RECOVERY_BASELINE_POLICY)) == "5b94f7fef0fb757681cebb5b2c3233e0f14aca393f2b74b268c79ba6cec2809d"
    before = timestamp(CLOSE_AVAILABLE_AT) - timedelta(microseconds=1)
    assert current_source_baseline_version(before) == INDUSTRY_BASELINE_POLICY
    assert current_source_baseline_version(CLOSE_AVAILABLE_AT) == CLOSE_BASELINE_POLICY
    with pytest.raises(ValueError, match="NOT_AVAILABLE_AT_CUTOFF"):
        require_baseline_policy(baseline_policy(version=CLOSE_BASELINE_POLICY), market="TWSE", cutoff=before)


def test_no_economic_scoring_or_risk_limits_are_relaxed():
    old, new = (baseline_policy(version=v) for v in (INDUSTRY_BASELINE_POLICY, CLOSE_BASELINE_POLICY))
    for key in ("config", "model_spec", "native_risk_limits", "valuation_policy", "valuation_input_policy",
                "entry_admission_policy", "score_policy", "score_adjustment_policy", "quality_policy", "industry_score_policy"):
        assert old[key] == new[key]
    assert old["reference_policy"]["structural_rules"] == new["reference_policy"]["structural_rules"]
    assert "native_feature_version" not in old
    assert new["native_feature_version"] == CLOSE_SUPPORT_FEATURE_VERSION


def test_confirmed_close_context_plan_and_veto_share_known_floor_without_rr_manipulation():
    bars = structural_bars().iloc[:120]
    cutoff = str(bars.Date.iloc[-1]) + "T13:30:00+08:00"
    prepared = prepare_baseline(bars, market="TWSE", cutoff=cutoff, policy_version=CLOSE_BASELINE_POLICY,
        industry_input=dict(symbol="2330", sector="SEMI"))
    old, _ = compute_features(bars)
    new, _ = compute_features(bars, feature_version=CLOSE_SUPPORT_FEATURE_VERSION)
    plan, audit = make_plan(prepared["selection"]["selected_setup"], new,
        prepared["policy"]["reference_policy"], "SYNTHETIC_ONLY", bars=bars, cutoff=cutoff)
    assert prepared["selection"]["selected_setup"] == "REVERSAL"
    assert prepared["features"]["support"] == new["support"] == audit["anchors"]["support"]["value"] == 90.
    assert old["support"] == 100. and "SUPPORT_FAILURE" in vetoes(old)
    assert "SUPPORT_FAILURE" not in vetoes(new) and "GAP_FAILURE" not in vetoes(new)
    assert plan.stop == 90. - .3 * new["atr"] and plan.target == 130.
    assert validate_band(plan, prepared["policy"]["reference_policy"])["state"] == "PASS"
    assert validate_band(plan, prepared["policy"]["reference_policy"], starter=True)["state"] == "PASS"


def test_profile_cache_keys_are_isolated_and_recompute_exactly():
    bars = structural_bars().iloc[:120]
    cutoff = str(bars.Date.iloc[-1]) + "T13:30:00+08:00"
    rows = bars.astype(object).where(pd.notna(bars), None).to_dict("records")
    _validated_features.cache_clear()
    first = _validated_features(canonical(rows), cutoff, FEATURE_VERSION)
    second = _validated_features(canonical(rows), cutoff, CLOSE_SUPPORT_FEATURE_VERSION)
    assert json.loads(first)["support"] == 100. and json.loads(second)["support"] == 90.
    assert _validated_features(canonical(rows), cutoff, FEATURE_VERSION) == first
    assert _validated_features.cache_info().misses == 2


def test_native_daily_binds_new_profile_receipts_and_keeps_source_unknowns():
    report = native()
    validate_report(report)
    diag = report["diagnostics"]
    assert diag["rule_baseline"]["policy"]["policy_version"] == CLOSE_BASELINE_POLICY
    assert diag["feature_audit"]["profile"] == CLOSE_SUPPORT_FEATURE_VERSION
    assert diag["evidence_binding"]["data_rules"] == CLOSE_SUPPORT_FEATURE_VERSION
    assert diag["native_evidence"]["produced_receipts"]["SAFETY"]["gate_version"] == CLOSE_SAFETY_PRODUCER_VERSION
    assert report["pae_authorization"]["live_order_authorized"] is False
    assert diag["native_evidence"]["produced_receipts"]["VALUATION"]["valuation_diagnostics"]["comparables"]["status"] == "INSUFFICIENT"


@pytest.mark.parametrize("kind", ["profile", "support_policy", "features", "data_rules"])
def test_rehashed_or_edited_timing_claims_never_pass_validation(kind):
    report = deepcopy(native())
    diag = report["diagnostics"]
    if kind == "profile": diag["feature_audit"]["profile"] = FEATURE_VERSION
    elif kind == "support_policy": diag["feature_audit"]["support_policy"]["additional_delay_sessions"] = 1
    elif kind == "features": diag["features"]["support"] = 999.
    else: diag["evidence_binding"]["data_rules"] = FEATURE_VERSION
    with pytest.raises(ValueError): validate_report(report)


def test_cash_changes_only_authorization_not_technical_score_or_gates():
    m = market()
    full, empty = native(m), native(replace(m, les_v6_portfolio=dict(m.les_v6_portfolio, available_cash=0.)))
    assert full["envelope"] == empty["envelope"]
    assert full["diagnostics"] == empty["diagnostics"]
    assert empty["pae_authorization"]["authorized"] is False


def monthly_selection(*, observed_at=None, status="PARTIAL"):
    row = dict(schema="les-v7-monthly-valuation-selection/v1", symbol="2330", market="TWSE",
        observed_at=observed_at or CLOSE_AVAILABLE_AT, capture_evidence_id=identity("SYNTHETIC_CAPTURE_ONLY"),
        source_id=identity("SYNTHETIC_MONTHLY_STOCK_ONLY"), status=status, price_unit="TWD",
        rows=[dict(Date="2023-10-31", Close=80.), dict(Date="2026-09-30", Close=90.)])
    return dict(row, selection_id=identity(row))


def test_independent_monthly_prices_do_not_extend_or_mutate_technical_history():
    from tests.strategies.test_v7_entry_policy_r9 import graded_market
    m = market()
    before = deepcopy(m)
    sources = deepcopy(m.native_source_selections)
    selected = monthly_selection()
    known = dict(zip(m.dates[0], (v / 1000. for v in m.closes[0])))
    for row in selected["rows"]:
        if row["Date"] in known: row["Close"] = known[row["Date"]]
    selected["selection_id"] = identity({k: v for k, v in selected.items() if k != "selection_id"})
    sources["2330"]["valuation_price_history"] = selected
    # Pure projection test: explicitly injected SYNTHETIC ONLY verifier. Real
    # capture/source identity is checked separately with retained originals.
    changed = capture_comparison_contexts(replace(m, native_source_selections=sources),
                                         verify_valuation_history=lambda value: value)
    issuer = changed.native_source_selections["2330"]["r10_valuation_context"]["issuer"]
    assert issuer["market_rows"][0]["Date"] == "2023-10-31"
    assert issuer["market_rows"][-1]["Date"] == m.dates[0][-1]
    assert changed.dates == m.dates == before.dates and changed.closes == m.closes == before.closes
    assert m.native_source_selections == before.native_source_selections
    assert "valuation_history_selection" in issuer["market_evidence"]
    # Explicit pre-R12 paths never consume the new optional source.
    old = graded_market()
    assert capture_comparison_contexts(old) is old


def test_future_conflict_and_unknown_monthly_history_never_supply_clearance():
    rows = [dict(Date="2026-10-14", Close=100.)]
    source = dict(source_available_at=CLOSE_AVAILABLE_AT, source_final=True, source_record_ids=[identity(rows)])
    at = (timestamp(CLOSE_AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="SELECTION_INVALID"):
        _independent_prices(monthly_selection(observed_at="2030-01-01T00:00:00Z"),
            rows=rows, symbol="2330", cutoff=at, source=source)
    conflict = monthly_selection()
    conflict["rows"].append(dict(Date="2026-10-14", Close=999.))
    conflict["selection_id"] = identity({k: v for k, v in conflict.items() if k != "selection_id"})
    same, invalid = _independent_prices(conflict, rows=rows, symbol="2330", cutoff=at, source=source,
                                       verify_selection=lambda value: value)
    assert same == rows and invalid["source_final"] is False
    assert invalid["valuation_history_conflicts"] == [dict(date="2026-10-14", strategy_close=100., monthly_close=999.)]
    same, invalid = _independent_prices(monthly_selection(status="UNKNOWN"), rows=rows,
        symbol="2330", cutoff=at, source=source)
    assert same == rows and invalid["source_final"] is False


def test_runtime_retains_shared_originals_and_preserves_future_claim_for_cutoff_rejection(tmp_path):
    from types import SimpleNamespace
    from tests.test_v7_valuation_history import database, projection, rehash_projection, REFERENCE, OBSERVED
    from infrastructure.crawler.valuation_history import collect_valuation_history, load_valuation_history
    from infrastructure.runtime.native_source_capture import NativeSourceCapture
    path = database(tmp_path / "history.sqlite3")
    calendar = projection(months=36)
    calendar["days"].update({"2026-05-29": "OPEN", "2026-06-30": "OPEN"})
    rehash_projection(calendar)
    evidence = collect_valuation_history(path, symbols=["2330", "2454"], reference_session=REFERENCE,
        observed_at=OBSERVED, calendar_projection=calendar)
    capture = NativeSourceCapture(tmp_path / "sources", valuation_history_database=path,
        valuation_history_collector=lambda *args, **kwargs: evidence)
    capture.market = "TWSE"
    crawler = SimpleNamespace(stocknumbers=["2330", "2454"], daily_dates=[[REFERENCE], [REFERENCE]])
    capture._collect_valuation_history(crawler, reference_session=REFERENCE)
    current = capture.selections(crawler, cutoff=OBSERVED)
    for symbol in crawler.stocknumbers:
        selected = current[symbol]["valuation_price_history"]
        assert selected["capture_evidence_id"] == evidence["evidence_id"]
        assert "documents" not in selected and "records" not in selected
        restored = load_valuation_history(selected["manifest"], tmp_path / "sources" / "valuation-history")
        assert selected["rows"] == restored["by_symbol"][symbol]["rows"]
    before_at = "2026-07-15T04:59:59Z"
    before = capture.selections(crawler, cutoff=before_at)
    assert all("valuation_price_history" in item for item in before.values())
    with pytest.raises(ValueError, match="SELECTION_INVALID"):
        _independent_prices(before["2330"]["valuation_price_history"], rows=[dict(Date=REFERENCE, Close=1.)],
            symbol="2330", cutoff=before_at, source={})


@pytest.mark.parametrize("qs,expected", [(80., "BUY"), (60., "BUY_SMALL")])
def test_r12_rules_emit_positive_decisions_only_when_declared_synthetic_conditions_pass(qs, expected):
    from dataclasses import asdict
    from domain.les_v7.contracts import ReferencePlan
    from domain.les_v7.rule_baseline import score_baseline, select_baseline_context
    from tests.strategies.test_v7_rule_baseline import unit_features
    from tests.strategies.test_v7_entry_policy_r9 import quality, valuation, gates
    f = unit_features()
    prepared = dict(policy=baseline_policy(version=CLOSE_BASELINE_POLICY), features=f,
        selection=select_baseline_context(f, policy_version=CLOSE_BASELINE_POLICY).record())
    plan = ReferencePlan("REFERENCE_H20", "SYNTHETIC_ONLY", "a" * 64, "REVERSAL",
        101., 99., 109., 2., (101., 101.5), 1, .1, .1)
    receipt = valuation(1.15)
    result = score_baseline(prepared, plan=plan, quality=quality(qs), gates=gates(receipt),
        native_features=dict(close=101., spread=.003, adv=100_000_000.), risks=[],
        standard=True, starter_gate=True, confirmed_gate=True, valuation_receipt=receipt,
        entry_binding=dict(symbol="2330", cutoff=CLOSE_AVAILABLE_AT))
    assert result["decision"] == expected
    assert result["score"]["final_score"] == 84.
    assert result["entry_admission"]["state"] == "ALLOWED"
    # Unit-level conditions are not real original-source certification,
    # account-bound PAE permission, broker authority, fills or profits.
    assert asdict(plan)["stop"] == 99. and asdict(plan)["target"] == 109.

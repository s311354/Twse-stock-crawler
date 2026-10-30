"""SYNTHETIC ONLY: registered policy/source boundaries, not repaired evidence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pandas as pd
import pytest

from application.evidence_context import ReadOnlyEvidence
from application.les_v7 import analyze_request, prepare_quality_context
from application.low_entry_v7_daily import daily_report, project_report
from application.market_history import MarketHistoryAssembler
from domain.financial_taxonomy import TAXONOMY_MAPPING_VERSION
from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.les_v7.contracts import identity
from domain.les_v7.engine import evaluate
from domain.les_v7.financial_sources import audit_proof, financial_supplement, FINANCIAL_MAPPING, PAYMENT_POLICY
from domain.les_v7.quality_policy import (
    BUILTIN_QUALITY_POLICY, R1_PROFILE, R2_PROFILE, r2_quality_policy as default_quality_policy,
    native_financial_safety, quality_freshness, validate_quality_policy,
)
from domain.strategies.low_entry_v6_filings import ixbrl_record
from tests.strategies.test_daily_v7_pipeline import synthetic_market
from tests.strategies.test_low_entry_score_v7 import CUTOFF, POLICY, bars, bundle, filings


def native_original(opinion="UnqualifiedOpinionAbstract", *, audit=True):
    """Run the actual native mapper over typed synthetic original contexts."""
    end, start = "2026-06-30", "2026-01-01"
    instant = {
        "ifrs-full:Assets": 100000., "ifrs-full:CurrentAssets": 50000.,
        "ifrs-full:Liabilities": 20000., "ifrs-full:CurrentLiabilities": 10000.,
        "ifrs-full:Equity": 80000., "ifrs-full:CashAndCashEquivalents": 20000.,
        "ifrs-full:Borrowings": 5000., "ifrs-full:CurrentLeaseLiabilities": 0.,
        "ifrs-full:NoncurrentLeaseLiabilities": 0.,
        "tifrs-SCF:CashAndCashEquivalentsAtBeginningOfPeriod": 16000.,
        "tifrs-SCF:CashAndCashEquivalentsAtEndOfPeriod": 20000.,
    }
    duration = {
        "ifrs-full:Revenue": 20000., "ifrs-full:CostOfRevenue": 8000.,
        "ifrs-full:ProfitLossFromOperatingActivities": 6000., "ifrs-full:ProfitLoss": 4000.,
        "ifrs-full:CashFlowsFromUsedInOperatingActivities": 5000.,
        "ifrs-full:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": 1000.,
        "ifrs-full:IncomeTaxExpense": 500., "ifrs-full:ProfitLossBeforeTax": 5000.,
        "tifrs-SCF:NetCashFlowsFromUsedInInvestingActivities": -1000.,
        "tifrs-SCF:CashFlowsFromUsedInFinancingActivities": 0.,
        "ifrs-full:EffectOfExchangeRateChangesOnCashAndCashEquivalents": 0.,
        "ifrs-full:IncreaseDecreaseInCashAndCashEquivalents": 4000.,
    }
    facts = [FiledFact(concept, value * 1000, None, None, None, end,
        unit="iso4217:TWD", context_id="SYNTHETIC_ONLY_INSTANT") for concept, value in instant.items()]
    facts += [FiledFact(concept, value * 1000, None, start, end, None,
        unit="iso4217:TWD", context_id="SYNTHETIC_ONLY_YTD") for concept, value in duration.items()]
    facts += [FiledFact("ifrs-full:BasicEarningsLossPerShare", 1., None, start, end, None,
        unit="iso4217:TWD/xbrli:shares", context_id="SYNTHETIC_ONLY_EPS")]
    if audit:
        facts += [FiledFact("tifrs-ar:" + opinion, None, "Y", None, end, None,
                            context_id="SYNTHETIC_ONLY_CURRENT_AUDIT")]
    filing = FiledFinancials(2026, 2, "2026-08-14T10:00:00+08:00", 1, tuple(facts),
        content_sha256=identity(dict(synthetic_facts=[str(f) for f in facts])),
        source_url="https://mops.twse.com.tw/SYNTHETIC_ONLY_ORIGINAL",
        period_end=end, taxonomy_schema_refs=("https://mops.twse.com.tw/tifrs-ci-cr-2026-03-31.xsd",))
    row = ixbrl_record("SYNTHETIC_ONLY", filing)
    row["audit_evidence"] = audit_proof(filing, symbol=row["symbol"], original_verified=True)
    return row, filing


def financial_proof(row):
    return native_financial_safety(row, symbol=row["symbol"], market="TWSE", cutoff=CUTOFF,
                                  policy=default_quality_policy())


def supplement_original():
    _, filing = native_original()
    end, start = "2026-06-30", "2026-01-01"
    extra = {"ifrs-full:EquityAttributableToOwnersOfParent": (78000., True),
        "ifrs-full:ProfitLossAttributableToOwnersOfParent": (3900., False),
        "tifrs-notes:InterestExpense_n": (200., False),
        "ifrs-full:PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities": (100., False),
        "tifrs-bsci-ci:AccountsReceivableNet": (1000., True), "ifrs-full:Inventories": (3000., True),
        "ifrs-full:DividendsPaidClassifiedAsFinancingActivities": (0., False),
        "ifrs-full:PurchaseOfTreasuryShares": (0., False)}
    additional = tuple(FiledFact(concept, value * 1000., None, None if instant else start,
        None if instant else end, end if instant else None, unit="iso4217:TWD", context_id="SYNTHETIC_ONLY_EXTRA")
        for concept, (value, instant) in extra.items())
    filing = replace(filing, facts=filing.facts + additional,
                     content_sha256=identity(dict(synthetic_extra=[str(f) for f in filing.facts + additional])))
    row = ixbrl_record("SYNTHETIC_ONLY", filing)
    return row, financial_supplement(filing, symbol=row["symbol"], original_verified=True)


def test_registered_explicit_r2_is_stable_detached_and_independent_of_training():
    a, b = default_quality_policy(), default_quality_policy()
    assert a == b and a is not b
    assert a["latest_required_period"] == "LATEST_VERIFIED_PUBLISHED_PERIOD"
    assert a["profile_id"] == R2_PROFILE
    assert a["overdue_filing_verification"] == "NOT_VERIFIED_NO_OFFICIAL_SCHEDULE_EVIDENCE"
    assert "model_bundle" not in a
    assert a["policy_sha256"] == identity({k: v for k, v in a.items() if k != "policy_sha256"})
    with pytest.raises(FrozenInstanceError): BUILTIN_QUALITY_POLICY.policy_id = "tampered"
    a["source_mapping_id"] = "tampered"
    assert default_quality_policy() == b


@pytest.mark.parametrize("key", ["policy_id", "source_mapping_id", "source_mapping_sha256", "policy_sha256",
                               "freshness_max_age_days", "filing_schedule_version", "capex_rule"])
def test_registered_policy_rejects_unregistered_or_changed_mapping(key):
    policy = default_quality_policy()
    policy[key] = "SYNTHETIC_TAMPER"
    with pytest.raises(ValueError): validate_quality_policy(policy, CUTOFF, market="TWSE")


def test_legacy_policy_is_unchanged_and_no_implicit_domain_upgrade():
    policy = validate_quality_policy(POLICY, CUTOFF, market="TWSE")
    assert policy == POLICY and policy is not POLICY
    context = dict(symbol="SYNTHETIC_ONLY", market="TWSE", as_of=CUTOFF, filings=filings())
    raw = evaluate(bars(), context)
    assert raw["diagnostics"]["quality"]["profile_id"] == R1_PROFILE
    assert "quality_extension_schema" not in raw


def test_period_resolves_cutoff_visible_original_not_today_or_deadline():
    row, _ = native_original()
    future = deepcopy(row)
    future.update(period_end="2026-09-30", available_at="2026-11-14T10:00:00+08:00")
    policy = default_quality_policy([row, future], symbol=row["symbol"], market="TWSE", cutoff=CUTOFF)
    assert policy["latest_required_period"] == "2026Q2"
    assert policy["period_selection_status"] == "VERIFIED_PUBLISHED_ORIGINAL_PERIOD"
    assert policy["policy_sha256"] == default_quality_policy()["policy_sha256"]
    at_release = "2026-08-14T09:59:00+08:00"
    pending = default_quality_policy([row], symbol=row["symbol"], market="TWSE", cutoff=at_release)
    assert pending["latest_required_period"] is None


def test_unverified_newest_catalog_period_never_falls_back_to_older_original():
    row, _ = native_original()
    pending = dict(row, period_end="2026-09-30", available_at="2026-10-10T10:00:00+08:00", field_evidence={})
    result = default_quality_policy([row, pending], symbol=row["symbol"], market="TWSE", cutoff=CUTOFF)
    assert result["latest_required_period"] is None
    assert result["period_selection_status"] == "LATEST_PERIOD_ORIGINAL_UNVERIFIED"
    assert result["period_selection_reasons"] == ["LATEST_PERIOD_ORIGINAL_UNVERIFIED:2026Q3"]


def test_explicit_resolved_period_cannot_cross_cutoff_or_source_selection():
    row, _ = native_original()
    wrong = dict(default_quality_policy(), latest_required_period="2026Q3")
    with pytest.raises(ValueError, match="REQUIRED_PERIOD_MISMATCH"):
        validate_quality_policy(wrong, CUTOFF, market="TWSE", records=[row], symbol=row["symbol"])
    with pytest.raises(ValueError, match="FUTURE_REQUIRED_PERIOD"):
        validate_quality_policy(dict(wrong, latest_required_period="2026Q4"), CUTOFF)


def test_native_financial_proof_uses_actual_mapper_cash_anchor_and_debt_aggregate():
    row, _ = native_original()
    before = deepcopy(row)
    proof = financial_proof(row)
    assert proof["state"] == "PASS", proof
    assert proof["gate_id"] == "FINANCIAL_SAFETY"
    assert proof["input_ids"] == [row["content_sha256"]]
    assert proof["source_available_at"] == "2026-08-14T02:00:00+00:00"
    assert row == before
    assert row["field_evidence"]["cash_beginning"]["basis"] == "CASH_FLOW_ANCHOR"
    assert row["field_evidence"]["cash_beginning"]["reported_instant"] == row["period_end"]
    assert row["field_evidence"]["interest_bearing_debt"]["used_fields"] == [
        "borrowings", "current_lease_liabilities", "noncurrent_lease_liabilities"]


@pytest.mark.parametrize("context_date", ["2026-06-30", "2026-08-11"])
def test_wrong_scope_report_body_cannot_certify_native_financial_safety(context_date):
    row, filing = native_original()
    facts = tuple(replace(f, end_date=context_date) if f.concept.startswith("tifrs-ar:") else f for f in filing.facts)
    body = FiledFact("tifrs-ar:AccountantsReportBody", None,
        "會計師核閱報告：民國115年6月30日之個體財務報表。本公司另編製合併財務報表。",
        None, context_date, None)
    source = replace(filing, facts=(*facts, body))
    row["audit_evidence"] = audit_proof(source, symbol=row["symbol"], original_verified=True)
    assert row["audit_evidence"]["status"] == "PENDING_VERIFICATION"
    assert financial_proof(row)["state"] == "UNKNOWN"


@pytest.mark.parametrize("damage", ["audit_missing", "audit_future", "audit_wrong_source", "audit_prior_context",
    "anchor_missing", "anchor_fake_duration", "anchor_wrong_window", "cash_source", "debt_incomplete"])
def test_native_financial_proof_never_promotes_missing_or_unbound_source(damage):
    row, _ = native_original()
    row["Quality Gate"] = row["safety_status"] = "PASS"  # Explicit old flags must have no authority.
    if damage == "audit_missing": row.pop("audit_evidence")
    elif damage == "audit_future": row["audit_evidence"]["available_at"] = "2026-10-13T00:00:00Z"
    elif damage == "audit_wrong_source": row["audit_evidence"]["content_sha256"] = "f" * 64
    elif damage == "audit_prior_context": row["audit_evidence"]["contexts"][0]["end_date"] = "2025-06-30"
    elif damage == "anchor_missing": row["field_evidence"].pop("cash_beginning")
    elif damage == "anchor_fake_duration": row["field_evidence"]["cash_beginning"]["basis"] = "YTD"
    elif damage == "anchor_wrong_window": row["field_evidence"]["cash_beginning"]["cash_flow_period_start"] = "2026-04-01"
    elif damage == "cash_source": row["field_evidence"]["cash"]["taxonomy_mapping_version"] = "unregistered"
    else: row["field_evidence"]["interest_bearing_debt"]["used_fields"] = []
    assert financial_proof(row)["state"] == "UNKNOWN"


def test_confirmed_adverse_audit_or_balance_conflict_is_hard_fail_not_data_gap():
    adverse, _ = native_original("AdverseOpinionAbstract")
    assert financial_proof(adverse)["state"] == "FAIL"
    row, _ = native_original()
    row["total_equity"] = 1000.
    row["field_evidence"]["total_equity"]["contexts"][0]["numeric_value"] = 1000000.
    assert financial_proof(row)["state"] == "FAIL"


def test_source_bound_audit_attachment_is_r2_only_and_does_not_mutate_filings():
    row, _ = native_original()
    audit = row.pop("audit_evidence")
    original = ReadOnlyEvidence(row)
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=(original,), quality_policy=default_quality_policy())
    prepare_quality_context(context, financial_audits={row["content_sha256"]: audit})
    assert context["quality_policy"] == default_quality_policy()
    assert context["filings"][0]["audit_evidence"] == audit
    assert "audit_evidence" not in original and "receipts" not in context
    old = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=(original,), quality_policy=POLICY)
    assert prepare_quality_context(old, financial_audits={row["content_sha256"]: "bad"}) is old
    assert old["filings"] == (original,)


def test_audit_attachment_conflict_or_wrong_source_refuses_instead_of_replacing():
    row, _ = native_original()
    proof = deepcopy(row["audit_evidence"])
    proof["source_url"] = "https://mops.twse.com.tw/OTHER_SYNTHETIC_ORIGINAL"
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=[row])
    with pytest.raises(ValueError, match="SOURCE_BINDING_MISMATCH"):
        prepare_quality_context(context, financial_audits={row["content_sha256"]: proof})
    proof["source_url"] = row["source_url"]
    proof["status"] = "PENDING_VERIFICATION"
    with pytest.raises(ValueError, match="SOURCE_CONFLICT"):
        prepare_quality_context(context, financial_audits={row["content_sha256"]: proof})


def test_native_supplement_thin_merge_preserves_core_original_and_source_proofs():
    row, supplement = supplement_original()
    before = deepcopy(row)
    frozen = ReadOnlyEvidence(row)
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=(frozen,))
    prepare_quality_context(context, financial_supplements={row["content_sha256"]: supplement})
    merged = context["filings"][0]
    assert merged["parent_equity"] == 78000. and merged["interest_expense"] == 200.
    assert merged["accounts_receivable"] == 1000.
    assert merged["receivable_basis"] == "REPORTED_ORDINARY_NET_TRADE_EXCLUDING_RELATED_PARTIES"
    assert merged["financial_mapping_version"] == FINANCIAL_MAPPING
    assert merged["payment_policy"] == PAYMENT_POLICY
    assert merged["capex"] == before["capex"]
    assert merged["field_evidence"]["capex"] is frozen["field_evidence"]["capex"]
    assert identity(merged["field_evidence"]["cfo"]) == identity(before["field_evidence"]["cfo"])
    assert row == before and "parent_equity" not in row
    # Re-enter after native freezing: tuple/list are the same JSON proof,
    # rather than a false source conflict caused by container representation.
    context["filings"] = tuple(ReadOnlyEvidence(r) for r in context["filings"])
    prior_hash = identity(context["filings"])
    prepare_quality_context(context, financial_supplements={row["content_sha256"]: supplement})
    assert identity(context["filings"]) == prior_hash


@pytest.mark.parametrize("damage", ["source", "scope", "unit", "future", "mapping", "extra_field", "value", "proof"])
def test_supplement_source_policy_or_existing_value_conflicts_refuse(damage):
    row, supplement = supplement_original()
    if damage == "source": supplement["content_sha256"] = "a" * 64
    elif damage == "scope": supplement["scope"] = "individual"
    elif damage == "unit": supplement["unit"] = "yuan"
    elif damage == "future": supplement["available_at"] = "2026-10-13T00:00:00Z"
    elif damage == "mapping": supplement["financial_mapping_version"] = "not_registered"
    elif damage == "extra_field": supplement["fields"]["cfo"] = 1e9
    elif damage == "value": row["parent_equity"] = 1.
    else: row["field_evidence"]["parent_equity"] = dict(supplement["field_evidence"]["parent_equity"], reason="OTHER_PROOF")
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=[row])
    with pytest.raises(ValueError, match="FINANCIAL_SUPPLEMENT"):
        prepare_quality_context(context, financial_supplements={row["content_sha256"]: supplement})


def test_unverified_supplement_keeps_gap_and_explicit_r1_does_not_consume_it():
    row, supplement = supplement_original()
    supplement["status"] = "PENDING_VERIFICATION"
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=[row])
    prepare_quality_context(context, financial_supplements={row["content_sha256"]: supplement})
    result = context["filings"][0]
    assert "parent_equity" not in result and "payment_policy" not in result
    assert result["financial_supplement_status"]["status"] == "PENDING_VERIFICATION"
    old = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=[row], quality_policy=POLICY)
    prepare_quality_context(old, financial_supplements={row["content_sha256"]: "invalid"})
    assert old["filings"] == [row]


def test_explicit_public_r2_quality_readiness_does_not_manufacture_market_receipts_or_model():
    row, _ = native_original()
    payload = dict(bars=bars().to_dict("records"), context=dict(
        symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=[row], quality_policy=default_quality_policy()))
    before = deepcopy(payload)
    report = analyze_request(payload)["reports"][0]
    assert report["diagnostics"]["quality"]["profile_id"] == R2_PROFILE
    assert report["diagnostics"]["quality"]["financial_safety"]["state"] == "PASS"
    assert report["diagnostics"]["readiness"]["model"]["state"] == "NOT_READY"
    assert {r["state"] for r in report["diagnostics"]["gate_receipts"]} == {"UNKNOWN"}
    assert report["envelope"]["score"]["final_score"] is None
    assert project_report(report)["Quality Gate"] == "DATA_INSUFFICIENT"
    assert payload == before


def test_daily_entry_retains_explicit_registered_r2_and_precise_projection():
    row, _ = native_original()
    market = replace(synthetic_market(), les_v7_bundle_json=None,
        fundamental_snapshots={row["symbol"]: {"sector": "24", "les_v6_filings": [row],
            "les_v7_context": {"quality_policy": default_quality_policy()}}})
    report = daily_report(MarketHistoryAssembler().build(market, 0))
    assert report["diagnostics"]["quality"]["profile_id"] == R2_PROFILE
    assert project_report(report)["Quality Gate"] == report["diagnostics"]["quality"]["status"]
    assert report["pae_authorization"]["authorized"] is False


def test_r2_public_cli_and_kernel_share_exact_report(tmp_path, capsys):
    from interface.les_v7 import main
    row, _ = native_original()
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=[row], quality_policy=default_quality_policy())
    raw = bars(90)
    explicit = dict(context, quality_policy=default_quality_policy())
    kernel = evaluate(raw, explicit)
    payload = dict(bars=raw.to_dict("records"), context=context)
    expected = analyze_request(payload)
    assert expected["reports"][0] == kernel
    path = tmp_path / "SYNTHETIC_ONLY_request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "SYNTHETIC_ONLY_result.json"
    assert main([str(path), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert json.loads(output.read_text()) == expected


def test_r2_daily_and_public_full_business_report_match_and_stale_fail_closed():
    row, _ = native_original()
    market = replace(synthetic_market(), les_v7_bundle_json=None,
        fundamental_snapshots={row["symbol"]: {"sector": "24", "les_v6_filings": [row],
            "les_v7_context": {"quality_policy": default_quality_policy()}}})
    history = MarketHistoryAssembler().build(market, 0)
    raw = history.reindex(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Trades", "Amount",
                                   "Bid", "Ask", "PE", "TAIEX Close"]).copy()
    raw.attrs = {}
    for field in ("Open", "High", "Low", "Close", "Bid", "Ask"): raw[field] /= 1000.
    context = dict(symbol=row["symbol"], market="TWSE", as_of=CUTOFF, filings=[row], model_bundle=None,
                   external_evidence_ids=[], receipts={}, quality_policy=default_quality_policy())
    expected = analyze_request(dict(bars=raw.to_dict("records"), context=context))["reports"][0]
    actual = daily_report(history)
    assert actual.pop("pae_authorization")["authorized"] is False
    assert actual["diagnostics"].pop("daily_input_errors") == []
    assert actual == expected
    history.attrs["les_v6_analysis_session"] = "2026-10-13"
    stale = daily_report(history)
    assert stale["diagnostics"]["daily_input_errors"] == ["STALE_MARKET_DATA"]
    assert stale["envelope"]["strategy"]["strategy_decision"] in {"DATA_INSUFFICIENT", "NO_BUY"}
    assert stale["diagnostics"]["readiness"]["execution"]["state"] == "BLOCKED"
    assert stale["diagnostics"]["readiness"]["execution"]["order_authority"] is False


def test_explicit_r1_bundle_report_retains_original_business_and_identity():
    context = dict(symbol="SYNTHETIC_ONLY", market="TWSE", as_of=CUTOFF, filings=filings(),
        model_bundle=bundle(), quality_policy=deepcopy(POLICY), receipts={})
    expected = evaluate(bars(), context)
    assert analyze_request(dict(bars=bars().to_dict("records"), context=context))["reports"][0] == expected
    assert "quality_extension_schema" not in expected


@pytest.mark.parametrize("age,expected", [(180, 100.), (181, 100. * 94 / 95), (275, 0.), (276, 0.)])
def test_period_freshness_policy_is_explicit_not_today_schedule(age, expected):
    cutoff = (pd.Timestamp("2026-06-30", tz="Asia/Taipei") + pd.Timedelta(days=age)).isoformat()
    actual_age, freshness = quality_freshness("2026Q2", cutoff, default_quality_policy())
    assert actual_age == age  # Period dates/freshness use the Taiwanese calendar.
    _, at_noon = quality_freshness("2026Q2", (pd.Timestamp(cutoff) + pd.Timedelta(hours=12)).isoformat(), default_quality_policy())
    assert at_noon == pytest.approx(expected)

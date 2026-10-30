"""SYNTHETIC ONLY: native evidence contracts, not official source completeness."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pandas as pd
import pytest

from application.evidence_context import ReadOnlyEvidence
from domain.les_v7.contracts import GateReceipt, ReferencePlan, canonical, identity, timestamp
from domain.les_v7.decision import make_plan, vetoes
from domain.les_v7.features import compute_features
from domain.les_v7.native_evidence import (
    SOURCE_SCHEMA, REQUIRED_GATES, produce_native_receipts, validate_native_receipts,
)
from domain.trading_restrictions import CURRENT_RESTRICTION_SCOPE, RESTRICTION_ENDPOINTS
from tests.strategies.test_low_entry_score_v7 import bars, bundle, CUTOFF


def inputs():
    data = bars(90)
    data["PE"] = 12.
    f, _ = compute_features(data)
    rows = data.astype(object).where(pd.notna(data), None).to_dict("records")
    for row in rows:
        row["Date"] = pd.Timestamp(row["Date"]).date().isoformat()
    base = dict(symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=timestamp(CUTOFF).isoformat(),
        bars_hash=identity(rows), financials_hash=identity([]))
    quality = dict(core_inputs_pass=True)
    reference_policy = bundle()["payload"]["reference_policy"]
    plan = asdict(make_plan("REVERSAL", f, reference_policy, identity(base)))
    bindings = {gate: base for gate in REQUIRED_GATES}
    bindings["VALUATION"] = dict(base, price=f["close"], quality_result_id=identity(quality))
    bindings["STRESS_RISK"] = dict(base, plan=plan)
    packet = dict(schema=SOURCE_SCHEMA, symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=base["cutoff"],
        expected_session="2026-10-12", historical_replay=False, market_rows=rows,
        market_evidence=dict(symbol="SYNTHETIC_ONLY", market="TWSE", session_date="2026-10-12", source="TWSE",
            source_available_at=base["cutoff"], source_final=True, price_unit="TWD", volume_unit="shares",
            amount_unit="TWD", bars_sha256=identity(rows), source_record_ids=[identity(rows)],
            evidence_origin="SYNTHETIC_ONLY_CAPTURE"),
        restriction_evidence=dict(source="TWSE_OPENAPI", session_date="2026-10-12", observation_date="2026-10-12",
            scope=CURRENT_RESTRICTION_SCOPE, available_at=base["cutoff"], restricted=False,
            endpoints=list(RESTRICTION_ENDPOINTS)), valuation_filings=[])
    return dict(symbol="SYNTHETIC_ONLY", market="TWSE", cutoff=base["cutoff"], bars=data,
        features=f, quality=quality, plan=plan, risks=vetoes(f), bindings=bindings, source_packet=packet,
        reference_policy=reference_policy)


def bound(item, gate):
    result = produce_native_receipts(**item)
    return GateReceipt.bind(result[gate], gate_id=gate, inputs=item["bindings"][gate], cutoff=item["cutoff"])


def test_native_receipts_are_derived_not_copied_and_inputs_are_immutable():
    item = inputs()
    original = canonical(item["source_packet"])
    item["source_packet"] = ReadOnlyEvidence(item["source_packet"])
    result = produce_native_receipts(**item)
    assert set(result) == set(REQUIRED_GATES)
    assert bound(item, "DATA_INTEGRITY").state == "PASS"
    assert bound(item, "TRADING_STATUS").state == "PASS"
    assert bound(item, "SAFETY").state in {"PASS", "FAIL"}
    assert bound(item, "STRESS_RISK").state == "PASS"
    assert bound(item, "VALUATION").state == "UNKNOWN"  # No annual source is invented.
    assert canonical(item["source_packet"]) == original
    assert result["DATA_INTEGRITY"]["proof_record_ids"] != ["PASS"]


@pytest.mark.parametrize("changed", [
    {"source_available_at": "2026-10-13T00:00:00Z"}, {"source_final": False},
    {"source_available_at": None}, {"bars_sha256": "0" * 64}, {"volume_unit": "lots"},
    {"market": "TPEX"}, {"symbol": "OTHER"}, {"source_record_ids": []},
])
def test_missing_invalid_future_market_proof_never_passes(changed):
    item = inputs()
    item["source_packet"]["market_evidence"].update(changed)
    for gate in ("DATA_INTEGRITY", "SAFETY", "STRESS_RISK", "VALUATION"):
        assert bound(item, gate).state == "UNKNOWN"


def test_legacy_pass_flags_cannot_supply_native_sources():
    item = inputs()
    item["source_packet"]["market_evidence"] = {}
    item["source_packet"]["restriction_evidence"] = {}
    item["source_packet"].update(data_integrity="PASS", no_veto=True, valuation="PASS", risk="PASS")
    assert all(value is None for value in produce_native_receipts(**item).values())


def test_empty_complete_restrictions_differ_from_missing_or_partial_evidence():
    item = inputs()
    assert bound(item, "TRADING_STATUS").state == "PASS"
    item["source_packet"]["restriction_evidence"]["endpoints"].pop()
    assert bound(item, "TRADING_STATUS").state == "UNKNOWN"
    item["source_packet"]["restriction_evidence"] = {}
    assert bound(item, "TRADING_STATUS").state == "UNKNOWN"
    item = inputs()
    item["source_packet"]["restriction_evidence"]["restricted"] = True
    assert bound(item, "TRADING_STATUS").state == "FAIL"


def test_current_official_final_cache_is_not_original_historical_observation():
    item = inputs()
    item["source_packet"]["historical_replay"] = True
    assert bound(item, "DATA_INTEGRITY").state == "UNKNOWN"
    item["source_packet"]["market_evidence"]["availability_basis"] = "ORIGINAL_OBSERVATION"
    assert bound(item, "DATA_INTEGRITY").state == "PASS"


def test_known_bad_safety_is_fail_but_missing_veto_operand_is_unknown():
    item = inputs()
    item["features"]["support"] = item["features"]["close"] + 100
    item["risks"] = vetoes(item["features"])
    item["plan"] = None
    item["bindings"]["STRESS_RISK"] = dict(item["bindings"]["DATA_INTEGRITY"], plan=None)
    assert bound(item, "SAFETY").state == "FAIL"
    item = inputs()
    item["features"]["new_lows3"] = None
    item["risks"] = vetoes(item["features"])
    assert bound(item, "SAFETY").state == "UNKNOWN"


@pytest.mark.parametrize("key,value,state", [("spread", None, "UNKNOWN"), ("adv", None, "UNKNOWN"),
    ("spread", .02, "FAIL"), ("adv", 1., "FAIL")])
def test_stress_risk_missing_prices_liquidity_or_capacity_never_pass(key, value, state):
    item = inputs()
    item["features"][key] = value
    assert bound(item, "STRESS_RISK").state == state


def annual(year, revenue):
    end, start = f"{year}-12-31", f"{year}-01-01"
    row = dict(symbol="SYNTHETIC_ONLY", market="TWSE", source="MOPS_IXBRL", scope="consolidated",
        currency="TWD", unit="thousand", period_end=end, flow_basis="YTD",
        available_at=f"{year + 1}-03-31T10:00:00+08:00", revision_id=f"SYNTHETIC_ONLY_{year}",
        content_sha256=identity(dict(synthetic_only=True, year=year, revenue=revenue)),
        revenue=revenue, operating_income=revenue * .3, net_income=revenue * .2,
        dividends=100., diluted_shares=100000.)
    proofs = {}
    for field in ("revenue", "operating_income", "net_income", "dividends", "diluted_shares"):
        shares = field == "diluted_shares"
        proofs[field] = dict({key: row[key] for key in ("symbol", "source", "scope", "currency", "content_sha256",
            "revision_id", "available_at")}, status="RESOLVED", basis="YTD", start_date=start, end_date=end,
            unit="shares" if shares else "thousand", contexts=[dict(context_id="SYNTHETIC_ONLY_ANNUAL",
                dimensions=[], start_date=start, end_date=end, unit="xbrli:shares" if shares else "iso4217:TWD",
                numeric_value=row[field] if shares else row[field] * 1000)])
    row["field_evidence"] = proofs
    return row


def test_valuation_reuses_scenario_arithmetic_only_after_original_source_checks():
    item = inputs()
    item["source_packet"]["valuation_filings"] = [annual(2023, 20000.), annual(2024, 24000.), annual(2025, 28800.)]
    assert bound(item, "VALUATION").state == "PASS"
    item["source_packet"]["valuation_filings"][-1]["field_evidence"]["dividends"]["contexts"][0]["unit"] = "xbrli:shares"
    # An invalid latest original cannot silently choose older annual earnings.
    assert bound(item, "VALUATION").state == "UNKNOWN"
    item["source_packet"]["valuation_filings"] = [annual(2025, 28800.)]
    item["source_packet"]["valuation_filings"][0]["field_evidence"].pop("dividends")
    assert bound(item, "VALUATION").state == "UNKNOWN"


def test_source_rows_and_native_binding_are_not_interchangeable():
    item = inputs()
    item["source_packet"]["market_rows"][-1]["Close"] += .1
    with pytest.raises(ValueError, match="MARKET_ROWS_BINDING"):
        produce_native_receipts(**item)


def test_future_and_conflicting_annual_versions_do_not_supply_missing_proof():
    item = inputs()
    years = [annual(2023, 20000.), annual(2024, 24000.), annual(2025, 28800.)]
    item["source_packet"]["valuation_filings"] = years
    before = produce_native_receipts(**item)["VALUATION"]
    future = annual(2025, 999999.)
    future["available_at"] = "2026-10-13T00:00:00Z"
    item["source_packet"]["valuation_filings"].append(future)
    assert produce_native_receipts(**item)["VALUATION"] == before
    conflict = annual(2025, 999999.)
    item["source_packet"]["valuation_filings"].append(conflict)
    assert bound(item, "VALUATION").state == "UNKNOWN"


def test_daily_native_adapter_uses_strategy_policy_without_fitted_model_or_fake_pass():
    from application.low_entry_v7_daily import daily_report
    from application.market_history import MarketHistoryAssembler
    from domain.les_v7.quality_policy import default_quality_policy
    from domain.les_v7.strategy_policy import make_strategy_policy
    from tests.strategies.test_daily_v7_pipeline import synthetic_market
    from tests.strategies.test_v7_reported_quality import source_records
    model = bundle()["payload"]
    policy = make_strategy_policy(policy_version="SYNTHETIC_ONLY", available_at="2026-01-03T00:00:00Z",
        quality_policy=default_quality_policy(),
        **{key: model[key] for key in ("context_rules", "trigger_rules", "reference_policy")})
    originals = source_records()
    market = replace(synthetic_market(), stock_numbers=["2330"], les_v7_bundle_json=None,
        signal_dates={"2330": CUTOFF},
        fundamental_snapshots={"2330": {"sector": "24", "les_v6_filings": originals,
                                      "les_v6_mops_filings": originals}})
    history = MarketHistoryAssembler().build(market, 0)
    history.attrs.update(les_v7_native_evidence=True, les_v7_strategy_policy_json=json.dumps(policy),
        les_v7_market_evidence=dict(symbol="2330", market="TWSE", session_date="2026-10-12", source="TWSE",
            source_available_at=timestamp(CUTOFF).isoformat(), source_final=True,
            price_unit="TWD", volume_unit="shares", amount_unit="TWD", availability_basis="CURRENT_OBSERVATION",
            evidence_origin="SYNTHETIC_ONLY_CAPTURE"),
        trading_restriction=inputs()["source_packet"]["restriction_evidence"])
    report = daily_report(history)
    gates = {row["gate_id"]: row["state"] for row in report["diagnostics"]["gate_receipts"]}
    assert gates["DATA_INTEGRITY"] == gates["TRADING_STATUS"] == "PASS"
    assert gates["VALUATION"] == "UNKNOWN"  # No source-bound annual shares/dividend original.
    assert report["envelope"]["quality"]["status"] == "PASS"
    assert report["diagnostics"]["reference_plan"] is not None
    assert report["envelope"]["score"]["final_score"] is None
    assert report["envelope"]["strategy"]["strategy_decision"] == "DATA_INSUFFICIENT"
    assert report["pae_authorization"]["authorized"] is False
    validate_native_receipts(report)


def test_report_recomputes_native_receipts_and_detects_rehashed_pass():
    item = inputs()
    produced = produce_native_receipts(**item)
    report = dict(symbol=item["symbol"], envelope=dict(event=dict(cutoff_at=item["cutoff"])), diagnostics=dict(
        native_evidence=dict(schema="les-v7-native-evidence/v1", source_packet=item["source_packet"],
            source_id=identity(item["source_packet"]), produced_receipts=produced,
            reference_policy=item["reference_policy"]),
        features=item["features"], quality=item["quality"], reference_plan=item["plan"],
        gate_bindings=item["bindings"], gate_receipts=[asdict(GateReceipt.bind(produced[gate], gate_id=gate,
            inputs=item["bindings"][gate], cutoff=item["cutoff"])) for gate in REQUIRED_GATES]))
    validate_native_receipts(report)
    changed = deepcopy(report)
    changed["diagnostics"]["native_evidence"]["produced_receipts"]["STRESS_RISK"]["state"] = "FAIL"
    with pytest.raises(ValueError, match="RECEIPT_RECOMPUTATION"):
        validate_native_receipts(changed)
    changed = deepcopy(report)
    changed["diagnostics"]["features"]["slope20"] = 999.
    with pytest.raises(ValueError, match="FEATURE_RECOMPUTATION"):
        validate_native_receipts(changed)


def test_daily_packet_excludes_valid_future_current_capture_before_hashing():
    from application.low_entry_v7_daily import _native_source_packet
    item = inputs()
    raw = item["bars"]
    context = dict(symbol=item["symbol"], market=item["market"], as_of=item["cutoff"], filings=[])
    history = raw.copy()
    history.attrs = {"les_v6_analysis_session": "2026-10-12"}
    empty = _native_source_packet(history, raw, context)
    future = deepcopy(item["source_packet"]["restriction_evidence"])
    future["available_at"] = "2026-10-13T00:00:00Z"
    history.attrs["trading_restriction"] = future
    future_market = deepcopy(item["source_packet"]["market_evidence"])
    future_market["source_available_at"] = future["available_at"]
    history.attrs["les_v7_market_evidence"] = future_market
    assert identity(_native_source_packet(history, raw, context)) == identity(empty)
    future["available_at"] = "INVALID_UNVERIFIED"
    history.attrs["trading_restriction"] = future
    assert _native_source_packet(history, raw, context)["restriction_evidence"]["available_at"] == "INVALID_UNVERIFIED"


def test_native_packet_cannot_claim_financial_capture_after_evaluation_or_for_replay():
    item = inputs()
    item["source_packet"]["financial_capture_cutoff"] = "2027-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="FINANCIAL_CAPTURE_CUTOFF_MISMATCH"):
        produce_native_receipts(**item)
    item["source_packet"]["financial_capture_cutoff"] = item["cutoff"]
    item["source_packet"]["historical_replay"] = True
    with pytest.raises(ValueError, match="FINANCIAL_CAPTURE_CUTOFF_MISMATCH"):
        produce_native_receipts(**item)

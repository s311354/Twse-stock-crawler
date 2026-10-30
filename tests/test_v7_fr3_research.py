"""SYNTHETIC PORT TESTS ONLY, not a trained FR3 bundle or market efficacy.

FrozenBundle/classifier/prediction/financial/factor ports below are unit-test
doubles. Native bar selection, FR3 source anchors/risk/cost arithmetic, bound
gate receipts, scoring, account validation and full report recomputation remain
real. Separate negative tests use the actual legacy FrozenBundle loader.
"""
from copy import deepcopy

import pandas as pd
import pytest

from application import fr3_research as app
from domain.les_v7.contracts import canonical, identity
from domain.les_v7.features import select_bars
from domain.les_v7.fr3_plan import policy as reference_policy
from domain.les_v7.fr3_policy import AVAILABLE_AT, eps_policy
from domain.les_v7.localization import Forecast
from domain.les_v7.native_evidence import _receipt, produce_native_receipts as native_producer
from domain.les_v7.quality_policy import r6_quality_policy
from tests.test_v7_fr3_plan import reference as old_reference
from tests.test_v7_plan_r16 import synthetic_calendar
from tests.test_v7_plan_r18 import neutral_bars


CUTOFF = "2026-10-24T06:00:00+00:00"


def _record(value):
    return dict(payload=value, sha256=identity(value))


class SyntheticFrozenPort:
    """Explicit mock: does NOT validate OOF/training or claim a real model."""
    def __init__(self, value):
        self.payload = deepcopy(value["payload"])
        self.sha256 = value["sha256"]

    @classmethod
    def load(cls, value):
        if value["sha256"] != identity(value["payload"]):
            raise ValueError("SYNTHETIC_TEST_PORT_HASH_MISMATCH")
        assert value["payload"].get("SYNTHETIC_PORT_TEST_ONLY") is True
        return cls(value)

    def record(self):
        return deepcopy(self.payload)


@pytest.fixture
def research_request(monkeypatch):
    data = neutral_bars()
    data["Date"] = pd.bdate_range(end="2026-10-23", periods=len(data)).strftime("%Y-%m-%d")
    # Known stronger current mark than the fixed conditional limit. This is a
    # deliberately synthetic exposure-cap check, not an edited real quote.
    data.loc[len(data)-1, ["Open", "High", "Low", "Close", "Bid", "Ask"]] = [104., 105., 103., 104., 103.9, 104.]
    data["Volume"] = 1_000_000.
    data["Amount"] = data["Close"] * data["Volume"]
    universe = [["2330", "TWSE"]]
    reference = reference_policy(old_reference(), available_at=AVAILABLE_AT)
    model = dict(SYNTHETIC_PORT_TEST_ONLY=True, reference_policy=reference, quality_policy=r6_quality_policy(),
        training_pipeline_schema="LES_V7_OFFLINE_TRAINING_R1", runtime_provenance_schema="LES_V7_RUNTIME_PROVENANCE_R2",
        research_only=True, approved_for_live=False, trained_until="2026-10-24T05:30:00+00:00",
        training_manifest=dict(full_universe_sha256=identity(universe), dataset_kind="SYNTHETIC_TEST_ONLY"),
        context_rules={"REVERSAL": dict(field="close", op="lt", value=0.),
            "TREND_PULLBACK": dict(field="close", op="gt", value=0.),
            "LEADERSHIP_CONTINUATION": dict(field="close", op="lt", value=0.)},
        feature_names=["slope5", "dfa512"], feature_bounds=[[-10., 10.], [0., 3.]],
        weights={name:[1/6]*6 for name in ("REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION")})
    rows = select_bars(data, CUTOFF).astype(object).where(pd.notna(data), None).to_dict("records")
    for row in rows: row["Date"] = str(row["Date"])[:10]
    calendar = synthetic_calendar(data)
    calendar["days"].update({str(day.date()): "OPEN" if day.weekday() < 5 else "CLOSED"
        for day in pd.date_range("2026-10-21", "2026-10-24")})
    packet = dict(schema="les-v7-native-source/v1", dataset_kind="SYNTHETIC_TEST_ONLY",
        symbol="2330", market="TWSE", cutoff=CUTOFF, filings=[], market_rows=rows,
        quote_market_rows=rows, expected_session="2026-10-23",
        calendar_evidence=calendar,
        market_evidence=dict(symbol="2330", market="TWSE", source="TWSE_DAILY",
            session_date="2026-10-23", price_unit="TWD", volume_unit="shares", amount_unit="TWD",
            source_final=True, bars_sha256=identity(rows), source_record_ids=[identity(rows), "SYNTHETIC_TEST_ONLY"],
            source_available_at="2026-10-23T05:30:00+00:00"))
    item = dict(bars=data.to_dict("records"), context=dict(symbol="2330", market="TWSE", as_of=CUTOFF,
        filings=[], native_evidence_source=packet, external_evidence_ids=["SYNTHETIC_TEST_ONLY"]),
        account=dict(value=1_000_000., available_cash=1_000_000., intended_weight=.25,
            capacity_weight=.25, positions=[], as_of=CUTOFF))
    monkeypatch.setattr(app, "FrozenBundle", SyntheticFrozenPort)
    from domain.les_v7 import training
    monkeypatch.setattr(training, "select_context", lambda *_a: ("TREND_PULLBACK", ("TREND_PULLBACK",)))
    monkeypatch.setattr(app, "evaluate_financials", lambda *_a, **_kw:
        dict(status="PASS", quality_score=80., SYNTHETIC_PORT_TEST_ONLY=True))

    def forecast(_model, _features, plan, **_options):
        if plan is None:
            return Forecast("NOT_READY", reasons=("MODEL_OR_REFERENCE_PLAN_NOT_READY",))
        return Forecast("VALID", quantiles=tuple(plan.entry-x*plan.atr for x in (.3, .2, .1)),
            timing=tuple([.2]*4+[.2/17]*17), p_fill=.8, p_rebound=.8, p_entry=.7,
            calibration_version="SYNTHETIC_PORT_NOT_A_TRAINED_CALIBRATOR")
    monkeypatch.setattr(app, "predict", forecast)
    monkeypatch.setattr(app, "factor_vector", lambda *_a, **_kw:
        ((.9,)*6, dict(SYNTHETIC_PORT_TEST_ONLY=True)))
    return dict(schema=app.REQUEST_SCHEMA, universe=universe, model_bundle=_record(model), securities=[item])


def allow_synthetic_gate_ports(monkeypatch):
    """Bound test receipts only; never a claim of verified market/financial data."""
    def produce(**options):
        result = {}
        for gate, binding in options["bindings"].items():
            receipt = _receipt(gate, "PASS", binding=binding,
                source=dict(SYNTHETIC_TEST_ONLY=True, symbol=options["symbol"], gate=gate),
                available=options["cutoff"], cutoff=options["cutoff"], reasons=[],
                result=dict(SYNTHETIC_PORT_TEST_ONLY=True), producer_version="SYNTHETIC_PORT_TEST_ONLY")
            if gate == "VALUATION":
                receipt["valuation_diagnostics"] = dict(policy_id=identity(eps_policy()),
                    scenarios=[dict(SYNTHETIC_PORT_TEST_ONLY=True)], gate_checks=dict(
                        BASE_PRICE_CAGR_GTE_REQUIRED_RETURN=True, BASE_FAIR_PRICE_GT_CURRENT_PRICE=True,
                        BEAR_LOSS_LTE_MAXIMUM=True, BEAR_LOSS_LTE_HARD_MAXIMUM=True))
            result[gate] = receipt
        return result
    monkeypatch.setattr(app, "produce_native_receipts", produce)


def first(payload):
    return app.analyze_fr3_request(payload)["reports"][0]


def test_public_request_and_full_reexecution_preserve_inputs_and_never_fill(research_request, monkeypatch):
    allow_synthetic_gate_ports(monkeypatch)
    before = canonical(research_request)
    result = app.analyze_fr3_request(research_request)
    report = result["reports"][0]
    assert canonical(research_request) == before
    assert report["decision"]["strategy_decision"] == "BUY_SMALL", report["decision"]
    assert report["score"]["final_score"] == pytest.approx(90.)
    assert report["execution"]["state"] == "QUOTA_ONLY_UNFILLED"
    assert report["execution"]["indicative_shares"] > 0
    assert not report["execution"]["fill_eligible"] and report["execution"]["filled_shares"] == 0
    assert not report["execution"]["live_order_authorized"]
    assert report["source_authenticity"] == "SOURCE_BINDINGS_VALIDATED_NOT_INDEPENDENTLY_CERTIFIED"
    assert app.validate_fr3_report(report, research_request["securities"][0], research_request["model_bundle"], universe=research_request["universe"])
    assert app.analyze_fr3_request(research_request) == result


@pytest.mark.parametrize("damage", ["score", "forecast", "quality", "gate", "plan", "account", "component", "scope"])
def test_public_validation_recomputes_every_business_surface_not_only_hashes(research_request, monkeypatch, damage):
    allow_synthetic_gate_ports(monkeypatch)
    report = first(research_request)
    if damage == "score": report["score"]["final_score"] += 1
    elif damage == "forecast": report["forecast"]["p_fill"] -= .01
    elif damage == "quality": report["quality"]["quality_score"] += 1
    elif damage == "gate": report["gate_receipts"][0]["state"] = "UNKNOWN"
    elif damage == "plan": report["plan"]["stop"] += .1
    elif damage == "account": report["execution"]["indicative_shares"] += 1
    elif damage == "component": report["components"]["entry"]["component_id"] = "0"*64
    else: report["universe_id"] = "0"*64
    report["report_id"] = identity({k:v for k,v in report.items() if k != "report_id"})
    with pytest.raises(ValueError, match="FR3_REPORT_RECOMPUTATION_MISMATCH"):
        app.validate_fr3_report(report, research_request["securities"][0], research_request["model_bundle"], universe=research_request["universe"])


def test_real_native_producer_keeps_missing_evidence_unknown_even_at_high_score(research_request, monkeypatch):
    monkeypatch.setattr(app, "produce_native_receipts", native_producer)
    report = first(research_request)
    assert report["score"]["final_score"] == pytest.approx(90.)
    assert any(row["state"] == "UNKNOWN" for row in report["gate_receipts"])
    assert report["decision"]["strategy_decision"] not in {"BUY", "BUY_SMALL"}
    assert not report["execution"]["allocation_authorized"]


@pytest.mark.parametrize("flag", ["receipts", "rule_baseline", "strategy_policy"])
def test_external_prebound_gates_or_rule_mode_are_not_accepted(research_request, flag):
    research_request["securities"][0]["context"][flag] = {"SYNTHETIC_UNTRUSTED_PASS": True}
    with pytest.raises(ValueError, match="FR3_NATIVE_SOURCES_REQUIRED"):
        first(research_request)


def test_future_model_is_rejected_before_prediction(research_request, monkeypatch):
    model = deepcopy(research_request["model_bundle"]["payload"])
    model["trained_until"] = "2099-01-01T00:00:00Z"
    research_request["model_bundle"] = _record(model)
    monkeypatch.setattr(app, "predict", lambda *_a, **_kw: pytest.fail("future model predicted"))
    with pytest.raises(ValueError, match="MODEL_OR_POLICY_NOT_AVAILABLE"):
        first(research_request)


def test_legacy_bundle_without_native_training_provenance_is_rejected_unmocked():
    from tests.test_v7_fr2_policy import request as old_request
    item = old_request()
    model = item["context"].pop("model_bundle")
    payload = dict(schema=app.REQUEST_SCHEMA, model_bundle=model, universe=[["2330", "TWSE"]], securities=[item])
    with pytest.raises(ValueError, match="FR3_FROZEN_UNIVERSE_ID_MISMATCH|FR3_FROZEN_PLAN_AND_R6_MODEL_REQUIRED"):
        first(payload)


@pytest.mark.parametrize("damage", ["missing", "wrong", "duplicate", "outside"])
def test_full_frozen_scope_is_required_and_each_security_must_belong(research_request, damage):
    if damage == "missing": research_request.pop("universe")
    elif damage == "wrong": research_request["universe"] = [["OTHER", "TWSE"]]
    elif damage == "duplicate": research_request["universe"] *= 2
    else: research_request["securities"][0]["context"]["symbol"] = "OTHER"
    with pytest.raises(ValueError, match="FR3_.*UNIVERSE"):
        first(research_request)


def test_missing_account_risk_stays_unknown_and_limits_do_not_change_score(research_request, monkeypatch):
    allow_synthetic_gate_ports(monkeypatch)
    original = first(research_request)
    changed = deepcopy(research_request)
    account = changed["securities"][0]["account"]
    account.update(positions=[dict(symbol="OTHER", shares=10)], available_cash=999_000.)
    report = first(changed)
    assert report["score"] == original["score"]
    assert report["decision"] == original["decision"]
    assert report["execution"]["state"] == "UNKNOWN"
    assert "EXISTING_POSITION_RISK_UNKNOWN" in report["execution"]["reason_codes"]
    assert not report["execution"]["allocation_authorized"]
    changed = deepcopy(research_request)
    changed["securities"][0]["account"]["capacity_weight"] = .001
    small = first(changed)
    assert small["score"] == original["score"]
    assert small["execution"]["indicative_shares"] < original["execution"]["indicative_shares"]


def test_current_mark_not_lower_limit_price_bounds_total_exposure(research_request, monkeypatch):
    allow_synthetic_gate_ports(monkeypatch)
    report = first(research_request)
    assert report["plan"]["entry"] < report["features"]["close"]
    assert report["execution"]["indicative_shares"] * report["features"]["close"] <= 10_000.


def test_inference_uses_the_exact_frozen_factor_mapping_parameter(research_request, monkeypatch):
    allow_synthetic_gate_ports(monkeypatch)
    expected = research_request["model_bundle"]["payload"]["reference_policy"]["leadership_rr_min"]
    calls = []
    def factors(*_args, **options):
        calls.append(options["leadership_rr_min"])
        return (.9,)*6, dict(SYNTHETIC_PORT_TEST_ONLY=True)
    monkeypatch.setattr(app, "factor_vector", factors)
    first(research_request)
    assert calls == [expected]


def test_quality_reject_is_no_buy_not_watch_at_high_score(research_request, monkeypatch):
    allow_synthetic_gate_ports(monkeypatch)
    monkeypatch.setattr(app, "evaluate_financials", lambda *_a, **_kw:
        dict(status="REJECT", quality_score=90., SYNTHETIC_PORT_TEST_ONLY=True))
    report = first(research_request)
    assert report["decision"]["strategy_decision"] == "NO_BUY"
    assert not report["execution"]["allocation_authorized"]


def test_valuation_admission_unknown_blocks_even_if_a_future_native_port_says_pass(research_request, monkeypatch):
    allow_synthetic_gate_ports(monkeypatch)
    monkeypatch.setattr(app, "valuation_admission", lambda _value:
        dict(state="UNKNOWN", warning=False, penalty=0., cap_multiplier=0.))
    report = first(research_request)
    assert report["decision"]["strategy_decision"] == "DATA_INSUFFICIENT"
    assert "FR3_VALUATION_ADMISSION_UNKNOWN" in report["decision"]["reason_codes"]


def test_future_bar_does_not_change_selected_business_research(research_request, monkeypatch):
    allow_synthetic_gate_ports(monkeypatch)
    before = first(research_request)
    future = deepcopy(research_request["securities"][0]["bars"][-1])
    future.update(Date="2026-10-26", Open=300., High=301., Low=299., Close=300.)
    research_request["securities"][0]["bars"].append(future)
    after = first(research_request)
    for field in ("features", "feature_audit", "plan", "quality", "forecast", "factor_ledger", "score",
                  "gate_receipts", "produced_receipts", "decision", "execution", "evidence_id"):
        assert after[field] == before[field], field
    assert after["request_item_id"] != before["request_item_id"]


def test_duplicate_security_cutoff_and_supplied_quote_are_refused(research_request):
    research_request["securities"].append(deepcopy(research_request["securities"][0]))
    with pytest.raises(ValueError, match="FR3_DUPLICATE_SECURITY_CUTOFF"):
        first(research_request)
    research_request["securities"].pop()
    research_request["securities"][0]["quote"] = dict(SYNTHETIC_ONLY=True)
    with pytest.raises(ValueError, match="QUOTA_ONLY"):
        first(research_request)

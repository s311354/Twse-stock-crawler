"""Explicit FR3 current-data research composition using the native v7 kernels.

Not a second feature/Quality/model engine, daily deployment, or broker. Frozen-
fit historical prediction comparisons use the separate prepared-row interface;
this public request never backdates publication or accepts prebound PASS gates.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, ROUND_FLOOR
import json

import pandas as pd

from application.les_v7 import prepare_quality_context
from domain.les_v7.contracts import GateReceipt, canonical, identity, timestamp
from domain.les_v7.decision import factor_vector, score_once, vetoes
from domain.les_v7.fr3_plan import PLAN_VERSION, is_fr3_reference, validate_band
from domain.les_v7.fr3_policy import (AVAILABLE_AT, INPUT_VERSION, components, entry_policy,
    eps_policy, execution_policy, valuation_admission)
from domain.les_v7.localization import FrozenBundle, localization_gate, predict
from domain.les_v7.native_evidence import _market_source, _receipt, produce_native_receipts
from domain.les_v7.quality import evaluate_financials
from domain.les_v7.quality_policy import r6_quality_policy
from domain.les_v7.reference_path import estimate_reference_costs
from domain.les_v7.source_policy import source_policy, SIGNED_VERSION
from domain.les_v7.training import prepare_reference
from domain.les_v7.valuation_input_policy import input_policy

REQUEST_SCHEMA = "les-v7-fr3-request/v1"
REPORT_SCHEMA = "les-v7-fr3-report/v1"
RESULT_SCHEMA = "les-v7-fr3-research-result/v1"
REQUIRED_GATES = ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK")


def _plain(value):
    return json.loads(canonical(value))


def _scope_members(model, universe):
    """A full frozen universe, not a caller-selected winner list or hash flag."""
    if (not isinstance(universe, list) or not 1 <= len(universe) <= 2048
            or any(not isinstance(row, list) or len(row) != 2
                or not isinstance(row[0], str) or not row[0] or row[0].strip() != row[0]
                or row[1] not in {"TWSE", "TPEX"} for row in universe)
            or len({tuple(row) for row in universe}) != len(universe)):
        raise ValueError("FR3_EXPLICIT_FROZEN_UNIVERSE_REQUIRED")
    if identity(universe) != model.get("training_manifest", {}).get("full_universe_sha256"):
        raise ValueError("FR3_FROZEN_UNIVERSE_ID_MISMATCH")
    return frozenset(tuple(row) for row in universe)


def _account_assessment(plan, reference, features, account, *, cutoff, decision, warning):
    """Indicative incremental quota only; execution observations are not invented."""
    from domain.les_v7.fr2_execution import _account
    result = dict(schema="les-v7-fr3-account-assessment/v1", state="UNKNOWN",
        supplied_account=_plain(account), account=None, indicative_shares=0,
        allocation_authorized=False, fill_eligible=False, filled_shares=0,
        live_order_authorized=False, price_condition="NEXT_OFFICIAL_OPEN_NOT_ASSESSED",
        reason_codes=[])
    try:
        snapshot = _account(account, cutoff=timestamp(cutoff))
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        snapshot = None
        result["reason_codes"].append("ACCOUNT_INVALID:" + str(error))
    if snapshot is None:
        result["reason_codes"].append("ACCOUNT_INPUTS_UNKNOWN")
        return result
    result["account"] = snapshot
    if any(row["risk_evidence"] is None for row in snapshot["positions"]):
        result["reason_codes"].append("EXISTING_POSITION_RISK_UNKNOWN")
        return result
    if decision != "BUY_SMALL" or plan is None:
        result.update(state="FAIL", reason_codes=["SIGNAL_NOT_EXECUTABLE"])
        return result
    policy = execution_policy(reference_version=reference["version"], available_at=reference["available_at"])
    if policy["version"] != execution_policy()["version"] and plan.context == "LEADERSHIP_CONTINUATION":
        result["price_condition"] = "NEXT_THREE_OFFICIAL_OPENS_NOT_ASSESSED"
    symbols = {row["symbol"] for row in snapshot["positions"]}
    if len(symbols) >= policy["maximum_holdings"] and features["symbol"] not in symbols:
        result.update(state="FAIL", reason_codes=["PORTFOLIO_NO_NEW_SYMBOL_SLOT"])
        return result
    dec = lambda v: Decimal(str(v))
    nav, cash = dec(snapshot["value"]), dec(snapshot["available_cash"])
    entry = dec(plan.band[1])
    # A lower conditional entry is not a lower current marked exposure. Keep
    # the declared total weight cap at least as conservative as the FR2 port.
    exposure_price = max(entry, dec(features["close"]))
    current_weight = current_risk = portfolio_risk = Decimal(0)
    for row in snapshot["positions"]:
        proof, shares = row["risk_evidence"], dec(row["shares"])
        risk = shares * (dec(proof["mark_price"])-dec(proof["stop_price"])+dec(proof["cost_stop_per_share"])) / nav
        portfolio_risk += risk
        if row["symbol"] == features["symbol"]:
            current_weight += shares * max(dec(proof["mark_price"]), exposure_price) / nav
            current_risk += risk
    cap = min(dec(policy["valuation_warning_cap"] if warning else policy["total_symbol_cap"]),
              dec(snapshot["intended_weight"]), dec(snapshot["capacity_weight"]))
    exposure = max(Decimal(0), cap-current_weight) * nav
    risk_budget = min(max(Decimal(0), dec(policy["planned_symbol_nav_risk"])-current_risk),
                      max(Decimal(0), dec(policy["planned_portfolio_nav_risk"])-portfolio_risk)) * nav
    adv = features.get("adv")
    if adv is None or adv <= 0:
        result["reason_codes"] = ["ADV_UNKNOWN"]
        return result
    upper = int(min(exposure/exposure_price, cash/entry, dec(adv)*dec(policy["maximum_adv_fraction"])/entry,
                    risk_budget/(entry-dec(plan.stop))).to_integral_value(rounding=ROUND_FLOOR))

    def costs_for(n):
        return estimate_reference_costs(float(entry), plan.stop, plan.target, n,
                                        reference["cost_path_policy"])

    def affordable(n):
        c = costs_for(n)
        return (dec(n)*entry+dec(c["detail"]["buy_fee"]) <= cash
            and dec(n)*(entry-dec(plan.stop)+dec(c["cost_stop_per_share"])) <= risk_budget)

    # Fees include minima and rounding: do not scale a reference one-share fee.
    lo, hi = 0, max(0, upper)
    while lo < hi:
        mid = (lo+hi+1)//2
        if affordable(mid): lo = mid
        else: hi = mid-1
    shares = lo
    if shares:
        costs = costs_for(shares)
        rr = (dec(plan.target)-entry-dec(costs["cost_up_per_share"])) / (
            entry-dec(plan.stop)+dec(costs["cost_stop_per_share"]))
        if rr < Decimal("1.2"):
            shares = 0
            result["reason_codes"].append("ACTUAL_SHARE_COST_NET_RR_FAIL")
    result.update(state="QUOTA_ONLY_UNFILLED" if shares else "FAIL", indicative_shares=shares,
        allocation_authorized=shares > 0, total_weight_cap=float(cap),
        account_id=identity(snapshot), execution_policy_id=identity(policy))
    if not shares and not result["reason_codes"]:
        result["reason_codes"].append("CASH_LIQUIDITY_OR_RISK_BUDGET_EXHAUSTED")
    return result


def _evaluate(item, model_record, *, universe):
    request = deepcopy(item)
    if request.get("quote") is not None:
        raise ValueError("FR3_CURRENT_REQUEST_IS_QUOTA_ONLY_USE_REFERENCE_PATH_FOR_OFFLINE_FILLS")
    account = request.get("account")
    if not isinstance(account, dict):
        raise ValueError("FR3_EXPLICIT_ACCOUNT_REQUIRED")
    context = request["context"]
    if context.get("receipts") or context.get("rule_baseline") or context.get("strategy_policy"):
        raise ValueError("FR3_NATIVE_SOURCES_REQUIRED_NO_PREBOUND_GATES_OR_RULE_MODE")
    cutoff = timestamp(context["as_of"]).isoformat()
    symbol, market = context["symbol"], context["market"]
    if not isinstance(symbol, str) or not symbol or symbol.strip() != symbol or market not in {"TWSE", "TPEX"}:
        raise ValueError("FR3_EXPLICIT_SECURITY_MARKET_REQUIRED")
    bundle = FrozenBundle.load(model_record)
    model = bundle.record()
    if (symbol, market) not in _scope_members(model, universe):
        raise ValueError("FR3_SECURITY_NOT_IN_FROZEN_UNIVERSE")
    if (not is_fr3_reference(model["reference_policy"])
            or identity(model["quality_policy"]) != identity(r6_quality_policy())
            or model.get("training_pipeline_schema") != "LES_V7_OFFLINE_TRAINING_R1"
            or model.get("runtime_provenance_schema") != "LES_V7_RUNTIME_PROVENANCE_R2"
            or model.get("research_only") is not True or model.get("approved_for_live") is not False):
        raise ValueError("FR3_FROZEN_PLAN_AND_R6_MODEL_REQUIRED")
    admission = entry_policy(reference_version=model["reference_policy"]["version"],
        available_at=model["reference_policy"]["available_at"])
    if max(timestamp(model["trained_until"]), timestamp(AVAILABLE_AT),
            timestamp(admission["available_at"])) > timestamp(cutoff):
        raise ValueError("FR3_CURRENT_MODEL_OR_POLICY_NOT_AVAILABLE_USE_RETROSPECTIVE_COMPARISON")
    context.update(model_bundle=model_record, quality_policy=model["quality_policy"])
    prepare_quality_context(context)
    filings = [r for r in context.get("filings", ()) if timestamp(r["available_at"]) <= timestamp(cutoff)]
    packet = _plain(context.get("native_evidence_source"))
    if not isinstance(packet, dict):
        raise ValueError("FR3_NATIVE_SOURCE_PACKET_REQUIRED")
    if identity(packet.get("filings", [])) != identity(filings):
        raise ValueError("FR3_FINANCIAL_PACKET_BINDING_MISMATCH")
    history = pd.DataFrame(request["bars"])
    for name in ("Symbol", "symbol", "stock_no", "證券代號"):
        if name in history and not history[name].map(str).eq(symbol).all():
            raise ValueError("FR3_BAR_SECURITY_MISMATCH")
    if "Market" in history and not history.Market.eq(market).all():
        raise ValueError("FR3_BAR_MARKET_MISMATCH")
    reference = prepare_reference(symbol, market, history, cutoff, model,
        list(context.get("external_evidence_ids", ())) + [identity(packet)],
        session_evidence=packet.get("calendar_evidence"))
    features, plan = reference["features"], reference["plan"]
    from domain.les_v7.features import select_bars
    bars = select_bars(history, cutoff)
    rows = bars.astype(object).where(pd.notna(bars), None).to_dict("records")
    for row in rows: row["Date"] = str(row["Date"])[:10]
    base = dict(symbol=symbol, market=market, cutoff=cutoff,
                bars_hash=identity(rows), financials_hash=identity(filings))
    quality = evaluate_financials(filings, symbol=symbol, market=market, cutoff=cutoff,
                                 policy=model["quality_policy"])
    binding = dict(SAFETY=base, DATA_INTEGRITY=base, TRADING_STATUS=base,
        VALUATION=dict(base, price=features.get("close"), quality_result_id=identity(quality)),
        STRESS_RISK=dict(base, plan=asdict(plan) if plan else None))
    risks = vetoes(features)
    receipts = produce_native_receipts(symbol=symbol, market=market, cutoff=cutoff,
        bars=bars, features=features, quality=quality, plan=None, risks=risks,
        bindings=binding, source_packet=packet, reference_policy=None,
        valuation_policy=eps_policy(), valuation_input_policy=input_policy(version=INPUT_VERSION),
        source_policy=source_policy(version=SIGNED_VERSION))
    # Native market/identity/restriction/financial producers remain authoritative;
    # only the new plan risk is evaluated by the new versioned plan kernel.
    market_source = _market_source(packet, symbol=symbol, market=market, cutoff=cutoff, rows=rows)
    risk = validate_band(plan, model["reference_policy"])
    spread, adv = features.get("spread"), features.get("adv")
    known = spread is not None and adv is not None and adv > 0
    liquid = known and spread <= .005 and (plan is not None and plan.entry*plan.reference_shares/adv <= .01)
    state = "FAIL" if risk["state"] == "FAIL" or plan is not None and known and not liquid else (
        "PASS" if risk["state"] == "PASS" and liquid else "UNKNOWN")
    receipts["STRESS_RISK"] = (_receipt("STRESS_RISK", state, binding=binding["STRESS_RISK"],
        source=dict(market_source=market_source, reference_policy=model["reference_policy"],
                    plan=asdict(plan) if plan else None),
        available=max(market_source["source_available_at"], model["reference_policy"]["available_at"], key=timestamp),
        cutoff=cutoff, reasons=[] if state == "PASS" else list(risk["reasons"])+["FR3_PLAN_OR_LIQUIDITY_"+state],
        result=dict(plan_risk=risk, liquidity_known=known, liquid=bool(liquid)),
        producer_version=("FR3_SOURCE_BOUND_PLAN_RISK_R1" if model["reference_policy"]["version"] == PLAN_VERSION
                          else "FR3_SOURCE_BOUND_PLAN_RISK_L2")) if market_source is not None else None)
    gates = [GateReceipt.bind(receipts.get(k), gate_id=k, inputs=binding[k], cutoff=cutoff) for k in REQUIRED_GATES]
    forecast = predict(bundle, features, plan, cutoff=cutoff)
    factors, ledger = (factor_vector(reference["context"], features, forecast, plan,
        leadership_rr_min=model["reference_policy"].get("leadership_rr_min"))
        if plan and forecast.status == "VALID" else (None, {}))
    value = valuation_admission(receipts.get("VALUATION"))
    penalties = {"valuation_warning": value["penalty"]} if value["warning"] else {}
    weights = model.get("weights", {}).get(reference["context"])
    score = score_once(factors, weights or (), penalties, {})
    reasons = [k.gate_id+"_"+k.state for k in gates if k.state != "PASS" and not (
        k.gate_id == "VALUATION" and value["warning"])]
    if value["state"] not in {"PASS", "ALLOWED_WITH_WARNING"}:
        reasons.append("FR3_VALUATION_ADMISSION_" + value["state"])
    if quality["status"] != "PASS" or (quality.get("quality_score") or 0) < admission["minimum_quality_score"]:
        reasons.append("QUALITY_NOT_PASS")
    localized = bool(plan and localization_gate(forecast, plan, confirmed=False))
    if not localized: reasons.append("FR3_LOCALIZATION_NOT_PASS")
    if score["final_score"] is None or score["final_score"] < admission["small_minimum"]:
        reasons.append("FR3_SCORE_BELOW_SMALL_MINIMUM")
    reasons.extend(risks)
    decision = "BUY_SMALL" if not reasons else "NO_BUY" if (quality["status"] == "REJECT"
        or any(k.state == "FAIL" for k in gates) or risks) else "DATA_INSUFFICIENT" if (
        any(k.state == "UNKNOWN" for k in gates) or quality["status"] in {"DATA_INSUFFICIENT", "UNKNOWN", "CONFIG_INCOMPLETE"}
        or forecast.status != "VALID" or value["state"] == "UNKNOWN") else "WATCH_ONLY"
    report = dict(schema=REPORT_SCHEMA, symbol=symbol, market=market, cutoff=cutoff,
        universe_id=identity(universe), universe_policy="EXACT_FROZEN_FULL_UNIVERSE",
        signal_date=reference["feature_audit"].get("cutoff_session"),
        model_id=bundle.sha256, evidence_id=identity(dict(base, source_id=identity(packet))),
        components=components(reference_version=model["reference_policy"]["version"],
            available_at=model["reference_policy"]["available_at"]), request_item_id=identity(item),
        features=features, feature_audit=reference["feature_audit"],
        context=reference["context"], quality=quality, plan=asdict(plan) if plan else None,
        forecast=forecast.record(), factor_ledger=ledger, score=score,
        gate_receipts=[asdict(v) for v in gates], produced_receipts=receipts,
        decision=dict(strategy_decision=decision, reason_codes=sorted(set(reasons)),
                      entry_policy_id=identity(admission)),
        execution=_account_assessment(plan, model["reference_policy"], dict(features, symbol=symbol),
            account, cutoff=cutoff, decision=decision, warning=value["warning"]),
        research_only=True, approved_for_live=False, execution_action="NO_ACTION",
        source_authenticity="SOURCE_BINDINGS_VALIDATED_NOT_INDEPENDENTLY_CERTIFIED",
        historical_deployment_claim=False)
    return dict(report, report_id=identity(report))


def analyze_fr3_request(payload):
    if not isinstance(payload, dict) or payload.get("schema") != REQUEST_SCHEMA:
        raise ValueError("FR3_EXPLICIT_REQUEST_SCHEMA_REQUIRED")
    requests = payload.get("securities")
    if not isinstance(requests, list) or not 1 <= len(requests) <= 2048:
        raise ValueError("FR3_BOUNDED_SECURITIES_REQUIRED")
    model = payload["model_bundle"]
    universe = payload.get("universe")
    _scope_members(FrozenBundle.load(model).record(), universe)
    reports, seen = [], set()
    for request in requests:
        key = (request["context"]["symbol"], request["context"]["market"],
               timestamp(request["context"]["as_of"]).isoformat())
        if key in seen: raise ValueError("FR3_DUPLICATE_SECURITY_CUTOFF")
        seen.add(key)
        reports.append(_evaluate(request, model, universe=universe))
    result = dict(schema=RESULT_SCHEMA, request_id=identity(payload), reports=reports,
        allocation_scope="INDEPENDENT_RESEARCH_QUOTAS_NOT_SIMULTANEOUS_ACCOUNT_APPROVAL",
        research_only=True, approved_for_live=False, execution_action="NO_ACTION")
    return dict(result, result_id=identity(result))


def validate_fr3_report(report, request_item, model_record, *, universe=None):
    """Re-execute all business fields, gates, forecasts and account checks."""
    expected = _evaluate(request_item, model_record, universe=universe)
    if canonical(report) != canonical(expected):
        raise ValueError("FR3_REPORT_RECOMPUTATION_MISMATCH")
    return True

"""Serialized v6 evidence validation shared by reporting and execution."""
import datetime as dt
from collections.abc import Mapping
from math import isclose

from domain.trading_restrictions import restriction_check
from domain.strategies.low_entry_score_v6 import parameter_version
from domain.strategies.low_entry_v6_quality import MINIMUM_ENTRY_QUALITY, QUALITY_MINIMUM_CONFIDENCE, QualityResult, number, quality_evidence_complete
from domain.strategies.low_entry_v6_signals import (
    EXECUTABLE_SIGNALS, SETUPS, LowEntryV6Config, hard_vetoes, risk_pass,
    decide, score_setups, trade_plan, triggers, veto_evidence_complete, VETO_BOOLEAN_FIELDS,
    relaxed_hybrid_config,
)


def report_config(report: Mapping, config: LowEntryV6Config | None = None) -> LowEntryV6Config:
    """Only select policy versions, never adopt caller-lowered thresholds/costs.

    Custom configuration still requires an explicit matching validator config.
    Full parameter and release identity is checked by report_failure.
    """
    if config is not None:
        return config
    params = report.get("parameters", {}) if isinstance(report, Mapping) else {}
    version = params.get("decision_policy_version", 2) if isinstance(params, Mapping) else 2
    correlation = params.get("correlation_policy_version", 2) if isinstance(params, Mapping) else 2
    leadership = params.get("leadership_policy_version", 2) if isinstance(params, Mapping) else 2
    factory = relaxed_hybrid_config if version == 4 else LowEntryV6Config
    return factory(decision_policy_version=version, correlation_policy_version=correlation,
                   leadership_policy_version=leadership)


def report_weights(report: Mapping) -> Mapping | None:
    """Select the published artifact, never substitute today's fitted weights."""
    decision = report.get("decision", {})
    if not isinstance(decision, Mapping):
        raise ValueError("INVALID_PUBLISHED_DECISION")
    weighting = decision.get("weighting")
    if weighting is None:
        if decision.get("policy_version") == "context-aware-convex-score/v1":
            raise ValueError("MISSING_FROZEN_WEIGHTS")
        return None
    if not isinstance(weighting, Mapping) or not isinstance(weighting.get("artifact"), Mapping):
        raise ValueError("MISSING_FROZEN_WEIGHTS")
    return weighting["artifact"]


def _minimum_score(signal: str, setup: str, regime: str | None, config: LowEntryV6Config) -> float:
    minimum = config.buy_score_minimum if signal == "BUY" else {
        "REVERSAL": config.reversal_starter_score_minimum,
        "TREND_PULLBACK": config.pullback_starter_score_minimum,
        "LEADERSHIP_CONTINUATION": config.leadership_starter_score_minimum,
    }[setup]
    return minimum + (config.risk_off_score_increment if regime == "RISK_OFF" and config.decision_policy_version == 1 else 0)


def entry_gate_checks(report: Mapping, *, config: LowEntryV6Config | None = None) -> dict[str, str]:
    """One read-only five-gate view for CSV, HTML and execution validation.

    These are completed-day setup gates, not live quote/order authorization.
    Missing evidence is UNKNOWN, not PASS. The published score is preserved;
    its minimum comes from the versioned policy, never a caller-lowered floor.
    """
    def mapping(key):
        value = report.get(key)
        return value if isinstance(value, Mapping) else {}

    data, q, d, f, e = (mapping(key) for key in ("data", "quality", "decision", "technical", "entry"))
    gates = dict.fromkeys(("quality", "no_veto", "score", "trigger", "risk"), "UNKNOWN")
    try:
        config = report_config(report, config)
    except (TypeError, ValueError):
        return gates
    signal, setup = d.get("signal"), d.get("setup")
    confidence, qs = number(data.get("confidence")), number(q.get("score"))
    coverage = q.get("coverage") if isinstance(q.get("coverage"), Mapping) else {}
    if q.get("status") in {"REJECT", "WATCH_QUALITY", "PROVISIONAL", "SECTOR_UNSUPPORTED"} or data.get("integrity") == "FAIL":
        gates["quality"] = "FAIL"
    elif q.get("status") == "PASS" and confidence is not None and qs is not None:
        quality = QualityResult(q["status"], qs, confidence, data.get("integrity"),
                                q.get("components", {}), coverage, q.get("metrics", {}))
        quality_ok = (data.get("integrity") == "PASS" and data.get("completed_history") is True
                      and data.get("market_aligned") is True and QUALITY_MINIMUM_CONFIDENCE <= confidence <= 100
                      and MINIMUM_ENTRY_QUALITY <= qs <= 100
                      and quality_evidence_complete(quality))
        gates["quality"] = "PASS" if quality_ok else "FAIL"

    veto = mapping("downtrend_gate")
    if veto.get("hard_veto") is True or d.get("hard_vetoes"):
        gates["no_veto"] = "FAIL"
    elif (veto.get("hard_veto") is False and d.get("hard_vetoes") == []
          and data.get("completed_history") is True and data.get("market_aligned") is True):
        if (veto_evidence_complete(dict(f))
                and all(isinstance(f.get(key), bool) for key in VETO_BOOLEAN_FIELDS)):
            gates["no_veto"] = "FAIL" if hard_vetoes(dict(f)) else "PASS"

    if config.decision_policy_version in {2, 3, 4}:
        return _unified_gate_checks(report, config, gates)
    if signal not in EXECUTABLE_SIGNALS:
        # Below-floor rows often stop before trigger/risk evaluation. Do not
        # misrepresent their initialized False values as assessed failures.
        gates.update(score="NOT_EVALUATED", trigger="NOT_EVALUATED", risk="NOT_EVALUATED")
        return gates
    if setup not in SETUPS:
        return gates
    score, required_score = number(d.get("score")), number(d.get("required_score"))
    minimum = _minimum_score(signal, setup, f.get("market_regime"), config)
    if score is not None and required_score is not None:
        computed = score_setups(dict(f), trade_plan(dict(f), config=config), config=config)[setup]
        score_ok = (d.get("score_basis") not in {"PARTIAL_DATA", "INSUFFICIENT_HISTORY"}
                    and minimum <= required_score <= score <= 100 and computed["valid"]
                    and computed["entry_tier"] in {"CONFIRMED", "STARTER_ONLY"}
                    and number(computed["score"]) is not None
                    and isclose(score, computed["score"], rel_tol=1e-10, abs_tol=1e-10)
                    and d.get("formal_score") == computed["formal_score"]
                    and d.get("estimated_score") == computed["estimated_score"]
                    and d.get("score_coverage") == computed["score_coverage"]
                    and d.get("score_evidence") == computed["score_evidence"]
                    and d.get("entry_tier") == computed["entry_tier"]
                    and (signal != "BUY" or computed["strict_valid"]))
        gates["score"] = "PASS" if score_ok else "FAIL"
    key = "confirmed_trigger" if signal == "BUY" else "starter_trigger"
    if isinstance(d.get(key), bool):
        starter, confirmed, demand = triggers(setup, dict(f))
        computed = confirmed if signal == "BUY" else starter
        risk_off_confirmation = (f.get("market_regime") != "RISK_OFF" or (
            all(number(f.get(key)) is not None and number(f[key]) > 0
                for key in ("rs20", "rs_slope5", "resilience")) and demand >= 3))
        gates["trigger"] = "PASS" if d[key] and computed and risk_off_confirmation and e.get(key, True) is True else "FAIL"

    if d.get("risk_gate") == "FAIL":
        gates["risk"] = "FAIL"
    elif d.get("risk_gate") == "PASS":
        plan = trade_plan(dict(f), config=config, setup=setup)
        def same_value(key):
            expected, actual = number(plan.get(key)), number(e.get(key))
            if key not in e:
                return False
            if expected is None or actual is None:
                return plan.get(key) is None and e[key] is None
            return isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10)
        plan_matches = all(same_value(key) for key in (
            "entry_price", "stop_price", "target_price", "stop_distance_atr",
            "stop_distance_fraction", "reward_risk", "net_reward_risk",
        )) and plan["stop_source"] == e.get("stop_source") and e.get("rr_basis") == plan["rr_basis"]
        risk_ok = risk_pass(setup, signal, dict(f), plan, mapping("valuation").get("interpretation"), config)[0]
        trigger_price, high = number(e.get("trigger_price")), number(f.get("high"))
        gates["risk"] = "PASS" if (plan_matches and risk_ok and plan["liquidity_pass"]
                                    and high is not None and high > 0 and trigger_price == high) else "FAIL"
    return gates


def _unified_gate_checks(report: Mapping, config: LowEntryV6Config, gates: dict) -> dict:
    """Reproduce the SAME score/cap/decision function from frozen evidence."""
    try:
        q, data, d, f = (report[k] for k in ("quality", "data", "decision", "technical"))
        if not f and d.get("evidence_status") == "UNAVAILABLE":
            return gates
        quality = QualityResult(status=q["status"], score=q["score"], confidence=data["confidence"],
            integrity=data["integrity"], components=q.get("components", {}), coverage=q.get("coverage", {}),
            metrics=q.get("metrics", {}), reasons=tuple(q.get("reasons", [])))
        weights = report_weights(report)
        expected = decide(quality, dict(f), valuation=report.get("valuation", {}).get("interpretation"),
            config=config, industry_mode=report.get("industry_profile", {}).get("mode", "UNSUPPORTED"),
            convex_weights=weights, as_of=report.get("decision_time"))
        keys = ("policy_version", "signal", "setup", "score", "raw_score", "penalty", "adjusted_score", "safety_cap",
                "final_score", "penalties", "safety_caps", "formal_score", "estimated_score", "score_coverage",
                "score_evidence", "budget_groups", "soft_conditions", "entry_tier", "entry_checks", "required_score",
                "risk_fraction", "position_multiplier", "reasons", "rr_policy", "setup_decisions", "comparison",
                "execution_score", "selection", "correlation_adjustment")
        if config.decision_policy_version in {3, 4}:
            keys = tuple(key for key in keys if key != "setup_decisions") + (
                "score_model", "score_setup", "score_basis", "evidence_status", "technical_observation")
            single_ledger = ("setup_decisions" not in d and report.get("setups") == dict(
                selection=expected["selection"], selected_setup=expected["setup"]))
            if weights is not None:
                keys += ("weighting", "research_only", "execution_policy", "requires_execution_check",
                         "performance_validated")
                single_ledger = single_ledger and report.get("parameter_version") == parameter_version(config, weights)
        else:
            single_ledger = True
        matches = single_ledger and all(key in d and d[key] == expected[key] for key in keys)
        gates["score"] = "PASS" if matches and expected["signal"] in EXECUTABLE_SIGNALS else "FAIL"
        # Compatibility key: validates the scored trigger diagnostics, not a
        # second soft entry gate. The next-session price trigger is unchanged.
        gates["trigger"] = "PASS" if all(d.get(key) == expected[key] == report["entry"].get(key)
            for key in ("starter_trigger", "confirmed_trigger")) else "FAIL"
        plan = expected["plan"]
        matched_plan = all(key in report["entry"] and report["entry"][key] == value for key, value in plan.items())
        gates["risk"] = "PASS" if (matched_plan and d.get("risk_gate") == expected["risk_gate"] == "PASS"
            and plan["liquidity_pass"] and report["entry"].get("trigger_price") == f.get("high")) else "FAIL"
    except (KeyError, TypeError, ValueError):
        gates.update(score="UNKNOWN", trigger="UNKNOWN", risk="UNKNOWN")
    return gates


def report_failure(report: object, *, config: LowEntryV6Config | None = None,
                   allow_replay: bool = False) -> str | None:
    """Validate serialized evidence before trusting an executable label."""
    try:
        decision = report.get("decision") if isinstance(report, dict) else None
        if isinstance(decision, Mapping) and decision.get("research_only") is True:
            return "RESEARCH_ONLY_SCORE_INTERVENTION"
        config = report_config(report, config)
        if not isinstance(report, dict) or report.get("model_version") != "6.0.0":
            return "INVALID_V6_REPORT"
        if report.get("evaluation_mode") == "HISTORICAL_REPLAY" and not allow_replay:
            return "HISTORICAL_REPLAY_NOT_EXECUTABLE"
        if (report.get("parameter_version") != parameter_version(config, report_weights(report))
                or report.get("parameters") != vars(config)):
            return "EXECUTION_CONFIG_MISMATCH"
        industry = report.get("industry_profile")
        if not isinstance(industry, Mapping) or industry.get("mode") != "ACTIVE":
            return "INDUSTRY_PROFILE_NOT_ACTIVE"
        data, quality, decision = report["data"], report["quality"], report["decision"]
        if "restriction_evidence" in data:
            market_info = report.get("market", {})
            checked = restriction_check(data["restriction_evidence"], market_data_date=report["signal_date"],
                                        decision_time=report["decision_time"],
                                        market=market_info.get("exchange", "TWSE") if isinstance(market_info, Mapping) else None,
                                        allow_historical=allow_replay and report.get("evaluation_mode") == "HISTORICAL_REPLAY",
                                        symbol=report.get("symbol"))
            if checked["restricted"] is not False:
                return "INVALID_PUBLISHED_RESTRICTION_EVIDENCE"
        if (data["integrity"] != "PASS" or data["completed_history"] is not True
                or data["market_aligned"] is not True or number(data.get("confidence")) is None
                or not QUALITY_MINIMUM_CONFIDENCE <= data["confidence"] <= 100 or quality["status"] != "PASS"
                or number(quality.get("score")) is None or not MINIMUM_ENTRY_QUALITY <= quality["score"] <= 100):
            return "MISSING_CRITICAL_DATA_OR_QUALITY_FAIL"
        if not quality_evidence_complete(QualityResult(quality["status"], quality["score"],
                data["confidence"], data["integrity"], quality.get("components", {}),
                quality["coverage"], quality.get("metrics", {}))):
            return "MISSING_CRITICAL_DATA_OR_QUALITY_FAIL"
        if (report["downtrend_gate"]["hard_veto"] is not False
                or decision["hard_vetoes"] != [] or decision["risk_gate"] != "PASS"):
            return "HARD_VETO_OR_RISK_FAIL"
        if (decision["signal"] not in {"BUY", "BUY_SMALL"}
                or number(decision.get("score")) is None
                or number(decision.get("required_score")) is None
                or decision["score"] < decision["required_score"]):
            return "INELIGIBLE_PUBLISHED_SIGNAL"
        if (decision.get("entry_tier") not in {"CONFIRMED", "STARTER_ONLY"}
                or (decision["signal"] == "BUY" and decision["entry_tier"] != "CONFIRMED")):
            return "INVALID_SETUP_ENTRY_TIER"
        if (report["entry"]["rule"] != "NEXT_SESSION_BREAKS_SIGNAL_HIGH"
                or report["entry"]["is_fill"] is not False):
            return "INVALID_ENTRY_RULE"
        gates = entry_gate_checks(report, config=config)
        score_failure = "V6_SCORE_OR_SETUP_MISMATCH"
        risk_failure = "V6_PUBLISHED_RISK_MISMATCH"
        if decision.get("setup") in SETUPS:
            minimum = _minimum_score(decision["signal"], decision["setup"], report.get("technical", {}).get("market_regime"), config)
            if decision["required_score"] < minimum or decision["score"] < minimum:
                score_failure = "V6_SCORE_BELOW_REQUIRED"
            if gates["risk"] == "FAIL":
                plan = trade_plan(report["technical"], config=config, setup=decision["setup"])
                if (report["entry"].get("stop_source") != plan["stop_source"]
                        or report["entry"].get("stop_price") != plan["stop_price"]):
                    # Preserve the existing execution reason when earlier frozen
                    # evidence validation catches a changed stop first.
                    risk_failure = "PUBLISHED_STOP_MISMATCH"
        for gate, failure in (
            ("quality", "MISSING_CRITICAL_DATA_OR_QUALITY_FAIL"),
            ("no_veto", "HARD_VETO_OR_RISK_FAIL"),
            ("score", score_failure),
            ("trigger", "V6_TRIGGER_NOT_CONFIRMED"),
            ("risk", risk_failure),
        ):
            if gates[gate] != "PASS":
                return failure
        dt.date.fromisoformat(report["signal_date"])
        return None
    except (KeyError, TypeError, ValueError):
        return "MISSING_CRITICAL_DATA_OR_QUALITY_FAIL"

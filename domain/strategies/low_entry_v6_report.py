"""Serialized v6 evidence validation shared by reporting and execution."""
import datetime as dt
from collections.abc import Mapping
from math import isclose

from domain.trading_restrictions import restriction_check
from domain.strategies.low_entry_score_v6 import PARAMETER_VERSION
from domain.strategies.low_entry_v6_quality import MINIMUM_ENTRY_QUALITY, number
from domain.strategies.low_entry_v6_signals import (
    EXECUTABLE_SIGNALS, SETUPS, LowEntryV6Config, hard_vetoes, risk_pass,
    score_setups, trade_plan, triggers,
)


def _minimum_score(signal: str, setup: str, regime: str | None, config: LowEntryV6Config) -> float:
    minimum = config.buy_score_minimum if signal == "BUY" else {
        "REVERSAL": config.reversal_starter_score_minimum,
        "TREND_PULLBACK": config.pullback_starter_score_minimum,
        "LEADERSHIP_CONTINUATION": config.leadership_starter_score_minimum,
    }[setup]
    return minimum + (config.risk_off_score_increment if regime == "RISK_OFF" else 0)


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
    config = config or LowEntryV6Config()
    signal, setup = d.get("signal"), d.get("setup")
    confidence, qs = number(data.get("confidence")), number(q.get("score"))
    coverage = q.get("coverage") if isinstance(q.get("coverage"), Mapping) else {}
    minima = {"balance_sheet": 70, "profitability": 70, "growth": 60, "return_on_capital": 60}
    coverage_values = {key: number(coverage.get(key)) for key in minima}
    if q.get("status") in {"REJECT", "WATCH_QUALITY", "PROVISIONAL", "SECTOR_UNSUPPORTED"} or data.get("integrity") == "FAIL":
        gates["quality"] = "FAIL"
    elif (q.get("status") == "PASS" and confidence is not None and qs is not None
          and all(value is not None for value in coverage_values.values())):
        quality_ok = (data.get("integrity") == "PASS" and data.get("completed_history") is True
                      and data.get("market_aligned") is True and 70 <= confidence <= 100
                      and MINIMUM_ENTRY_QUALITY <= qs <= 100
                      and all(minima[key] <= value <= 100 for key, value in coverage_values.items()))
        gates["quality"] = "PASS" if quality_ok else "FAIL"

    veto = mapping("downtrend_gate")
    if veto.get("hard_veto") is True or d.get("hard_vetoes"):
        gates["no_veto"] = "FAIL"
    elif (veto.get("hard_veto") is False and d.get("hard_vetoes") == []
          and data.get("completed_history") is True and data.get("market_aligned") is True):
        required = ("new_lows3", "clv", "ema5_slope", "ema20_slope", "atr_expansion",
                    "r5", "atr_percent", "close", "open", "volume_ratio", "gap",
                    "support", "rs20", "rs_slope5", "resilience")
        if (all(number(f.get(key)) is not None for key in required)
                and all(isinstance(f.get(key), bool) for key in ("support_break", "new_low", "undercut_reclaim"))
                and f.get("market_regime") in {"RISK_ON", "NEUTRAL", "RISK_OFF"}):
            gates["no_veto"] = "FAIL" if hard_vetoes(dict(f)) else "PASS"

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
                    and isclose(score, computed["score"], rel_tol=1e-10, abs_tol=1e-10)
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


def report_failure(report: object, *, config: LowEntryV6Config | None = None,
                   allow_replay: bool = False) -> str | None:
    """Validate serialized evidence before trusting an executable label."""
    config = config or LowEntryV6Config()
    try:
        if not isinstance(report, dict) or report.get("model_version") != "6.0.0":
            return "INVALID_V6_REPORT"
        if report.get("evaluation_mode") == "HISTORICAL_REPLAY" and not allow_replay:
            return "HISTORICAL_REPLAY_NOT_EXECUTABLE"
        if report.get("parameter_version") != PARAMETER_VERSION or report.get("parameters") != vars(config):
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
                or not 70 <= data["confidence"] <= 100 or quality["status"] != "PASS"
                or number(quality.get("score")) is None or not MINIMUM_ENTRY_QUALITY <= quality["score"] <= 100):
            return "MISSING_CRITICAL_DATA_OR_QUALITY_FAIL"
        if any(number(quality["coverage"].get(key)) is None or quality["coverage"][key] < minimum
               for key, minimum in (("balance_sheet", 70), ("profitability", 70), ("growth", 60), ("return_on_capital", 60))):
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

"""Daily LES v6 boundary: raw prices/PIT filings in, versioned decisions out.

The v5 output is retained only as separately attributed valuation/legacy evidence
for PAE. It must never supply the published v6 entry decision or position cap.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping

import pandas as pd

from domain.strategy import get_strategy
from domain.fundamentals import evaluate_quality_gate
from domain.strategies.low_entry_score_v4 import ANNUAL_FUNDAMENTALS_ATTR, FUNDAMENTAL_SNAPSHOT_ATTR, FUNDAMENTAL_STATUS_ATTR, SIGNAL_DATE_ATTR
from domain.strategies.low_entry_score_v6 import PARAMETER_VERSION, V6_CONTEXT_ATTR, attach_low_entry_v6_context
from domain.strategies.low_entry_v6_signals import LowEntryV6Config
from domain.strategies.low_entry_v6_quality import number, verification_diagnostics
from domain.strategies.low_entry_v6_report import entry_gate_checks
from domain.strategies.low_entry_v6_watch import screen_leadership


V6_DAILY_COLUMNS = [
    "LES v6 Score", "LES v6 Decision", "LES v6 Setup", "LES v6 Required Score",
    "LES v6 Trigger", "LES v6 Report", "Quality Score", "Quality Status", "Quality Gate",
    "LES v6 Score Gate", "LES v6 Trigger Gate", "No Veto",
    "Data Confidence", "Falling Knife Veto", "LES Parameter Version",
    "Entry Trigger Price",
    "LES v6 Watch Status", "LES v6 Watch Patterns", "LES v6 Verification Action",
]
# Only these compatibility columns are produced by v6. Legacy v5 diagnostic
# columns retain their original names, values and producer attribution.
V6_PROJECTED_COLUMNS = set(V6_DAILY_COLUMNS) | {
    "Industry Quality Profile", "Industry Profile Version", "Industry Profile Mode", "Industry Shadow Decision",
    "低點分數", "低點決策", "低點理由", "LES Model Version", "策略停損", "策略停利",
    "建議股數", "Risk Gate", "Risk Reasons", "Falling Knife State", "No Veto",
    "Liquidity Status", "Entry Candidate", "Entry Rule", "Entry Trigger Price",
    "RewardRisk", "ATR14", "Position Fraction",
}


def unavailable_report(symbol: str, name: str, signal_date: str, as_of: object, reason: str) -> dict:
    return {
        "symbol": symbol, "name": name, "model_version": "6.0.0",
        "parameter_version": PARAMETER_VERSION, "parameters": vars(LowEntryV6Config()),
        "signal_date": signal_date, "decision_time": str(as_of),
        "data": {"integrity": "UNKNOWN", "confidence": 0, "completed_history": False, "market_aligned": False},
        "quality": {"status": "DATA_INSUFFICIENT", "score": None, "coverage": {}, "components": {}, "metrics": {}, "reasons": [reason]},
        "valuation": {"interpretation": "NOT_AVAILABLE"}, "market": {"regime": "UNKNOWN"},
        "downtrend_gate": {"hard_veto": False, "reasons": [], "warnings": []},
        "technical": {}, "setups": {}, "liquidity": {"passed": False},
        "decision": {"signal": "DATA_INSUFFICIENT", "score": None, "setup": None,
                     "risk_gate": "NOT_EVALUATED", "reasons": [reason], "required_score": None},
        "entry": {"rule": "NEXT_SESSION_BREAKS_SIGNAL_HIGH", "is_fill": False},
        "sizing": {"indicative_shares": 0, "portfolio_data_pass": False, "reason": reason},
    }


def daily_report(history: pd.DataFrame, *, watch: dict | None = None) -> dict:
    report = _daily_report(history)
    if history.attrs.get("replay_as_of"):
        report["evaluation_mode"] = "HISTORICAL_REPLAY"
        report["replay_as_of"] = history.attrs["replay_as_of"]
    if history.attrs.get("market_source") == "TPEX":
        report["market"]["exchange"] = "TPEX"
    expected = history.attrs.get("les_v6_analysis_session") or report["signal_date"]
    session = pd.to_datetime(expected, errors="coerce")
    report["data"]["analysis_session_date"] = session.date().isoformat() if pd.notna(session) else ""
    report["data"]["market_data_date"] = report["signal_date"]
    if watch is None and pd.notna(session):
        watch = screen_leadership([history], session_date=session.date().isoformat(),
                                  as_of=history.attrs.get(SIGNAL_DATE_ATTR))[0]
    snapshot = history.attrs.get(FUNDAMENTAL_SNAPSHOT_ATTR, {})
    verification = verification_diagnostics(snapshot.get("les_v6_filings", []),
                                          symbol=report["symbol"], as_of=report["decision_time"],
                                          market=history.attrs.get("market_source", "TWSE"))
    before = history.attrs.get("les_v6_research_context", {}).get("before_refresh", {})
    d, q = report["decision"], report["quality"]
    hard = report["downtrend_gate"]["hard_veto"] or q["status"] == "REJECT"
    action = "HARD_GATE_REVIEW" if hard else verification["next_action"]
    if not hard and not verification["backfill_needed"] and not verification["local_review_required"]:
        action = "AWAIT_NEXT_SESSION_EXECUTION_CHECK" if d["signal"] in {"BUY", "BUY_SMALL"} else "REEVALUATED_NOT_EXECUTABLE"
    report["research"] = {
        "watch": watch or {"status": "UNAVAILABLE", "patterns": [], "priority_eligible": False},
        "verification": verification, "before_refresh": before,
        "refresh_status": snapshot.get("remote_refresh_status") or history.attrs.get(FUNDAMENTAL_STATUS_ATTR),
        "next_action": action, "is_entry_signal": False,
        "blocking_reasons": list(dict.fromkeys(list(d.get("reasons", [])) + list(q.get("reasons", [])) + list(q.get("warnings", [])))),
        "reassessment": "CURRENT_PLAN_ONLY_NO_BACKDATED_ENTRY",
    }
    return report


def _daily_report(history: pd.DataFrame) -> dict:
    identity = history.attrs.get("les_v6_daily_identity", {})
    snapshot = history.attrs.get(FUNDAMENTAL_SNAPSHOT_ATTR, {})
    symbol, name = str(identity.get("symbol", "")), str(identity.get("name", ""))
    as_of = history.attrs.get(SIGNAL_DATE_ATTR)
    dates = pd.to_datetime(history.get("Date", pd.Series(dtype=object)), errors="coerce")
    date = dates.iloc[-1].date().isoformat() if len(dates) and pd.notna(dates.iloc[-1]) else ""
    if history.empty or dates.isna().any() or not date:
        timestamp = pd.to_datetime(as_of, utc=True, errors="coerce")
        date = timestamp.tz_convert("Asia/Taipei").date().isoformat() if pd.notna(timestamp) else ""
        return unavailable_report(symbol, name, date, as_of, "MISSING_DAILY_DATES")
    expected = history.attrs.get("les_v6_analysis_session")
    session = pd.to_datetime(expected, errors="coerce")
    if expected and pd.isna(session):
        return unavailable_report(symbol, name, date, as_of, "INVALID_ANALYSIS_SESSION")
    if pd.notna(session) and dates.iloc[-1].date() < session.date():
        # Preserve the real bar date. Stale prices cannot become a current
        # candidate merely because a batch is evaluated or resent today.
        return unavailable_report(symbol, name, date, as_of, "STALE_MARKET_DATA")
    raw = history.copy(deep=True)
    for field in ("Open", "High", "Low", "Close", "Bid", "Ask"):
        raw[field] = pd.to_numeric(raw[field], errors="coerce") / 1000.0
    # Restrictions are decision-time context, not a fabricated historical bar.
    raw["Restricted"] = pd.Series(None, index=raw.index, dtype=object)
    restriction = history.attrs.get("trading_restriction", {})
    # Strategy, report and execution validation bind evidence to this market;
    # official TPEx lists are usable, but TWSE evidence cannot certify TPEx.
    sector = str(snapshot.get("sector") or identity.get("sector") or "UNKNOWN").upper()
    sector = sector.removeprefix("STOCKTYPE.")
    # Keep the official company sector: aggregate universe names must not erase
    # the per-industry policy identity before strategy evaluation.
    if snapshot.get("security_type") not in (None, "COMMON"):
        sector = "UNSUPPORTED_SECURITY"
    filings = snapshot.get("les_v6_filings", [])
    if not isinstance(filings, (list, tuple)) or any(not isinstance(row, Mapping) for row in filings):
        return unavailable_report(symbol, name, date, as_of, "INVALID_PIT_FILINGS")
    raw = attach_low_entry_v6_context(
        raw, symbol=symbol, name=name, sector=sector, as_of=as_of,
        market=identity.get("market", "TWSE"),
        filings=filings, portfolio=history.attrs.get("les_v6_portfolio", {}),
    )
    risks = [key.upper() for key in ("going_concern", "material_accounting_issue") if snapshot.get(key) is True]
    audit = snapshot.get("current_audit_opinion", snapshot.get("audit_opinion"))
    if audit in {"ADVERSE", "DISCLAIMER"}:
        risks.append(f"AUDIT_{audit}")
    # A known but unresolved qualification is not an optional missing field.
    # Removing a false material-misstatement flag must not silently clear it.
    review = snapshot.get("audit_review_required") is True or (
        audit in {"UNKNOWN", "QUALIFIED", "QUALIFIED_UNKNOWN"}
        and snapshot.get("audit_evidence_available") is True
    )
    raw.attrs[V6_CONTEXT_ATTR]["hard_risk_evidence"] = {
        "available_at": snapshot.get("available_at"), "reasons": risks,
        "review_reasons": ["AUDIT_REVIEW_REQUIRED"] if review else [],
    }
    raw.attrs[V6_CONTEXT_ATTR]["restriction_evidence"] = restriction
    raw.attrs[V6_CONTEXT_ATTR]["historical_replay"] = bool(history.attrs.get("replay_as_of"))
    try:
        report = get_strategy("low_entry_score_v6").run_latest(raw).iloc[-1]["LES v6 Report"]
    except ValueError as error:
        return unavailable_report(symbol, name, date, as_of, f"V6_INPUT_INVALID:{error}")
    if report["data"]["restriction_check"]["status"] == "UNKNOWN":
        report["decision"]["reasons"].append("TRADING_RESTRICTION_UNKNOWN")
    if report["industry_profile"]["name"] in {"ELEC_PROFILE", "ELECCOMP_PROFILE"}:
        # Industry research evidence is distinct from v6's native quality score.
        # Re-evaluate from cached PIT inputs, not from a prior strategy's score.
        industry = evaluate_quality_gate(
            history.attrs.get(ANNUAL_FUNDAMENTALS_ATTR, pd.DataFrame()),
            dict(snapshot, sector=sector), report["decision_time"],
        )
        report["industry_profile"]["quality_evaluation"] = {
            "status": industry.status.value, "score": industry.score,
            "data_confidence": industry.data_confidence,
            "available_weight": industry.available_metric_weight,
            "qa_flags": industry.qa_flags, "hard_red_flags": industry.hard_red_flags,
            "metrics": {
                item.name: {"value": value if value is not None and math.isfinite(value) else None,
                            "points": item.earned, "maximum": item.maximum,
                            "available": item.available}
                for item in industry.metric_scores
                for value in [industry.metrics.get(item.name)]
            },
        }
    return report


def project_report(report: Mapping) -> dict:
    d, q, e, t = (report[key] for key in ("decision", "quality", "entry", "technical"))
    gates = entry_gate_checks(report)
    signal, score = d["signal"], d.get("score")
    executable = signal in {"BUY", "BUY_SMALL"}
    p = report["parameters"]
    cap = (p["maximum_position_weight"] * (p["starter_fraction"] if signal == "BUY_SMALL" else 1)
           * d.get("position_multiplier", 1)) if executable else 0.0
    def scaled(value):
        n = number(value)
        return n * 1000 if n is not None else None
    return {
        "Industry Quality Profile": report.get("industry_profile", {}).get("name", "UNSUPPORTED"),
        "Industry Profile Version": report.get("industry_profile", {}).get("version", "N/A"),
        "Industry Profile Mode": report.get("industry_profile", {}).get("mode", "UNSUPPORTED"),
        "Industry Shadow Decision": report.get("industry_profile", {}).get("shadow_signal"),
        "LES Model Version": report["model_version"], "LES Parameter Version": report["parameter_version"],
        "LES v6 Report": json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        "LES v6 Score": score, "低點分數": score, "LES v6 Decision": signal, "低點決策": signal,
        "LES v6 Setup": d.get("setup"), "LES v6 Required Score": d.get("required_score"),
        "LES v6 Trigger": bool(d.get("confirmed_trigger") if signal == "BUY" else d.get("starter_trigger")),
        "低點理由": ";".join(d.get("reasons", [])), "Quality Score": q.get("score"),
        "Quality Status": q["status"], "Data Confidence": report["data"]["confidence"],
        "Quality Gate": gates["quality"], "LES v6 Score Gate": gates["score"],
        "LES v6 Trigger Gate": gates["trigger"],
        "Risk Gate": d["risk_gate"], "Risk Reasons": ";".join(d.get("reasons", [])),
        "Falling Knife Veto": report["downtrend_gate"]["hard_veto"],
        "Falling Knife State": "HARD_NO_BUY" if report["downtrend_gate"]["hard_veto"] else "CLEAR",
        "No Veto": {"PASS": True, "FAIL": False}.get(gates["no_veto"]),
        "Liquidity Status": "PASS" if report["liquidity"]["passed"] else "FAIL",
        "策略停損": scaled(e.get("stop_price")), "策略停利": scaled(e.get("target_price")),
        "Entry Candidate": scaled(e.get("entry_price")), "Entry Trigger Price": scaled(e.get("trigger_price")),
        "Entry Rule": e["rule"], "ATR14": scaled(t.get("atr14")), "RewardRisk": e.get("reward_risk"),
        "Position Fraction": cap, "建議股數": report["sizing"]["indicative_shares"],
        "LES v6 Watch Status": report.get("research", {}).get("watch", {}).get("status", "UNAVAILABLE"),
        "LES v6 Watch Patterns": ";".join(report.get("research", {}).get("watch", {}).get("patterns", [])),
        "LES v6 Verification Action": report.get("research", {}).get("next_action", "NOT_EVALUATED"),
    }


def overlay_daily_v6(histories, legacy: pd.DataFrame) -> pd.DataFrame:
    result = legacy.copy()
    if not histories:
        return result
    sessions = [pd.to_datetime(history.attrs.get("les_v6_analysis_session") or
                              (history["Date"].iloc[-1] if len(history) and "Date" in history else None),
                              errors="coerce") for history in histories]
    session = max((date for date in sessions if pd.notna(date)), default=pd.NaT)
    cutoffs = [pd.to_datetime(history.attrs.get(SIGNAL_DATE_ATTR), utc=True, errors="coerce") for history in histories]
    cutoff = max((date for date in cutoffs if pd.notna(date)), default=pd.NaT)
    watches = screen_leadership(histories, session_date=session, as_of=cutoff)
    projected = pd.DataFrame([project_report(daily_report(history, watch=watch))
                              for history, watch in zip(histories, watches)], index=result.index)
    for name in projected:
        result[name] = projected[name]
    return result

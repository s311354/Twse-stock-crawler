"""Daily LES v6 boundary: raw prices/PIT filings in, versioned decisions out.

The v5 output is retained only as separately attributed valuation/legacy evidence
for PAE. It must never supply the published v6 entry decision or position cap.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from types import MappingProxyType

import pandas as pd

from domain.strategy import get_strategy
from domain.fundamentals import evaluate_quality_gate
from domain.strategies.low_entry_score_v4 import ANNUAL_FUNDAMENTALS_ATTR, FUNDAMENTAL_SNAPSHOT_ATTR, FUNDAMENTAL_STATUS_ATTR, SIGNAL_DATE_ATTR
from domain.strategies.low_entry_score_v6 import V6_CONTEXT_ATTR, attach_low_entry_v6_context, parameter_version
from domain.strategies.low_entry_v6_signals import relaxed_hybrid_config
from domain.strategies.low_entry_v6_decision import decision_from_score, safety_limits
from domain.strategies.low_entry_v6_hybrid import factor_budgets, model_spec, select_context
from domain.strategies.low_entry_v6_quality import QUALITY_POLICY_VERSION, number, verification_diagnostics
from domain.strategies.low_entry_v6_report import entry_gate_checks
from domain.strategies.low_entry_v6_watch import screen_leadership
from application.leadership_screen import screen_once
from application.daily_performance import count_work, measure_stage
from application.evidence_context import StockEvidence
from application.daily_work import current_work
from application.replay_contracts import digest
from domain.numeric_history import metadata_free_frame


# Explicit deployment policy. Keep the legacy plugin/config defaults intact for
# archived callers; the daily pipeline and replay manifest share this identity.
DAILY_V6_STRATEGY = "low_entry_score_v6_hybrid"
DAILY_V6_CONFIG = relaxed_hybrid_config()
DAILY_V6_PARAMETER_VERSION = parameter_version(DAILY_V6_CONFIG)


V6_DAILY_COLUMNS = [
    "LES v6 Raw Score", "LES v6 Penalty", "LES v6 Adjusted Score", "LES v6 Safety Cap", "LES v6 Final Score",
    "LES v6 Formal Score", "LES v6 Estimated Score", "LES v6 Score Coverage",
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
    spec = model_spec(DAILY_V6_CONFIG)
    blocked_cap, _ = safety_limits(DAILY_V6_CONFIG)
    return {
        "symbol": symbol, "name": name, "model_version": "6.0.0",
        "parameter_version": DAILY_V6_PARAMETER_VERSION, "parameters": vars(DAILY_V6_CONFIG),
        "signal_date": signal_date, "decision_time": str(as_of),
        "data": {"integrity": "UNKNOWN", "confidence": 0, "completed_history": False, "market_aligned": False},
        "quality": {"status": "DATA_INSUFFICIENT", "score": None, "coverage": {}, "components": {}, "metrics": {}, "reasons": [reason]},
        "valuation": {"interpretation": "NOT_AVAILABLE"}, "market": {"regime": "UNKNOWN"},
        "downtrend_gate": {"hard_veto": False, "reasons": [], "warnings": []},
        "technical": {}, "setups": {}, "liquidity": {"passed": False},
        "decision": {"signal": decision_from_score(None), "score": None, "setup": None,
                     "policy_version": spec["policy_version"], "evidence_status": "UNAVAILABLE",
                     "selection": select_context({}).record(), "score_model": spec,
                     "budget_groups": {factor: {"budget": budget, "earned": 0., "available_weight": 0.}
                                       for factor, budget in factor_budgets(DAILY_V6_CONFIG)},
                     "formal_score": None, "estimated_score": None, "score_coverage": 0.,
                     "raw_score": None, "penalty": 0., "adjusted_score": None, "safety_cap": blocked_cap, "final_score": None,
                     "penalties": [], "safety_caps": [{"reason": reason, "maximum": blocked_cap}],
                     "risk_gate": "NOT_EVALUATED", "reasons": [reason], "required_score": None},
        "entry": {"rule": "NEXT_SESSION_BREAKS_SIGNAL_HIGH", "is_fill": False},
        "sizing": {"indicative_shares": 0, "portfolio_data_pass": False, "reason": reason},
    }


def daily_report(history: pd.DataFrame, *, watch: dict | None = None, features=None,
                 evidence: StockEvidence | None = None) -> dict:
    if evidence is not None:
        evidence.validate(history)
    report = _daily_report(history, features=features, evidence=evidence)
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
    snapshot = evidence.snapshot if evidence is not None else history.attrs.get(FUNDAMENTAL_SNAPSHOT_ATTR, {})
    records = snapshot.get("les_v6_filings", ())
    market = history.attrs.get("market_source", "TWSE")
    def verify():
        # Diagnostics can retain nested ReadOnlyEvidence source proofs. Convert
        # them losslessly BEFORE the JSON memo boundary, on both cache paths;
        # never stringify proofs or relax the native evidence validators.
        return _plain_evidence(verification_diagnostics(
            records, symbol=report["symbol"], as_of=report["decision_time"], market=market,
            preprocessing=evidence.quality_preprocessing if evidence is not None else None))
    with measure_stage("quality_diagnostics"):
        work = current_work()
        if work is not None and work.enabled:
            with measure_stage("quality_diagnostics_key"):
                if evidence is not None:
                    try:
                        content_key = evidence.diagnostics_content_key(
                            symbol=report["symbol"], market=market, as_of=report["decision_time"])
                    except (ValueError, TypeError, AttributeError):
                        # Failed selection diagnostics include best-effort raw
                        # inventory. Keep the old complete-content key for it.
                        content_key = evidence.filings_content_key()
                else:
                    content_key = digest(_plain_evidence(records))
                key = (content_key, report["symbol"], report["decision_time"], market,
                       QUALITY_POLICY_VERSION)
            verification = work.memo_json("quality_diagnostics", key, verify)
        else:
            # Disabled caching must not pay a hash/conversion cost for a key
            # that cannot be used; native verification still runs unchanged.
            verification = verify()
    research = evidence.research_context if evidence is not None else history.attrs.get("les_v6_research_context", {})
    before = research.get("before_refresh", {})
    d, q = report["decision"], report["quality"]
    hard = report["downtrend_gate"]["hard_veto"] or q["status"] == "REJECT"
    action = "HARD_GATE_REVIEW" if hard else verification["next_action"]
    if not hard and not verification["backfill_needed"] and not verification["local_review_required"]:
        action = "AWAIT_NEXT_SESSION_EXECUTION_CHECK" if d["signal"] in {"BUY", "BUY_SMALL"} else "REEVALUATED_NOT_EXECUTABLE"
    report["research"] = {
        "watch": watch or {"status": "UNAVAILABLE", "patterns": [], "priority_eligible": False},
        "verification": verification, "before_refresh": _plain_evidence(before),
        "refresh_status": snapshot.get("remote_refresh_status") or history.attrs.get(FUNDAMENTAL_STATUS_ATTR),
        "next_action": action, "is_entry_signal": False,
        "blocking_reasons": list(dict.fromkeys(list(d.get("reasons", [])) + list(q.get("reasons", [])) + list(q.get("warnings", [])))),
        "reassessment": "CURRENT_PLAN_ONLY_NO_BACKDATED_ENTRY",
    }
    return report


def _plain_evidence(value):
    kind = type(value)
    if kind in (str, int, float, bool, type(None)):
        return value
    if kind is dict or isinstance(value, Mapping):
        return {key: _plain_evidence(item) for key, item in value.items()}
    if kind in (list, tuple) or isinstance(value, (list, tuple)):
        return [_plain_evidence(item) for item in value]
    return value


def _daily_report(history: pd.DataFrame, *, features=None, evidence: StockEvidence | None = None) -> dict:
    from domain.industry_benchmark import INDUSTRY_BENCHMARK_ATTR
    identity = history.attrs.get("les_v6_daily_identity", {})
    snapshot = evidence.snapshot if evidence is not None else history.attrs.get(FUNDAMENTAL_SNAPSHOT_ATTR, {})
    symbol, name = str(identity.get("symbol", "")), str(identity.get("name", ""))
    as_of = history.attrs.get(SIGNAL_DATE_ATTR)
    raw = metadata_free_frame(history)
    dates = pd.to_datetime(raw.get("Date", pd.Series(dtype=object)), errors="coerce")
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
    filings = snapshot.get("les_v6_filings", ())
    if not isinstance(filings, (list, tuple)) or any(not isinstance(row, Mapping) for row in filings):
        return unavailable_report(symbol, name, date, as_of, "INVALID_PIT_FILINGS")
    benchmark = evidence.industry_benchmark if evidence is not None else history.attrs.get(INDUSTRY_BENCHMARK_ATTR)
    if evidence is None:
        raw = attach_low_entry_v6_context(
            raw, symbol=symbol, name=name, sector=sector, as_of=as_of,
            market=identity.get("market", "TWSE"), filings=filings,
            portfolio=history.attrs.get("les_v6_portfolio", {}), industry_benchmark=benchmark,
        )
        context = raw.attrs[V6_CONTEXT_ATTR]
    else:
        context = {"symbol": symbol, "name": name, "sector": sector, "as_of": as_of,
                   "market": identity.get("market", "TWSE"), "filings": filings,
                   "portfolio": history.attrs.get("les_v6_portfolio", {}),
                   "industry_benchmark": benchmark}
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
    context["hard_risk_evidence"] = {
        "available_at": snapshot.get("available_at"), "reasons": risks,
        "review_reasons": ["AUDIT_REVIEW_REQUIRED"] if review else [],
    }
    context["restriction_evidence"] = restriction
    # Optional instrumentation only; domain does not depend on application
    # observability and this callback is not serialized into native results.
    context["_quality_stage"] = lambda: measure_stage("quality_gate")
    context["_stage"] = measure_stage
    if evidence is not None:
        context["_quality_preprocessing"] = evidence.quality_preprocessing
    context["historical_replay"] = bool(history.attrs.get("replay_as_of"))
    try:
        if features is None:
            count_work("v6_fallback_feature_builds")
        # Financial evidence is already recursively frozen. Keep native JSON
        # containers for restriction proofs: their existing validators require
        # dict/list and must still reject malformed inputs, not be loosened.
        report = get_strategy(DAILY_V6_STRATEGY).run_latest(
            raw, features=features, context=MappingProxyType(context) if evidence is not None else context,
        ).iloc[-1]["LES v6 Report"]
    except ValueError as error:
        return unavailable_report(symbol, name, date, as_of, f"V6_INPUT_INVALID:{error}")
    if report["data"]["restriction_check"]["status"] == "UNKNOWN":
        report["decision"]["reasons"].append("TRADING_RESTRICTION_UNKNOWN")
    if report["industry_profile"]["name"] in {"ELEC_PROFILE", "ELECCOMP_PROFILE"}:
        # Industry research evidence is distinct from v6's native quality score.
        # Re-evaluate from cached PIT inputs, not from a prior strategy's score.
        with measure_stage("les_industry_quality"):
            industry = evaluate_quality_gate(
                evidence.annual() if evidence is not None else history.attrs.get(ANNUAL_FUNDAMENTALS_ATTR, pd.DataFrame()),
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
        "LES v6 Raw Score": d.get("raw_score"), "LES v6 Penalty": d.get("penalty"),
        "LES v6 Adjusted Score": d.get("adjusted_score"), "LES v6 Safety Cap": d.get("safety_cap"),
        "LES v6 Final Score": d.get("final_score"),
        "LES v6 Formal Score": d.get("formal_score"),
        "LES v6 Estimated Score": d.get("estimated_score"),
        "LES v6 Score Coverage": d.get("score_coverage"),
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


def build_daily_v6(histories, *, feature_frames=None, watches=None, reuse_watch=True, evidence=None) -> pd.DataFrame:
    """Independent latest-session entry output; no legacy result is required."""
    if not histories:
        return pd.DataFrame(columns=sorted(V6_PROJECTED_COLUMNS))
    if feature_frames is None:
        feature_frames = [None] * len(histories)
    if len(feature_frames) != len(histories):
        raise ValueError("DAILY_V6_FEATURE_COUNT_MISMATCH")
    if evidence is None:
        evidence = [None] * len(histories)
    if len(evidence) != len(histories):
        raise ValueError("DAILY_V6_EVIDENCE_COUNT_MISMATCH")
    sessions = [pd.to_datetime(history.attrs.get("les_v6_analysis_session") or
                              (metadata_free_frame(history)["Date"].iloc[-1] if len(history) and "Date" in history else None),
                              errors="coerce") for history in histories]
    session = max((date for date in sessions if pd.notna(date)), default=pd.NaT)
    cutoffs = [pd.to_datetime(history.attrs.get(SIGNAL_DATE_ATTR), utc=True, errors="coerce") for history in histories]
    cutoff = max((date for date in cutoffs if pd.notna(date)), default=pd.NaT)
    if watches is None:
        watches, _ = screen_once(histories, session_date=session, as_of=cutoff, screener=screen_leadership, reuse=reuse_watch)
    if len(watches) != len(histories):
        raise ValueError("DAILY_V6_WATCH_COUNT_MISMATCH")
    return pd.DataFrame([project_report(daily_report(history, watch=watch, features=features, evidence=packet))
                         for history, watch, features, packet in zip(histories, watches, feature_frames, evidence, strict=True)])


def overlay_daily_v6(histories, legacy: pd.DataFrame) -> pd.DataFrame:
    """Compatibility adapter for archived callers, not the daily entry pipeline."""
    result = legacy.copy()
    if not histories:
        return result
    projected = build_daily_v6(histories)
    if len(projected) != len(result):
        raise ValueError("DAILY_V6_ROW_COUNT_MISMATCH")
    projected.index = result.index
    for name in projected:
        result[name] = projected[name]
    return result

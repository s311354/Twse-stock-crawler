"""One research engine for CLI and BankEmulator; no fills or notifications."""
from __future__ import annotations

import datetime as dt
import enum
import json
import math
from dataclasses import asdict, replace
from typing import Callable, Protocol

from application.analysis_dataset import AnalysisDatasetBuilder
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from application.replay_contracts import REPLAY_SCHEMA, ReplayRequest, ReplayVersions, aware_timestamp, digest
from application.replay_inputs import ReplayInputs
from application.research_evidence import RAW_EVIDENCE_SCHEMA
from application.research_audit import coverage_audit
from domain.services import max_profit, max_profit_unlimited, max_profit_k_transactions, max_profit_with_fee
from domain.strategies.low_entry_v6_report import entry_gate_checks, report_failure
from domain.model_contracts import DecisionField, FieldSource, FieldUnit


def json_value(value):
    if callable(value):
        return {"callable": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_value(v) for v in (sorted(value) if isinstance(value, (set, frozenset)) else value)]
    if hasattr(value, "item"):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ReplayEvidencePort(Protocol):
    def select(self, request: ReplayRequest) -> ReplayInputs: ...


class ReplayStatePort(Protocol):
    def session(self, request: ReplayRequest, trading_sessions: tuple[str, ...]): ...


class StrictReplayEngine:
    def __init__(self, evidence: ReplayEvidencePort, state: ReplayStatePort,
                 versions: Callable[[], ReplayVersions], *, builder: AnalysisDatasetBuilder | None = None):
        self.evidence, self.state, self.versions = evidence, state, versions
        self.builder = builder or AnalysisDatasetBuilder()

    def run(self, request: ReplayRequest) -> dict:
        return self.run_with_coverage(request)[0]

    def run_with_coverage(self, request: ReplayRequest) -> tuple[dict, dict]:
        """One selection for research and coverage; internal inputs stay out of JSON."""
        if request.versions != self.versions():
            raise ValueError("REPLAY_MODEL_OR_CODE_VERSION_MISMATCH")
        selected = self.evidence.select(request)
        with self.state.session(request, selected.sessions) as session:
            if session.cached is not None:
                # Restore derived diagnostic annotations without altering the
                # independently selected/account-validated inputs.
                selected.diagnostics.update(session.cached['diagnostics'])
                return session.cached, coverage_audit(request, selected, session.cached)
            market = selected.market
            profits = []
            for prices in market.opens:
                fee_profit = max_profit_with_fee(prices, 300)
                profits.append([round(max_profit_unlimited(prices), 2), round(max_profit(prices), 2),
                                round(max_profit_k_transactions(5, prices), 2), round(fee_profit, 2),
                                round(fee_profit / prices[-1], 2) if prices else None])
            decisions = self.builder.build_decisions(market, profits) if market.stock_count else ()
            calibrated = []
            for decision in decisions:
                calibration = selected.calibrations.get(str(decision.value("證券代號")))
                if calibration is not None:
                    decision = replace(decision, fields=decision.fields + (DecisionField(
                        name="Half Kelly Weight", value=float(calibration["half_kelly_weight"]),
                        source=FieldSource.KELLY_CALIBRATION, available_at=aware_timestamp(calibration["available_at"]),
                        confidence=100.0, unit=FieldUnit.RATIO, model_version=calibration["model_version"]),))
                calibrated.append(decision)
            decisions = tuple(calibrated)
            native = {str(d.value("證券代號")): d for d in decisions}
            allocator = PortfolioAllocationV2Pipeline(persistence=session, portfolio_id=request.replay_id)
            context = PortfolioRuntimeContext(
                account_equity=request.account.account_equity, current_cash=request.account.cash,
                positions=tuple(h.position() for h in request.account.holdings),
                default_sector=request.sector, sector_by_symbol={s:request.sector for s in request.symbols})
            missing = (any(selected.diagnostics["missing_bars"].values()) or
                       bool(selected.diagnostics.get("calendar_unknown_dates")))
            raw_context_missing = (selected.diagnostics.get("evidence_schema") == RAW_EVIDENCE_SCHEMA and
                bool(selected.diagnostics.get("missing_taiex_dates") or
                     selected.diagnostics.get("missing_breadth_dates") or selected.diagnostics["invalid_bars"]))
            missing = missing or raw_context_missing
            # Evidence integrity is independent of native scores/Hard Gates.
            # A conflicting publication cannot authorize research allocation,
            # even if another source happens to keep the raw Quality score high.
            publication_conflicts = set()
            for symbol in request.symbols:
                errors = selected.diagnostics.get("raw_filing_errors", {}).get(symbol, [])
                if any(e.get("reason") == "CONFLICTING_PUBLICATION_PROOFS" and
                       (not e.get("conflict_available_at") or aware_timestamp(e["conflict_available_at"]) <= request.as_of)
                       for e in errors) or any(e.get("reason") == "CONFLICTING_FILING_VERSION"
                           for e in selected.diagnostics.get("excluded_evidence", {}).get(symbol, [])):
                    publication_conflicts.add(symbol)
            publication_assumptions = selected.diagnostics.get("publication_time_assumptions", {})
            if selected.account_validation["status"] != "PASS" or missing or publication_conflicts:
                # Preserve raw LES/EREM signals, but incomplete price histories
                # cannot create portfolio actions or persistence successes.
                decisions = allocator.enrich(decisions, context,
                    blocked_reason="REPLAY_PUBLICATION_VERSION_CONFLICT" if publication_conflicts else
                                   "REPLAY_MARKET_GAP_OR_ACCOUNT_MARK_UNKNOWN")
            elif decisions:
                decisions = allocator.enrich(decisions, context)
            rows, models = [], []
            for decision in decisions:
                symbol = str(decision.value("證券代號"))
                row = {field.name: json_value(field.value) for field in decision.fields}
                rows.append(row)
                report = json.loads(decision.value("LES v6 Report"))
                restriction = selected.diagnostics["restrictions"][symbol]
                gates = entry_gate_checks(report)
                validation_error = report_failure(report, allow_replay=True)
                allocation_status = "PASS" if decision.value("PAE Binding") == "NON_BINDING" else (
                    "PASS" if decision.value("PAE Decision") in {
                        "STARTER", "ADD", "STRONG_ADD", "ROTATE_25", "ROTATE_50", "ROTATE_75", "FAST_TRACK_ROTATE"
                    } and (decision.value("PAE Target Weight") or 0) > 0 else "FAIL")
                gates.update(restrictions="PASS" if restriction["status"] == "CLEAR" else restriction["status"],
                             account=selected.account_validation["status"],
                             market_history="UNKNOWN" if (selected.diagnostics["missing_bars"][symbol] or
                                 selected.diagnostics.get("calendar_unknown_dates")) else "PASS",
                             published_report="PASS" if validation_error is None else "FAIL",
                             allocation=allocation_status,
                             kelly="PASS" if decision.value("Half Kelly Weight") is not None else "UNKNOWN")
                gates["calendar"] = "UNKNOWN" if selected.diagnostics.get("calendar_unknown_dates") else "PASS"
                if selected.diagnostics.get("evidence_schema") == RAW_EVIDENCE_SCHEMA:
                    gates["market_context"] = "UNKNOWN" if raw_context_missing else "PASS"
                # An explicitly accepted PDF timestamp is a disclosed research
                # assumption, not a failed Hard Gate or a VERIFIED publication.
                warnings = (["ASSUMED_NOT_VERIFIED:OFFICIAL_PDF_UPLOAD_TIME"]
                            if symbol in publication_assumptions else [])
                blockers = [f"{name}:{status}" for name, status in gates.items() if status != "PASS"]
                if symbol in publication_conflicts:
                    blockers.append("PUBLICATION_VERSION_CONFLICT:UNKNOWN")
                if validation_error:
                    blockers.append(validation_error)
                selected.diagnostics["financials"][symbol].update(
                    coverage=report.get("quality", {}).get("coverage", {}),
                    data_confidence=report.get("data", {}).get("confidence"))
                models.append(dict(symbol=symbol, signal_date=decision.value("日期"),
                    raw=dict(les=native[symbol].value("低點決策"), les_score=native[symbol].value("低點分數"),
                             erem=native[symbol].value("EREM 決策訊號"), erem_score=native[symbol].value("EREM 分數"),
                             pae=decision.value("PAE Decision")),
                    hard_gates=gates, blocking_reasons=blockers,
                    evidence_warnings=warnings,
                    publication_time_status=("ASSUMED_NOT_VERIFIED" if warnings else "NO_PDF_TIME_ASSUMPTION"),
                    research_entry_eligible=not blockers and native[symbol].value("低點決策") in {"BUY", "BUY_SMALL"},
                    live_order_authorized=False, is_fill=False,
                    allocation={k:v for k,v in row.items() if k.startswith("PAE ")},
                    les_report=report,
                    erem={k:v for k,v in row.items() if k.startswith("EREM ")},
                    field_evidence=[json_value(asdict(field)) for field in decision.fields]))
            for symbol in sorted(set(request.symbols) - native.keys()):
                models.append(dict(symbol=symbol, signal_date=None, raw=dict(les=None, erem=None, pae=None),
                    hard_gates={"market_history":"UNKNOWN"}, blocking_reasons=["NO_BARS_IN_EXPLICIT_RANGE"],
                    research_entry_eligible=False, live_order_authorized=False, is_fill=False))
            by_symbol = {item["symbol"]: item for item in models}
            plans = []
            for row in rows:
                if row.get("PAE Role") == "CANDIDATE" and row.get("PAE Atomic Group"):
                    plans.append(dict(kind="EXIT_THEN_ENTER", atomic_group=row["PAE Atomic Group"],
                        exit_symbol=row.get("PAE Incumbent"), enter_symbol=row["證券代號"],
                        target_weight=row.get("PAE Target Weight"),
                        research_approved=by_symbol[row["證券代號"]]["research_entry_eligible"],
                        requires_external_exit_confirmation=True, live_order_authorized=False))
            result = json_value(dict(schema=REPLAY_SCHEMA, evaluation_mode="HISTORICAL_REPLAY",
                model_policy=("CURRENT_PINNED_MODEL_WITH_PDF_UPLOAD_TIME_ASSUMPTION" if publication_assumptions
                              else "CURRENT_PINNED_MODEL_ON_HISTORICALLY_AVAILABLE_EVIDENCE"),
                research_eligibility_policy="DECLARED_PDF_TIME_ASSUMPTION_NON_BLOCKING/v1",
                replay_id=request.replay_id, as_of=request.as_of, history_start=request.history_start,
                history_end=request.history_end, inclusive=True, signal_date=selected.signal_date,
                market=request.market, sector=request.sector, symbols=sorted(request.symbols),
                versions=asdict(request.versions), evidence_id=request.evidence_id,
                selected_evidence_sha256=digest(json_value(selected.evidence)), evidence=selected.evidence,
                account=asdict(request.account), account_sha256=digest(asdict(request.account)),
                account_validation=selected.account_validation,
                pae_state=dict(previous_result_sha256=session.previous_result_sha256,
                               observations=session.observations,
                               persistence=getattr(session, "reads", [])),
                decisions=models, allocation_plans=plans, rows=rows, diagnostics=selected.diagnostics,
                status="NO_TRADING_SESSION" if not selected.signal_date else "RESEARCH_COMPLETED",
                live_order_authorized=False, execution_contract="EXIT_THEN_ENTER_REQUIRES_EXTERNAL_ATOMIC_FILL_CONFIRMATION"))
            # Recheck code/config drift before committing any research state.
            if request.versions != self.versions():
                raise ValueError("REPLAY_VERSION_CHANGED_DURING_EVALUATION")
            session.finish(result)
            return result, coverage_audit(request, selected, result)

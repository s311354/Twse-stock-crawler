"""Quality-First Adaptive Entry v6: opt-in Strategy plugin and execution contract."""

from __future__ import annotations

from contextlib import nullcontext

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import time
from dataclasses import replace
from math import isfinite

import numpy as np
import pandas as pd

from domain.features import FeatureFrame, SharedFeaturePipeline
from domain.numeric_history import metadata_free_frame
from domain.industry_benchmark import aligned_index
from domain.fundamental_models import FundamentalSnapshot
from domain.industry_admission import industry_entry_signal, industry_profile_metadata
from domain.industry_profiles import IndustryProfileRegistry
from domain.models import Stocktype
from domain.strategy import Strategy
from domain.trading_restrictions import restriction_check
from domain.strategies.low_entry_v6_features import build_features
from domain.strategies.low_entry_v6_quality import evaluate_quality, number, snapshot_record
from domain.strategies.low_entry_v6_payoff import payoff_scenarios, resistance_target_research
from domain.strategies.low_entry_v6_valuation import valuation_context
from domain.strategies.low_entry_v6_leadership import recognition
from domain.strategies.low_entry_v6_signals import (
    EXECUTABLE_SIGNALS, SETUPS, LowEntryV6Config, decide, net_reward_risk, risk_pass,
    size_position, trade_plan, technical_observation,
)


V6_CONTEXT_ATTR = "les_v6_context"
PARAMETER_VERSION = "2026-10-04-v6-pit-reported-period-quality-v8"


def parameter_version(config: LowEntryV6Config, convex_weights: Mapping | None = None) -> str:
    if config.decision_policy_version in {3, 4}:
        from domain.strategies.low_entry_v6_hybrid import model_spec
        hybrid_version = model_spec(config)["parameter_version"]
        if convex_weights is not None:
            if config.decision_policy_version != 3:
                raise ValueError("CONVEX_WEIGHTS_REQUIRE_BASELINE_HYBRID_POLICY")
            from domain.strategies.low_entry_v6_weights import FrozenWeights
            return hybrid_version + ":convex:" + FrozenWeights.from_record(dict(convex_weights)).sha256
        return hybrid_version
    return PARAMETER_VERSION


V6_OUTPUT_COLUMNS = (
    "LES v6 Raw Score", "LES v6 Penalty", "LES v6 Adjusted Score", "LES v6 Safety Cap", "LES v6 Final Score",
    "LES v6 Formal Score", "LES v6 Estimated Score", "LES v6 Score Coverage",
    "LES v6 Score", "LES v6 Decision", "LES v6 Setup", "Quality Score", "Data Confidence",
    "Quality Status", "Falling Knife Veto", "Risk Gate", "建議股數", "LES v6 Report",
)
SUPPORTED_SECTORS = {v.name for v in Stocktype} | {
    "ELECTRONICS", "MANUFACTURING", "SOFTWARE", "DISTRIBUTION", "GENERAL_NON_FINANCIAL",
}


def attach_low_entry_v6_context(data: pd.DataFrame, *, symbol: str, sector: str,
                               as_of: str, filings: Sequence[Mapping | FundamentalSnapshot],
                               market: str = "TWSE", taiex: pd.DataFrame | None = None,
                               portfolio: Mapping | None = None, name: str = "",
                               industry_benchmark: Mapping | None = None,
                               convex_weights: Mapping | None = None) -> pd.DataFrame:
    """Attach explicit PIT context, optionally joining benchmark by exact session date.

    Canonical amounts are raw TWD, volume is shares. Financial amounts may use
    a consistent declared unit. This adapter never forwards legacy v5 quality.
    """
    result = data.copy(deep=True)
    if taiex is not None:
        if "Date" not in result or not {"Date", "Close"}.issubset(taiex.columns):
            raise ValueError("Stock and TAIEX require Date for exact alignment")
        dates = pd.to_datetime(taiex["Date"], errors="raise").dt.normalize()
        if dates.duplicated().any():
            raise ValueError("Duplicate TAIEX session")
        benchmark = pd.Series(pd.to_numeric(taiex["Close"], errors="coerce").to_numpy(), index=dates)
        result["TAIEX Close"] = pd.to_datetime(result["Date"]).dt.normalize().map(benchmark)
    result.attrs[V6_CONTEXT_ATTR] = {
        "symbol": str(symbol), "name": name, "sector": sector, "market": market,
        "as_of": as_of, "portfolio": dict(portfolio or {}),
        "filings": [snapshot_record(row, market=market) if isinstance(row, FundamentalSnapshot) else deepcopy(dict(row)) for row in filings],
        "industry_benchmark": deepcopy(industry_benchmark),
    }
    if convex_weights is not None:
        from domain.strategies.low_entry_v6_weights import FrozenWeights
        result.attrs[V6_CONTEXT_ATTR]["convex_weights"] = FrozenWeights.from_record(dict(convex_weights)).record()
    return result


def _clean(value):
    """JSON-safe output: missing numerics are null, never NaN or fabricated zero."""
    kind = type(value)
    # Most financial proof leaves are already Python scalars. Avoid repeated
    # ABC/attribute checks and float conversion for every such leaf; retain
    # the generic path for NumPy scalars, readonly mappings and legacy types.
    if kind in (str, int, bool, type(None)):
        return value
    if kind is float:
        return value if isfinite(value) else None
    if kind is dict or isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if kind in (list, tuple) or isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and number(value) is None:
        return None
    return value


def valuation_evidence(reported_pe: object, pe_history: pd.Series, eps: object) -> dict:
    """Unknown EPS cannot award cheapness or erase an observed expensive warning.

    The official market P/E is separate evidence from our quarterly EPS adapter;
    retain its conservative penalty without pretending EPS was verified.
    """
    market_pe, earnings = number(reported_pe), number(eps)
    market_pe = market_pe if market_pe is not None and market_pe > 0 else None
    history = pe_history.map(number)
    valid = history[history.notna() & history.gt(0)]
    market_rank = (100 * ((valid < market_pe).sum() + .5 * (valid == market_pe).sum()) / 60
                   if market_pe is not None and len(history) == len(valid) == 60 else None)
    pe = market_pe if earnings is not None and earnings > 0 else None
    percentile = market_rank if pe is not None else None
    interpretation = ("NOT_AVAILABLE" if percentile is None else "INEXPENSIVE" if percentile <= 30
                      else "NEUTRAL" if percentile <= 70 else "EXPENSIVE")
    conservative_warning = earnings is None and market_rank is not None and market_rank > 70
    if conservative_warning:
        interpretation = "EXPENSIVE"
    return {"pe": pe, "pe_percentile_60": percentile, "interpretation": interpretation,
            "reported_pe": market_pe, "reported_pe_percentile_60": market_rank,
            "eps_status": "UNKNOWN" if earnings is None else "POSITIVE" if earnings > 0 else "NONPOSITIVE",
            "expensive_warning_basis": "REPORTED_PE_EPS_UNVERIFIED" if conservative_warning else None}


class LowEntryScoreV6Strategy(Strategy):
    name = "low_entry_score_v6"
    version = "6.0.0"
    description = "Quality-first adaptive reversal / pullback / leadership; BUY and BUY_SMALL only"

    def __init__(self, config: LowEntryV6Config | None = None):
        self.config = config or LowEntryV6Config()

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.run_with_features(data, SharedFeaturePipeline().build(data))

    def run_latest(self, data: pd.DataFrame, *, features: FeatureFrame | None = None,
                   context: Mapping | None = None) -> pd.DataFrame:
        """Daily snapshot: warm indicators over history, evaluate only at explicit as_of.

        Historical run() retains close-time PIT evaluation. Snapshot availability
        may be after the final close, e.g. a financial filing published that evening.
        """
        return self.run_with_features(data, features if features is not None else SharedFeaturePipeline().build(data),
                                      latest_only=True, context=context)

    def run_with_features(self, data: pd.DataFrame, features: FeatureFrame, *, latest_only: bool = False,
                          context: Mapping | None = None) -> pd.DataFrame:
        context = data.attrs.get(V6_CONTEXT_ATTR, {}) if context is None else context
        observer = context.get("_stage")
        stage = observer if callable(observer) else lambda name: nullcontext()
        data = metadata_free_frame(data)
        result = (data.tail(1) if latest_only else data).copy(deep=True)
        if data.empty:
            for column in V6_OUTPUT_COLUMNS:
                result[column] = pd.Series(index=result.index, dtype=object)
            return result
        if context.get("convex_weights") is not None:
            from domain.strategies.low_entry_v6_weights import FrozenWeights
            if self.config.decision_policy_version != 3:
                raise ValueError("CONVEX_WEIGHTS_REQUIRE_HYBRID_POLICY")
            FrozenWeights.from_record(context["convex_weights"]).require_compatible(self.config, context.get("as_of"))
        as_of = pd.to_datetime(context.get("as_of"), utc=True, errors="coerce")
        if "Date" not in data or pd.isna(as_of):
            raise ValueError("LES v6 requires Date and an explicit as_of timestamp")
        dates = pd.to_datetime(data["Date"], errors="raise")
        if dates.dt.tz is not None:
            raise ValueError("Date must be an exchange-local session date without timezone")
        if not result.index.is_unique or not dates.eq(dates.dt.normalize()).all() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError("Dates must be unique, increasing session dates")
        closes = dates.dt.tz_localize("Asia/Taipei") + pd.Timedelta(hours=13, minutes=30)
        if (closes > as_of).any():
            raise ValueError("INCOMPLETE_OR_FUTURE_DAILY_BAR")
        if "Complete" in data and not data["Complete"].eq(True).all():
            raise ValueError("INCOMPLETE_DAILY_BAR")
        with stage("les_feature_extension"):
            frame = build_features(data, features)
        rows = []
        positions = (len(frame) - 1,) if latest_only else range(len(frame))
        for position in positions:
            feature = frame.iloc[position]
            decision_time = as_of if latest_only else closes.iloc[position].tz_convert("UTC")
            quality_stage = context.get("_quality_stage")
            with quality_stage() if callable(quality_stage) else nullcontext():
                quality = evaluate_quality(context.get("filings", []), symbol=str(context.get("symbol", "")),
                                           market=context.get("market", "TWSE"), as_of=decision_time,
                                           preprocessing=context.get("_quality_preprocessing"))
            risk_evidence = context.get("hard_risk_evidence", {})
            risk_at = pd.to_datetime(risk_evidence.get("available_at"), utc=True, errors="coerce")
            if pd.notna(risk_at) and risk_at <= decision_time and risk_evidence.get("reasons"):
                quality = replace(quality, status="REJECT", reasons=tuple(risk_evidence["reasons"]))
            elif quality.status != "REJECT":
                if pd.isna(risk_at) and (risk_evidence.get("reasons") or risk_evidence.get("review_reasons")):
                    quality = replace(quality, status="DATA_INSUFFICIENT", reasons=("AUDIT_EVIDENCE_DATE_UNKNOWN",))
                elif pd.notna(risk_at) and risk_at <= decision_time and risk_evidence.get("review_reasons"):
                    quality = replace(quality, status="DATA_INSUFFICIENT", reasons=tuple(risk_evidence["review_reasons"]))
            f = feature.to_dict()
            if self.config.decision_policy_version in {3, 4}:
                # Added only to the Hybrid contract; legacy report payloads stay stable.
                prior_rsi = number(frame["rsi14"].iloc[position - 1]) if position else None
                current_rsi = number(f.get("rsi14"))
                f["rsi_change"] = current_rsi - prior_rsi if current_rsi is not None and prior_rsi is not None else None
                f["score_availability"] = dict(f.get("score_availability", {}))
                f["score_availability"].update(rsi_change=f["rsi_change"] is not None,
                    rsi14=current_rsi is not None,
                    volume_ma20_strict=f["score_availability"].get("volume_ratio", False))
            restriction = None
            if "restriction_evidence" in context:
                restriction = restriction_check(
                    context["restriction_evidence"], market_data_date=dates.iloc[position].date().isoformat(),
                    decision_time=decision_time, market=context.get("market", "TWSE"),
                    allow_historical=context.get("historical_replay") is True,
                    symbol=str(context.get("symbol", "")),
                )
                f["restricted"] = restriction["restricted"]
                if self.config.decision_policy_version in {3, 4}:
                    # Daily raw bars intentionally have no restriction flag.
                    # Replace their availability mask only after the separate
                    # cutoff/market-aware evidence check, never from a quote or
                    # the absence of a symbol in an unverified list.
                    f["score_availability"]["restricted"] = restriction["restricted"] is not None
            eps = quality.metrics.get("basic_eps_ttm")
            history = pd.to_numeric(frame["pe"].iloc[max(0, position - 59):position + 1], errors="coerce")
            valuation = valuation_evidence(f.get("pe"), history, eps)
            sector = str(context.get("sector", "UNKNOWN")).upper()
            start = max(0, position - 60)
            industry_index, industry_evidence = aligned_index(context.get("industry_benchmark"),
                dates.iloc[start:position + 1], symbol=str(context.get("symbol", "")),
                sector=sector, market=context.get("market", "TWSE"), as_of=str(decision_time))
            stock = frame["close"].iloc[start:position + 1]
            f["score_availability"] = dict(f.get("score_availability", {}))
            for period in (20, 60):
                paired = pd.concat([stock.tail(period + 1), industry_index.tail(period + 1)], axis=1)
                known = len(paired) == period + 1 and np.isfinite(paired).all().all() and paired.gt(0).all().all()
                f[f"industry_rs{period}"] = (paired.iloc[-1, 0] / paired.iloc[0, 0]
                    - paired.iloc[-1, 1] / paired.iloc[0, 1]) if known else None
                f["score_availability"][f"industry_rs{period}"] = known
            f["industry_status"] = industry_evidence["status"]
            profile = IndustryProfileRegistry.resolve(sector)
            industry_mode = industry_profile_metadata(profile).get("mode", "UNSUPPORTED") if profile else "UNSUPPORTED"
            with stage("les_decision"):
                decision = decide(quality, f, valuation=valuation["interpretation"], config=self.config,
                                  industry_mode=industry_mode, convex_weights=context.get("convex_weights"),
                                  as_of=decision_time.isoformat())
            if self.config.decision_policy_version == 1 and profile is None and sector not in SUPPORTED_SECTORS:
                decision.update(signal="NO_MODEL", reasons=["SECTOR_UNSUPPORTED"], risk_fraction=0.)
                decision["technical_observation"] = technical_observation(
                    quality, f, decision, valuation=valuation["interpretation"], config=self.config,
                )
            shadow_signal = decision["signal"]
            admitted = (industry_entry_signal(shadow_signal, profile)
                        if self.config.decision_policy_version == 1 else shadow_signal)
            if admitted != shadow_signal:
                decision.update(signal=admitted, risk_fraction=0., position_multiplier=0.)
                decision["reasons"] = list(decision["reasons"]) + [
                    "INDUSTRY_PROFILE_SHADOW_MODE" if profile else "SECTOR_UNSUPPORTED"
                ]
            account = context.get("portfolio", {})
            shares = size_position(decision, portfolio_value=account.get("value"), available_cash=account.get("available_cash"),
                                   intended_weight=account.get("intended_weight"), portfolio_capacity_weight=account.get("capacity_weight"),
                                   amount_ma20=f.get("amount_ma20"), config=self.config)
            with stage("les_portfolio_valuation"):
                portfolio_valuation = valuation_context(context.get("filings", []),
                    symbol=str(context.get("symbol", "")), market=context.get("market", "TWSE"),
                    as_of=decision_time, quality=quality,
                    preprocessing=context.get("_quality_preprocessing"))
            report = {
                "symbol": str(context.get("symbol", "")), "name": context.get("name", ""),
                "model_version": self.version, "parameter_version": parameter_version(self.config, context.get("convex_weights")),
                "parameters": vars(self.config),
                "industry_profile": dict(industry_profile_metadata(profile), shadow_signal=shadow_signal),
                "signal_date": dates.iloc[position].date().isoformat(), "decision_time": decision_time.isoformat(),
                "data": {"integrity": quality.integrity, "confidence": quality.confidence,
                         "completed_history": bool(f["history_ready"]), "market_aligned": bool(f["market_ready"]),
                         "stock_history_observations": f.get("stock_history_observations"), "stock_history_required": 60,
                         "market_history_observations": f.get("market_history_observations"), "market_history_required": 21},
                "quality": {"score": quality.score, "status": quality.status, "components": quality.components,
                            **{f"{key}_score": value for key, value in quality.components.items()},
                            "coverage": quality.coverage, "metrics": quality.metrics,
                            "warnings": quality.warnings, "reasons": quality.reasons, "optional": quality.optional},
                "valuation": valuation,
                "portfolio_valuation": portfolio_valuation,
                "market": {"regime": f["market_regime"], "exchange": context.get("market", "TWSE")},
                "relative_strength": dict({key: f[key] for key in ("rs20", "rs60", "rs_slope5")}, correction_resilience=f["resilience"]),
                "leadership": recognition(f), "industry_benchmark": industry_evidence,
                "setups": (dict(selection=decision["selection"], selected_setup=decision["setup"])
                    if self.config.decision_policy_version in {3, 4} else
                    dict(decision["setups"], **{f"{key.lower()}_score": value["score"] for key, value in decision["setups"].items()}, selected_setup=decision["setup"], les_v6=decision["score"])),
                "downtrend_gate": {"hard_veto": bool(decision["hard_vetoes"]), "reasons": decision["hard_vetoes"],
                                   "warnings": [reason for reason, active in (
                                       ("ATR_EXPANSION", number(f.get("atr_expansion")) is not None and f["atr_expansion"] > 1.2),
                                       ("WEAK_VOLUME_WITHOUT_SUPPORT_BREAK", f["clv"] < -.5 and f["volume_ratio"] > 1.5 and not f["support_break"]),
                                   ) if active]},
                "technical": f,
                "entry": dict(decision["plan"], starter_trigger=decision["starter_trigger"], confirmed_trigger=decision["confirmed_trigger"],
                              rule="NEXT_SESSION_BREAKS_SIGNAL_HIGH", trigger_price=f.get("high"),
                              earliest_execution="NEXT_TRADING_SESSION", is_fill=False),
                "payoff_scenarios": payoff_scenarios(decision["plan"], buy_cost_rate=self.config.buy_fee_rate,
                                                     sell_cost_rate=self.config.sell_cost_rate),
                "target_policy_research": resistance_target_research(
                    decision["plan"], f, buy_cost_rate=self.config.buy_fee_rate,
                    sell_cost_rate=self.config.sell_cost_rate),
                "liquidity": {"passed": decision["plan"]["liquidity_pass"], "spread": f.get("spread"), "trading_value_ma20": f.get("amount_ma20")},
                "decision": {key: value for key, value in decision.items() if key not in {"setups", "plan"}},
                "sizing": {"indicative_shares": shares, "portfolio_data_pass": shares > 0,
                           "reason": "AWAIT_FRESH_EXECUTION_CHECK" if shares > 0 else "NO_CAPACITY_OR_MISSING_PORTFOLIO_DATA"},
            }
            if context.get("historical_replay") is True:
                report["evaluation_mode"] = "HISTORICAL_REPLAY"
                report["replay_as_of"] = str(decision_time)
            if context.get("convex_weights") is not None:
                # Weight provenance must not erase the historical replay boundary.
                report["weight_evaluation_mode"] = "AS_OF_SNAPSHOT" if latest_only else "CLOSE_TIME_REPLAY"
                report.setdefault("evaluation_mode", "CONVEX_WEIGHT_SNAPSHOT" if latest_only else "CONVEX_WEIGHT_CLOSE_TIME_REPLAY")
            report["decision"]["conviction"] = "HIGH" if decision["signal"] == "BUY" else "STARTER" if decision["signal"] == "BUY_SMALL" else "NONE"
            if restriction is not None:
                report["data"].update(restriction_check=restriction,
                                      restriction_evidence=deepcopy(context["restriction_evidence"]))
            with stage("les_report_normalize"):
                clean_report = _clean(report)
            rows.append({
                "LES v6 Raw Score": decision.get("raw_score"),
                "LES v6 Penalty": decision.get("penalty"),
                "LES v6 Adjusted Score": decision.get("adjusted_score"),
                "LES v6 Safety Cap": decision.get("safety_cap"),
                "LES v6 Final Score": decision.get("final_score"),
                "LES v6 Formal Score": decision["formal_score"],
                "LES v6 Estimated Score": decision["estimated_score"],
                "LES v6 Score Coverage": decision["score_coverage"],
                "LES v6 Score": decision["score"], "LES v6 Decision": decision["signal"], "LES v6 Setup": decision["setup"],
                "Quality Score": quality.score, "Data Confidence": quality.confidence, "Quality Status": quality.status,
                "Falling Knife Veto": bool(decision["hard_vetoes"]), "Risk Gate": decision["risk_gate"],
                "建議股數": shares, "LES v6 Report": clean_report,
            })
        for column in V6_OUTPUT_COLUMNS:
            result[column] = [row[column] for row in rows]
        return result


class LowEntryScoreV6HybridStrategy(LowEntryScoreV6Strategy):
    """Context-first fixed factors, explicitly deployed by the daily pipeline."""
    name = "low_entry_score_v6_hybrid"
    description = "Context-first six-factor score; uncalibrated research candidate"

    def __init__(self, config: LowEntryV6Config | None = None):
        from domain.strategies.low_entry_v6_signals import relaxed_hybrid_config
        config = config or relaxed_hybrid_config()
        if config.decision_policy_version not in {3, 4}:
            raise ValueError("HYBRID_PLUGIN_REQUIRES_POLICY_3_OR_4")
        super().__init__(config)


class LowEntryScoreV6HybridLegacyStrategy(LowEntryScoreV6HybridStrategy):
    """Explicit 75/65 comparison policy; never migrate archived decisions."""
    name = "low_entry_score_v6_hybrid_legacy"
    description = "Baseline Hybrid 75/65 with original six-factor budgets"

    def __init__(self):
        super().__init__(LowEntryV6Config(decision_policy_version=3))


def assess_execution(report: Mapping, quote: Mapping, *, now: str,
                     trading_sessions: Sequence[str], portfolio: Mapping,
                     config: LowEntryV6Config | None = None, hard_risk: bool = False) -> dict:
    """No broker calls. Revalidate the published signal without recalculating LES.

    trading_sessions must come from the caller's verified exchange calendar.
    Portfolio capacity already incorporates holdings, pending orders, PAE,
    sector and correlation constraints; no capacity is inferred from cash.
    """
    from domain.strategies.low_entry_v6_report import report_config, report_weights
    def skip(reason):
        return {"action": "SKIP", "shares": 0, "reason": reason}
    try:
        config = report_config(report, config)
    except (TypeError, ValueError):
        return skip("EXECUTION_CONFIG_MISMATCH")
    if report.get("evaluation_mode") == "HISTORICAL_REPLAY":
        return skip("HISTORICAL_REPLAY_NOT_EXECUTABLE")
    decision = deepcopy(report.get("decision", {}))
    if decision.get("signal") not in EXECUTABLE_SIGNALS or report.get("quality", {}).get("status") != "PASS" or report.get("downtrend_gate", {}).get("hard_veto") is not False or hard_risk:
        return skip("INELIGIBLE_PUBLISHED_SIGNAL_OR_HARD_RISK")
    try:
        expected_version = parameter_version(config, report_weights(report))
    except (TypeError, ValueError, KeyError):
        return skip("PARAMETER_VERSION_MISMATCH")
    if report.get("parameter_version") != expected_version:
        return skip("PARAMETER_VERSION_MISMATCH")
    if report.get("parameters") != vars(config):
        return skip("EXECUTION_CONFIG_MISMATCH")
    if (decision.get("entry_tier") not in {"CONFIRMED", "STARTER_ONLY"}
            or (decision["signal"] == "BUY" and decision["entry_tier"] != "CONFIRMED")):
        return skip("INVALID_SETUP_ENTRY_TIER")
    if "restriction_evidence" in report.get("data", {}):
        market_info = report.get("market", {})
        checked = restriction_check(report["data"]["restriction_evidence"],
                                    market_data_date=report.get("signal_date", ""),
                                    decision_time=report.get("decision_time"),
                                    market=market_info.get("exchange", "TWSE") if isinstance(market_info, Mapping) else None)
        if checked["restricted"] is not False:
            return skip("INVALID_PUBLISHED_RESTRICTION_EVIDENCE")
    # Also protect direct plugin callers, not only the stdin execution bridge.
    # Import at call time because serialized-report validation uses this module's
    # version contract. It does not recalculate unfinished intraday indicators.
    from domain.strategies.low_entry_v6_report import report_failure
    failure = report_failure(report, config=config)
    if failure:
        return skip(failure)
    timestamp = pd.to_datetime(quote.get("timestamp"), utc=True, errors="coerce")
    current = pd.to_datetime(now, utc=True, errors="coerce")
    if pd.isna(timestamp) or pd.isna(current) or not 0 <= (current - timestamp).total_seconds() <= config.quote_max_age_seconds:
        return skip("STALE_OR_MISSING_QUOTE")
    published = pd.to_datetime(report.get("decision_time"), utc=True, errors="coerce")
    if pd.isna(published) or published > current:
        return skip("FUTURE_OR_MISSING_PUBLISHED_DECISION")
    sessions = list(trading_sessions)
    if sessions != sorted(set(sessions)):
        return skip("INVALID_SESSION_CALENDAR")
    try:
        expected = sessions[sessions.index(report["signal_date"]) + 1]
    except (ValueError, IndexError, KeyError):
        return skip("NO_VERIFIED_NEXT_SESSION")
    local = current.tz_convert("Asia/Taipei")
    quote_local = timestamp.tz_convert("Asia/Taipei")
    if (local.date().isoformat() != expected or quote_local.date() != local.date()
            or not time(9) <= local.time() <= time(13, 30)
            or not time(9) <= quote_local.time() <= time(13, 30)):
        return skip("OUTSIDE_NEXT_SESSION")
    price, bid, ask = (number(quote.get(key)) for key in ("price", "bid", "ask"))
    if str(quote.get("symbol")) != report.get("symbol") or price is None or price <= 0 or bid is None or ask is None or not 0 < bid <= ask or quote.get("restricted") is not False:
        return skip("INVALID_QUOTE_OR_RESTRICTION")
    trigger = number(report.get("entry", {}).get("trigger_price"))
    if trigger is None or price <= trigger:
        return skip("TRIGGER_NOT_REACHED")
    price = max(price, ask)
    f = dict(report["technical"], spread=(ask - bid) / ((ask + bid) / 2), restricted=False)
    if config.decision_policy_version in {3, 4}:
        from domain.strategies.low_entry_v6_hybrid import known_features
        f = known_features(f)
    atr = number(f.get("atr14"))
    if atr is None or atr <= 0:
        return skip("INVALID_ATR")
    f["ema20_extension"] = (price - f["ema20"]) / atr
    f["breakout_extension"] = (price - f["hh20"]) / atr
    plan = trade_plan(f, price, config=config, setup=decision["setup"])
    if (plan.get("stop_source") != report["entry"].get("stop_source")
            or plan.get("stop_price") != report["entry"].get("stop_price")):
        return skip("PUBLISHED_STOP_MISMATCH")
    # A fixed planned resistance cannot disappear just because a gap passed it.
    target = report["entry"].get("target_price")
    if target is not None:
        plan["target_price"] = target
        plan["reward_risk"] = (target - price) / (price - plan["stop_price"]) if plan["stop_price"] is not None and price > plan["stop_price"] else None
    plan["net_reward_risk"] = net_reward_risk(plan, config)
    risk_setup = decision["setup"]
    if config.decision_policy_version in {3, 4} and risk_setup == "NO_CLEAR_CONTEXT":
        risk_setup = "REVERSAL"
    valid, _, _ = risk_pass(risk_setup, decision["signal"], f, plan, report["valuation"]["interpretation"], config)
    if not valid or not plan["liquidity_pass"]:
        return skip("REPRICED_RISK_OR_LIQUIDITY_FAIL")
    decision["plan"] = plan
    shares = size_position(decision, portfolio_value=portfolio.get("value"), available_cash=portfolio.get("available_cash"),
                           intended_weight=portfolio.get("intended_weight"), portfolio_capacity_weight=portfolio.get("capacity_weight"),
                           amount_ma20=f.get("amount_ma20"), config=config)
    if shares <= 0:
        return skip("ZERO_CAPACITY_OR_MISSING_PORTFOLIO_DATA")
    return {"action": "ELIGIBLE", "signal": decision["signal"], "shares": shares,
            "reference_price": price, "stop_price": plan["stop_price"], "is_fill": False,
            "payoff_scenarios": payoff_scenarios(plan, buy_cost_rate=config.buy_fee_rate,
                                                 sell_cost_rate=config.sell_cost_rate),
            "reason": "FRESH_TRIGGER_RISK_LIQUIDITY_PORTFOLIO_PASS"}


def gate_funnel(reports: Sequence[Mapping]) -> dict[str, int]:
    """Count each symbol once at a single decision date, not warm-up/history rows."""
    keys = [report["symbol"] for report in reports]
    if len(keys) != len(set(keys)) or len({r["signal_date"] for r in reports}) > 1:
        raise ValueError("Funnel requires one report per symbol on the same signal date")
    counts = Counter({key: 0 for key in (
        "universe_count", "data_integrity_pass_count", "quality_pass_count", "provisional_count",
        "watch_quality_count", "downtrend_veto_count", "risk_gate_fail_count", "liquidity_fail_count",
    )})
    prefixes = dict(zip(SETUPS, ("reversal", "pullback", "leadership")))
    for prefix in prefixes.values():
        for signal in ("watch", "armed", "buy_small", "buy"):
            counts[f"{prefix}_{signal}_count"] = 0
    for report in reports:
        counts["universe_count"] += 1
        counts["data_integrity_pass_count"] += report["data"]["integrity"] == "PASS" and report["data"]["completed_history"] and report["data"]["market_aligned"]
        for status, key in (("PASS", "quality_pass_count"), ("PROVISIONAL", "provisional_count"), ("WATCH_QUALITY", "watch_quality_count")):
            counts[key] += report["quality"]["status"] == status
        counts["downtrend_veto_count"] += report["downtrend_gate"]["hard_veto"]
        counts["risk_gate_fail_count"] += report["decision"]["risk_gate"] == "FAIL"
        counts["liquidity_fail_count"] += not report["liquidity"]["passed"]
        decision = report["decision"]
        if decision["setup"] in prefixes and decision["signal"] in {"WATCH", "ARMED", "BUY_SMALL", "BUY"}:
            counts[f"{prefixes[decision['setup']]}_{decision['signal'].lower()}_count"] += 1
    return dict(counts)

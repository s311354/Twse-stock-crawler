"""Recompute crawler PAE output for one owner-scoped paper account.

The crawler remains the source of LES/EREM/PAE policy. This bridge only adapts
its latest analysis CSV and the web app's current account state to the typed
PAE aggregate, then emits a bounded JSON contract for the Node service.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .contract import CONTRACT_NAME as INTEGRATION_CONTRACT_NAME
from .contract import CONTRACT_VERSION as INTEGRATION_CONTRACT_VERSION
from .contract import PACKAGE_VERSION as INTEGRATION_PACKAGE_VERSION

MAX_INPUT_BYTES = 512 * 1024
MAX_DATASET_BYTES = 64 * 1024 * 1024
MAX_DATASET_ROWS = 20_000
MAX_OUTPUT_LINES = 24
REQUIRED_HISTORY_SESSIONS = 500
PAE_EXECUTABLE_ACTIONS = {
    "ADD",
    "FAST_TRACK_ROTATE",
    "ROTATE_25",
    "ROTATE_50",
    "ROTATE_75",
    "STARTER",
    "STRONG_ADD",
}
ACTION_PRIORITY = {
    "EXIT": 0,
    "EXIT_AND_REALLOCATE": 0,
    "EXIT_TO_CASH": 0,
    "FAST_TRACK_ROTATE": 1,
    "ROTATE_75": 1,
    "ROTATE_50": 1,
    "ROTATE_25": 1,
    "STRONG_ADD": 2,
    "ADD": 2,
    "STARTER": 2,
    "REDUCE_25": 3,
    "HOLD": 4,
    "HOLD_REVIEW": 4,
    "HOLD_NO_ADD": 4,
    "WATCH": 5,
    "STRATEGIC_WATCH": 5,
    "CASH": 6,
    "DATA_INSUFFICIENT": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crawler-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Crawler checkout root. Defaults to the installed package root.",
    )
    parser.add_argument(
        "--contract-version",
        required=True,
        type=int,
        choices=(INTEGRATION_CONTRACT_VERSION,),
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--history-db", required=True, type=Path)
    parser.add_argument("--market-type", default="13")
    parser.add_argument("--industry", default="電子工業")
    return parser.parse_args()


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    rendered = str(value).strip()
    return rendered if rendered and rendered.lower() != "nan" else default


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return text(value).lower() in {"1", "true", "yes", "pass"}


def clean(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def load_request() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("REQUEST_TOO_LARGE")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("REQUEST_MUST_BE_OBJECT")
    return payload


def load_history_sessions(path: Path, market_type: str) -> int:
    if not path.is_file():
        return 0
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """
            SELECT dataset, COUNT(DISTINCT trade_date)
            FROM market_history_cache
            WHERE schema_version = 1
              AND ((dataset = 'STOCK_TABLE' AND market_type = ?)
                   OR (dataset = 'TAIEX' AND market_type = 'IND'))
            GROUP BY dataset
            """,
            (market_type,),
        ).fetchall()
    counts = {str(dataset): int(count) for dataset, count in rows}
    return min(counts.get("STOCK_TABLE", 0), counts.get("TAIEX", 0))


def position_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    account = payload.get("account")
    if not isinstance(account, dict):
        raise ValueError("ACCOUNT_REQUIRED")
    positions = account.get("positions", [])
    if not isinstance(positions, list) or len(positions) > 100:
        raise ValueError("ACCOUNT_POSITIONS_INVALID")
    result: dict[str, dict[str, Any]] = {}
    for item in positions:
        if not isinstance(item, dict):
            raise ValueError("ACCOUNT_POSITION_INVALID")
        code = text(item.get("code"))
        shares = finite(item.get("shares"))
        buy_price = finite(item.get("buyPrice"))
        current_price = finite(item.get("currentPrice"))
        entry_les_score = finite(
            item.get("entryLesScore", item.get("signalScore"))
        )
        entry_decision = text(item.get("entryDecision")).upper().replace("-", "_").replace(" ", "_")
        entry_decision = entry_decision or None
        buy_time = text(item.get("buyTime"))
        try:
            parsed_buy_time = dt.datetime.fromisoformat(
                buy_time.replace("Z", "+00:00")
            )
            if parsed_buy_time.tzinfo is not None:
                parsed_buy_time = parsed_buy_time.astimezone(
                    dt.timezone(dt.timedelta(hours=8))
                )
            buy_date = parsed_buy_time.date()
        except (TypeError, ValueError):
            buy_date = None
        if (
            not code.isdigit()
            or code in result
            or shares is None
            or shares <= 0
            or buy_price is None
            or buy_price <= 0
            or current_price is None
            or current_price <= 0
            or entry_les_score is None
            or entry_les_score < 0
            or entry_les_score > 100
            or entry_decision not in {None, "BUY", "EARLY_BUY", "BUY_SMALL"}
            or buy_date is None
        ):
            raise ValueError("ACCOUNT_POSITION_INVALID")
        result[code] = {
            **item,
            "shares": shares,
            "buyPrice": buy_price,
            "currentPrice": current_price,
            "entryLesScore": entry_les_score,
            "entryDecision": entry_decision,
            "buyDate": buy_date,
        }
    return result


def apply_owner_erem(frame: Any, positions: dict[str, dict[str, Any]], pd: Any) -> None:
    """Re-anchor shared EREM market scores to one owner's entry evidence."""
    if not positions:
        return
    parsed_dates = pd.to_datetime(frame["日期"], errors="coerce")
    scores = pd.to_numeric(frame["EREM 分數"], errors="coerce")
    for code, position in positions.items():
        indices = frame.index[frame["證券代號"].astype("string").str.strip() == code]
        ordered = sorted(indices, key=lambda index: parsed_dates.loc[index])
        exit_streak = 0
        active = [
            index
            for index in ordered
            if not pd.isna(parsed_dates.loc[index])
            and parsed_dates.loc[index].date() >= position["buyDate"]
        ]
        for holding_days, index in enumerate(active):
            signal_date = parsed_dates.loc[index]
            current_score = finite(scores.loc[index])
            if pd.isna(signal_date) or current_score is None:
                continue
            entry_weight = max(0.10, 0.40 * math.exp(-holding_days / 20.0))
            holding_confidence = max(
                0.0,
                min(
                    100.0,
                    (1.0 - entry_weight) * current_score
                    + entry_weight * position["entryLesScore"],
                ),
            )
            exit_streak = exit_streak + 1 if holding_confidence < 40.0 else 0
            entry_decision = position.get("entryDecision")
            buy_grace = (
                entry_decision == "BUY"
                and position["entryLesScore"] >= 75.0
                and holding_days < 3
            )
            buy_small_grace = (
                entry_decision == "BUY_SMALL"
                and holding_days < 2
            )
            early_buy_grace = (
                entry_decision == "EARLY_BUY"
                and holding_days < 2
            )
            grace_profile = (
                "BUY_3_SESSION" if buy_grace
                else "EARLY_BUY_2_SESSION" if early_buy_grace
                else "BUY_SMALL_2_SESSION" if buy_small_grace
                else "NONE"
            )
            if holding_confidence < 40.0:
                decision = "EXIT"
                reduction = 100.0
            elif holding_confidence < 55.0:
                if buy_grace or early_buy_grace or buy_small_grace:
                    decision = "HOLD_REVIEW"
                    reduction = 0.0
                else:
                    continuous = 25.0 + ((54.0 - holding_confidence) / 14.0 * 25.0)
                    reduction = max(25.0, min(50.0, math.floor((continuous + 2.5) / 5.0) * 5.0))
                    decision = f"REDUCE {int(reduction)}%"
            elif holding_confidence < 85.0:
                decision = "HOLD"
                reduction = 0.0
            else:
                decision = "STRONG HOLD"
                reduction = 0.0
            frame.at[index, "EREM Holding Confidence"] = round(holding_confidence, 2)
            frame.at[index, "EREM Entry LES"] = round(position["entryLesScore"], 2)
            frame.at[index, "EREM Entry Decision"] = entry_decision
            frame.at[index, "EREM Grace Profile"] = grace_profile
            frame.at[index, "EREM Holding Days"] = holding_days
            frame.at[index, "EREM Entry Weight"] = round(entry_weight, 4)
            frame.at[index, "EREM Exit Streak Count"] = exit_streak
            frame.at[index, "EREM 決策訊號"] = decision
            frame.at[index, "EREM 建議減倉比例"] = reduction


def persistence_evidence(payload: dict[str, Any], evidence_type: Any) -> tuple[Any, ...]:
    values = payload.get("persistence", [])
    if not isinstance(values, list) or len(values) > 500:
        raise ValueError("PERSISTENCE_INVALID")
    output = []
    for item in values:
        if not isinstance(item, dict):
            continue
        incumbent = text(item.get("incumbentSymbol"))
        challenger = text(item.get("challengerSymbol"))
        observations = item.get(
            "recentPasses",
            item.get("recentExcessOpportunities", []),
        )
        if not incumbent or not challenger or not isinstance(observations, list):
            continue
        parsed = tuple(
            entry if isinstance(entry, bool) else bool(finite(entry, 0.0) > 0)
            for entry in observations[-3:]
        )
        if parsed:
            output.append(evidence_type(incumbent, challenger, parsed))
    return tuple(output)


def nullable_gate(value: Any) -> str:
    cleaned = clean(value)
    if cleaned is None or text(cleaned).upper() in {"", "N/A", "NONE", "NOT_APPLICABLE"}:
        return "NOT_APPLICABLE"
    return "PASS" if boolean(cleaned) else "FAIL"


def quality_status(value: Any) -> str:
    normalized = text(value, "UNKNOWN").upper()
    return {
        "QUALITY_CONFIRMED": "PASS",
        "QUALITY_PROVISIONAL": "PROVISIONAL",
        "QUALITY_WATCH": "WATCH_QUALITY",
        "QUALITY_REJECT": "REJECT",
    }.get(normalized, normalized)


def security_v2(source: Any, position: dict[str, Any] | None, account_equity: float, security_type: Any) -> Any:
    code = text(source.get("證券代號"))
    current_price = finite(source.get("Current Price"), finite(source.get("收盤")))
    current_weight = finite(source.get("Current Portfolio Weight"), 0.0) or 0.0
    position_cap = finite(source.get("Position Fraction"))
    executable_shares = finite(source.get("Executable Shares"), 0.0) or 0.0
    base_confidence = finite(source.get("Base Data Confidence"))
    extended_confidence = finite(source.get("Extended Data Confidence"))
    confidences = [value for value in (base_confidence, extended_confidence) if value is not None]
    data_confidence = min(confidences) if confidences else None
    quality_confidence = finite(source.get("Quality Confidence"), data_confidence)
    hard_rejects = [
        item.strip()
        for item in text(source.get("Quality Hard Rejects"), "NONE").split("|")
        if item.strip() and item.strip().upper() not in {"NONE", "N/A", "NAN"}
    ]
    liquidity = text(source.get("Liquidity Status")).upper()
    if liquidity not in {"PASS", "FAIL", "UNKNOWN"}:
        risk_reasons = text(source.get("Risk Gate Reasons")).upper()
        liquidity = "FAIL" if "LIQUIDITY" in risk_reasons else "PASS" if executable_shares > 0 else "UNKNOWN"
    stop_price = finite(source.get("策略停損"))
    erem_confidence = finite(source.get("EREM Holding Confidence"))
    return security_type(
        stock_id=code,
        sector_id=text(source.get("Sector"), "ELECTRONICS"),
        correlation_cluster_id=text(source.get("Correlation Cluster"), text(source.get("Sector"), "ELECTRONICS")),
        quality_status=quality_status(source.get("LES v5 Final Quality Status")),
        quality_score=finite(source.get("LES v5 Final Quality Score")),
        quality_confidence=quality_confidence,
        data_confidence=data_confidence,
        valuation_status=text(source.get("Valuation Model Status"), "UNKNOWN"),
        valuation_gate=text(source.get("Valuation Gate"), "UNKNOWN"),
        bear_cagr=finite(source.get("Bear Case Expected CAGR")),
        base_cagr=finite(source.get("Base Case Expected CAGR")),
        bull_cagr=finite(source.get("Bull Case Expected CAGR")),
        valuation_confidence=finite(source.get("Valuation Confidence")),
        business_uncertainty_penalty=finite(source.get("Business Uncertainty Penalty"), 0.0) or 0.0,
        les_decision=text(source.get("低點決策"), "DATA_INSUFFICIENT"),
        risk_gate=text(source.get("Risk Gate"), "UNKNOWN"),
        falling_knife_state=text(source.get("Falling Knife State"), "UNKNOWN"),
        liquidity_status=liquidity,
        executable_shares=executable_shares,
        les_position_cap=position_cap,
        risk_budget_weight=finite(source.get("Risk Budget Weight"), position_cap),
        liquidity_weight_cap=finite(source.get("Liquidity Weight Cap"), position_cap),
        half_kelly_weight=finite(source.get("Half Kelly Weight")),
        erem_decision=text(source.get("EREM 決策訊號"), "DATA_INSUFFICIENT").replace(" ", "_").replace("%", ""),
        erem_holding_confidence=erem_confidence,
        erem_data_sufficient=erem_confidence is not None,
        current_weight=max(0.0, min(1.0, current_weight)),
        current_price=current_price,
        stop_price=stop_price,
        fundamental_hard_exit=bool(hard_rejects),
        strategy_stop_breached=bool(position and current_price and stop_price and current_price <= stop_price),
        cyclical_state=text(source.get("Cyclical State"), "CYCLICAL_CLEAR"),
    )


def row_payload(source: Any, output: Any) -> dict[str, Any]:
    average_volume_20 = finite(source.get("EREM Volume MA20"))
    atr14 = finite(source.get("ATR14"))
    strategy_stop = finite(source.get("策略停損"))
    strategy_target = finite(source.get("策略停利"))
    confirmed_swing_low = (
        strategy_stop + 0.3 * atr14
        if strategy_stop is not None and atr14 is not None and atr14 > 0
        else None
    )
    quality_gate = nullable_gate(output.get("PAE Quality Dominance Gate"))
    cer_gate = nullable_gate(output.get("PAE CER Dominance Gate"))
    dual_gate = (
        "PASS" if quality_gate == "PASS" and cer_gate == "PASS"
        else "FAIL" if "FAIL" in {quality_gate, cer_gate}
        else "NOT_APPLICABLE"
    )
    reduction_fraction = finite(source.get("EREM 建議減倉比例"), 0.0)
    if reduction_fraction is not None and reduction_fraction > 1.0:
        reduction_fraction /= 100.0
    return {
        "code": text(source.get("證券代號")),
        "name": text(source.get("證券名稱"), text(source.get("證券代號"))),
        "date": text(source.get("日期"))[:10],
        "role": text(output.get("PAE Role"), "CANDIDATE"),
        "action": text(output.get("PAE Decision"), "DATA_INSUFFICIENT"),
        "reason": text(output.get("PAE Reason"), "UNAVAILABLE"),
        "close": finite(source.get("收盤")),
        "rewardRisk": finite(source.get("RewardRisk")),
        "averageVolume20Lots": (
            average_volume_20 / 1000.0
            if average_volume_20 is not None and average_volume_20 > 0
            else None
        ),
        "volumeRatio": finite(source.get("Volume Ratio")),
        "lowEntry": {
            "decision": text(source.get("低點決策"), "DATA_INSUFFICIENT"),
            "score": finite(source.get("低點分數")),
            "qualityStatus": text(source.get("LES v5 Final Quality Status"), "DATA_INSUFFICIENT"),
            "qualityScore": finite(source.get("LES v5 Final Quality Score")),
            "qualityHardRejects": [
                item.strip()
                for item in text(source.get("Quality Hard Rejects"), "NONE").split("|")
                if item.strip() and item.strip().upper() != "NONE"
            ],
            "valuationGate": text(source.get("Valuation Gate"), "UNKNOWN"),
            "valuationStatus": text(source.get("Valuation Model Status"), "UNKNOWN"),
            "riskGate": text(source.get("Risk Gate"), "UNKNOWN"),
            "fallingKnifeState": text(source.get("Falling Knife State"), "DATA_INSUFFICIENT"),
            "atr14": atr14,
            "confirmedSwingLow": confirmed_swing_low,
            "entryPrice": finite(source.get("Entry Candidate")),
            "positionCap": finite(source.get("Position Fraction"), 0.0),
            "priorHigh60": strategy_target,
            "strategyStop": strategy_stop,
            "strategyTarget": strategy_target,
            "ema20": finite(source.get("EREM EMA20")),
            "supportPrice": confirmed_swing_low,
        },
        "erem": {
            "holdingConfidence": finite(source.get("EREM Holding Confidence")),
            "decision": text(source.get("EREM 決策訊號"), "DATA_INSUFFICIENT"),
            "reductionFraction": reduction_fraction,
            "exitStreak": int(finite(source.get("EREM Exit Streak Count"), 0.0) or 0),
            "entryLesScore": finite(source.get("EREM Entry LES")),
            "entryDecision": text(source.get("EREM Entry Decision")) or None,
            "graceProfile": text(source.get("EREM Grace Profile"), "NONE"),
            "holdingDays": int(finite(source.get("EREM Holding Days"), 0.0) or 0)
            if finite(source.get("EREM Holding Days")) is not None
            else None,
        },
        "pae": {
            "status": text(output.get("PAE Status"), "DATA_INSUFFICIENT"),
            "binding": text(output.get("PAE Binding"), "UNKNOWN"),
            "mode": text(output.get("PAE Mode"), "UNKNOWN"),
            "cer": finite(output.get("PAE CER")),
            "requiredReturn": finite(output.get("PAE Required Return")),
            "qualityScore": finite(output.get("PAE Quality Score")),
            "capitalClaim": finite(output.get("PAE Capital Claim")),
            "currentWeight": finite(output.get("PAE Current Weight"), 0.0),
            "targetWeight": finite(output.get("PAE Target Weight"), 0.0),
            "targetCashWeight": finite(output.get("PAE Target Cash Weight")),
            "executableCandidate": boolean(output.get("PAE Executable Candidate")),
            "portfolioRiskGate": "PASS" if boolean(output.get("PAE Executable Candidate")) else "FAIL",
            "qualityDominanceGate": quality_gate,
            "cerDominanceGate": cer_gate,
            "dualDominanceGate": dual_gate,
            "persistenceGate": boolean(output.get("PAE Persistence Gate")),
            "obviousOpportunity": boolean(output.get("PAE Obvious Opportunity")),
            "highQualityProtected": False,
            "strategyStopBreached": False,
            "eremExitPersistent": False,
            "trendBroken": False,
            "valuationBroken": False,
            "exitConfirmationCount": 0,
            "alternative": text(output.get("PAE Incumbent")) or None,
            "incumbent": text(output.get("PAE Incumbent")) or None,
            "excessOpportunity": finite(output.get("PAE Excess Opportunity")),
            "portfolioImprovement": clean(output.get("PAE Portfolio Improvement")),
            "executionStatus": text(output.get("PAE Execution Status"), "NOT_REQUIRED"),
            "atomicGroup": text(output.get("PAE Atomic Group")) or None,
            "blockingReasons": [
                item for item in text(output.get("PAE Blocking Reasons")).split(";") if item
            ],
        },
    }


def select_output_lines(
    lines: list[dict[str, Any]],
    maximum: int = MAX_OUTPUT_LINES,
) -> list[dict[str, Any]]:
    """Keep owner holdings and LES entry signals ahead of PAE watch rows.

    PAE may correctly block a Low-Entry BUY/EARLY_BUY/BUY_SMALL candidate for this owner,
    but the research UI still needs that published LES decision and its blocking
    reason.  Selecting only the highest-CER PAE rows can otherwise discard every
    entry signal before the owner-specific plan is built.
    """
    holdings = [item for item in lines if item["role"] == "HOLDING"]
    candidates = [item for item in lines if item["role"] != "HOLDING"]
    entry_signals = [
        item
        for item in candidates
        if item.get("lowEntry", {}).get("decision") in {"BUY", "EARLY_BUY", "BUY_SMALL"}
    ]
    entry_codes = {item["code"] for item in entry_signals}
    pae_signals = [
        item
        for item in candidates
        if item.get("action") in PAE_EXECUTABLE_ACTIONS
        and item["code"] not in entry_codes
    ]
    priority_codes = entry_codes | {item["code"] for item in pae_signals}
    other_candidates = [item for item in candidates if item["code"] not in priority_codes]
    available = max(0, maximum - len(holdings))
    selected_candidates = (entry_signals + pae_signals + other_candidates)[:available]
    return (holdings + selected_candidates)[:maximum]


def main() -> int:
    args = parse_args()
    crawler_root = args.crawler_root.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    history_path = args.history_db.expanduser().resolve()
    if not crawler_root.is_dir() or not dataset_path.is_file():
        raise ValueError("CRAWLER_ARTIFACT_UNAVAILABLE")
    if dataset_path.stat().st_size > MAX_DATASET_BYTES:
        raise ValueError("DATASET_TOO_LARGE")

    sys.path.insert(0, str(crawler_root))
    import pandas as pd
    from domain.strategies.expected_return_exit_v3 import EREM_V3_MODEL_VERSION
    from domain.strategies.low_entry_v5_contracts import LES_V5_MODEL_VERSION
    from domain.strategies.portfolio_allocation_v2 import (
        PAE_V2_MODEL_VERSION,
        PersistenceEvidenceV2,
        PortfolioAllocationEngineV2,
        PortfolioAllocationRequestV2,
        PortfolioSecurityV2,
    )

    payload = load_request()
    owner_id = text(payload.get("ownerId"), "owner:anonymous")
    if ":" not in owner_id or len(owner_id) > 192:
        raise ValueError("OWNER_ID_INVALID")
    positions = position_map(payload)
    account = payload["account"]
    cash = finite(account.get("cash"))
    if cash is None or cash < 0:
        raise ValueError("ACCOUNT_CASH_INVALID")

    frame = pd.read_csv(dataset_path, dtype={"證券代號": "string"})
    if frame.empty or len(frame.index) > MAX_DATASET_ROWS:
        raise ValueError("DATASET_ROWS_INVALID")
    required = {
        "日期", "證券代號", "證券名稱", "收盤", "低點決策",
        "LES Model Version", "EREM 分數", "EREM Holding Confidence", "EREM 決策訊號",
        "PAE Model Version", "Risk Gate", "Falling Knife State",
        "LES v5 Final Quality Status", "Valuation Gate", "EREM Volume MA20",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("DATASET_COLUMNS_MISSING=" + ",".join(missing))
    frame["日期"] = frame["日期"].astype("string").str.slice(0, 10)
    frame["證券代號"] = frame["證券代號"].astype("string").str.strip()
    apply_owner_erem(frame, positions, pd)
    data_date = str(frame["日期"].max())
    frame = frame[frame["日期"] == data_date].copy()
    # The crawler's StrategyDecision CSV preserves its explicit
    # TWD_SCALED_1000 contract. The PAE DataFrame compatibility boundary uses
    # real TWD values, so normalize only the declared price fields here.
    for price_column in (
        "收盤", "ATR14", "Entry Candidate", "策略停損", "策略停利",
        "EREM EMA20",
    ):
        if price_column in frame.columns:
            frame[price_column] = pd.to_numeric(
                frame[price_column], errors="coerce"
            ) / 1000.0
    if frame["證券代號"].duplicated().any():
        # The legacy report can repeat an identical configured holding after
        # merging candidates and holdings. Collapse only byte-equivalent rows;
        # conflicting duplicate decisions remain a fail-closed input error.
        duplicates = frame[frame["證券代號"].duplicated(keep=False)]
        for _, group in duplicates.groupby("證券代號", dropna=False):
            comparable = group.fillna("__NULL__").astype(str)
            if len(comparable.drop_duplicates().index) != 1:
                raise ValueError("DATASET_CONFLICTING_DUPLICATE_SYMBOLS")
        frame = frame.drop_duplicates(subset=["證券代號"], keep="last").copy()
    missing_holdings = sorted(set(positions) - set(frame["證券代號"].tolist()))
    if missing_holdings:
        raise ValueError("HOLDINGS_MISSING_FROM_ANALYSIS=" + ",".join(missing_holdings))

    versions = {
        "les": sorted(set(frame["LES Model Version"].dropna().astype(str))),
        "pae": sorted(set(frame["PAE Model Version"].dropna().astype(str))),
    }
    if versions["les"] not in ([LES_V5_MODEL_VERSION], ["6.0.0"]) or versions["pae"] != [PAE_V2_MODEL_VERSION]:
        raise ValueError("MODEL_VERSION_MISMATCH")

    v6_reports = None
    execution_sessions = None
    if payload.get("lesV6ReportsFile") or versions["les"] == ["6.0.0"]:
        from .les_v6_execution import is_supported_execution_symbol, load_reports, next_sessions
        from .les_v6_premarket import apply_reports
        if versions["les"] == ["6.0.0"]:
            if "LES v6 Report" not in frame:
                raise ValueError("MISSING_V6_REPORTS")
            v6_reports = {
                str(row["證券代號"]): json.loads(row["LES v6 Report"])
                for _, row in frame.iterrows()
                if is_supported_execution_symbol(str(row["證券代號"]))
            }
            for code, report in v6_reports.items():
                if report.get("symbol") != code or report.get("signal_date") != data_date or report.get("model_version") != "6.0.0":
                    raise ValueError("V6_REPORT_IDENTITY_OR_DATE_MISMATCH")
            if payload.get("lesV6ReportsFile") and load_reports(payload["lesV6ReportsFile"], data_date) != v6_reports:
                raise ValueError("V6_CSV_ARTIFACT_MISMATCH")
        else:
            v6_reports = load_reports(payload["lesV6ReportsFile"], data_date)
        execution_sessions = next_sessions(data_date, payload["tradeDate"])
        frame = apply_reports(frame, v6_reports)

    close_by_code = {
        text(row["證券代號"]): finite(row["收盤"])
        for _, row in frame.iterrows()
    }
    market_values = {
        code: item["shares"] * item["currentPrice"]
        for code, item in positions.items()
    }
    account_equity = cash + sum(market_values.values())
    if account_equity <= 0:
        raise ValueError("ACCOUNT_EQUITY_INVALID")

    frame["Portfolio Role"] = frame["證券代號"].map(
        lambda code: "HOLDING" if str(code) in positions else "CANDIDATE"
    )
    frame["Account Equity"] = account_equity
    frame["Current Cash"] = cash
    frame["Current Shares"] = frame["證券代號"].map(
        lambda code: positions.get(str(code), {}).get("shares", 0.0)
    )
    frame["Average Cost"] = frame["證券代號"].map(
        lambda code: positions.get(str(code), {}).get("buyPrice")
    )
    frame["Current Price"] = frame["證券代號"].map(
        lambda code: positions.get(str(code), {}).get("currentPrice", close_by_code.get(str(code)))
    )
    frame["Current Market Value"] = frame["證券代號"].map(
        lambda code: market_values.get(str(code), 0.0)
    )
    frame["Current Portfolio Weight"] = frame["Current Market Value"] / account_equity
    position_caps = pd.to_numeric(frame.get("Position Fraction", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    frame["Maximum Weight"] = position_caps
    prices = pd.to_numeric(frame["Current Price"], errors="coerce")
    les_decisions = frame["低點決策"].astype("string").str.upper()
    executable = (
        (account_equity * position_caps / prices)
        .where(les_decisions.isin(["BUY", "EARLY_BUY", "BUY_SMALL"]) & prices.gt(0), 0.0)
        .fillna(0.0)
        .apply(math.floor)
    )
    frame["Executable Shares"] = executable

    securities = [
        security_v2(
            row,
            positions.get(text(row["證券代號"])),
            account_equity,
            PortfolioSecurityV2,
        )
        for _, row in frame.iterrows()
    ]
    holdings = tuple(item for item in securities if item.stock_id in positions)
    candidates = tuple(item for item in securities if item.stock_id not in positions)
    request = PortfolioAllocationRequestV2(
            portfolio_id=owner_id,
            decision_time=dt.datetime.fromtimestamp(
                dataset_path.stat().st_mtime,
                tz=dt.timezone.utc,
            ),
            account_equity=account_equity,
            available_cash=cash,
            holdings=holdings,
            candidates=candidates,
            persistence=persistence_evidence(payload, PersistenceEvidenceV2),
    )
    engine = PortfolioAllocationEngineV2()
    decision = engine.allocate(request)
    output = decision.to_frame().set_index("Stock Code")
    source_by_code = {text(row["證券代號"]): row for _, row in frame.iterrows()}
    lines = [row_payload(source_by_code[str(code)], row) for code, row in output.iterrows()]
    if v6_reports is not None:
        for line in lines:
            line["lowEntry"]["modelVersion"] = "6.0.0"
            line["lowEntry"]["v6Report"] = v6_reports.get(line["code"])
    lines.sort(key=lambda item: (
        0 if item["role"] == "HOLDING" else 1,
        ACTION_PRIORITY.get(item["action"], 8),
        -(item["pae"]["cer"] if item["pae"]["cer"] is not None else -999.0),
        item["code"],
    ))
    selected = select_output_lines(lines)
    observations = [
        {
            "incumbentSymbol": item.incumbent_stock_id,
            "challengerSymbol": item.challenger_stock_id,
            "excessOpportunity": 1.0 if item.passed else 0.0,
        }
        for item in engine.persistence_observations(request)
    ][:32]
    history_sessions = load_history_sessions(history_path, str(args.market_type))
    result = {
        "schemaVersion": 1,
        "integration": {
            "name": INTEGRATION_CONTRACT_NAME,
            "version": INTEGRATION_CONTRACT_VERSION,
            "packageVersion": INTEGRATION_PACKAGE_VERSION,
        },
        "generatedAt": dt.datetime.fromtimestamp(
            dataset_path.stat().st_mtime,
            tz=dt.timezone.utc,
        ).isoformat(),
        "dataDate": data_date,
        "industry": {"code": str(args.market_type), "label": str(args.industry)},
        "historySessions": history_sessions,
        "requiredHistorySessions": REQUIRED_HISTORY_SESSIONS,
        "historyComplete": history_sessions >= REQUIRED_HISTORY_SESSIONS,
        "models": {
            "les": "6.0.0" if v6_reports is not None else LES_V5_MODEL_VERSION,
            "erem": EREM_V3_MODEL_VERSION,
            "pae": PAE_V2_MODEL_VERSION,
        },
        "mode": decision.mode.value,
        "binding": decision.binding.value,
        "targetCashWeight": clean(decision.cash_weight_after),
        "portfolioExpectedCer": clean(
            decision.cash_weight_after * 0.02
            + sum(
                item.target_weight * float(item.evaluation.cer or 0.0)
                for item in decision.lines
            )
        ),
        "portfolioReason": ";".join(decision.reason_codes),
        "observations": observations,
        "lines": selected,
    }
    if execution_sessions is not None:
        result["executionSessions"] = execution_sessions
        result["executionCalendarSource"] = "TWSE_OFFICIAL"
    sys.stdout.write(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    return 0


def cli() -> int:
    try:
        return main()
    except Exception as error:  # Boundary: return only a safe reason, never input data.
        sys.stderr.write(f"TWSE_RESEARCH_BRIDGE_FAILED:{type(error).__name__}:{error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())

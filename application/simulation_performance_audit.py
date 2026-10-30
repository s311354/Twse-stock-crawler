"""Read-only attribution of exported Bank results, never a return optimizer.

This diagnostic cannot infer counterfactual fills or certify out-of-sample
performance. All arithmetic uses the supplied report, not today's market data.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import io
import re


SCHEMA = "les-simulation-performance-audit/v1"
ZERO = Decimal(0)
RESIDUAL_FLOOR_TWD = Decimal("3000")


def decompose_erem_reduction(*, command, shares_before, actual_shares, price,
                             reason="EREM_REDUCE", residual_floor=RESIDUAL_FLOOR_TWD):
    """Local quantity reconstruction, not an alternative portfolio simulation.

    Use the recorded EREM command, actual pre-fill position and actual opening
    price. Do not recompute EREM from today's data or infer unrecorded signals.
    The comparison policy reconstructs current Bank arithmetic; its match alone
    does not prove that this exact source version ran historically.
    """
    result = dict(schema="erem-bank-reduction-decomposition/v1",
        comparison_policy="CEILING_REDUCTION_THEN_STRICT_RESIDUAL_FLOOR",
        historical_bank_source_binding="NOT_VERIFIED_BY_QUANTITY_MATCH_ALONE",
        counterfactual_account_return="NOT_COMPUTED", residual_floor_twd=str(residual_floor))
    if reason != "EREM_REDUCE":
        return dict(result, status="NOT_APPLICABLE", classification="OTHER_EXIT_REASON")
    match = re.fullmatch(r"REDUCE\s+(\d+(?:\.\d+)?)%", command) if isinstance(command, str) else None
    if not match or any(value is None for value in (shares_before, actual_shares, price)):
        return dict(result, status="UNKNOWN", classification="MISSING_REDUCTION_INPUT")
    try:
        held, actual, opening, floor = map(number, (shares_before, actual_shares, price, residual_floor))
        fraction = number(match[1]) / 100
        if (held <= 0 or actual <= 0 or actual > held or opening <= 0 or floor < 0
                or held != held.to_integral_value() or actual != actual.to_integral_value()
                or not 0 < fraction <= 1):
            raise ValueError("INVALID_REDUCTION_INPUT")
    except ValueError:
        return dict(result, status="UNKNOWN", classification="INVALID_REDUCTION_INPUT")
    model_qty = min(held, max(Decimal(1), (held*fraction).to_integral_value(rounding=ROUND_CEILING)))
    residual_shares = held-model_qty
    residual_value = residual_shares*opening
    bank_qty = held if residual_value < floor else model_qty
    if actual != bank_qty:
        classification, status = "EXECUTION_QUANTITY_MISMATCH", "CONFLICT"
    elif bank_qty > model_qty:
        classification, status = "BANK_RESIDUAL_FULL_EXIT_MATCH", "MATCH"
    elif model_qty == held:
        classification, status = "MODEL_FULL_EXIT", "MATCH"
    else:
        classification, status = "MODEL_PARTIAL_REDUCTION", "MATCH"
    return dict(result, status=status, classification=classification,
        erem_command=command, shares_before=int(held), opening_price_twd=str(opening),
        erem_quantity=int(model_qty), erem_residual_shares=int(residual_shares),
        erem_residual_value_twd=str(residual_value), bank_rule_quantity=int(bank_qty),
        actual_quantity=int(actual), bank_rule_additional_shares=int(bank_qty-model_qty),
        bank_rule_additional_notional_twd=str((bank_qty-model_qty)*opening),
        residual_floor_triggered=residual_value < floor)


def number(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("SIMULATION_AUDIT_INVALID_NUMBER") from error
    if not result.is_finite():
        raise ValueError("SIMULATION_AUDIT_NONFINITE_NUMBER")
    return result


def section(rows, header):
    indices = [index for index, row in enumerate(rows) if row[:len(header)] == header]
    if len(indices) != 1:
        raise ValueError("SIMULATION_AUDIT_SECTION_MISSING_OR_DUPLICATED:" + str(header))
    return indices[0]


def audit_simulation_report(csv_text: str, *, bank_run: dict | None = None) -> dict:
    """Reconcile a multi-section export; attach matching saved-run evidence.

    CSV amounts are rounded to cents, so aggregation residuals are disclosed
    with a bound derived from the number of rounded observations.
    """
    rows = list(csv.reader(io.StringIO(csv_text.lstrip("\ufeff"))))
    account_start = section(rows, ["帳戶摘要", "金額／數值"])
    daily_start = section(rows, ["日期", "紀錄類型", "資產淨值", "可用現金"])
    trade_start = section(rows, ["日期", "代號", "名稱", "方向"])
    metadata_start = section(rows, ["研究與模型資訊"])
    if not account_start < daily_start < trade_start < metadata_start:
        raise ValueError("SIMULATION_AUDIT_SECTION_ORDER_INVALID")
    account = {row[0]: number(row[1]) for row in rows[account_start+1:daily_start]
               if len(row) == 2 and row[0] not in {"每日資產紀錄"}}
    required = {"初始資金", "資產淨值", "可用現金", "持股市值", "投資總損益",
                "已實現損益", "未實現損益", "累計交易費用", "成交筆數"}
    if not required <= set(account) or account["初始資金"] <= 0:
        raise ValueError("SIMULATION_AUDIT_REQUIRED_ACCOUNT_INPUT_MISSING")
    daily = [row for row in rows[daily_start+1:trade_start] if len(row) == 4 and row[1] == "收盤"]
    if not daily:
        raise ValueError("SIMULATION_AUDIT_DAILY_NAV_REQUIRED")
    trades = []
    headers = rows[trade_start]
    for row in rows[trade_start+1:metadata_start]:
        if not row:
            continue
        if len(row) != len(headers) or row[3] not in {"買入", "賣出"}:
            raise ValueError("SIMULATION_AUDIT_TRADE_ROW_INVALID")
        item = dict(zip(headers, row))
        for key in ("股數", "成交價", "金額", "成本"):
            item[key] = number(item[key])
        if (item["股數"] <= 0 or item["股數"] != item["股數"].to_integral_value()
                or item["成交價"] <= 0 or item["成本"] < 0
                or abs(item["股數"] * item["成交價"] - item["金額"]) > Decimal(".005")):
            raise ValueError("SIMULATION_AUDIT_TRADE_UNITS_OR_AMOUNT_CONFLICT")
        item["損益"] = number(item["損益"]) if item["方向"] == "賣出" else None
        trades.append(item)
    if len(trades) != account["成交筆數"]:
        raise ValueError("SIMULATION_AUDIT_TRADE_COUNT_CONFLICT")
    metadata = defaultdict(list)
    for row in rows[metadata_start+1:]:
        if len(row) >= 2:
            metadata[row[0]].append(row[1:])
    buys = [row for row in trades if row["方向"] == "買入"]
    sells = [row for row in trades if row["方向"] == "賣出"]
    total = lambda data, key: sum((row[key] for row in data), ZERO)
    capital, nav = account["初始資金"], account["資產淨值"]
    pnl = nav - capital
    reconstructed_cash = capital - total(buys, "金額") + total(sells, "金額") - total(trades, "成本")
    checks = {}
    for key, residual, observations in (
        ("cash", reconstructed_cash - account["可用現金"], len(trades)+2),
        ("nav", nav - account["可用現金"] - account["持股市值"], 3),
        ("pnl", pnl - account["投資總損益"], 3),
        ("realized", total(sells, "損益") - account["已實現損益"], len(sells)+1),
        ("fees", total(trades, "成本") - account["累計交易費用"], len(trades)+1),
        ("pnl_components", account["已實現損益"] + account["未實現損益"] - pnl, 4),
        ("ending_nav", number(daily[-1][2]) - nav, 2),
    ):
        tolerance = Decimal(".005") * observations
        checks[key] = dict(residual_twd=str(residual), rounding_bound_twd=str(tolerance),
                           status="TIES" if residual == 0 else "ROUNDING_ONLY" if abs(residual) <= tolerance else "CONFLICT")
    by_symbol = defaultdict(list)
    for row in trades:
        by_symbol[row["代號"]].append(row)
    positions = []
    for symbol, group in sorted(by_symbol.items()):
        bought = total([row for row in group if row["方向"] == "買入"], "股數")
        sold = total([row for row in group if row["方向"] == "賣出"], "股數")
        if sold > bought:
            raise ValueError("SIMULATION_AUDIT_OPENING_HOLDINGS_OR_SHORTS_NOT_SUPPORTED")
        positions.append(dict(symbol=symbol, name=group[0]["名稱"], bought_shares=int(bought),
            sold_shares=int(sold), remaining_shares=int(bought-sold),
            realized_twd=str(total([row for row in group if row["方向"] == "賣出"], "損益")),
            listed_cost_twd=str(total(group, "成本")),
            entry_scores=[row.get("Entry LES") for row in group if row["方向"] == "買入"],
            exit_reasons=[row["原因"] for row in group if row["方向"] == "賣出"]))
    closed = [row for row in positions if row["remaining_shares"] == 0]
    winners = [number(row["realized_twd"]) for row in closed if number(row["realized_twd"]) > 0]
    losers = [number(row["realized_twd"]) for row in closed if number(row["realized_twd"]) < 0]
    exposures, peak, drawdown = [], capital, ZERO
    for row in daily:
        value, cash = number(row[2]), number(row[3])
        if value <= 0:
            raise ValueError("SIMULATION_AUDIT_NONPOSITIVE_DAILY_NAV")
        exposures.append((value-cash)/value*100)
        peak = max(peak, value)
        drawdown = min(drawdown, (value/peak-1)*100)
    result = dict(schema=SCHEMA, report_text_sha256=hashlib.sha256(csv_text.encode()).hexdigest(),
        readiness="NOT_SUFFICIENT_TO_RETUNE_OR_CLAIM_OOS_PROFIT", accounting_checks=checks,
        account=dict(initial_twd=str(capital), nav_twd=str(nav), net_pnl_twd=str(pnl),
            net_return_pct=str(pnl/capital*100), listed_cost_twd=str(account["累計交易費用"]),
            price_and_mark_pnl_before_listed_cost_twd=str(pnl+account["累計交易費用"]),
            cost_bridge="ACCOUNTING_ADD_BACK_ONLY_NOT_A_COUNTERFACTUAL_BACKTEST",
            mean_close_exposure_pct=str(sum(exposures)/len(exposures)), max_close_exposure_pct=str(max(exposures)),
            max_close_drawdown_pct=str(drawdown)),
        sample=dict(close_sessions=len(daily), buy_fills=len(buys), sell_fills=len(sells),
            closed_symbols=len(closed), open_symbols=len(positions)-len(closed),
            winning_closed_symbols=len(winners), losing_closed_symbols=len(losers),
            closed_symbol_win_rate_pct=str(Decimal(len(winners))/len(closed)*100) if closed else None,
            closed_symbol_profit_factor=str(sum(winners)/-sum(losers)) if losers else None,
            independent_sample_count="NOT_ESTABLISHED_PARTIAL_EXITS_AND_SHARED_MARKET_EXPOSURE"),
        exit_attribution=[dict(reason=reason, sell_fills=len(group), net_realized_twd=str(total(group, "損益")))
            for reason in sorted({row["原因"] for row in sells})
            for group in [[row for row in sells if row["原因"] == reason]]],
        positions=positions, source_metadata=dict(metadata),
        required_next_evidence=["CUTOFF_LOCAL_RAW_SOURCES", "SAME_SOURCE_R18_R19_COUNTERFACTUALS",
            "ENTRY_TIME_FIXED_PLAN_AND_LIMITS", "DAILY_CANDIDATES_AND_EREM", "COMPANY_ACTIONS_AND_EXECUTION_COSTS"],
        policy_changes_applied=False, profit_guaranteed=False)
    if bank_run is not None:
        _attach_saved_run(result, bank_run, trades, account, metadata)
    if any(check["status"] == "CONFLICT" for check in checks.values()):
        result["readiness"] = "ACCOUNTING_CONFLICT_REQUIRES_REVIEW"
    return result


def _attach_saved_run(result, bank_run, trades, account, metadata):
    """Join the exact export owner, not a similarly dated simulation."""
    job = metadata.get("模擬工作", [[None]])[0][0]
    if not job or job != bank_run.get("jobId"):
        raise ValueError("SIMULATION_AUDIT_BANK_RUN_ID_MISMATCH")
    summary = bank_run.get("summary", {})
    if (len(bank_run.get("trades", [])) != len(trades)
            or abs(number(summary.get("nav")) - account["資產淨值"]) > Decimal(".005")):
        raise ValueError("SIMULATION_AUDIT_BANK_RUN_EXPORT_CONFLICT")
    for row, raw in zip(trades, bank_run["trades"]):
        if (row["日期"] != raw.get("date") or row["代號"] != raw.get("code")
                or {"買入": "BUY", "賣出": "SELL"}[row["方向"]] != raw.get("side")
                or row["股數"] != number(raw.get("shares"))
                or abs(row["成交價"] - number(raw.get("price"))) > Decimal(".005")
                or abs(row["成本"] - number(raw.get("fee"))) > Decimal(".005")):
            raise ValueError("SIMULATION_AUDIT_BANK_RUN_TRADE_CONFLICT")
    policies, decisions, executions = Counter(), Counter(), Counter()
    exits, held = [], Counter()
    scopes = Counter()
    for day in bank_run.get("daily", []):
        signals = day.get("decisions", [])
        scopes[len(signals)] += 1
        for signal in signals:
            policies[signal.get("parameterVersion", "UNRECORDED")] += 1
            decisions[signal.get("decision", "UNRECORDED")] += 1
        for action in day.get("execution", []):
            executions[(action.get("status"), action.get("reason"))] += 1
        for fill in day.get("fills", []):
            if fill.get("side") == "BUY":
                held[fill["code"]] += fill["shares"]
                continue
            if fill.get("side") != "SELL":
                continue
            matches = [signal for signal in signals if signal.get("code") == fill.get("code")]
            signal = matches[0] if len(matches) == 1 else {}
            before = held[fill["code"]]
            reduce = re.fullmatch(r"REDUCE (\d+)%", signal.get("eremDecision") or "")
            fraction = Decimal(reduce[1])/100 if reduce else None
            intended = ((number(before)*fraction).to_integral_value(rounding=ROUND_CEILING)
                        if fraction is not None and before > 0 else None)
            fully_closed_from_partial = bool(fill["reason"] == "EREM_REDUCE" and before == fill["shares"]
                and intended is not None and 0 < intended < before)
            exits.append(dict(date=day["date"], symbol=fill["code"], reason=fill["reason"],
                shares=fill["shares"], realized_twd=fill.get("pnl"),
                shares_before=before if before > 0 else None,
                requested_reduce_shares_with_ceiling=int(intended) if intended is not None else None,
                partial_signal_executed_as_full_exit=fully_closed_from_partial,
                holding_confidence=signal.get("hc"), erem_decision=signal.get("eremDecision"),
                erem_reasons=signal.get("eremReasons"), signal_date=signal.get("signalDate"),
                reduction_decomposition=decompose_erem_reduction(command=signal.get("eremDecision"),
                    shares_before=before, actual_shares=fill["shares"], price=fill.get("price"),
                    reason=fill["reason"]),
                source_match="MATCHED" if signal else "MISSING_OR_CONFLICTING"))
            held[fill["code"]] -= fill["shares"]
    groups = defaultdict(list)
    for row in exits:
        groups[row["reduction_decomposition"]["classification"]].append(row)
    comparison = [dict(classification=key, fills=len(group),
        actual_realized_pnl_twd=str(sum((number(row["realized_twd"]) for row in group), ZERO)),
        additional_sold_notional_twd=(str(sum((number(row["reduction_decomposition"]["bank_rule_additional_notional_twd"])
            for row in group), ZERO)) if all(row["reduction_decomposition"]["status"] == "MATCH" for row in group) else None),
        pnl_interpretation="ACTUAL_FILL_GROUP_ONLY_NOT_CAUSAL_POLICY_EFFECT")
        for key, group in sorted(groups.items())]
    result["saved_run"] = dict(job_id=job, versions=bank_run.get("versions"),
        evidence_policy=bank_run.get("evidencePolicy"), reference_date=bank_run.get("referenceDate"),
        parameter_version_counts=dict(policies), decision_counts=dict(decisions),
        daily_scope_counts={str(key): value for key, value in sorted(scopes.items())},
        execution_observations=[dict(status=key[0], reason=key[1], count=count)
                                for key, count in sorted(executions.items())],
        exits=exits,
        reduction_comparison=comparison,
        partial_reductions_executed_as_full_exits=sum(row["partial_signal_executed_as_full_exit"] for row in exits),
        source_retention="READ_ONLY_SAVED_RUN_NOT_NEW_NATIVE_RECOMPUTATION")

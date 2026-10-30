"""Public-fee, fixed-fill accounting sensitivities; never new account returns."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_UP
import datetime as dt
import re

from application.replay_contracts import digest


VERSION = "PUBLIC_ODD_LOT_FEE_SENSITIVITY_R1"
SCENARIOS = (
    dict(key="PUBLIC_UNDISCOUNTED_MIN1", rate="0.001425", minimum="1",
         basis="YUANTA_ODD_LOT_PUBLIC_RATE_WITH_NO_DISCOUNT_ASSUMED", sources=["yuanta_odd", "yuanta_accounts"]),
    dict(key="PUBLIC_UNDISCOUNTED_MIN20", rate="0.001425", minimum="20",
         basis="BANKTAIWAN_PUBLIC_UNDISCOUNTED_SMALL_TURNOVER_REFERENCE", sources=["banktaiwan_faq", "banktaiwan_business"]),
    dict(key="PUBLIC_ELECTRONIC_MIN20", rate="0.000498", minimum="20",
         basis="BANKTAIWAN_PUBLIC_ELECTRONIC_REFERENCE", sources=["banktaiwan_business"]),
)


def number(value):
    if isinstance(value, bool) or value is None:
        raise ValueError("FEE_RESEARCH_FINITE_NUMBER_REQUIRED")
    try:
        value = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise ValueError("FEE_RESEARCH_FINITE_NUMBER_REQUIRED") from error
    if not value.is_finite():
        raise ValueError("FEE_RESEARCH_FINITE_NUMBER_REQUIRED")
    return value


def quote_order(side, fills, *, rate, minimum, commission_rounding="FLOOR"):
    """Quote one explicit order, grouping settlement by identical price.

    Separate orders must be quoted separately. No fees are charged to unfilled
    orders. Taxes use unrounded gross as an explicit reference convention; an
    actual broker settlement receipt is still needed to certify cash movements.
    """
    if side not in {"BUY", "SELL"} or not isinstance(fills, (tuple, list)):
        raise ValueError("FEE_RESEARCH_ORDER_SIDE_OR_FILLS_INVALID")
    rate, minimum = number(rate), number(minimum)
    if (not 0 <= rate <= Decimal(".01") or not 0 <= minimum <= 1000
            or minimum != minimum.to_integral_value()):
        raise ValueError("FEE_RESEARCH_RATE_OR_MINIMUM_INVALID")
    if commission_rounding not in {"FLOOR", "HALF_UP"}:
        raise ValueError("FEE_RESEARCH_ROUNDING_REQUIRED")
    groups = defaultdict(lambda: Decimal(0))
    for fill in fills:
        qty, price = number(fill["shares"]), number(fill["price"])
        if qty <= 0 or qty != qty.to_integral_value() or price <= 0 or price != price.quantize(Decimal(".01")):
            raise ValueError("FEE_RESEARCH_QUANTITY_OR_TWD_PRICE_INVALID")
        groups[price] += qty * price
    gross = sum(groups.values(), Decimal(0))
    settled = sum((value.to_integral_value(rounding=ROUND_DOWN) for value in groups.values()), Decimal(0))
    rounding = ROUND_DOWN if commission_rounding == "FLOOR" else ROUND_HALF_UP
    commission = max(minimum, (gross * rate).to_integral_value(rounding=rounding)) if fills else Decimal(0)
    tax = (gross * Decimal(".003")).to_integral_value(rounding=ROUND_DOWN) if side == "SELL" else Decimal(0)
    cash = -settled - commission if side == "BUY" else settled - commission - tax
    return dict(gross_raw=str(gross), settlement_gross=str(settled), commission=str(commission),
        sell_tax=str(tax), fees=str(commission + tax), cash_delta=str(cash),
        filled=bool(fills), price_groups=len(groups),
        settlement_policy="SUM_FLOOR_GROSS_PER_ORDER_IDENTICAL_PRICE_GROUP",
        commission_rounding=commission_rounding, tax_rounding="FLOOR_TWD",
        tax_basis="RAW_EXECUTED_GROSS_REFERENCE_NOT_BROKER_RECEIPT")


def compare_public_fees(run):
    """Keep original quantities/prices/marks fixed; disclose lost causality.

    Each saved trade row is treated as one order solely for sensitivity. The
    original file has no exchange order IDs. This does not replay candidates,
    EREM, PAE, fills, minimum-fee sizing, corporate actions or later holdings.
    """
    if not isinstance(run, dict) or not isinstance(run.get("jobId"), str):
        raise ValueError("FEE_RESEARCH_SAVED_RUN_REQUIRED")
    fills = run.get("trades")
    if not isinstance(fills, list) or not fills:
        raise ValueError("FEE_RESEARCH_NONEMPTY_SAVED_TRADES_REQUIRED")
    keys, sides = set(), defaultdict(set)
    recorded = Decimal(0)
    for row in fills:
        if (not isinstance(row, dict) or row.get("side") not in {"BUY", "SELL"}
                or not isinstance(row.get("code"), str) or not re.fullmatch(r"\d{4}", row["code"])
                or type(row.get("id")) is not int or row["id"] <= 0 or row["id"] in keys):
            raise ValueError("FEE_RESEARCH_TRADE_ID_OR_IDENTITY_INVALID")
        if not isinstance(row.get("date"), str) or dt.date.fromisoformat(row["date"]).isoformat() != row["date"]:
            raise ValueError("FEE_RESEARCH_TRADE_DATE_INVALID")
        keys.add(row["id"])
        sides[(row["date"], row["code"])].add(row["side"])
        quote_order(row["side"], [row], rate=".001425", minimum="1")
        if abs(number(row["amount"]) - number(row["shares"]) * number(row["price"])) > Decimal(".005"):
            raise ValueError("FEE_RESEARCH_AMOUNT_CONFLICT")
        fee = number(row["fee"])
        if fee < 0:
            raise ValueError("FEE_RESEARCH_NEGATIVE_RECORDED_FEE")
        recorded += fee
    if any(value == {"BUY", "SELL"} for value in sides.values()):
        raise ValueError("FEE_RESEARCH_SAME_DAY_OFFSET_NOT_SUPPORTED")
    summary = run.get("summary", {})
    if abs(recorded - number(summary.get("fees"))) > Decimal(".005"):
        raise ValueError("FEE_RESEARCH_RECORDED_COST_CONFLICT")
    capital, nav = number(run.get("initialCapital")), number(summary.get("nav"))
    if capital <= 0 or nav <= 0:
        raise ValueError("FEE_RESEARCH_POSITIVE_CAPITAL_AND_NAV_REQUIRED")
    pnl = nav - capital
    scenarios = []
    for parameters in SCENARIOS:
        for rounding in ("FLOOR", "HALF_UP"):
            rows, cash_adjustment = [], Decimal(0)
            commissions = taxes = Decimal(0)
            for fill in fills:
                quote = quote_order(fill["side"], [fill], rate=parameters["rate"],
                    minimum=parameters["minimum"], commission_rounding=rounding)
                gross, original_fee = number(fill["shares"]) * number(fill["price"]), number(fill["fee"])
                old_cash = -gross - original_fee if fill["side"] == "BUY" else gross - original_fee
                cash_adjustment += number(quote["cash_delta"]) - old_cash
                commissions += number(quote["commission"])
                taxes += number(quote["sell_tax"])
                rows.append(dict(trade_id=fill["id"], date=fill["date"], symbol=fill["code"],
                    side=fill["side"], shares=fill["shares"], price=fill["price"], recorded_fee=str(original_fee),
                    **quote))
            scenario = dict(parameters, commission_rounding=rounding,
                rounding_status="FLOOR_PUBLIC_REFERENCE" if rounding == "FLOOR" else "ILLUSTRATIVE_ROUNDING_SENSITIVITY_NOT_PUBLISHED_BROKER_POLICY",
                commission_twd=str(commissions), sell_tax_twd=str(taxes),
                total_fee_twd=str(commissions + taxes), fixed_fill_cash_change_twd=str(cash_adjustment),
                conditional_pnl_bridge_twd=str((pnl + cash_adjustment).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)),
                conditional_pnl_bridge_unrounded_twd=str(pnl + cash_adjustment),
                result_type="FIXED_FILL_ACCOUNTING_BRIDGE_NOT_REEVOLVED_ACCOUNT_RETURN", rows=rows)
            scenario["scenario_id"] = digest(scenario)
            scenarios.append(scenario)
    result = dict(schema="public-exit-fee-sensitivity/v1", policy=VERSION, source_job_id=run["jobId"],
        original_account=dict(capital_twd=str(capital), nav_twd=str(nav), pnl_twd=str(pnl),
            listed_cost_twd=str(recorded)), scenarios=scenarios,
        presentation_rounding="PNL_BRIDGE_HALF_UP_CENTS_WITH_UNROUNDED_VALUE_RETAINED",
        execution_diagnostic=dict(trade_rows=len(fills),
            odd_lot_sized_rows=sum(0 < number(row["shares"]) < 1000 for row in fills),
            ordinary_open_proxy_rows=sum(row.get("priceBasis") == "OPEN" for row in fills),
            order_ids_available=False, order_grouping="ONE_SAVED_TRADE_ROW_AS_ONE_ORDER_ASSUMPTION",
            odd_lot_fills_verified=False, slippage_calibrated=False,
            corporate_action_accounting_verified=False),
        assumptions=["ORDINARY_TWD_STOCK_TAX_NOT_ETF_DAY_TRADE_OR_SPECIAL_TAX",
            "PUBLICLY_OBSERVED_NOW_NOT_HISTORICAL_BROKER_CONTRACT",
            "NO_DISCOUNT_OR_FUTURE_REBATE_CREDIT_ASSUMED", "FIXED_ORIGINAL_FILLS_AND_END_MARKS",
            "TAX_MINIMUM_ZERO_AND_RAW_GROSS_TAX_BASIS_ARE_EXPLICIT_REFERENCE_CONVENTIONS",
            "MINIMUM_FEE_AND_SETTLEMENT_GROUPING_REQUIRE_ORDER_IDS_FOR_REAL_EXECUTION"],
        full_account_comparison_complete=False, policy_promotion="NOT_ASSESSABLE_DO_NOT_SWITCH",
        approved_for_live=False, orders=False, email=False)
    result["comparison_id"] = digest(result)
    return deepcopy(result)

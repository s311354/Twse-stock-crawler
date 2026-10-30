"""Synthetic accounting/transport tests; not investment-performance evidence."""
from copy import deepcopy
import json

import pytest

from application.simulation_performance_audit import audit_simulation_report, decompose_erem_reduction
from interface.simulation_audit import main


CSV = """虛擬投資帳戶模擬區間結果報告
帳戶摘要,金額／數值
初始資金,1000.00
資產淨值,1009.37
可用現金,1009.37
持股市值,0.00
投資總損益,9.37
已實現損益,9.37
未實現損益,0.00
累計交易費用,0.63
成交筆數,2

每日資產紀錄
日期,紀錄類型,資產淨值,可用現金
2026-09-01,初始帳戶,1000.00,1000.00
2026-09-01,收盤,999.86,899.86
2026-09-02,收盤,1009.37,1009.37

交易歷史,全部成交,2
日期,代號,名稱,方向,股數,成交價,金額,成本,損益,原因,Entry LES,Entry Decision
2026-09-01,TEST_A,合成測試,買入,10,10.00,100.00,0.14,,BUY_SMALL,65,BUY_SMALL
2026-09-02,TEST_A,合成測試,賣出,10,11.00,110.00,0.49,9.37,EREM_REDUCE,65,BUY_SMALL

研究與模型資訊
模式,SYNTHETIC_ONLY
模擬工作,SYNTHETIC_JOB
資料政策,ANALYSIS_TIME_DATA_ON_HISTORICAL_BARS/v1
"""


def test_cost_bridge_uses_nav_once_and_never_double_subtracts_costs():
    audit = audit_simulation_report(CSV)
    assert audit["account"]["net_pnl_twd"] == "9.37"
    assert audit["account"]["net_return_pct"] == "0.93700"
    assert audit["account"]["price_and_mark_pnl_before_listed_cost_twd"] == "10.00"
    assert all(check["status"] == "TIES" for check in audit["accounting_checks"].values())
    assert audit["sample"]["closed_symbols"] == 1
    assert audit["sample"]["closed_symbol_win_rate_pct"] == "100"
    assert audit["sample"]["closed_symbol_profit_factor"] is None
    assert audit["policy_changes_applied"] is False and audit["profit_guaranteed"] is False
    assert "NOT_SUFFICIENT" in audit["readiness"]


def test_reconciliation_exposes_conflict_not_healthy_zero():
    audit = audit_simulation_report(CSV.replace("可用現金,1009.37", "可用現金,1008.37"))
    assert audit["readiness"] == "ACCOUNTING_CONFLICT_REQUIRES_REVIEW"
    assert audit["accounting_checks"]["cash"]["status"] == "CONFLICT"


def test_rounding_residual_is_disclosed_and_bounded():
    audit = audit_simulation_report(CSV.replace("已實現損益,9.37", "已實現損益,9.38"))
    assert audit["accounting_checks"]["realized"]["residual_twd"] == "-0.01"
    assert audit["accounting_checks"]["realized"]["status"] == "ROUNDING_ONLY"


@pytest.mark.parametrize("change", [
    ("初始資金,1000.00", "初始資金,NaN"),
    ("TEST_A,合成測試,買入,10", "TEST_A,合成測試,買入,10.5"),
    ("買入,10,10.00,100.00", "買入,10,10.00,1000.00"),
    ("成交筆數,2", "成交筆數,3"),
])
def test_invalid_values_do_not_silently_produce_metrics(change):
    with pytest.raises(ValueError, match="SIMULATION_AUDIT"):
        audit_simulation_report(CSV.replace(*change))


def saved_run():
    return dict(jobId="SYNTHETIC_JOB", summary=dict(nav=1009.37), trades=[
        dict(date="2026-09-01", code="TEST_A", side="BUY", shares=10, price=10, fee=.14),
        dict(date="2026-09-02", code="TEST_A", side="SELL", shares=10, price=11, fee=.49),
    ], daily=[dict(date="2026-09-02", decisions=[dict(code="TEST_A", decision="WATCH",
        parameterVersion="SYNTHETIC_R18", hc=54., eremDecision="REDUCE 25%")],
        execution=[dict(status="NOT_FILLED", reason="SYNTHETIC_OUTSIDE_BAND")],
        fills=[dict(side="SELL", code="TEST_A", reason="EREM_REDUCE", shares=10, price=11., pnl=9.37)])],
        evidencePolicy="SYNTHETIC_NON_OOS")


def test_exact_saved_run_join_exposes_policy_and_exit_fields():
    audit = audit_simulation_report(CSV, bank_run=saved_run())
    assert audit["saved_run"]["parameter_version_counts"] == {"SYNTHETIC_R18": 1}
    assert audit["saved_run"]["exits"][0]["holding_confidence"] == 54.
    assert audit["saved_run"]["source_retention"].endswith("NOT_NEW_NATIVE_RECOMPUTATION")
    wrong = deepcopy(saved_run())
    wrong["jobId"] = "OTHER_JOB"
    with pytest.raises(ValueError, match="RUN_ID_MISMATCH"):
        audit_simulation_report(CSV, bank_run=wrong)
    wrong = deepcopy(saved_run())
    wrong["trades"][0]["shares"] = 11
    with pytest.raises(ValueError, match="TRADE_CONFLICT"):
        audit_simulation_report(CSV, bank_run=wrong)


def test_partial_reduce_executed_as_full_exit_is_separate_from_model_decision():
    raw = saved_run()
    raw["daily"].insert(0, dict(date="2026-09-01", fills=[dict(side="BUY", code="TEST_A", shares=10)]))
    audit = audit_simulation_report(CSV, bank_run=raw)
    exit = audit["saved_run"]["exits"][0]
    assert exit["shares_before"] == 10 and exit["requested_reduce_shares_with_ceiling"] == 3
    assert exit["shares"] == 10 and exit["partial_signal_executed_as_full_exit"] is True
    assert audit["saved_run"]["partial_reductions_executed_as_full_exits"] == 1
    assert exit["reduction_decomposition"]["classification"] == "BANK_RESIDUAL_FULL_EXIT_MATCH"
    assert exit["reduction_decomposition"]["erem_residual_value_twd"] == "77.0"
    group = audit["saved_run"]["reduction_comparison"][0]
    assert group["additional_sold_notional_twd"] == "77.0"
    assert group["pnl_interpretation"].endswith("NOT_CAUSAL_POLICY_EFFECT")


@pytest.mark.parametrize("price,actual,classification", [
    ("100", 20, "MODEL_PARTIAL_REDUCTION"),  # exactly 3000: strict < does not trigger
    ("99.99", 50, "BANK_RESIDUAL_FULL_EXIT_MATCH"),
    ("100.01", 20, "MODEL_PARTIAL_REDUCTION"),
])
def test_residual_floor_boundary_reproduces_strict_less_than(price, actual, classification):
    row = decompose_erem_reduction(command="REDUCE 40%", shares_before=50,
                                  actual_shares=actual, price=price)
    assert row["classification"] == classification and row["status"] == "MATCH"
    assert row["erem_quantity"] == 20
    assert row["counterfactual_account_return"] == "NOT_COMPUTED"


@pytest.mark.parametrize("command,held,actual,classification", [
    ("REDUCE 25%", 21, 21, "BANK_RESIDUAL_FULL_EXIT_MATCH"),
    ("REDUCE 100%", 21, 21, "MODEL_FULL_EXIT"),
    ("REDUCE 25%", 1, 1, "MODEL_FULL_EXIT"),  # rounding itself consumes the one share
    ("REDUCE 25.5%", 21, 21, "BANK_RESIDUAL_FULL_EXIT_MATCH"),
    ("REDUCE 25%", 21, 7, "EXECUTION_QUANTITY_MISMATCH"),
    ("REDUCE 0%", 21, 21, "INVALID_REDUCTION_INPUT"),
    ("REDUCE 101%", 21, 21, "INVALID_REDUCTION_INPUT"),
    (None, 21, 21, "MISSING_REDUCTION_INPUT"),
])
def test_model_rounding_full_exit_and_missing_sources_are_not_confused(command, held, actual, classification):
    row = decompose_erem_reduction(command=command, shares_before=held, actual_shares=actual, price="116.50")
    assert row["classification"] == classification


def test_other_exit_reasons_and_unknown_price_cannot_be_attributed_to_residual_rule():
    row = decompose_erem_reduction(command="HOLD", shares_before=21, actual_shares=21,
                                  price="116.50", reason="RISK_STOP")
    assert row["status"] == "NOT_APPLICABLE"
    row = decompose_erem_reduction(command="REDUCE 25%", shares_before=21, actual_shares=21, price=None)
    assert row["status"] == "UNKNOWN"


def test_cli_output_is_write_once_and_source_is_not_changed(tmp_path, capsys):
    source, output = tmp_path / "source.csv", tmp_path / "audit.json"
    source.write_text(CSV, encoding="utf-8")
    args = ["--report", str(source), "--output", str(output)]
    assert main(args) == main(args) == 0
    assert source.read_text(encoding="utf-8") == CSV
    assert json.loads(output.read_text())["account"]["net_pnl_twd"] == "9.37"
    output.write_text("{}")
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        main(args)

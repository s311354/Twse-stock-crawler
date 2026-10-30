"""Synthetic admission audit tests, not actual four-account efficacy evidence."""
from copy import deepcopy
import gzip
import json

import pytest

from application.exit_comparison_readiness import audit_exit_comparison, inspect_execution_source
from interface.exit_comparison_audit import main


CSV = """帳戶摘要,金額／數值
初始資金,1000000
資產淨值,1000009.37
可用現金,1000009.37
持股市值,0
投資總損益,9.37
已實現損益,9.37
未實現損益,0
累計交易費用,0.63
成交筆數,2
每日資產紀錄
日期,紀錄類型,資產淨值,可用現金
2026-09-01,收盤,999999.86,999899.86
2026-09-02,收盤,1000009.37,1000009.37
交易歷史
日期,代號,名稱,方向,股數,成交價,金額,成本,損益,原因,Entry LES,Entry Decision
2026-09-01,TEST,合成測試,買入,10,10,100,0.14,,BUY_SMALL,65,BUY_SMALL
2026-09-02,TEST,合成測試,賣出,10,11,110,0.49,9.37,EREM_REDUCE,65,BUY_SMALL
研究與模型資訊
模擬工作,SYNTHETIC_ONLY
"""
ENGINE = """BUY_FEE = .001425
SELL_FEE = .004425
def simulate(start, research=None):
    if (pos['shares'] - qty)*q['open'] < 3000:
        qty = pos['shares']
"""
PRICES = "EXECUTION_POLICY = 'SYNTHETIC_ONLY'"


def sources():
    run = dict(jobId="SYNTHETIC_ONLY", versions=dict(code="synthetic", parameters="fixed"),
        universe=["TEST"], initialCapital=1_000_000,
        initialAccount=dict(cash=1_000_000, nav=1_000_000, holdings=[]),
        evidencePolicy="ANALYSIS_TIME_DATA_ON_HISTORICAL_BARS/v1", referenceDate="2026-10-21",
        executionPolicy="SYNTHETIC_ONLY", executionAssumptions=dict(slippageBps=0),
        summary=dict(nav=1000009.37), daily=[], trades=[])
    for day in (1, 2):
        position = dict(code="TEST", shares=10, entryDate="2026-09-01", entryLES=65., entryDecision="BUY_SMALL")
        account = dict(cutoff=f"2026-09-{day:02}T13:30:00+08:00", holdings=[] if day == 1 else [position])
        fill = dict(date=f"2026-09-{day:02}", code="TEST", side="BUY" if day == 1 else "SELL",
            reason="BUY_SMALL" if day == 1 else "EREM_REDUCE", shares=10,
            price=10 if day == 1 else 11, fee=.14 if day == 1 else .49, pnl=None if day == 1 else 9.37)
        run["trades"].append(fill)
        run["daily"].append(dict(date=fill["date"], openingSnapshot=account,
            research=dict(accountSnapshot=deepcopy(account)), fills=[deepcopy(fill)],
            decisions=[dict(code="TEST", eremDecision="REDUCE 25%", hc=50.,
                eremReasons=[f"HoldingDays={day-1}；GraceProfile=NONE"])], execution=[]))
    manifest = dict(schema="bank-date-current/v1", replayId=run["jobId"],
        **{key: deepcopy(run[key]) for key in ("versions", "evidencePolicy", "referenceDate", "executionPolicy")},
        symbols=["TEST"], dates=[row["date"] for row in run["daily"]], marketHash="SYNTHETIC_HASH")
    return run, manifest


def test_four_arms_are_isolated_specs_not_fake_zero_return_accounts():
    run, manifest = sources()
    before = deepcopy((run, manifest))
    result = audit_exit_comparison(CSV, run, manifest, engine_text=ENGINE, prices_text=PRICES)
    assert (run, manifest) == before
    assert result["status"] == "BLOCKED_NOT_A_FOUR_ACCOUNT_BACKTEST"
    assert not result["approved_for_live"] and result["validated_oos_account_sessions"] == 0
    arms = result["arms"]
    assert [(a["erem_policy"], a["retain_residual_liquidation"]) for a in arms] == [
        ("NATIVE", True), ("R1", True), ("NATIVE", False), ("R1", False)]
    assert len({a["account_id"] for a in arms}) == len({a["pae_namespace"] for a in arms}) == 4
    assert all(a["result"] is None and a["net_return"] is None and a["status"] == "NOT_RUN" for a in arms)
    arms[0]["initial_account"]["holdings"].append("MUTATION_TEST")
    assert arms[1]["initial_account"]["holdings"] == []
    assert result["contract"]["initial_account"]["holdings"] == []
    result["contract"]["scope"]["universe"].append("MUTATION_TEST")
    result["contract"]["scope"]["source_versions"]["code"] = "MUTATION_TEST"
    assert (run, manifest) == before
    assert len(result["gaps"]) == 6
    assert result["first_changed_recorded_sell"]["date"] == "2026-09-02"


@pytest.mark.parametrize("field,value", [
    ("replayId", "OTHER"), ("versions", {}), ("symbols", []),
    ("dates", []), ("referenceDate", "2026-10-22"), ("executionPolicy", "OTHER"),
])
def test_source_mismatch_cannot_authorize_comparison(field, value):
    run, manifest = sources()
    manifest[field] = value
    with pytest.raises(ValueError, match="EXIT_COMPARISON_MANIFEST"):
        audit_exit_comparison(CSV, run, manifest, engine_text=ENGINE, prices_text=PRICES)


def test_missing_cash_or_fabricated_empty_account_is_not_accepted():
    run, manifest = sources()
    run["initialAccount"]["holdings"] = None
    with pytest.raises(ValueError, match="EXPLICIT_MILLION_EMPTY_ACCOUNT"):
        audit_exit_comparison(CSV, run, manifest, engine_text=ENGINE, prices_text=PRICES)


def test_missing_action_evidence_never_becomes_no_actions_or_ready():
    run, manifest = sources()
    run["corporateActions"] = []
    result = audit_exit_comparison(CSV, run, manifest, engine_text=ENGINE, prices_text=PRICES)
    assert "CORPORATE_ACTION_LEDGER_NOT_VALIDATED" in {row["code"] for row in result["gaps"]}
    assert not result["approved_for_live"]


def test_source_inspection_preserves_unknown_and_historical_distinction():
    source = inspect_execution_source(ENGINE, PRICES)
    assert source["buy_fee_rate"] == .001425 and source["residual_floor_twd"] == 3000
    assert source["simulate_parameters"] == ["start", "research"]
    assert source["historical_source_binding"].startswith("NOT_ESTABLISHED")
    assert inspect_execution_source("def simulate(): pass", PRICES)["residual_floor_twd"] is None
    assert inspect_execution_source(ENGINE.replace("3000", "2000"), PRICES)["residual_floor_twd"] == 2000


def test_deterministic_audit_source_and_policy_changes_have_new_identity():
    run, manifest = sources()
    a = audit_exit_comparison(CSV, run, manifest, engine_text=ENGINE, prices_text=PRICES)
    b = audit_exit_comparison(CSV, run, manifest, engine_text=ENGINE, prices_text=PRICES)
    c = audit_exit_comparison(CSV, run, manifest, engine_text=ENGINE.replace("3000", "2000"), prices_text=PRICES)
    assert a == b and a["audit_id"] != c["audit_id"]
    assert a["contract_id"] == c["contract_id"]  # requested policies remain fixed


def test_cli_is_idempotent_and_never_overwrites_sources_or_prior_audit(tmp_path, capsys):
    run, manifest = sources()
    files = dict(report=CSV.encode(), bank_run=gzip.compress(json.dumps(run).encode()),
        bank_manifest=json.dumps(manifest).encode(), bank_engine=ENGINE.encode(), bank_prices=PRICES.encode())
    paths, args = {}, []
    for key, value in files.items():
        path = tmp_path / (key + (".gz" if key == "bank_run" else ".txt"))
        path.write_bytes(value)
        paths[key] = path
        args += ["--" + key.replace("_", "-"), str(path)]
    output = tmp_path / "audit.json"
    assert main(args + ["--output", str(output)]) == 0
    assert main(args + ["--output", str(output)]) == 0
    assert all(path.read_bytes() == files[key] for key, path in paths.items())
    with pytest.raises(SystemExit):
        main(args + ["--output", str(paths["report"])])
    output.write_text("{}")
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        main(args + ["--output", str(output)])

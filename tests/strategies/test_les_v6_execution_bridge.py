from copy import deepcopy
import json
import subprocess
import sys

import pandas as pd
import pytest

from aistock_twse_integration.les_v6_execution import execute_request, load_reports, next_sessions
from aistock_twse_integration.les_v6_premarket import apply_reports
from test_low_entry_score_v6 import execution_report, ACCOUNT


def request():
    return dict(report=execution_report(), quote={"symbol": "2330", "timestamp": "2026-09-14T09:10+08:00",
                "price": 100.5, "bid": 100.4, "ask": 100.5, "restricted": False},
                now="2026-09-14T09:10:30+08:00", trading_sessions=["2026-09-11", "2026-09-14"],
                portfolio=ACCOUNT, hard_risk=False)


@pytest.mark.parametrize("path,value", [
    (("data",), {}), (("data", "integrity"), "UNKNOWN"), (("data", "confidence"), None),
    (("quality", "coverage"), {}), (("quality", "status"), "UNKNOWN"),
    (("downtrend_gate", "hard_veto"), True), (("decision", "hard_vetoes"), ["NEW_LOW"]),
    (("decision", "risk_gate"), "FAIL"), (("decision", "signal"), "WATCH"),
    (("parameters",), {}), (("entry", "rule"), "IMMEDIATE"),
])
def test_high_score_cannot_offset_missing_data_or_veto(path, value):
    payload = request()
    report = payload["report"]
    report["decision"]["score"] = 100
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert execute_request(payload)["action"] == "SKIP"


def test_real_stdin_worker_is_read_only_and_fails_closed():
    result = subprocess.run([sys.executable, "-m", "aistock_twse_integration.les_v6_execution"],
                            input=json.dumps(request()), capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["action"] == "ELIGIBLE"
    payload = request()
    payload["quote"]["ask"] = 130
    assert execute_request(payload)["reason"] == "REPRICED_RISK_OR_LIQUIDITY_FAIL"
    payload["quote"]["timestamp"] = "2026-09-14T09:08+08:00"
    assert execute_request(payload)["reason"] == "STALE_OR_MISSING_QUOTE"
    payload = request()
    payload["portfolio"] = dict(ACCOUNT, available_cash=1)
    assert execute_request(payload)["shares"] == 0
    assert execute_request({})["action"] == "SKIP"


def test_calendar_does_not_assume_a_weekday_is_open():
    class Calendar:
        def is_trading_day(self, day):
            return day.isoformat() in {"2026-09-11", "2026-09-15"}
    assert next_sessions("2026-09-11", "2026-09-15", Calendar()) == ["2026-09-11", "2026-09-15"]
    with pytest.raises(ValueError, match="WRONG_SESSION"):
        next_sessions("2026-09-11", "2026-09-14", Calendar())
    class Unknown:
        def is_trading_day(self, day):
            raise RuntimeError("CALENDAR_UNAVAILABLE")
    with pytest.raises(RuntimeError, match="UNAVAILABLE"):
        next_sessions("2026-09-11", "2026-09-14", Unknown())


def test_premarket_adapter_preserves_cer_and_blocks_missing_or_invalid_reports(tmp_path):
    report = execution_report()
    report["liquidity"] = {"passed": True}
    source = pd.DataFrame([{"證券代號": code, "低點決策": "BUY", "策略停損": 90., "Position Fraction": .25,
                            "Valuation Gate": "PASS", "Base Case Expected CAGR": .15} for code in ("2330", "2454")])
    actual = apply_reports(source, {"2330": report})
    assert actual.iloc[0]["低點決策"] in {"BUY", "BUY_SMALL"}
    assert actual.iloc[0]["Base Case Expected CAGR"] == .15
    assert actual.iloc[1]["低點決策"] == "DATA_INSUFFICIENT"
    assert actual.iloc[1]["策略停損"] == 90  # Existing holding protection survives missing entry data.
    assert actual.iloc[1]["Position Fraction"] == 0
    broken = deepcopy(report)
    broken["data"] = {}
    assert apply_reports(source, {"2330": broken}).iloc[0]["低點決策"] == "NO_BUY"
    artifact = tmp_path / "reports.json"
    artifact.write_text(json.dumps({"model": "low_entry_score_v6", "reports": [report]}))
    assert load_reports(str(artifact), "2026-09-11")["2330"] == report
    with pytest.raises(ValueError, match="DATE_MISMATCH"):
        load_reports(str(artifact), "2026-09-14")

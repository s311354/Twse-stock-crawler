"""Adapt published v6 reports BEFORE owner-specific PAE allocation.

EREM and PAE valuation/CER evidence remain the existing crawler's responsibility.
Missing v6 reports block new money, not the existing holding's protective exits.
"""
from .les_v6_execution import report_failure


def apply_reports(frame, reports):
    frame = frame.copy()
    for index, row in frame.iterrows():
        report = reports.get(str(row["證券代號"]))
        frame.at[index, "低點決策"] = "DATA_INSUFFICIENT"
        frame.at[index, "Position Fraction"] = 0.0
        if report is None:
            continue
        decision, quality, technical = report.get("decision", {}), report.get("quality", {}), report.get("technical", {})
        # Keep the typed holding/EREM contract even for a non-executable report.
        frame.at[index, "低點分數"] = decision.get("score")
        frame.at[index, "低點決策"] = decision.get("signal", "DATA_INSUFFICIENT")
        for column, value in {
            "LES v5 Final Quality Status": quality.get("status", "UNKNOWN"),
            "LES v5 Final Quality Score": quality.get("score"),
            "Quality Confidence": report.get("data", {}).get("confidence"),
            "Base Data Confidence": report.get("data", {}).get("confidence"),
            "Extended Data Confidence": report.get("data", {}).get("confidence"),
            "Risk Gate": decision.get("risk_gate", "UNKNOWN"),
            "Falling Knife State": "CLEAR" if report.get("downtrend_gate", {}).get("hard_veto") is False else "HARD_NO_BUY",
            "Liquidity Status": "PASS" if report.get("liquidity", {}).get("passed") is True else "UNKNOWN",
        }.items():
            frame.at[index, column] = value
        if report_failure(report):
            # Never let a superficially valid BUY string enter PAE.
            frame.at[index, "低點決策"] = "NO_BUY"
            continue
        entry, parameters = report["entry"], report["parameters"]
        for column, value in {
            "策略停損": entry.get("stop_price"), "策略停利": entry.get("target_price"),
            "Entry Candidate": entry.get("entry_price"), "RewardRisk": entry.get("reward_risk"),
            "ATR14": technical.get("atr14"), "EREM EMA20": technical.get("ema20"),
        }.items():
            frame.at[index, column] = value
        starter = parameters["starter_fraction"] if decision["signal"] == "BUY_SMALL" else 1.0
        frame.at[index, "Position Fraction"] = parameters["maximum_position_weight"] * starter * decision["position_multiplier"]
    return frame

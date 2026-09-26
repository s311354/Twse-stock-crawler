"""OTC data/quality failures retain measured scores, never entry authority."""
import json
from dataclasses import replace

import pytest

from tests.strategies.test_daily_v6_pipeline import daily_market, build
from infrastructure.report.low_entry_v6_renderer import setup_score_summary


@pytest.mark.parametrize("has_financials", [False, True])
def test_semi_daily_output_keeps_actual_scores_while_restriction_is_unknown(has_financials):
    base = daily_market()
    snapshot = dict(base.fundamental_snapshots["2330"])
    snapshot["les_v6_filings"] = [dict(row, symbol="3105", market="TPEX", source="MOPS_IXBRL")
                                  for row in snapshot["les_v6_filings"]]
    market = replace(base, stock_numbers=["3105"], market_source="TPEX", default_sector="SEMI",
                     fundamental_snapshots={"3105": snapshot} if has_financials else {},
                     signal_dates={"3105": base.signal_dates["2330"]}, trading_restrictions={})

    _, decisions = build(market)
    row = decisions[0]
    report = json.loads(row.value("LES v6 Report"))
    decision = report["decision"]

    assert report["market"]["exchange"] == "TPEX"
    assert report["quality"]["status"] == ("PASS" if has_financials else "DATA_INSUFFICIENT")
    assert decision["score"] == row.value("LES v6 Score") > 0
    assert decision["score_setup"] in {"REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION"}
    assert decision["score"] == report["setups"][decision["score_setup"]]["score"]
    assert "N/A" not in setup_score_summary(report)
    assert report["data"]["restriction_check"]["status"] == "UNKNOWN"
    assert decision["signal"] not in {"BUY", "BUY_SMALL"}
    assert report["sizing"]["indicative_shares"] == 0
    assert row.value("建議股數") == row.value("Position Fraction") == 0

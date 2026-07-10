import pandas as pd

from infrastructure.report.html_renderer import RendererFactory


def test_renderer_factory_renders_low_entry_v3_html() -> None:
    renderer = RendererFactory.get_renderer("low_entry_score_v3")
    html = renderer.render(
        pd.DataFrame(
            [
                {
                    "證券代號": "2382",
                    "證券名稱": "廣達",
                    "收盤": 109000.0,
                    "低點分數": 75,
                    "EREM 分數": 82,
                    "EREM Holding Confidence": 84,
                    "EREM Entry LES": 87,
                    "EREM 決策訊號": "HOLD",
                    "EREM RewardRisk": 1.8,
                    "EREM Relative Strength60": 0.08,
                    "EREM Higher High": 1,
                    "EREM Higher Low": 1,
                    "EREM Turnover Strength": 1.35,
                    "EREM VWAP60": 106000.0,
                    "EREM Highest120": 122000.0,
                    "EREM UpperBound": 121500.0,
                    "EMA Alignment": 1,
                    "Price Above EMA20": 1,
                    "RSI14": 52.5,
                    "MACD Bullish": 1,
                    "Near Lower Band": 0,
                    "ATR Contraction": 1,
                    "Volume Above MA20": 1,
                    "OBV Rising": 1,
                    "Support Holding": 1,
                    "Higher Low": 1,
                    "策略停損": 105000.0,
                    "策略停利": 115000.0,
                    "低點決策": "BUY",
                    "低點理由": "趨勢+25；決策=BUY",
                    "EREM 理由": "趨勢延續+20；決策=HOLD",
                }
            ]
        )
    )

    assert "Low-Entry Model v3" in html
    assert "EREM v3" in html
    #assert "持有信心" in html
    #assert "RS60" in html
    #assert "VWAP60" in html
    #assert "EREM最高120日" in html
    #assert "預期報酬出場判斷決策" in html
    assert "2382" in html
    assert "BUY" in html
    assert "HOLD" in html


def test_renderer_factory_exposes_expected_return_exit_v3_renderer() -> None:
    assert RendererFactory.get_renderer("expected_return_exit_v3") is not None

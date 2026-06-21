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
                }
            ]
        )
    )

    assert "Low-Entry Model v3" in html
    assert "EMA多頭排列" in html
    assert "2382" in html
    assert "BUY" in html

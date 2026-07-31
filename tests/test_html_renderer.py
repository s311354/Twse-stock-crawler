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


def test_renderer_factory_renders_low_entry_v4_gates() -> None:
    renderer = RendererFactory.get_renderer("low_entry_score_v4")
    rendered = renderer.render(
        pd.DataFrame(
            [
                {
                    "證券代號": "2330",
                    "證券名稱": "台積電",
                    "收盤": 1000.0,
                    "Quality Gate": "PASS",
                    "Quality Score": 84,
                    "Data Confidence": 91,
                    "Available Metric Weight": 89,
                    "Market Data Gate": "PASS",
                    "Fundamental Available At": "2026-07-21T03:00:00+00:00",
                    "Fundamental Availability Source": "FIRST_OBSERVED_DOWNLOAD",
                    "Fundamental Revision": 2,
                    "低點分數": 82,
                    "No Veto": 1,
                    "LES v4 Trigger": 1,
                    "Risk Gate": "PASS",
                    "RewardRisk": 2.2,
                    "RS20": 0.03,
                    "策略停損": 960.0,
                    "策略停利": 1088.0,
                    "低點決策": "BUY",
                    "EREM 決策訊號": "HOLD",
                    "低點理由": "所有進場 Gate 通過",
                    "EREM 理由": "持有",
                }
            ]
        )
    )

    assert "Low-Entry Model v4" in rendered
    assert "Quality Gate" in rendered
    assert "Data Confidence" in rendered
    assert "可用QS權重" in rendered
    assert "Risk Gate" in rendered
    assert "Market Data Gate" in rendered
    assert "FIRST_OBSERVED_DOWNLOAD" in rendered
    assert "財報版本" in rendered
    assert "WATCH_ONLY" in rendered
    assert "2330" in rendered
    assert "BUY" in rendered


def test_renderer_factory_renders_low_entry_v4_1_components() -> None:
    renderer = RendererFactory.get_renderer("low_entry_score_v4_1")
    rendered = renderer.render(
        pd.DataFrame(
            [
                {
                    "證券代號": "2330",
                    "證券名稱": "台積電",
                    "收盤": 1000.0,
                    "LES Profile": "BALANCED_PROFILE",
                    "Quality Gate": "PASS",
                    "Quality Score": 84,
                    "Data Confidence": 91,
                    "Current Audit Opinion": "REVIEWED_UNMODIFIED",
                    "Comparative Audit Opinion": (
                        "QUALIFIED_SCOPE_LIMITATION"
                    ),
                    "Provisional Buy Eligible": 0,
                    "Market Data Gate": "PASS",
                    "低點分數": 82,
                    "市場環境分數": 15,
                    "價格位置分數": 12,
                    "止跌結構分數": 25,
                    "買盤確認分數": 16,
                    "動能分數": 8,
                    "執行風控分數": 6,
                    "No Veto": 1,
                    "LES v4.1 Technical Trigger": 1,
                    "LES v4.1 Strict Trigger": 1,
                    "LES v4.1 Trigger": 1,
                    "Liquidity Gate": "PASS",
                    "Risk Gate": "PASS",
                    "Strict Risk Gate": "PASS",
                    "RewardRisk": 2.2,
                    "Position Fraction": 1 / 3,
                    "策略停損": 960.0,
                    "策略停利": 1088.0,
                    "低點決策": "BUY",
                    "EREM 決策訊號": "HOLD",
                    "低點理由": "所有進場 Gate 通過",
                    "EREM 理由": "持有",
                }
            ]
        )
    )

    assert "Low-Entry Model v4.1" in rendered
    assert "動能" in rendered
    assert "Technical Trigger" in rendered
    assert "Liquidity Gate" in rendered
    assert "BALANCED_PROFILE" in rendered
    assert "REVIEWED_UNMODIFIED" in rendered
    assert "QUALIFIED_SCOPE_LIMITATION" in rendered
    assert "BUY_SMALL" in rendered
    assert "BUY" in rendered

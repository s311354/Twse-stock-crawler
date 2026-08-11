import datetime

from domain.model_contracts import DecisionField
from domain.model_contracts import FieldSource
from domain.model_contracts import FieldUnit
from domain.model_contracts import StrategyDecision
from infrastructure.report.html_renderer import RendererFactory
from textmewhenitsdone import StockhtmlMessage


EVALUATED_AT = datetime.datetime(
    2026,
    7,
    25,
    13,
    30,
    tzinfo=datetime.timezone(datetime.timedelta(hours=8)),
)


def decisions(values: dict[str, object]) -> tuple[StrategyDecision, ...]:
    fields = tuple(
        DecisionField(
            name=name,
            value=value,
            source=FieldSource.LES_V5,
            available_at=EVALUATED_AT,
            confidence=100,
            unit=FieldUnit.TEXT,
            model_version="renderer-test-v1",
        )
        for name, value in values.items()
    )
    return (
        StrategyDecision(
            strategy_name="renderer_test",
            evaluated_at=EVALUATED_AT,
            fields=fields,
        ),
    )


def test_renderer_factory_renders_low_entry_v3_html() -> None:
    renderer = RendererFactory.get_renderer("low_entry_score_v3")
    html = renderer.render(
        decisions(
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
        decisions(
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
        decisions(
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
                    "低點理由": (
                        "Quality=PASS(QS=84；DC=91)；"
                        "分數=15/12/25/16/8/6；Profile=BALANCED_PROFILE；"
                        "Veto=NONE；Trigger=YES；Risk=PASS/PASS(PASS)；"
                        "決策=BUY"
                    ),
                    "EREM 理由": "持有",
            }
        )
    )

    assert "Low-Entry Model v4.1" in rendered
    assert "動能" in rendered
    assert "Technical Trigger" in rendered
    assert "Liquidity Gate" in rendered
    assert "BALANCED_PROFILE" in rendered
    assert "REVIEWED_UNMODIFIED" in rendered
    assert "QUALIFIED_SCOPE_LIMITATION" in rendered
    assert "決策快速表" in rendered
    assert "PASS，Score 65–74" in rendered
    assert "等待 Trigger／Risk 完成" in rendered
    assert "PROVISIONAL BUY_SMALL" in rendered
    assert "<br>" in rendered
    assert "&#8226; </span>分數=15/12/25/16/8/6" in rendered
    assert "&#8226; </span>Veto=NONE" in rendered
    assert "BUY_SMALL" in rendered
    assert "BUY" in rendered


def test_renderer_factory_renders_low_entry_v5_quality_and_panic_state() -> None:
    renderer = RendererFactory.get_renderer("low_entry_score_v5")
    rendered = renderer.render(
        decisions(
            {
                    "證券代號": "3036A",
                    "證券名稱": "文曄甲特",
                    "Security Type": "PREFERRED",
                    "Issuer Symbol": "3036",
                    "收盤": 1000.0,
                    "LES v5 Final Quality Status": "QUALITY_CONFIRMED",
                    "LES v5 Final Quality Score": 82,
                    "Base Quality Status": "BASE_PASS",
                    "Base Quality Score": 80,
                    "Base Data Confidence": 92,
                    "Extended Quality Status": "EXTENDED_PASS",
                    "Extended Quality Score": 85,
                    "Extended Data Confidence": 88,
                    "Critical Field Coverage": 90,
                    "Quality Warnings": "TWSE_BASE_NOT_POINT_IN_TIME",
                    "Quality Hard Rejects": "NONE",
                    "Valuation Gate": "PASS",
                    "Valuation Model Status": "COMPLETE",
                    "Base Case Expected CAGR": 0.13,
                    "Base Case Fair Value": 1120.0,
                    "Bear Case Fair Value": 880.0,
                    "Bear Case Permanent Loss": 0.12,
                    "Bull Case Fair Value": 1420.0,
                    "Valuation Reasons": "NONE",
                    "Falling Knife State": "REVERSAL_WATCH",
                    "Panic Condition Count": 2,
                    "Reversal Confirmation Count": 4,
                    "Broad Rally Regime": "ACTIVE",
                    "Broad Rally Trigger": 1,
                    "Broad Rally Full BUY Trigger": 0,
                    "Broad Rally Absorption Trigger": 1,
                    "Broad Rally Risk Gate": "PASS",
                    "低點分數": 84,
                    "Required Technical Score": 80,
                    "LES v5 Trigger": 1,
                    "Risk Gate": "PASS",
                    "RewardRisk": 2.3,
                    "Position Fraction": 0.25,
                    "策略停損": 960,
                    "策略停利": 1090,
                    "Entry Rule": "下一交易日突破訊號日最高價",
                    "低點決策": "REVERSAL_WATCH",
                    "EREM 決策訊號": "HOLD",
                    "低點理由": (
                        "Quality=QUALITY_CONFIRMED；Valuation=PASS；"
                        "Falling=REVERSAL_WATCH；決策=REVERSAL_WATCH"
                    ),
                    "EREM 理由": "持有",
            }
        )
    )

    assert "Low-Entry Model v5" in rendered
    assert "Low-Entry Model v5｜進場模式" in rendered
    assert "EREM v3｜出場模式" in rendered
    assert "高分不能抵銷 Hard Reject" in rendered
    assert "40–54 REDUCE 25%" in rendered
    assert "低於 40 EXIT" in rendered
    assert "進場決策彙整（BUY / BUY_SMALL）" in rendered
    assert "本次沒有 BUY / BUY_SMALL 可執行訊號" in rendered
    assert "Quality-First Panic Reversal" in rendered
    assert "QUALITY_CONFIRMED" in rendered
    assert "REVERSAL_WATCH" in rendered
    assert "決策總覽" in rendered
    assert "個股依據與 QA" in rendered
    assert "Base FV" in rendered
    assert "1,120" in rendered
    assert "Bear Loss" in rendered
    assert "TWSE_BASE_NOT_POINT_IN_TIME" in rendered
    assert "BASE_PASS" in rendered
    assert "EXTENDED_PASS" in rendered
    assert "QS 80" in rendered
    assert "DC 92" in rendered
    assert "QS 85" in rendered
    assert "DC 88" in rendered
    assert "ACTIVE" in rendered
    assert "Absorption 1" in rendered
    assert "下一交易日突破訊號日最高價" in rendered
    assert "證券身分" in rendered
    assert "發行人 3036" in rendered
    assert "<th>Base Status</th>" not in rendered
    assert "<th>Quality Warnings</th>" not in rendered
    assert "<th>Breadth Full BUY</th>" not in rendered


def test_email_body_omits_legacy_estimated_max_profit_table() -> None:
    body = StockhtmlMessage.format(
        days=335,
        begin="2025-04-01",
        end="2026-08-03",
        entryanalysis="<section>LES v5 / EREM v3</section>",
    )

    assert "Here is the estimated max profit" not in body
    assert "$NTD" not in body
    assert "{stocktable}" not in body
    assert "LES v5 / EREM v3" in body


def test_low_entry_v5_renders_buy_decisions_in_a_separate_readable_summary() -> None:
    renderer = RendererFactory.get_renderer("low_entry_score_v5")
    buy = decisions(
        {
            "證券代號": "2330",
            "證券名稱": "台積電",
            "收盤": 1000.0,
            "Entry Candidate": 1010.0,
            "Entry Rule": "NEXT_SESSION_BREAKS_SIGNAL_HIGH",
            "Position Fraction": 1 / 3,
            "建議股數": 1000,
            "策略停損": 960.0,
            "策略停利": 1090.0,
            "RewardRisk": 2.3,
            "低點分數": 84,
            "Required Technical Score": 80,
            "LES v5 Trigger": 1,
            "Risk Gate": "PASS",
            "低點決策": "BUY",
            "EREM 決策訊號": "HOLD",
        }
    )
    buy_small = decisions(
        {
            "證券代號": "2317",
            "證券名稱": "鴻海",
            "收盤": 220.0,
            "Entry Candidate": 222.0,
            "Entry Rule": "NEXT_SESSION_BREAKS_SIGNAL_HIGH",
            "Position Fraction": 0.125,
            "建議股數": 2000,
            "策略停損": 211.0,
            "策略停利": 239.0,
            "RewardRisk": 2.0,
            "低點分數": 81,
            "Required Technical Score": 80,
            "LES v5 Trigger": 1,
            "Risk Gate": "PASS",
            "低點決策": "BUY_SMALL",
            "EREM 決策訊號": "HOLD",
        }
    )
    watch = decisions(
        {
            "證券代號": "2382",
            "證券名稱": "廣達",
            "收盤": 300.0,
            "低點決策": "WATCH",
            "EREM 決策訊號": "HOLD",
        }
    )

    rendered = renderer.render(watch + buy_small + buy)
    entry_section = rendered.split(
        'data-section="entry-decisions"',
        1,
    )[1].split("</section>", 1)[0]

    assert "共 <strong>2</strong> 檔" in entry_section
    assert "BUY <strong>1</strong>" in entry_section
    assert "BUY_SMALL <strong>1</strong>" in entry_section
    assert "2330" in entry_section
    assert "2317" in entry_section
    assert "2382" not in entry_section
    assert entry_section.index("2330") < entry_section.index("2317")
    assert "觸發 1,010" in entry_section
    assert "1,000 股" in entry_section
    assert "停損 960" in entry_section
    assert "目標 1,090" in entry_section
    assert "RR 2.3" in entry_section
    assert "LES 84 / 門檻 80" in entry_section
    assert "BUY_SMALL 為縮小部位訊號" in entry_section

import datetime

from domain.model_contracts import DecisionField
from domain.model_contracts import FieldSource
from domain.model_contracts import FieldUnit
from domain.model_contracts import StrategyDecision
from infrastructure.report.html_renderer import PortfolioAllocationRenderer
from infrastructure.report.html_renderer import (
    PortfolioAllocationV1Renderer,
)
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


def test_portfolio_renderer_is_version_neutral_with_v1_alias() -> None:
    assert PortfolioAllocationV1Renderer is PortfolioAllocationRenderer
    assert isinstance(
        RendererFactory.get_renderer("portfolio_allocation_v1"),
        PortfolioAllocationRenderer,
    )
    assert isinstance(
        RendererFactory.get_renderer("portfolio_allocation_v2"),
        PortfolioAllocationRenderer,
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


def test_renderer_factory_renders_compact_portfolio_allocation_summary() -> None:
    renderer = RendererFactory.get_renderer("portfolio_allocation_v1")
    rendered = renderer.render(
        decisions(
            {
                "證券代號": "2330",
                "證券名稱": "台積電",
                "PAE Status": "SUCCESS",
                "PAE Mode": "COLD_START",
                "PAE Portfolio Input": "SYMBOL_ONLY_EQUAL_WEIGHT",
                "PAE Role": "CANDIDATE",
                "PAE CER": 0.17,
                "PAE Quality Score": 90.0,
                "PAE Required Return": 0.10,
                "PAE Opportunity Edge": 0.07,
                "PAE Absolute Strategic Candidate": True,
                "PAE Strategic Candidate": True,
                "PAE Executable Candidate": True,
                "PAE Dual Dominance Candidate": None,
                "PAE Capital Claim": 0.055,
                "PAE Best Strategic Alternative": "2330",
                "PAE Best Executable Alternative": "2330",
                "PAE Current Weight": 0.0,
                "PAE Target Weight": 0.33,
                "PAE Weight Change": 0.33,
                "PAE Target Cash Weight": 0.67,
                "PAE Portfolio Expected CER": 0.17,
                "PAE Decision": "ADD",
                "PAE Reason": "COLD_START_EXECUTABLE_CANDIDATE",
            }
        )
    )

    assert "投資組合配置（PAE v1.2.1）" in rendered
    assert "COLD_START" in rendered
    assert "必要報酬" in rendered
    assert "目標現金" in rendered
    assert "持股基準" in rendered
    assert "SYMBOL_ONLY_EQUAL_WEIGHT" in rendered
    assert "進出場訊號 1" in rendered
    assert "已篩選 0" in rendered
    assert "2330" in rendered
    assert "ADD" in rendered


def test_portfolio_renderer_reads_v2_model_version_from_decision() -> None:
    renderer = RendererFactory.get_renderer("portfolio_allocation_v2")
    rendered = renderer.render(
        decisions(
            {
                "證券代號": "2330",
                "證券名稱": "台積電",
                "PAE Status": "SUCCESS",
                "PAE Mode": "DIRECT_SELECTION",
                "PAE Binding": "NON_BINDING",
                "PAE Role": "CANDIDATE",
                "PAE CER": 0.17,
                "PAE Required Return": 0.10,
                "PAE Strategic Candidate": True,
                "PAE Executable Candidate": True,
                "PAE Target Weight": 0.10,
                "PAE Target Cash Weight": 0.90,
                "PAE Decision": "STARTER",
                "PAE Reason": "LES_DIRECT_SELECTION_NON_BINDING",
                "PAE Model Version": "2.0.0",
                "PAE Schema Version": "pae.portfolio_decision.v2",
            }
        )
    )

    assert "投資組合配置（PAE v2.0.0）" in rendered
    assert "NON_BINDING" in rendered
    assert "DIRECT_SELECTION" in rendered
    assert "PAE v1.2.1" not in rendered


def test_portfolio_renderer_hides_non_actionable_candidates() -> None:
    renderer = RendererFactory.get_renderer("portfolio_allocation_v1")
    rows = tuple(
        decisions(
            {
                "證券代號": symbol,
                "證券名稱": f"候選{symbol}",
                "PAE Status": "SUCCESS",
                "PAE Mode": "STEADY_STATE",
                "PAE Role": "CANDIDATE",
                "PAE CER": cer,
                "PAE Quality Score": quality,
                "PAE Strategic Candidate": True,
                "PAE Executable Candidate": False,
                "PAE Capital Claim": 0.0,
                "PAE Best Strategic Alternative": "1001",
                "PAE Best Executable Alternative": None,
                "PAE Target Weight": 0.0,
                "PAE Decision": "STRATEGIC_WATCH",
                "PAE Reason": "STRATEGIC_NOT_EXECUTABLE_OR_SLOT_UNAVAILABLE",
            }
        )[0]
        for symbol, quality, cer in (
            ("1001", 90, 0.14),
            ("1002", 89, 0.13),
            ("1003", 88, 0.12),
            ("1004", 87, 0.11),
        )
    )

    rendered = renderer.render(rows)

    assert "STRATEGIC_WATCH" not in rendered
    assert "1001" not in rendered
    assert "本次沒有涉及進場、加碼、輪動、減碼或退出的 PAE 訊號" in rendered


def test_portfolio_renderer_only_shows_requested_action_signals() -> None:
    renderer = RendererFactory.get_renderer("portfolio_allocation_v1")
    allowed = (
        "ADD",
        "STRONG_ADD",
        "ROTATE_25",
        "ROTATE_50",
        "ROTATE_75",
        "FAST_TRACK_ROTATE",
        "EXIT",
        "EXIT_AND_REALLOCATE",
        "EXIT_TO_CASH",
        "REDUCE_25",
        "STARTER",
    )
    hidden = ("HOLD", "HOLD_REVIEW", "STRATEGIC_WATCH")
    rows = tuple(
        decisions(
            {
                "證券代號": str(2000 + index),
                "證券名稱": action,
                "PAE Status": "SUCCESS",
                "PAE Mode": "STEADY_STATE",
                "PAE Role": "CANDIDATE",
                "PAE Decision": action,
                "PAE Target Weight": 0.1,
            }
        )[0]
        for index, action in enumerate(allowed + hidden)
    )

    rendered = renderer.render(rows)

    for index, action in enumerate(allowed):
        assert action in rendered
        assert str(2000 + index) in rendered
    for index, action in enumerate(hidden, start=len(allowed)):
        assert action not in rendered
        assert str(2000 + index) not in rendered
    assert "進出場訊號 11" in rendered


def test_portfolio_renderer_explains_building_mode() -> None:
    renderer = RendererFactory.get_renderer("portfolio_allocation_v1")
    rendered = renderer.render(
        decisions(
            {
                "證券代號": "2330",
                "證券名稱": "台積電",
                "PAE Status": "SUCCESS",
                "PAE Mode": "BUILDING",
                "PAE Role": "HOLDING",
                "PAE Decision": "HOLD",
            }
        )
    )

    assert "BUILDING" in rendered
    assert "目前 1 檔有效持股" in rendered
    assert "不套用成對支配與輪動門檻" in rendered
    assert "2330" not in rendered
    assert "HOLD" not in rendered
    assert "本次沒有涉及進場、加碼、輪動、減碼或退出的 PAE 訊號" in rendered


def test_daily_email_renders_each_decision_card_and_stock_once() -> None:
    renderer = RendererFactory.get_renderer("daily_strategy_email")
    sample = decisions(
        {
            "證券代號": "2330",
            "證券名稱": "台積電",
            "收盤": 1000.0,
            "低點決策": "BUY",
            "EREM 決策訊號": "HOLD",
            "PAE Status": "SUCCESS",
            "PAE Mode": "STEADY_STATE",
            "PAE Portfolio Input": "EXPLICIT_SHARES",
            "PAE Role": "HOLDING",
            "PAE CER": 0.17,
            "PAE Required Return": 0.10,
            "PAE Current Weight": 0.25,
            "PAE Target Weight": 0.25,
            "PAE Weight Change": 0.0,
            "PAE Target Cash Weight": 0.10,
            "PAE Portfolio Expected CER": 0.17,
            "PAE Decision": "HOLD",
            "PAE Reason": "KEEP_POSITION",
        }
    )

    rendered = renderer.render(sample + sample)

    assert rendered.count('data-section="daily-decision-cards"') == 1
    assert rendered.count('data-section="entry-decisions"') == 1
    assert rendered.count('data-section="portfolio-allocation"') == 1
    assert rendered.index('data-section="entry-decisions"') < rendered.index(
        'data-section="portfolio-allocation"'
    )
    portfolio_card = rendered.split(
        'data-section="portfolio-allocation"',
        1,
    )[1].split("</section>", 1)[0]
    assert "2330" not in portfolio_card
    assert "HOLD" not in portfolio_card
    assert "分析 1 檔" in portfolio_card


def test_daily_email_stacks_decision_tables_with_labels_on_mobile() -> None:
    rendered = RendererFactory.get_renderer("daily_strategy_email").render(
        decisions(
            {
                "證券代號": "2330",
                "證券名稱": "台積電",
                "收盤": 1000.0,
                "低點分數": 82,
                "低點決策": "BUY",
                "EREM 分數": 78,
                "EREM 決策訊號": "HOLD",
                "PAE Status": "SUCCESS",
                "PAE Mode": "COLD_START",
                "PAE Role": "CANDIDATE",
                "PAE Quality Score": 88,
                "PAE CER": 0.16,
                "PAE Required Return": 0.10,
                "PAE Current Weight": 0.0,
                "PAE Target Weight": 0.30,
                "PAE Target Cash Weight": 0.70,
                "PAE Portfolio Expected CER": 0.16,
                "PAE Decision": "ADD",
                "PAE Reason": "COLD_START_EXECUTABLE",
            }
        )
    )

    assert '<main class="twse-daily-report">' in rendered
    assert "@media only screen and (max-width: 720px)" in rendered
    assert "grid-template-columns: minmax(104px, 35%) minmax(0, 1fr)" in rendered
    assert rendered.count('class="responsive-decision-table"') == 3
    assert 'aria-label="可執行進場決策"' in rendered
    assert 'aria-label="投資組合配置決策"' in rendered
    assert 'aria-label="Low-Entry 與 EREM 決策總覽"' in rendered
    for label in (
        "股票",
        "進場決策",
        "品質／CER",
        "PAE 決策",
        "EREM v3 持有分數",
        "EREM 出場決策",
    ):
        assert 'data-label="{}"'.format(label) in rendered


def test_low_entry_v5_explains_no_entry_state_with_top_candidates() -> None:
    renderer = RendererFactory.get_renderer("low_entry_score_v5")
    rows = tuple(
        decisions(values)[0]
        for values in (
            {
                "證券代號": "2330",
                "證券名稱": "台積電",
                "低點分數": 74,
                "Required Technical Score": 75,
                "低點決策": "WATCH",
                "LES v5 Final Quality Status": "QUALITY_CONFIRMED",
                "Valuation Gate": "PASS",
                "Falling Knife State": "CLEAR",
                "LES v5 Trigger": 0,
                "Risk Gate": "PASS",
            },
            {
                "證券代號": "2454",
                "證券名稱": "聯發科",
                "低點分數": 88,
                "Required Technical Score": 75,
                "低點決策": "NO_BUY",
                "LES v5 Final Quality Status": "QUALITY_CONFIRMED",
                "Valuation Gate": "FAIL",
                "Falling Knife State": "CLEAR",
                "LES v5 Trigger": 1,
                "Risk Gate": "PASS",
            },
            {
                "證券代號": "2317",
                "證券名稱": "鴻海",
                "低點分數": 51,
                "Required Technical Score": 75,
                "低點決策": "AVOID",
                "LES v5 Final Quality Status": "QUALITY_CONFIRMED",
                "Valuation Gate": "PASS",
                "Falling Knife State": "CLEAR",
                "LES v5 Trigger": 0,
                "Risk Gate": "FAIL",
            },
            {
                "證券代號": "2382",
                "證券名稱": "廣達",
                "低點決策": "DATA_INSUFFICIENT",
                "LES v5 Final Quality Status": "UNKNOWN",
                "Valuation Gate": "UNKNOWN",
                "Falling Knife State": "DATA_INSUFFICIENT",
                "Risk Gate": "UNKNOWN",
            },
        )
    )

    rendered = renderer.render_entry_summary(rows)

    assert "共分析 <strong>4</strong> 檔" in rendered
    assert "沒有 BUY / EARLY_BUY / BUY_SMALL 可執行訊號" in rendered
    assert "決策分布：AVOID <strong>1</strong>" in rendered
    assert "DATA_INSUFFICIENT <strong>1</strong>" in rendered
    assert "NO_BUY <strong>1</strong>" in rendered
    assert "WATCH <strong>1</strong>" in rendered
    assert 'aria-label="未執行候選診斷"' in rendered
    assert "最高 LES 非執行候選" in rendered
    assert rendered.index("2454") < rendered.index("2330")
    assert rendered.index("2330") < rendered.index("2317")
    assert "2382" not in rendered
    assert "實際 <strong>88</strong>／門檻 75｜高於門檻 13 分" in rendered
    assert "Valuation Gate" not in rendered
    assert "FAIL" in rendered
    assert "僅用於定位阻擋狀態，不代表可執行" in rendered
    for label in (
        "目前決策",
        "LES／門檻",
        "Falling-Knife",
        "Trigger／Risk",
    ):
        assert 'data-label="{}"'.format(label) in rendered


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
                    "EREM 分數": 72,
                    "EREM Holding Confidence": 76,
                    "EREM Entry LES": 84,
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
    assert "40–54 REDUCE 25%–50%（HC 越低減越多）" in rendered
    assert "低於 40 立即 EXIT" in rendered
    assert "BUY 且 Entry LES≥75 的一般 REDUCE 寬限 3 個持有交易日" in rendered
    assert "BUY_SMALL 寬限 2 日" in rendered
    assert "EARLY_BUY 採保守 2 個持有交易日寬限" in rendered
    assert "進場決策彙整（BUY / EARLY_BUY / BUY_SMALL）" in rendered
    assert "本次沒有 BUY / EARLY_BUY / BUY_SMALL 可執行訊號" in rendered
    assert "Quality-First Panic Reversal" in rendered
    assert "QUALITY_CONFIRMED" in rendered
    assert "REVERSAL_WATCH" in rendered
    assert "決策總覽" in rendered
    assert "LES v5 進場分數" in rendered
    assert "EREM v3 持有分數" in rendered
    assert "即時 <strong>72</strong>／100" in rendered
    summary_table = rendered.split(
        'aria-label="Low-Entry 與 EREM 決策總覽"',
        1,
    )[1]
    summary_row = summary_table.split("<tbody>", 1)[1].split(
        "</tbody>",
        1,
    )[0]
    assert "實際 <strong>84</strong>／100" in summary_row
    assert "本次門檻" not in summary_row
    assert "持有信心" not in summary_row
    assert "Entry LES" not in summary_row
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
    early_buy = decisions(
        {
            "證券代號": "2454",
            "證券名稱": "聯發科",
            "收盤": 1400.0,
            "Entry Candidate": 1410.0,
            "Entry Rule": "NEXT_DAY_REVERSAL_CONFIRMATION",
            "Position Fraction": 0.1625,
            "建議股數": 100,
            "策略停損": 1350.0,
            "策略停利": 1520.0,
            "RewardRisk": 2.1,
            "低點分數": 74,
            "Required Technical Score": 75,
            "LES v5 Trigger": 1,
            "Risk Gate": "PASS",
            "低點決策": "EARLY_BUY",
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

    rendered = renderer.render(watch + buy_small + early_buy + buy)
    entry_section = rendered.split(
        'data-section="entry-decisions"',
        1,
    )[1].split("</section>", 1)[0]

    assert "共 <strong>3</strong> 檔" in entry_section
    assert "BUY <strong>1</strong>" in entry_section
    assert "EARLY_BUY <strong>1</strong>" in entry_section
    assert "BUY_SMALL <strong>1</strong>" in entry_section
    assert "2330" in entry_section
    assert "2454" in entry_section
    assert "2317" in entry_section
    assert "2382" not in entry_section
    assert entry_section.index("2330") < entry_section.index("2454")
    assert entry_section.index("2454") < entry_section.index("2317")
    assert "觸發 1,010" in entry_section
    assert "1,000 股" in entry_section
    assert "停損 960" in entry_section
    assert "目標 1,090" in entry_section
    assert "RR 2.3" in entry_section
    assert "LES 進場分數 <strong>84</strong>" in entry_section
    assert "本次門檻 80｜高於門檻 4 分" in entry_section
    assert "EARLY_BUY 採 0.65 execution factor" in entry_section
    assert "BUY_SMALL 採縮小部位配置" in entry_section

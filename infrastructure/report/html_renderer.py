"""HTML renderers for strategy analysis sections."""

from __future__ import annotations

import html
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from tabulate import tabulate


@dataclass(frozen=True)
class MetricColumn:
    label: str
    column: str
    precision: int = 2


class HtmlRenderer(ABC):
    """Base renderer for strategy output embedded in email reports."""

    @abstractmethod
    def render(self, result: pd.DataFrame) -> str:
        pass


class LowEntryHtmlRenderer(HtmlRenderer):
    def __init__(
        self,
        title: str,
        description: str,
        metrics: list[MetricColumn],
        decision_note: str | None = None,
    ) -> None:
        self.title = title
        self.description = description
        self.metrics = metrics
        self.decision_note = decision_note or (
            "<strong>低點進場判斷決策</strong>：BUY &gt;= 75；WATCH = 60-74；WAIT &lt; 60。"
            "風控採 ATR14：策略停損 = 收盤 - 2*ATR14；策略停利 = 收盤 + 3*ATR14。"
        )

    def render(self, result: pd.DataFrame) -> str:
        rows = []

        for _, stock in result.iterrows():
            row = [
                "{} {}".format(html.escape(str(stock["證券代號"])), html.escape(str(stock["證券名稱"]))),
                self._format_value(stock["收盤"], 2),
            ]
            row.extend(self._format_value(stock.get(metric.column), metric.precision) for metric in self.metrics)
            row.extend([
                html.escape(str(stock.get("低點決策", "N/A"))),
                html.escape(str(stock.get("EREM 決策訊號", "N/A"))),
                html.escape(str(stock.get("低點理由", "N/A"))),
                html.escape(str(stock.get("EREM 理由", "N/A"))),
            ])
            rows.append(row)

        df_entry = pd.DataFrame(
            rows,
            columns=["股票", "現價"] + [metric.label for metric in self.metrics] + [
                "低點進場判斷決策",
                "預期報酬出場判斷決策",
                "低點理由",
                "EREM理由",
            ],
        )

        return """
        <h3>{}</h3>
        <p>{}</p>
        <p>{}</p>
        {}
        """.format(
            html.escape(self.title),
            self.description,
            self.decision_note,
            tabulate(df_entry, headers="keys", tablefmt="html", showindex=False),
        )

    def _format_value(self, value: object, precision: int = 2) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        if isinstance(value, str):
            return html.escape(value)

        rounded_value = round(float(value), precision)
        if precision == 0:
            return str(int(rounded_value))
        return str(rounded_value)


class LowEntryV1Renderer(LowEntryHtmlRenderer):
    def __init__(self) -> None:
        super().__init__(
            title="低點進場判斷（Low-Entry Model v1）",
            description="<strong>Score 規則 v1</strong>：價格位置、MA 趨勢、RSI14、布林下緣、成交量 Z-score、Hammer K 線、MACD、PE 估值濾網，合計 0-100 分。",
            metrics=[
                MetricColumn("低點分數", "低點分數"),
                MetricColumn("60日位置", "60日位置"),
                MetricColumn("RSI14", "RSI14"),
                MetricColumn("成交量Z", "成交量Z"),
                MetricColumn("MACD", "MACD"),
                MetricColumn("ATR14", "ATR14"),
                MetricColumn("策略停損", "策略停損"),
                MetricColumn("策略停利", "策略停利"),
            ],
        )


class LowEntryV2Renderer(LowEntryHtmlRenderer):
    def __init__(self) -> None:
        super().__init__(
            title="低點進場判斷（Low-Entry Model v2）",
            description="<strong>Score 規則 v2</strong>：60日低位加分、高位懲罰、MA20 初步反轉或站上 MA20、RSI14、布林下緣、下跌爆量/縮量止跌、Hammer K 線、MACD、PE 估值濾網，合計 -20 到 100 分。",
            metrics=[
                MetricColumn("低點分數", "低點分數"),
                MetricColumn("60日位置", "60日位置"),
                MetricColumn("Trend Reversal", "Trend Reversal", 0),
                MetricColumn("站上MA20", "Close Above MA20", 0),
                MetricColumn("RSI14", "RSI14"),
                MetricColumn("成交量Z", "成交量Z"),
                MetricColumn("下跌爆量", "Volume Spike Down", 0),
                MetricColumn("縮量止跌", "Volume Dry", 0),
                MetricColumn("MACD", "MACD"),
                MetricColumn("ATR14", "ATR14"),
                MetricColumn("策略停損", "策略停損"),
                MetricColumn("策略停利", "策略停利"),
            ],
        )


class ExpectedReturnExitV3Renderer(LowEntryHtmlRenderer):
    def __init__(self) -> None:
        super().__init__(
            title="低點進場判斷（Low-Entry Model v3）+ 預期報酬出場判斷（EREM v3）",
            description=(
                "<strong>Score 規則 v3</strong>：趨勢 25 分、動能 20 分、波動 20 分、量能 20 分、結構 15 分，合計 0-100 分。"
                "<br><strong>Score 規則 EREM v3</strong>：趨勢、動能、波動、籌碼、剩餘報酬、TAIEX 相對強弱、"
                "趨勢結構、Entry LES、週轉強度與 VWAP。規格原始權重合計 105 分，固定正規化為 0-100 分。"
            ),
            metrics=[
                MetricColumn("低點分數", "低點分數"),
                MetricColumn("EREM分數", "EREM 分數"),
                #MetricColumn("持有信心", "EREM Holding Confidence"),
                #MetricColumn("Entry LES", "EREM Entry LES"),
                #MetricColumn("EREM RR", "EREM RewardRisk"),
                #MetricColumn("RS60", "EREM Relative Strength60", 4),
                #MetricColumn("Higher High", "EREM Higher High", 0),
                #MetricColumn("Higher Low", "EREM Higher Low", 0),
                #MetricColumn("週轉強度", "EREM Turnover Strength", 4),
                #MetricColumn("VWAP60", "EREM VWAP60"),
                #MetricColumn("EREM最高120日", "EREM Highest120"),
                #MetricColumn("EREM上界", "EREM UpperBound"),
                MetricColumn("策略停損", "策略停損"),
                MetricColumn("策略停利", "策略停利"),
            ],
            decision_note=(
                "<strong>低點進場判斷決策</strong>：BUY &gt;= 75；WATCH = 60-74；WAIT &lt; 60。"
                "<br><strong>預期報酬出場判斷決策</strong>（依 Holding Confidence）：STRONG HOLD &gt;= 85；"
                "HOLD = 70-84；HOLD = 55-69；REDUCE 25% = 40-54；EXIT &lt; 40。"
                "此分級以降低不必要換手與過度交易。"
                "<br>風控採 ATR14：策略停損 = 收盤 - 2*ATR14；策略停利 = 收盤 + 3*ATR14。"
            ),
        )


class LowEntryV3Renderer(ExpectedReturnExitV3Renderer):
    """Backward-compatible renderer alias for the existing LES v3 report key."""


class LowEntryV4Renderer(LowEntryHtmlRenderer):
    def __init__(self) -> None:
        super().__init__(
            title="低點進場判斷（Low-Entry Model v4）+ 預期報酬出場判斷（EREM v3）",
            description=(
                "<strong>LES v4</strong>：Quality Score 只評估可計算財報指標；Data Confidence 獨立衡量"
                "覆蓋率、時效、驗證與來源可靠度。缺值不等於零分或安全，非 PASS 不得 BUY。"
            ),
            metrics=[
                MetricColumn("Quality Gate", "Quality Gate", 0),
                MetricColumn("Quality Score", "Quality Score"),
                MetricColumn("Data Confidence", "Data Confidence"),
                MetricColumn("可用QS權重", "Available Metric Weight"),
                MetricColumn("會計師意見", "Audit Opinion", 0),
                MetricColumn("Market Data Gate", "Market Data Gate", 0),
                MetricColumn("財報可用時間", "Fundamental Available At", 0),
                MetricColumn("時間來源", "Fundamental Availability Source", 0),
                MetricColumn("財報版本", "Fundamental Revision", 0),
                MetricColumn("LES v4 分數", "低點分數"),
                MetricColumn("No Veto", "No Veto", 0),
                MetricColumn("Trigger", "LES v4 Trigger", 0),
                MetricColumn("Risk Gate", "Risk Gate", 0),
                MetricColumn("RR", "RewardRisk"),
                MetricColumn("RS20", "RS20", 4),
                MetricColumn("策略停損", "策略停損"),
                MetricColumn("策略目標", "策略停利"),
            ],
            decision_note=(
                "<strong>Quality 狀態</strong>：QS &gt;= 70 且 DC &gt;= 70 才是 PASS；DC 50-69 為 PROVISIONAL；"
                "QS 55-69 為 WATCH_QUALITY；DC &lt; 50 為 UNKNOWN。"
                "範圍受限但僅涉及未核閱的非重要子公司／被投資公司時，標示 QUALIFIED_SCOPE_LIMITATION、"
                "DC 扣 10 分但不觸發 Hard Reject。"
                "<br><strong>LES v4 決策順序</strong>：REJECT &gt; DATA_INSUFFICIENT &gt; WATCH_ONLY &gt; NO_BUY &gt; AVOID &gt; WATCH &gt; ARMED &gt; BUY。"
                "BUY 必須同時符合 Quality Gate=PASS、Market Data Gate=PASS、No Veto、Score &gt;= 80、Trigger、RR &gt;= 2、"
                "停損距離 &lt;= 1.5 ATR 與 Liquidity Gate。"
                "<br><strong>成交規則</strong>：訊號日收盤確認，下一交易日突破訊號日最高價才成交；"
                "若跳空造成 Risk Gate 失效則取消，不追價。"
            ),
        )


class LowEntryV41Renderer(LowEntryHtmlRenderer):
    def __init__(self) -> None:
        super().__init__(
            title=(
                "低點進場判斷（Low-Entry Model v4.1）"
                "+ 預期報酬出場判斷（EREM v3）"
            ),
            description=(
                "<strong>LES v4.1 BALANCED_PROFILE</strong>：先通過 Quality Score 與 "
                "Data Confidence，再依市場環境 15、價格位置 15、止跌結構 30、"
                "買盤確認 20、動能 10、執行風控 10 計分。"
                "高分不能抵銷 Falling-Knife Veto。"
            ),
            metrics=[
                MetricColumn("Profile", "LES Profile", 0),
                MetricColumn(
                    "產業品質 Profile",
                    "Industry Quality Profile",
                    0,
                ),
                MetricColumn(
                    "產業模式",
                    "Industry Profile Mode",
                    0,
                ),
                MetricColumn(
                    "產業規則版本",
                    "Industry Profile Version",
                    0,
                ),
                MetricColumn("Quality Gate", "Quality Gate", 0),
                MetricColumn("Quality Score", "Quality Score"),
                MetricColumn("Data Confidence", "Data Confidence"),
                MetricColumn("本期意見", "Current Audit Opinion", 0),
                MetricColumn("前期意見", "Comparative Audit Opinion", 0),
                MetricColumn("Provisional Eligible", "Provisional Buy Eligible", 0),
                MetricColumn("Market Data Gate", "Market Data Gate", 0),
                MetricColumn("LES v4.1 分數", "低點分數"),
                MetricColumn("市場", "市場環境分數", 0),
                MetricColumn("位置", "價格位置分數", 0),
                MetricColumn("止跌", "止跌結構分數", 0),
                MetricColumn("買盤", "買盤確認分數", 0),
                MetricColumn("動能", "動能分數", 0),
                MetricColumn("執行", "執行風控分數", 0),
                MetricColumn("No Veto", "No Veto", 0),
                MetricColumn("Technical Trigger", "LES v4.1 Technical Trigger", 0),
                MetricColumn("Strict Trigger", "LES v4.1 Strict Trigger", 0),
                MetricColumn("Trigger", "LES v4.1 Trigger", 0),
                MetricColumn("Liquidity Gate", "Liquidity Gate", 0),
                MetricColumn("Risk Gate", "Risk Gate", 0),
                MetricColumn("Strict Risk", "Strict Risk Gate", 0),
                MetricColumn("RR", "RewardRisk"),
                MetricColumn("部位比例", "Position Fraction", 4),
                MetricColumn("策略停損", "策略停損"),
                MetricColumn("策略目標", "策略停利"),
            ],
            decision_note=(
                "<strong>決策順序</strong>：REJECT→NO_BUY；UNKNOWN→DATA_INSUFFICIENT；"
                "SECTOR_UNSUPPORTED→NO_MODEL；WATCH_QUALITY 原則上→WATCH_ONLY。"
                "CHEMICAL_PROFILE 為 ACTIVE：QS≥65 可進標準 BUY 流程；"
                "QS 60–64、DC≥75 僅可在 Strict Trigger、Strict Risk 與所有 "
                "Veto 通過時進入 1/8 部位 BUY_SMALL。"
                "<br><strong>BALANCED PASS</strong>：任一 Veto→NO_BUY；"
                "Score&lt;55→AVOID；55–64→WATCH；65–74→ARMED；"
                "Score≥75 且 Trigger、Risk Gate、RR≥1.7、停損距離≤1.8 ATR14 "
                "與流動性通過才是 BUY，初始部位上限為 1/3。"
                "<br><strong>PROVISIONAL BUY_SMALL</strong>：僅限 QS≥75、DC≥60、"
                "獲利與資產負債覆蓋率各≥70%、無 Hard Flag、無已確認負 CFO；"
                "並須 Score≥80、Strict Trigger、Strict Risk=PASS、RR≥2、"
                "停損距離≤1.5 ATR14，初始部位上限為 1/6。"
                "訊號於收盤確認，下一交易日突破訊號日最高價才考慮成交。"
            ),
        )


class RendererFactory:
    _renderers: dict[str, HtmlRenderer] = {
        "low_entry_score": LowEntryV1Renderer(),
        "low_entry_score_v2": LowEntryV2Renderer(),
        "low_entry_score_v3": LowEntryV3Renderer(),
        "low_entry_score_v4": LowEntryV4Renderer(),
        "low_entry_score_v4_1": LowEntryV41Renderer(),
        "expected_return_exit_v3": ExpectedReturnExitV3Renderer(),
    }

    @classmethod
    def get_renderer(cls, strategy_name: str) -> HtmlRenderer:
        try:
            return cls._renderers[strategy_name]
        except KeyError as error:
            raise KeyError("Unknown renderer '{}'. Available: {}".format(strategy_name, sorted(cls._renderers))) from error

"""HTML renderers for strategy analysis sections."""

from __future__ import annotations

import html
from abc import ABC, abstractmethod
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from tabulate import tabulate

from domain.model_contracts import StrategyDecision
from domain.strategies.portfolio_allocation_v1 import PAE_V1_MODEL_VERSION


@dataclass(frozen=True)
class MetricColumn:
    label: str
    column: str
    precision: int = 2


class HtmlRenderer(ABC):
    """Base renderer for strategy output embedded in email reports."""

    @abstractmethod
    def render(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> str:
        pass


class LowEntryHtmlRenderer(HtmlRenderer):
    _REASON_SECTION_BREAK = "__LOW_ENTRY_REASON_SECTION_BREAK__"

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

    def render(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> str:
        rows: list[list[str]] = []

        for decision in decisions:
            if not isinstance(decision, StrategyDecision):
                raise TypeError(
                    "HtmlRenderer accepts StrategyDecision objects only"
                )
            row = [
                "{} {}".format(
                    html.escape(str(decision.value("證券代號", "N/A"))),
                    html.escape(str(decision.value("證券名稱", "N/A"))),
                ),
                self._format_value(decision.value("收盤"), 2),
            ]
            row.extend(
                self._format_value(
                    decision.value(metric.column),
                    metric.precision,
                )
                for metric in self.metrics
            )
            row.extend([
                html.escape(str(decision.value("低點決策", "N/A"))),
                html.escape(str(decision.value("EREM 決策訊號", "N/A"))),
                self._format_low_entry_reason(
                    decision.value("低點理由", "N/A")
                ),
                html.escape(str(decision.value("EREM 理由", "N/A"))),
            ])
            rows.append(row)

        headers = (
            ["股票", "現價"]
            + [metric.label for metric in self.metrics]
            + [
                "低點進場判斷決策",
                "預期報酬出場判斷決策",
                "低點理由",
                "EREM理由",
            ]
        )

        table_html = tabulate(
            rows,
            headers=headers,
            tablefmt="html",
        ).replace(
            self._REASON_SECTION_BREAK,
            '<br><span style="display:inline-block;margin-top:4px">&#8226; </span>',
        )

        return """
        <h3>{}</h3>
        <div>{}</div>
        <div>{}</div>
        <div style="overflow-x:auto">{}</div>
        """.format(
            html.escape(self.title),
            self.description,
            self.decision_note,
            table_html,
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

    def _format_low_entry_reason(self, value: object) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        formatted = html.escape(str(value))
        section_labels = (
            "分數",
            "Valuation",
            "Penalty",
            "Falling",
            "Panic/Reversal",
            "Required",
            "Profile",
            "ProvisionalEligible",
            "Veto",
            "TechnicalTrigger",
            "StrictTrigger",
            "Trigger",
            "Liquidity",
            "Risk",
            "CHEMIProbe",
            "ELECProbe",
            "ELECRecovery",
            "BroadRally",
            "ELECRally",
            "SEMIProbe",
            "決策",
        )
        for label in section_labels:
            formatted = formatted.replace(
                "；{}=".format(label),
                "{}{}=".format(self._REASON_SECTION_BREAK, label),
            )
        return formatted


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
                '<section style="border-left:4px solid #1f6f5f;padding:8px 12px;'
                'margin:8px 0;background:#f4f8f6">'
                "<strong>LES v4.1 / BALANCED_PROFILE</strong>"
                '<ul style="margin:6px 0 0 18px;padding:0">'
                "<li><strong>先看品質</strong>：Quality Score 與 Data Confidence "
                "必須符合進場資格。</li>"
                "<li><strong>再算技術分數</strong>：市場環境 15、價格位置 15、"
                "止跌結構 30、買盤確認 20、動能 10、執行風控 10，合計 100 分。</li>"
                "<li><strong>最後檢查硬條件</strong>：高分不能抵銷 "
                "Falling-Knife Veto，也不能略過 Trigger、Risk 與流動性。</li>"
                "</ul></section>"
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
                '<section style="margin:10px 0">'
                "<h4 style=\"margin:0 0 6px\">決策快速表</h4>"
                '<table style="border-collapse:collapse" border="1" cellpadding="5">'
                "<tr><th>前提／分數</th><th>決策</th><th>必要條件</th></tr>"
                "<tr><td>Quality=REJECT</td><td>NO_BUY</td><td>不可被技術分數覆蓋</td></tr>"
                "<tr><td>Quality=UNKNOWN</td><td>DATA_INSUFFICIENT</td><td>補足資料後再評估</td></tr>"
                "<tr><td>產業不支援</td><td>NO_MODEL</td><td>不套用錯誤產業門檻</td></tr>"
                "<tr><td>Quality=WATCH_QUALITY</td><td>WATCH_ONLY</td>"
                "<td>原則上只觀察；符合產業專屬嚴格條件時才可走 BUY_SMALL</td></tr>"
                "<tr><td>PASS，Score &lt; 55</td><td>AVOID</td><td>不進場</td></tr>"
                "<tr><td>PASS，Score 55–64</td><td>WATCH</td><td>等待止跌與買盤確認</td></tr>"
                "<tr><td>PASS，Score 65–74</td><td>ARMED</td><td>等待 Trigger／Risk 完成</td></tr>"
                "<tr><td>PASS，Score ≥ 75</td><td>BUY</td><td>無 Veto；Trigger、Risk、"
                "Market Data 與流動性全部通過</td></tr>"
                "</table>"
                '<div style="margin-top:8px"><strong>標準 BUY 風控</strong>：'
                "RR ≥ 1.7、停損距離 ≤ 1.8 ATR14；初始部位上限 1/3。</div>"
                '<div style="margin-top:8px"><strong>CHEMICAL_PROFILE</strong>：'
                "QS ≥ 65 可走標準流程；QS 60–64 且 DC ≥ 75，只能在 Strict "
                "Trigger、Strict Risk 與所有 Veto 通過後，以 1/8 部位 BUY_SMALL。</div>"
                '<div style="margin-top:8px"><strong>PROVISIONAL BUY_SMALL</strong>：'
                "QS ≥ 75、DC ≥ 60、獲利與資產負債覆蓋率各 ≥ 70%，且無 Hard Flag "
                "或已確認負 CFO；另須 Score ≥ 80、Strict Trigger、Strict Risk=PASS、"
                "RR ≥ 2、停損距離 ≤ 1.5 ATR14，初始部位上限 1/6。</div>"
                '<div style="margin-top:8px"><strong>成交原則</strong>：訊號於收盤確認；'
                "下一交易日突破訊號日最高價才考慮成交。</div>"
                "</section>"
            ),
        )


class LowEntryV5Renderer(LowEntryHtmlRenderer):
    _ENTRY_DECISIONS = frozenset({"BUY", "BUY_SMALL"})

    def __init__(self) -> None:
        super().__init__(
            title=(
                "低點進場判斷（Low-Entry Model v5）"
                "+ 預期報酬出場判斷（EREM v3）"
            ),
            description=(
                '<section data-section="model-guide" style="margin:10px 0">'
                '<div style="border-left:5px solid #176b4d;background:#f3faf7;'
                'padding:10px 12px;margin-bottom:8px">'
                '<strong>Low-Entry Model v5｜進場模式</strong>'
                '<span style="color:#52665f">（Quality-First Panic Reversal）</span><br>'
                "品質與估值先過關，再等待恐慌賣壓後的反轉確認。"
                "只有 Trigger、Risk、流動性與市場資料 Gate 全部通過，"
                "才產生 BUY／BUY_SMALL；高分不能抵銷 Hard Reject 或 Falling-Knife Veto。"
                "</div>"
                '<div style="border-left:5px solid #356a9a;background:#f3f7fb;'
                'padding:10px 12px">'
                '<strong>EREM v3｜出場模式</strong><br>'
                "評估趨勢、動能、波動、籌碼與剩餘報酬，目的是讓強勢部位續抱、"
                "避免因正常回檔過度交易。85–100 STRONG HOLD；70–84 HOLD；"
                "55–69 HOLD；40–54 REDUCE 25%；低於 40 EXIT。"
                "</div></section>"
            ),
            metrics=[],
            decision_note=(
                '<div style="margin:8px 0;padding:9px 11px;background:#fff8e8;'
                'border:1px solid #ead7aa">'
                "<strong>執行重點</strong>：LES &lt;60 AVOID；60–69 WATCH；"
                "70 起 ARMED；正式 BUY 依 Quality 狀態要求 75 或 80 分，且 RR≥1.7。"
                "訊號於收盤確認，下一交易日突破訊號日最高價才考慮成交；"
                "若風控失效則取消，不追價。"
                "</div>"
            ),
        )

    def render(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> str:
        return self.render_analysis(decisions, include_entry_summary=True)

    def render_analysis(
        self,
        decisions: Sequence[StrategyDecision],
        *,
        include_entry_summary: bool,
    ) -> str:
        for decision in decisions:
            if not isinstance(decision, StrategyDecision):
                raise TypeError(
                    "HtmlRenderer accepts StrategyDecision objects only"
                )

        summary_rows = "".join(
            self._render_summary_row(decision) for decision in decisions
        )
        entry_summary = (
            self.render_entry_summary(decisions)
            if include_entry_summary
            else ""
        )
        detail_cards = "".join(
            self._render_detail_card(decision) for decision in decisions
        )
        return """
        <h3>{}</h3>
        <div>{}</div>
        {}
        <div>{}</div>
        <h4 style="margin:16px 0 6px">決策總覽</h4>
        <div style="overflow-x:auto">
          <table style="border-collapse:collapse;width:100%;font-size:13px" border="1" cellpadding="6">
            <thead style="background:#eaf2ef">
              <tr>
                <th>股票</th><th>現價</th><th>品質</th><th>估值</th>
                <th>反轉狀態</th><th>LES v5 進場分數</th>
                <th>EREM v3 持有分數</th><th>Trigger／Risk</th>
                <th>部位與價格</th><th>進場決策</th><th>EREM 出場決策</th>
              </tr>
            </thead>
            <tbody>{}</tbody>
          </table>
        </div>
        <h4 style="margin:18px 0 6px">個股依據與 QA</h4>
        <div>{}</div>
        """.format(
            html.escape(self.title),
            self.description,
            entry_summary,
            self.decision_note,
            summary_rows,
            detail_cards,
        )

    def render_entry_summary(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> str:
        """Render the executable entry card for composite email reports."""
        return self._render_entry_decision_summary(decisions)

    def _render_entry_decision_summary(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> str:
        selected = sorted(
            (
                decision
                for decision in decisions
                if self._entry_decision(decision) in self._ENTRY_DECISIONS
            ),
            key=lambda decision: (
                0 if self._entry_decision(decision) == "BUY" else 1,
                str(decision.value("證券代號", "")),
            ),
        )
        if not selected:
            return """
            <section data-section="entry-decisions" style="margin:14px 0;padding:12px;border:1px solid #d7dfdc;border-left:5px solid #6f7f79;background:#f7f9f8">
              <h3 style="margin:0 0 6px;color:#174f43">進場決策彙整（BUY / BUY_SMALL）</h3>
              <div>本次沒有 BUY / BUY_SMALL 可執行訊號。</div>
            </section>
            """

        buy_count = sum(
            self._entry_decision(decision) == "BUY"
            for decision in selected
        )
        buy_small_count = len(selected) - buy_count
        rows = "".join(
            self._render_entry_decision_row(decision)
            for decision in selected
        )
        return """
        <section data-section="entry-decisions" style="margin:14px 0;padding:12px;border:1px solid #b9d8cc;border-left:5px solid #176b4d;background:#f3faf7">
          <h3 style="margin:0 0 5px;color:#174f43">進場決策彙整（BUY / BUY_SMALL）</h3>
          <div style="margin-bottom:8px;color:#38554b">
            共 <strong>{}</strong> 檔｜BUY <strong>{}</strong>｜BUY_SMALL <strong>{}</strong>；僅列正式進場訊號。
          </div>
          <div style="overflow-x:auto">
            <table style="border-collapse:collapse;width:100%;font-size:13px;background:#fff" border="1" cellpadding="6">
              <thead style="background:#dcefe7">
                <tr>
                  <th>股票</th><th>進場決策</th><th>執行價格</th>
                  <th>部位</th><th>風控</th><th>訊號確認</th>
                </tr>
              </thead>
              <tbody>{}</tbody>
            </table>
          </div>
          <div style="margin-top:7px;font-size:12px;color:#52665f">
            BUY_SMALL 為縮小部位訊號；實際成交仍依 Entry Rule 與下一交易日價格確認。
          </div>
        </section>
        """.format(len(selected), buy_count, buy_small_count, rows)

    def _render_entry_decision_row(
        self,
        decision: StrategyDecision,
    ) -> str:
        stock = "<strong>{}</strong><br><small>{}</small>".format(
            self._safe_text(decision.value("證券代號")),
            self._safe_text(decision.value("證券名稱")),
        )
        prices = "現價 {}<br><strong>觸發 {}</strong><br><small>{}</small>".format(
            self._format_money(decision.value("收盤")),
            self._format_money(decision.value("Entry Candidate")),
            self._safe_text(decision.value("Entry Rule")),
        )
        position = "{}<br><small>{} 股</small>".format(
            self._format_percent(decision.value("Position Fraction")),
            self._format_shares(decision.value("建議股數")),
        )
        risk = "停損 {}<br>目標 {}<br><small>RR {}</small>".format(
            self._format_money(decision.value("策略停損")),
            self._format_money(decision.value("策略停利")),
            self._format_value(decision.value("RewardRisk"), 2),
        )
        confirmation = (
            "LES 進場分數 <strong>{}</strong><br>本次門檻 {}{}"
            "<br>Trigger {} / {}<br><small>EREM 出場決策 {}</small>"
        ).format(
            self._format_value(decision.value("低點分數"), 0),
            self._format_value(decision.value("Required Technical Score"), 0),
            self._score_gap_text(
                decision.value("低點分數"),
                decision.value("Required Technical Score"),
            ),
            self._format_flag(decision.value("LES v5 Trigger")),
            self._badge(decision.value("Risk Gate")),
            self._safe_text(decision.value("EREM 決策訊號")),
        )
        cells = (
            stock,
            self._entry_decision_badge(self._entry_decision(decision)),
            prices,
            position,
            risk,
            confirmation,
        )
        return "<tr>{}</tr>".format(
            "".join(
                '<td style="vertical-align:top">{}</td>'.format(cell)
                for cell in cells
            )
        )

    @staticmethod
    def _entry_decision(decision: StrategyDecision) -> str:
        return str(decision.value("低點決策", "")).strip().upper()

    @staticmethod
    def _entry_decision_badge(decision: str) -> str:
        colors = (
            ("#176b4d", "#dff3e9")
            if decision == "BUY"
            else ("#8a5700", "#fff0c2")
        )
        return (
            '<span style="display:inline-block;padding:3px 8px;'
            'border-radius:10px;font-weight:700;color:{};background:{}">{}</span>'
        ).format(colors[0], colors[1], html.escape(decision))

    def _format_shares(self, value: object) -> str:
        if self._is_missing(value):
            return "N/A"
        try:
            return "{:,.0f}".format(float(value))
        except (TypeError, ValueError):
            return html.escape(str(value))

    def _render_summary_row(self, decision: StrategyDecision) -> str:
        stock = "{} {}".format(
            html.escape(str(decision.value("證券代號", "N/A"))),
            html.escape(str(decision.value("證券名稱", "N/A"))),
        )
        quality = "{}<br><small>QS {}</small>".format(
            self._badge(decision.value("LES v5 Final Quality Status")),
            self._format_value(
                decision.value("LES v5 Final Quality Score"),
                0,
            ),
        )
        valuation = "{}<br><small>Base FV {}<br>CAGR {}</small>".format(
            self._badge(decision.value("Valuation Gate")),
            self._format_money(decision.value("Base Case Fair Value")),
            self._format_percent(decision.value("Base Case Expected CAGR")),
        )
        reversal = "{}<br><small>Panic {}／Reversal {}</small>".format(
            self._badge(decision.value("Falling Knife State")),
            self._format_value(decision.value("Panic Condition Count"), 0),
            self._format_value(
                decision.value("Reversal Confirmation Count"),
                0,
            ),
        )
        les_score = "實際 <strong>{}</strong>／100".format(
            self._format_value(decision.value("低點分數"), 0),
        )
        erem_score = "即時 <strong>{}</strong>／100".format(
            self._format_value(decision.value("EREM 分數"), 0),
        )
        trigger_risk = "Trigger {}<br>{}<br><small>RR {}</small>".format(
            self._format_flag(decision.value("LES v5 Trigger")),
            self._badge(decision.value("Risk Gate")),
            self._format_value(decision.value("RewardRisk"), 2),
        )
        position = "部位 {}<br><small>停損 {}<br>目標 {}</small>".format(
            self._format_percent(decision.value("Position Fraction")),
            self._format_money(decision.value("策略停損")),
            self._format_money(decision.value("策略停利")),
        )
        cells = (
            stock,
            self._format_money(decision.value("收盤")),
            quality,
            valuation,
            reversal,
            les_score,
            erem_score,
            trigger_risk,
            position,
            self._badge(decision.value("低點決策")),
            self._badge(decision.value("EREM 決策訊號")),
        )
        return "<tr>{}</tr>".format(
            "".join(
                '<td style="vertical-align:top">{}</td>'.format(cell)
                for cell in cells
            )
        )

    def _score_gap_text(self, score: object, threshold: object) -> str:
        if self._is_missing(score) or self._is_missing(threshold):
            return "｜差距 N/A"
        try:
            gap = float(score) - float(threshold)
        except (TypeError, ValueError):
            return "｜差距 N/A"
        if gap >= 0:
            return "｜高於門檻 {} 分".format(self._format_value(gap, 0))
        return "｜尚差 {} 分".format(self._format_value(abs(gap), 0))

    def _render_detail_card(self, decision: StrategyDecision) -> str:
        stock = "{} {}".format(
            html.escape(str(decision.value("證券代號", "N/A"))),
            html.escape(str(decision.value("證券名稱", "N/A"))),
        )
        rows = [
            self._detail_row(
                "品質證據",
                "Base {}｜QS {}｜DC {}<br>Extended {}｜QS {}｜DC {}"
                "<br>Critical Coverage {}".format(
                    self._badge(decision.value("Base Quality Status")),
                    self._format_value(decision.value("Base Quality Score"), 0),
                    self._format_value(decision.value("Base Data Confidence"), 0),
                    self._badge(decision.value("Extended Quality Status")),
                    self._format_value(
                        decision.value("Extended Quality Score"),
                        0,
                    ),
                    self._format_value(
                        decision.value("Extended Data Confidence"),
                        0,
                    ),
                    self._format_percent(
                        decision.value("Critical Field Coverage"),
                        whole_number=True,
                    ),
                ),
            ),
            self._detail_row(
                "品質 QA",
                "Warnings：{}<br>Hard Rejects：{}".format(
                    self._safe_text(decision.value("Quality Warnings")),
                    self._safe_text(decision.value("Quality Hard Rejects")),
                ),
            ),
            self._detail_row(
                "估值依據",
                "資料 {}｜Base FV {}｜Bear FV {}｜Bull FV {}｜Bear Loss {}"
                "<br>Valuation QA：{}".format(
                    self._badge(decision.value("Valuation Model Status")),
                    self._format_money(decision.value("Base Case Fair Value")),
                    self._format_money(decision.value("Bear Case Fair Value")),
                    self._format_money(decision.value("Bull Case Fair Value")),
                    self._format_percent(
                        decision.value("Bear Case Permanent Loss")
                    ),
                    self._safe_text(decision.value("Valuation Reasons")),
                ),
            ),
        ]
        security_type = self._safe_text(decision.value("Security Type"))
        if security_type not in {"N/A", "COMMON"}:
            rows.insert(
                0,
                self._detail_row(
                    "證券身分",
                    "市場代號 {}｜發行人 {}｜類型 {}".format(
                        self._safe_text(decision.value("證券代號")),
                        self._safe_text(decision.value("Issuer Symbol")),
                        security_type,
                    ),
                ),
            )
        breadth = self._breadth_summary(decision)
        if breadth:
            rows.append(self._detail_row("市場廣度", breadth))
        rows.extend([
            self._detail_row(
                "進場規則",
                self._safe_text(decision.value("Entry Rule")),
            ),
            self._detail_row(
                "低點理由",
                self._format_reason(decision.value("低點理由")),
            ),
            self._detail_row(
                "EREM 理由",
                self._format_reason(decision.value("EREM 理由")),
            ),
        ])
        return """
        <section style="border:1px solid #d7dfdc;border-radius:6px;margin:8px 0 14px;padding:10px;background:#fbfcfb">
          <h4 style="margin:0 0 8px;color:#174f43">{}</h4>
          <table style="border-collapse:collapse;width:100%;font-size:13px" cellpadding="5">
            {}
          </table>
        </section>
        """.format(stock, "".join(rows))

    @staticmethod
    def _detail_row(label: str, value: str) -> str:
        return (
            '<tr><th style="width:120px;text-align:left;vertical-align:top;'
            'border-top:1px solid #e5e9e7">{}</th>'
            '<td style="border-top:1px solid #e5e9e7">{}</td></tr>'
        ).format(html.escape(label), value)

    def _breadth_summary(self, decision: StrategyDecision) -> str:
        fields = (
            ("Tier", "Broad Rally Tier"),
            ("Regime", "Broad Rally Regime"),
            ("Trigger", "Broad Rally Trigger"),
            ("Full BUY", "Broad Rally Full BUY Trigger"),
            ("Absorption", "Broad Rally Absorption Trigger"),
            ("Risk", "Broad Rally Risk Gate"),
        )
        values = []
        for label, field_name in fields:
            value = decision.value(field_name)
            if self._is_missing(value):
                continue
            values.append("{} {}".format(html.escape(label), self._safe_text(value)))
        return "｜".join(values)

    def _badge(self, value: object) -> str:
        text = self._safe_text(value)
        normalized = text.upper().replace(" ", "_")
        if any(
            token in normalized
            for token in ("REJECT", "NO_BUY", "EXIT", "FAILED", "BLOCK")
        ):
            colors = ("#8f1d1d", "#fdecec")
        elif any(
            token in normalized
            for token in ("UNKNOWN", "INSUFFICIENT", "N/A", "BASE_ONLY")
        ):
            colors = ("#5f6368", "#eef0f2")
        elif any(
            token in normalized
            for token in ("WATCH", "ARMED", "PANIC", "REDUCE")
        ):
            colors = ("#8a5700", "#fff4d6")
        else:
            colors = ("#176b4d", "#e7f5ee")
        return (
            '<span style="display:inline-block;padding:2px 6px;border-radius:10px;'
            'font-weight:600;color:{};background:{}">{}</span>'
        ).format(colors[0], colors[1], text)

    def _format_flag(self, value: object) -> str:
        if self._is_missing(value):
            return "N/A"
        normalized = str(value).strip().upper()
        return "YES" if normalized in {"1", "TRUE", "YES", "PASS"} else "NO"

    def _format_money(self, value: object) -> str:
        if self._is_missing(value):
            return "N/A"
        try:
            return "{:,.2f}".format(float(value)).rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return html.escape(str(value))

    def _format_percent(
        self,
        value: object,
        *,
        whole_number: bool = False,
    ) -> str:
        if self._is_missing(value):
            return "N/A"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return html.escape(str(value))
        percent = numeric if whole_number or abs(numeric) > 1 else numeric * 100
        return "{:.1f}%".format(percent)

    def _format_reason(self, value: object) -> str:
        text = self._safe_text(value)
        if text == "N/A":
            return text
        return "&#8226; {}".format(text.replace("；", "<br>&#8226; "))

    def _safe_text(self, value: object) -> str:
        if self._is_missing(value):
            return "N/A"
        return html.escape(str(value))

    @staticmethod
    def _is_missing(value: object) -> bool:
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False


class PortfolioAllocationV1Renderer(HtmlRenderer):
    """Render the aggregate PAE decision without recalculating its rules."""

    _MATERIAL_ACTIONS = frozenset(
        {
            "STRONG_ADD",
            "ADD",
            "STARTER",
            "HOLD_REVIEW",
            "REDUCE_25",
            "ROTATE_25",
            "ROTATE_50",
            "ROTATE_75",
            "FAST_TRACK_ROTATE",
            "EXIT",
            "EXIT_AND_REALLOCATE",
            "EXIT_TO_CASH",
        }
    )

    def render(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> str:
        for decision in decisions:
            if not isinstance(decision, StrategyDecision):
                raise TypeError(
                    "HtmlRenderer accepts StrategyDecision objects only"
                )
        pae_decisions = self.latest_by_symbol(
            decision
            for decision in decisions
            if decision.value("PAE Status") is not None
        )
        if not pae_decisions:
            return ""

        first = pae_decisions[0]
        strategic_count = sum(
            self._truth(decision.value("PAE Strategic Candidate"))
            for decision in pae_decisions
        )
        executable_count = sum(
            self._truth(decision.value("PAE Executable Candidate"))
            for decision in pae_decisions
        )
        selected = self._selected(pae_decisions)
        rows = "".join(self._row(decision) for decision in selected)
        mode = self._text(first.value("PAE Mode"))
        mode_note = {
            "COLD_START": (
                "目前 0 檔有效持股；只有完整通過 Quality、Valuation、CER、LES 與風控的候選才配置資金。"
            ),
            "BUILDING": (
                "目前 1 檔有效持股，處於第二檔建構期；候選須通過絕對品質、估值、CER、LES 與風控門檻，但不套用成對支配與輪動門檻。"
            ),
            "STEADY_STATE": (
                "目前至少 2 檔有效持股；新標的必須通過品質優勢與 Switch Hurdle 的 CER 優勢，才可取得資金。"
            ),
        }.get(mode, "無法判定投資組合階段；本次結果僅供資料檢查。")
        candidate_label = (
            "Dual" if mode == "STEADY_STATE" else "Absolute"
        )
        summary_note = (
            "僅列持股、配置異動與最佳雙重支配候選。"
            if mode == "STEADY_STATE"
            else "僅列持股、配置異動與最佳絕對合格候選。"
        )
        table = (
            """
            <div style="overflow-x:auto">
              <table style="border-collapse:collapse;width:100%;font-size:13px;background:#fff" border="1" cellpadding="6">
                <thead style="background:#edf3f7">
                  <tr>
                    <th>股票／角色</th><th>品質／CER</th><th>候選資格</th>
                    <th>目前 → 目標</th><th>品質＋CER 支配證據</th><th>PAE 決策</th>
                  </tr>
                </thead>
                <tbody>{}</tbody>
              </table>
            </div>
            """.format(rows)
            if rows
            else (
                '<div style="padding:9px;background:#fff">'
                "本次沒有需要列出的持股或可執行配置；未使用資金維持現金。"
                "</div>"
            )
        )
        return """
        <section data-section="portfolio-allocation" style="margin:16px 0;padding:12px;border:1px solid #bfd0dc;border-left:5px solid #295f7a;background:#f5f9fb">
          <h3 style="margin:0 0 6px;color:#204f66">投資組合配置（PAE v{}）</h3>
          <div style="margin-bottom:9px;color:#3f5662">
            <strong>{}</strong>：{} PAE 消費 LES v5／EREM v3 已發布決策，不重算技術指標。
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:7px;margin-bottom:10px">
            {}
            {}
            {}
            {}
            {}
            {}
          </div>
          <div style="margin-bottom:8px;font-size:12px;color:#526770">
            分析 {} 檔｜{} Strategic {}｜{} Executable {}｜{}
          </div>
          {}
        </section>
        """.format(
            html.escape(PAE_V1_MODEL_VERSION),
            html.escape(mode or "DATA_INSUFFICIENT"),
            html.escape(mode_note),
            self._metric(
                "必要報酬",
                self._percent(first.value("PAE Required Return")),
            ),
            self._metric(
                "目標現金",
                self._percent(first.value("PAE Target Cash Weight")),
            ),
            self._metric(
                "組合 CER",
                self._percent(first.value("PAE Portfolio Expected CER")),
            ),
            self._metric(
                "最佳可執行",
                self._text(first.value("PAE Best Executable Alternative")),
            ),
            self._metric("狀態", self._text(first.value("PAE Status"))),
            self._metric(
                "持股基準",
                self._text(first.value("PAE Portfolio Input")),
            ),
            len(pae_decisions),
            candidate_label,
            strategic_count,
            candidate_label,
            executable_count,
            summary_note,
            table,
        )

    @classmethod
    def latest_by_symbol(
        cls,
        decisions: Iterable[StrategyDecision],
    ) -> tuple[StrategyDecision, ...]:
        """Keep one deterministic PAE row per stock code."""
        latest: dict[str, StrategyDecision] = {}
        for decision in decisions:
            symbol = cls._raw_text(decision.value("證券代號"))
            if not symbol:
                continue
            current = latest.get(symbol)
            if current is None or decision.evaluated_at >= current.evaluated_at:
                latest[symbol] = decision
        return tuple(latest[symbol] for symbol in sorted(latest))

    @staticmethod
    def _raw_text(value: object) -> str:
        if value is None:
            return ""
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    def _selected(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> tuple[StrategyDecision, ...]:
        first = decisions[0]
        alternatives = {
            self._text(first.value("PAE Best Strategic Alternative")),
            self._text(first.value("PAE Best Executable Alternative")),
        }
        selected = [
            decision
            for decision in decisions
            if self._text(decision.value("PAE Role")) == "HOLDING"
            or self._number(decision.value("PAE Target Weight")) > 0
            or self._text(decision.value("PAE Decision")) in self._MATERIAL_ACTIONS
            or self._text(decision.value("證券代號")) in alternatives
        ]
        return tuple(
            sorted(
                selected,
                key=lambda decision: (
                    0
                    if self._text(decision.value("PAE Role")) == "HOLDING"
                    else 1,
                    -self._number(decision.value("PAE Target Weight")),
                    self._text(decision.value("證券代號")),
                ),
            )[:12]
        )

    def _row(self, decision: StrategyDecision) -> str:
        stock = "<strong>{}</strong><br><small>{}</small><br>{}".format(
            self._text(decision.value("證券代號")),
            self._text(decision.value("證券名稱")),
            self._badge(decision.value("PAE Role")),
        )
        returns = "Quality <strong>{}</strong><br>CER <strong>{}</strong><br><small>Required {}｜Edge {}</small>".format(
            self._decimal(decision.value("PAE Quality Score"), 2),
            self._percent(decision.value("PAE CER")),
            self._percent(decision.value("PAE Required Return")),
            self._percent(decision.value("PAE Opportunity Edge")),
        )
        eligibility = "Strategic {}｜Executable {}<br><small>Dual {}｜Claim {}</small>".format(
            self._flag(decision.value("PAE Strategic Candidate")),
            self._flag(decision.value("PAE Executable Candidate")),
            self._flag(decision.value("PAE Dual Dominance Candidate")),
            self._decimal(decision.value("PAE Capital Claim"), 4),
        )
        weights = "{} → <strong>{}</strong><br><small>Δ {}</small>".format(
            self._percent(decision.value("PAE Current Weight")),
            self._percent(decision.value("PAE Target Weight")),
            self._signed_percent(decision.value("PAE Weight Change")),
        )
        replacement = "{} vs {}<br><small>Quality Δ {} / required {} ({})<br>CER excess {} / hurdle {} ({})<br>Dual {}｜Persistence {}</small>".format(
            self._text(decision.value("PAE Alternative")),
            self._text(decision.value("PAE Dominance Incumbent")),
            self._signed_decimal(decision.value("PAE Quality Advantage"), 2),
            self._decimal(decision.value("PAE Required Quality Advantage"), 2),
            self._text(decision.value("PAE Quality Dominance Gate")),
            self._percent(decision.value("PAE Excess Opportunity")),
            self._percent(decision.value("PAE Switch Hurdle")),
            self._text(decision.value("PAE CER Dominance Gate")),
            self._text(decision.value("PAE Dual Dominance Gate")),
            self._flag(decision.value("PAE Persistence Gate")),
        )
        final = (
            "{}<br><small>{}</small>"
            "<br><small>QualityProtect {} | Stop {} | EREM Persist {} | "
            "TrendBreak {} | ValuationBreak {} | Confirm {}/3</small>"
        ).format(
            self._badge(decision.value("PAE Decision")),
            self._text(decision.value("PAE Reason")),
            self._flag(decision.value("PAE High Quality Protected")),
            self._flag(decision.value("PAE Strategy Stop Breached")),
            self._flag(decision.value("PAE EREM Exit Persistent")),
            self._flag(decision.value("PAE Trend Broken")),
            self._flag(decision.value("PAE Valuation Broken")),
            self._decimal(decision.value("PAE Exit Confirmation Count"), 0),
        )
        return "<tr>{}</tr>".format(
            "".join(
                '<td style="vertical-align:top">{}</td>'.format(cell)
                for cell in (
                    stock,
                    returns,
                    eligibility,
                    weights,
                    replacement,
                    final,
                )
            )
        )

    @staticmethod
    def _metric(label: str, value: str) -> str:
        return (
            '<div style="min-width:125px;padding:7px 9px;border:1px solid #cad8e1;'
            'background:#fff"><small style="color:#62737c">{}</small><br>'
            "<strong>{}</strong></div>"
        ).format(html.escape(label), value)

    @staticmethod
    def _text(value: object) -> str:
        if value is None:
            return "N/A"
        try:
            if bool(pd.isna(value)):
                return "N/A"
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        return html.escape(text) if text else "N/A"

    @staticmethod
    def _number(value: object) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return numeric if pd.notna(numeric) else 0.0

    @classmethod
    def _percent(cls, value: object) -> str:
        if value is None:
            return "N/A"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return cls._text(value)
        if pd.isna(numeric):
            return "N/A"
        return "{:.1f}%".format(numeric * 100)

    @classmethod
    def _signed_percent(cls, value: object) -> str:
        if value is None:
            return "N/A"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return cls._text(value)
        if pd.isna(numeric):
            return "N/A"
        return "{:+.1f}%".format(numeric * 100)

    @classmethod
    def _decimal(cls, value: object, precision: int) -> str:
        if value is None:
            return "N/A"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return cls._text(value)
        if pd.isna(numeric):
            return "N/A"
        return ("{:.%df}" % precision).format(numeric)

    @classmethod
    def _signed_decimal(cls, value: object, precision: int) -> str:
        if value is None:
            return "N/A"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return cls._text(value)
        if pd.isna(numeric):
            return "N/A"
        return ("{:+.%df}" % precision).format(numeric)

    @classmethod
    def _flag(cls, value: object) -> str:
        return "YES" if cls._truth(value) else "NO"

    @staticmethod
    def _truth(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "pass"}

    @classmethod
    def _badge(cls, value: object) -> str:
        text = cls._text(value)
        normalized = html.unescape(text).upper()
        if any(token in normalized for token in ("EXIT", "FAIL", "INSUFFICIENT")):
            colors = ("#8f1d1d", "#fdecec")
        elif any(token in normalized for token in ("HOLD", "WATCH", "CANDIDATE")):
            colors = ("#755000", "#fff4d6")
        else:
            colors = ("#176b4d", "#e7f5ee")
        return (
            '<span style="display:inline-block;padding:2px 6px;border-radius:10px;'
            'font-weight:600;color:{};background:{}">{}</span>'
        ).format(colors[0], colors[1], text)


class DailyStrategyEmailRenderer(HtmlRenderer):
    """Compose each decision card once, followed by the detailed report."""

    def __init__(self) -> None:
        self.entry_renderer = LowEntryV5Renderer()
        self.portfolio_renderer = PortfolioAllocationV1Renderer()

    def render(
        self,
        decisions: Sequence[StrategyDecision],
    ) -> str:
        unique_decisions = self.portfolio_renderer.latest_by_symbol(decisions)
        entry_card = self.entry_renderer.render_entry_summary(unique_decisions)
        portfolio_card = self.portfolio_renderer.render(unique_decisions)
        details = self.entry_renderer.render_analysis(
            unique_decisions,
            include_entry_summary=False,
        )
        return """
        <section data-section="daily-decision-cards" style="margin:12px 0 18px">
          <h2 style="margin:0 0 8px;color:#263d47">今日決策卡</h2>
          {}
          {}
        </section>
        {}
        """.format(entry_card, portfolio_card, details)


class RendererFactory:
    _renderers: dict[str, HtmlRenderer] = {
        "low_entry_score": LowEntryV1Renderer(),
        "low_entry_score_v2": LowEntryV2Renderer(),
        "low_entry_score_v3": LowEntryV3Renderer(),
        "low_entry_score_v4": LowEntryV4Renderer(),
        "low_entry_score_v4_1": LowEntryV41Renderer(),
        "low_entry_score_v5": LowEntryV5Renderer(),
        "expected_return_exit_v3": ExpectedReturnExitV3Renderer(),
        "portfolio_allocation_v1": PortfolioAllocationV1Renderer(),
        "daily_strategy_email": DailyStrategyEmailRenderer(),
    }

    @classmethod
    def get_renderer(cls, strategy_name: str) -> HtmlRenderer:
        try:
            return cls._renderers[strategy_name]
        except KeyError as error:
            raise KeyError("Unknown renderer '{}'. Available: {}".format(strategy_name, sorted(cls._renderers))) from error

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
    def __init__(self, title: str, description: str, metrics: list[MetricColumn]) -> None:
        self.title = title
        self.description = description
        self.metrics = metrics

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
                html.escape(str(stock.get("低點理由", "N/A"))),
            ])
            rows.append(row)

        df_entry = pd.DataFrame(
            rows,
            columns=["股票", "現價"] + [metric.label for metric in self.metrics] + ["決策", "理由"],
        )

        return """
        <h3>{}</h3>
        <p>{}</p>
        <p><strong>決策</strong>：BUY &gt;= 75；WATCH = 60-74；WAIT &lt; 60。風控採 ATR14：策略停損 = 收盤 - 2*ATR14；策略停利 = 收盤 + 3*ATR14。</p>
        {}
        """.format(
            html.escape(self.title),
            self.description,
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


class LowEntryV3Renderer(LowEntryHtmlRenderer):
    def __init__(self) -> None:
        super().__init__(
            title="低點進場判斷（Low-Entry Model v3）",
            description="<strong>Score 規則 v3</strong>：趨勢 25 分、動能 20 分、波動 20 分、量能 20 分、結構 15 分，合計 0-100 分。",
            metrics=[
                MetricColumn("低點分數", "低點分數"),
                MetricColumn("EMA多頭排列", "EMA Alignment", 0),
                MetricColumn("站上EMA20", "Price Above EMA20", 0),
                MetricColumn("RSI14", "RSI14"),
                MetricColumn("MACD翻正", "MACD Bullish", 0),
                MetricColumn("近布林下緣", "Near Lower Band", 0),
                MetricColumn("ATR收斂", "ATR Contraction", 0),
                MetricColumn("量大於MA20", "Volume Above MA20", 0),
                MetricColumn("OBV上升", "OBV Rising", 0),
                MetricColumn("支撐未破", "Support Holding", 0),
                MetricColumn("Higher Low", "Higher Low", 0),
                MetricColumn("策略停損", "策略停損"),
                MetricColumn("策略停利", "策略停利"),
            ],
        )


class RendererFactory:
    _renderers: dict[str, HtmlRenderer] = {
        "low_entry_score": LowEntryV1Renderer(),
        "low_entry_score_v2": LowEntryV2Renderer(),
        "low_entry_score_v3": LowEntryV3Renderer(),
    }

    @classmethod
    def get_renderer(cls, strategy_name: str) -> HtmlRenderer:
        try:
            return cls._renderers[strategy_name]
        except KeyError as error:
            raise KeyError("Unknown renderer '{}'. Available: {}".format(strategy_name, sorted(cls._renderers))) from error

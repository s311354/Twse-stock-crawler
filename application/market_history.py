"""Pure assembly of per-stock strategy history frames."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field

import pandas as pd

from domain.market_breadth import MARKET_BREADTH_ATTR
from domain.market_breadth import MARKET_DATA_THROUGH_ATTR
from domain.market_breadth import MARKET_SESSION_FINAL_ATTR
from domain.strategies.low_entry_score_v4 import ACCOUNT_EQUITY_ATTR
from domain.strategies.low_entry_score_v4 import ANNUAL_FUNDAMENTALS_ATTR
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_SNAPSHOT_ATTR
from domain.strategies.low_entry_score_v4 import FUNDAMENTAL_STATUS_ATTR
from domain.strategies.low_entry_score_v4 import (
    FUNDAMENTAL_STATUS_DETAIL_ATTR,
)
from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
from domain.strategies.low_entry_score_v4 import SIGNAL_DATE_ATTR


NestedValues = Sequence[Sequence[object]]


@dataclass(frozen=True)
class MarketHistoryInput:
    """Read-only view of collected market and fundamental context."""

    stock_count: int
    stock_numbers: Sequence[str]
    stock_names: Sequence[str]
    dates: NestedValues
    opens: NestedValues
    highs: NestedValues
    lows: NestedValues
    closes: NestedValues
    volumes: NestedValues
    trade_counts: NestedValues
    trade_values: NestedValues
    bids: NestedValues
    asks: NestedValues
    pe_ratios: NestedValues
    taiex_closes: NestedValues
    annual_fundamentals: Mapping[str, pd.DataFrame]
    fundamental_snapshots: Mapping[str, Mapping[str, object]]
    fundamental_statuses: Mapping[str, str]
    fundamental_status_details: Mapping[str, str]
    signal_dates: Mapping[str, str]
    account_equity: float | None
    fallback_date: str = ""
    market_breadth_snapshots: Mapping[
        str,
        Mapping[str, object],
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stock_count < 0:
            raise ValueError("stock_count cannot be negative")
        per_stock_values = (
            ("stock_numbers", self.stock_numbers),
            ("stock_names", self.stock_names),
            ("dates", self.dates),
            ("opens", self.opens),
            ("highs", self.highs),
            ("lows", self.lows),
            ("closes", self.closes),
            ("volumes", self.volumes),
            ("trade_counts", self.trade_counts),
            ("trade_values", self.trade_values),
            ("bids", self.bids),
            ("asks", self.asks),
            ("pe_ratios", self.pe_ratios),
            ("taiex_closes", self.taiex_closes),
        )
        for name, values in per_stock_values:
            if len(values) != self.stock_count:
                raise ValueError(
                    "{} must match stock_count".format(name)
                )


class MarketHistoryAssembler:
    """Build canonical pandas inputs without crawler or infrastructure access."""

    def build(
        self,
        market: MarketHistoryInput,
        item: int,
    ) -> pd.DataFrame:
        if item < 0 or item >= market.stock_count:
            raise IndexError("stock history index is out of range")

        history_length = len(market.closes[item])
        required_values = (
            ("Open", market.opens[item]),
            ("High", market.highs[item]),
            ("Low", market.lows[item]),
            ("Volume", market.volumes[item]),
            ("PE", market.pe_ratios[item]),
            ("TAIEX Close", market.taiex_closes[item]),
        )
        for name, values in required_values:
            if len(values) != history_length:
                raise ValueError(
                    "{} history length must match Close".format(name)
                )

        def aligned(values: Sequence[object]) -> list[object]:
            if len(values) == history_length:
                return list(values)
            return [None] * history_length

        history = pd.DataFrame(
            {
                "Date": aligned(market.dates[item]),
                "Open": list(market.opens[item]),
                "High": list(market.highs[item]),
                "Low": list(market.lows[item]),
                "Close": list(market.closes[item]),
                "Volume": list(market.volumes[item]),
                "Trades": aligned(market.trade_counts[item]),
                "Amount": aligned(market.trade_values[item]),
                "Bid": aligned(market.bids[item]),
                "Ask": aligned(market.asks[item]),
                "PE": list(market.pe_ratios[item]),
                "TAIEX Close": list(market.taiex_closes[item]),
            }
        )
        stock_no = market.stock_numbers[item]
        dates = market.dates[item]
        market_data_through = str(dates[-1]) if dates else None
        history.attrs[SIGNAL_DATE_ATTR] = market.signal_dates.get(
            stock_no,
            (
                "{}T13:30:00+08:00".format(dates[-1])
                if dates
                else None
            ),
        )
        history.attrs[MARKET_DATA_THROUGH_ATTR] = market_data_through
        breadth_snapshot = (
            market.market_breadth_snapshots.get(market_data_through, {})
            if market_data_through
            else {}
        )
        if breadth_snapshot:
            normalized_breadth = dict(breadth_snapshot)
            history.attrs[MARKET_BREADTH_ATTR] = normalized_breadth
            history.attrs[MARKET_SESSION_FINAL_ATTR] = bool(
                normalized_breadth.get("source_final")
            )
        else:
            history.attrs[MARKET_SESSION_FINAL_ATTR] = False
        history.attrs[PRICE_SCALE_ATTR] = 1000.0
        if stock_no in market.annual_fundamentals:
            history.attrs[ANNUAL_FUNDAMENTALS_ATTR] = (
                market.annual_fundamentals[stock_no].copy()
            )
        if stock_no in market.fundamental_snapshots:
            history.attrs[FUNDAMENTAL_SNAPSHOT_ATTR] = dict(
                market.fundamental_snapshots[stock_no]
            )
        if stock_no in market.fundamental_statuses:
            history.attrs[FUNDAMENTAL_STATUS_ATTR] = (
                market.fundamental_statuses[stock_no]
            )
            history.attrs[FUNDAMENTAL_STATUS_DETAIL_ATTR] = (
                market.fundamental_status_details.get(stock_no, "")
            )
        if market.account_equity is not None:
            history.attrs[ACCOUNT_EQUITY_ATTR] = market.account_equity
        return history

    def build_all(
        self,
        market: MarketHistoryInput,
    ) -> list[pd.DataFrame]:
        return [
            self.build(market, item)
            for item in range(market.stock_count)
        ]


__all__ = [
    "MarketHistoryAssembler",
    "MarketHistoryInput",
]

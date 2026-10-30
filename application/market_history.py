"""Pure assembly of per-stock strategy history frames."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field

import pandas as pd

from domain.numeric_history import metadata_free_frame, SCREEN_SOURCE_ROWS_ATTR
from application.snapshot_retention import snapshot_metadata

from domain.market_breadth import MARKET_BREADTH_ATTR
from domain.market_breadth import MARKET_BREADTH_HISTORY_ATTR
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
    default_sector: str = "UNKNOWN"
    les_v6_portfolio: Mapping[str, object] = field(default_factory=dict)
    trading_restrictions: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    les_v6_research_context: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    fallback_date: str = ""
    entry_les_by_stock: Mapping[str, float] = field(default_factory=dict)
    entry_dates_by_stock: Mapping[
        str,
        datetime.date | str,
    ] = field(default_factory=dict)
    entry_decisions_by_stock: Mapping[str, str] = field(default_factory=dict)
    market_breadth_snapshots: Mapping[
        str,
        Mapping[str, object],
    ] = field(default_factory=dict)
    market_source: str = "TWSE"
    replay_as_of: str | None = None
    industry_benchmarks: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    entry_model: str = "v6"
    les_v7_bundle_json: str | None = None
    les_v7_strategy_policy_json: str | None = None
    les_v7_diagnostics_only: bool = False
    les_v7_native_evidence: bool = False
    les_v7_rule_baseline: bool = False
    market_observed_at: str | None = None
    fundamental_capture_cutoffs: Mapping[str, str] = field(default_factory=dict)
    les_v7_source_repair: bool = False
    native_source_selections: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    # Explicit retained observations, not a current clock inferred by replay.
    # Native validators still bind symbol, market, availability and bars hash.
    captured_market_evidence: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    # Explicit public research entry marker; appended for positional caller
    # compatibility. A generic model alone does not opt into a new policy.
    les_v7_research_metadata_json: str | None = None

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
        *,
        include_evidence: bool = True,
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
        entry_les = market.entry_les_by_stock.get(stock_no)
        entry_date = market.entry_dates_by_stock.get(stock_no)
        entry_decision = market.entry_decisions_by_stock.get(stock_no)
        if entry_date is not None:
            parsed_entry_date = pd.to_datetime(entry_date, errors="coerce")
            parsed_dates = pd.to_datetime(history["Date"], errors="coerce")
            if pd.notna(parsed_entry_date):
                holding_days = (
                    parsed_dates.dt.normalize()
                    - pd.Timestamp(parsed_entry_date).normalize()
                ).dt.days
                active_holding = holding_days >= 0
                history["Holding Days"] = holding_days.where(active_holding)
                if entry_les is not None:
                    # Entry LES applies only from the declared entry date.
                    history["Entry LES"] = pd.Series(
                        float(entry_les),
                        index=history.index,
                    ).where(active_holding)
                if entry_decision is not None:
                    history["Entry Decision"] = pd.Series(
                        str(entry_decision),
                        index=history.index,
                    ).where(active_holding)
        elif entry_les is not None:
            # Missing entry date remains supported. EREM applies only its 10%
            # floor so unknown recency cannot claim full new-entry protection.
            history["Entry LES"] = float(entry_les)
        if entry_date is None and entry_decision is not None:
            # Retain the diagnostic value, but EREM cannot grant entry grace
            # without a dated, scored execution record.
            history["Entry Decision"] = str(entry_decision)
        dates = market.dates[item]
        market_data_through = str(dates[-1]) if dates else None
        history.attrs[SIGNAL_DATE_ATTR] = market.signal_dates.get(
            stock_no,
            (
                "{}T13:30:00+08:00".format(dates[-1])
                if dates
                else "{}T13:30:00+08:00".format(market.fallback_date)
                if market.fallback_date
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
        # Preserve only breadth snapshots visible through this stock's signal
        # date. Multi-session regimes must be derived from point-in-time facts,
        # not from a narrative about market direction or future sessions.
        if include_evidence and market.market_breadth_snapshots:
            through_timestamp = pd.to_datetime(
                market_data_through,
                errors="coerce",
            )
            visible_breadth: dict[str, dict[str, object]] = {}
            for date, snapshot in market.market_breadth_snapshots.items():
                snapshot_date = pd.to_datetime(
                    snapshot.get("as_of") or date,
                    errors="coerce",
                )
                if (
                    pd.notna(through_timestamp)
                    and pd.notna(snapshot_date)
                    and snapshot_date <= through_timestamp
                ):
                    visible_breadth[str(date)] = dict(snapshot)
            history.attrs[MARKET_BREADTH_HISTORY_ATTR] = visible_breadth
        history.attrs[PRICE_SCALE_ATTR] = 1000.0
        if market.entry_model == "v7":
            history.attrs["daily_entry_model"] = "v7"
            history.attrs["les_v7_bundle_json"] = market.les_v7_bundle_json
            if market.les_v7_research_metadata_json:
                history.attrs["les_v7_research_metadata_json"] = market.les_v7_research_metadata_json
            if market.les_v7_strategy_policy_json:
                history.attrs["les_v7_strategy_policy_json"] = market.les_v7_strategy_policy_json
            history.attrs["les_v7_diagnostics_only"] = market.les_v7_diagnostics_only
            history.attrs["les_v7_native_evidence"] = market.les_v7_native_evidence
            history.attrs["les_v7_rule_baseline"] = market.les_v7_rule_baseline
            history.attrs["les_v7_source_repair"] = market.les_v7_source_repair
            if market.les_v7_source_repair:
                history.attrs["les_v7_native_sources"] = market.native_source_selections.get(stock_no, {})
            if stock_no in market.fundamental_capture_cutoffs:
                history.attrs["les_v7_fundamental_capture_cutoff"] = market.fundamental_capture_cutoffs[stock_no]
            if market.les_v7_native_evidence and stock_no in market.captured_market_evidence:
                history.attrs["les_v7_market_evidence"] = dict(market.captured_market_evidence[stock_no])
            elif market.les_v7_native_evidence and market.market_observed_at:
                history.attrs["les_v7_market_evidence"] = {
                    "symbol": stock_no, "market": market.market_source,
                    "session_date": market_data_through,
                    "source": market.market_source + "_DAILY",
                    "source_available_at": market.market_observed_at,
                    "price_unit": "TWD", "volume_unit": "shares", "amount_unit": "TWD",
                    "source_final": history.attrs[MARKET_SESSION_FINAL_ATTR],
                    "availability_basis": "CURRENT_OBSERVATION",
                    "evidence_origin": "CURRENT_OBSERVATION_OF_OFFICIAL_FINAL_DAILY_DATA_OR_CACHE",
                    "historical_original_observation_verified": False,
                }
        history.attrs["les_v6_daily_identity"] = {
            "symbol": stock_no, "name": market.stock_names[item],
            "sector": market.default_sector,
        }
        if market.market_source != "TWSE":
            history.attrs["les_v6_daily_identity"]["market"] = market.market_source
            history.attrs["market_source"] = market.market_source
        # The scheduled completed session belongs to the batch, not to each
        # stock's last quote (which may be absent from recent market tables).
        history.attrs["les_v6_analysis_session"] = market.fallback_date or max(
            (str(values[-1]) for values in market.dates if values), default=""
        )
        history.attrs["les_v6_portfolio"] = dict(market.les_v6_portfolio)
        if market.replay_as_of:
            history.attrs["replay_as_of"] = market.replay_as_of
        history.attrs["trading_restriction"] = dict(market.trading_restrictions.get(stock_no, {}))
        research = market.les_v6_research_context.get(stock_no, {})
        history.attrs["les_v6_research_context"] = dict(research) if include_evidence else {
            key: research[key] for key in ("watch", "watch_input_sha256", "watch_result_sha256") if key in research
        }
        from domain.industry_benchmark import INDUSTRY_BENCHMARK_ATTR, sector_code
        snapshot = snapshot_metadata(market.fundamental_snapshots, stock_no)
        sector = sector_code(snapshot.get("sector") or market.default_sector)
        if include_evidence:
            history.attrs[INDUSTRY_BENCHMARK_ATTR] = market.industry_benchmarks.get(sector)
        if include_evidence and stock_no in market.annual_fundamentals:
            history.attrs[ANNUAL_FUNDAMENTALS_ATTR] = (
                market.annual_fundamentals[stock_no].copy()
            )
        if include_evidence and stock_no in market.fundamental_snapshots:
            from application.snapshot_retention import materialize_history_snapshot
            history.attrs[FUNDAMENTAL_SNAPSHOT_ATTR] = dict(
                materialize_history_snapshot(market, stock_no, cutoff=history.attrs[SIGNAL_DATE_ATTR])
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

    def build_screen(self, market: MarketHistoryInput, item: int) -> pd.DataFrame:
        """Only the watch's 61 bars; retain original row count/index for its key.

        This is not a feature warm-up window. Strategies still build all bars.
        """
        history = self.build(market, item, include_evidence=False)
        # Copy AFTER slicing the metadata-free view. Otherwise a short slice
        # can retain the full history's ndarray backing storage under CoW.
        screen = metadata_free_frame(pd.DataFrame(history, copy=False).tail(61))
        screen.attrs = dict(history.attrs)
        screen.attrs[SCREEN_SOURCE_ROWS_ATTR] = len(history)
        return screen

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

"""Bounded stock-local transport input, not another research/model engine."""
from dataclasses import replace

from application.market_history import MarketHistoryInput
from domain.industry_benchmark import sector_code
from domain.numeric_history import metadata_free_frame


PER_STOCK_SEQUENCES = (
    "stock_numbers", "stock_names", "dates", "opens", "highs", "lows", "closes",
    "volumes", "trade_counts", "trade_values", "bids", "asks", "pe_ratios", "taiex_closes",
)
PER_STOCK_MAPPINGS = (
    "annual_fundamentals", "fundamental_snapshots", "fundamental_statuses",
    "fundamental_status_details", "signal_dates", "trading_restrictions",
    "les_v6_research_context", "entry_les_by_stock", "entry_dates_by_stock", "entry_decisions_by_stock",
)


def isolate_stock_market(market: MarketHistoryInput, item: int) -> MarketHistoryInput:
    """Keep global market context, but never send another issuer's financials.

    Source values are not modified. The process boundary copies this one packet;
    neither the full universe nor a live SQLite/HTTP/PAE object is transported.
    The watch itself was computed against the original whole universe.
    """
    if not 0 <= item < market.stock_count:
        raise IndexError("stock history index is out of range")
    symbol = market.stock_numbers[item]
    updates = {name: (getattr(market, name)[item],) for name in PER_STOCK_SEQUENCES}
    updates.update({name: {symbol: values[symbol]} if symbol in (values := getattr(market, name)) else {}
                    for name in PER_STOCK_MAPPINGS})
    if symbol in market.annual_fundamentals:
        # The native EvidenceContext already ignores annual DataFrame attrs;
        # don't transport unused metadata or its full financial graph.
        updates["annual_fundamentals"] = {symbol: metadata_free_frame(market.annual_fundamentals[symbol])}
    sector = sector_code(market.fundamental_snapshots.get(symbol, {}).get("sector") or market.default_sector)
    updates["industry_benchmarks"] = ({sector: market.industry_benchmarks[sector]}
                                      if sector in market.industry_benchmarks else {})
    # Do not recalculate the batch's completed session from just this stock.
    updates["fallback_date"] = market.fallback_date or max(
        (str(values[-1]) for values in market.dates if values), default="")
    return replace(market, stock_count=1, **updates)

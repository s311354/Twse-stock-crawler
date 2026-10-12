"""Non-trading leadership observations. Never supplies an entry decision or size."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from domain.security_identity import resolve_security_identity
from domain.trading_restrictions import restriction_check
from domain.strategies.low_entry_score_v4 import PRICE_SCALE_ATTR
from domain.strategies.low_entry_v6_quality import number
from domain.numeric_history import metadata_free_frame


@dataclass(frozen=True)
class LeadershipWatchConfig:
    """Research defaults, not calibrated return forecasts or relaxed BUY gates."""

    version: str = "leadership-watch-1"
    correction_drawdown: float = -.05
    minimum_rs20: float = .03
    maximum_base_range10: float = .08
    maximum_base_return10: float = .03
    maximum_stock_drawdown20: float = .05
    breakout_volume_ratio: float = 1.1
    minimum_amount_ma20: float = 30_000_000.
    minimum_peers: int = 3


def _observation(history, session, cutoff, config):
    identity = history.attrs.get("les_v6_daily_identity", {})
    symbol = str(identity.get("symbol", ""))
    result = {
        "version": config.version, "parameters": asdict(config),
        "symbol": symbol, "name": identity.get("name", ""),
        "analysis_session_date": session, "observed_at": cutoff.isoformat(),
        "market_data_date": None, "status": "UNAVAILABLE", "patterns": [],
        "priority_eligible": False, "reasons": [], "metrics": {},
        "peer_scope": "SELECTED_UNIVERSE_EXCLUDING_SELF", "peer_count": 0,
        "sector": str(identity.get("sector", "UNKNOWN")), "is_entry_signal": False,
    }

    def fail(reason):
        result["reasons"] = [reason]
        return result, None

    if not symbol or not resolve_security_identity(symbol, str(identity.get("name", ""))).low_entry_model_supported:
        return fail("UNSUPPORTED_SECURITY")
    required = {"Date", "Open", "High", "Low", "Close", "Volume", "Amount", "TAIEX Close"}
    if len(history) < 61 or not required.issubset(history):
        return fail("INCOMPLETE_WATCH_HISTORY")
    data = metadata_free_frame(history).tail(61)
    dates = pd.to_datetime(data["Date"], errors="coerce")
    if dates.isna().any() or dates.dt.tz is not None or dates.duplicated().any() or not dates.is_monotonic_increasing or not dates.eq(dates.dt.normalize()).all():
        return fail("INVALID_WATCH_DATES")
    result["market_data_date"] = dates.iloc[-1].date().isoformat()
    if result["market_data_date"] != session:
        return fail("STALE_MARKET_DATA")
    if ((dates.dt.tz_localize("Asia/Taipei") + pd.Timedelta(hours=13, minutes=30)) > cutoff).any() or ("Complete" in data and not data["Complete"].eq(True).all()):
        return fail("INCOMPLETE_DAILY_BAR")
    scale = number(history.attrs.get(PRICE_SCALE_ATTR, 1))
    if scale is None or scale <= 0:
        return fail("INVALID_PRICE_UNIT")
    for field in required - {"Date"}:
        data[field] = pd.to_numeric(data[field], errors="coerce")
    for field in ("Open", "High", "Low", "Close"):
        data[field] /= scale
    o, h, l, c, v, amount, market = (data[key] for key in ("Open", "High", "Low", "Close", "Volume", "Amount", "TAIEX Close"))
    if not np.isfinite(data[list(required - {"Date"})]).all().all() or (data[list(required - {"Date"})] <= 0).any().any():
        return fail("MISSING_OR_INACTIVE_MARKET_DATA")
    # Compare only prices: Series attrs may contain financial DataFrames whose
    # equality is ambiguous when pandas.concat tries to reconcile metadata.
    open_values, close_values = o.to_numpy(), c.to_numpy()
    if (h.to_numpy() < np.maximum(open_values, close_values)).any() or (l.to_numpy() > np.minimum(open_values, close_values)).any():
        return fail("INVALID_OHLC")
    restriction = restriction_check(history.attrs.get("trading_restriction", {}),
                                    market_data_date=session, decision_time=cutoff,
                                    market=identity.get("market", "TWSE"),
                                    allow_historical=bool(history.attrs.get("replay_as_of")),
                                    symbol=str(identity.get("symbol", "")))
    result["restriction_check"] = restriction
    if restriction["status"] == "UNKNOWN":
        return fail("TRADING_RESTRICTION_UNKNOWN")
    if restriction["restricted"]:
        return fail("TRADING_RESTRICTION")
    if amount.tail(20).mean() < config.minimum_amount_ma20:
        return fail("ILLIQUID_WATCH_HISTORY")

    stock_r20 = c.iloc[-1] / c.iloc[-21] - 1
    rs20 = stock_r20 - (market.iloc[-1] / market.iloc[-21] - 1)
    rs_slope = np.log((c.iloc[-1] / market.iloc[-1]) / (c.iloc[-6] / market.iloc[-6])) / 5
    # A 60-session peak is visible only after 60 observations; the trailing
    # 20-session correction uses a causal expanding peak within this 61-bar slice.
    correction = (market / market.cummax() - 1).tail(20).min()
    stock_dd = c.iloc[-1] / c.tail(20).max() - 1
    base_range = h.tail(10).max() / l.tail(10).min() - 1
    base_return = c.iloc[-1] / c.iloc[-11] - 1
    hh20, hh60 = h.iloc[-21:-1].max(), h.iloc[:-1].max()
    clv = (2 * c.iloc[-1] - h.iloc[-1] - l.iloc[-1]) / (h.iloc[-1] - l.iloc[-1]) if h.iloc[-1] > l.iloc[-1] else 0.
    volume_ratio = v.iloc[-1] / v.iloc[-21:-1].mean()
    support = l.iloc[-11:-1].min()
    result["metrics"] = {key: float(value) for key, value in dict(
        close=c.iloc[-1], stock_return20=stock_r20, rs20=rs20, rs_slope5=rs_slope,
        recent_market_drawdown=correction, stock_drawdown20=stock_dd,
        base_range10=base_range, base_return10=base_return, prior_high20=hh20,
        prior_high60=hh60, prior_base_low10=support, clv=clv,
        volume_ratio=volume_ratio, amount_ma20=amount.tail(20).mean(),
    ).items()}
    result["status"] = "NONE"
    strong = correction <= config.correction_drawdown and rs20 >= config.minimum_rs20 and rs_slope > 0
    if strong and base_range <= config.maximum_base_range10 and abs(base_return) <= config.maximum_base_return10 and stock_dd >= -config.maximum_stock_drawdown20 and l.iloc[-1] >= support:
        result["patterns"].append("RESILIENT_BASE")
    if strong and c.iloc[-1] > hh20 and clv >= .5 and volume_ratio >= config.breakout_volume_ratio:
        result["patterns"].append("LEADING_BREAKOUT_60" if c.iloc[-1] > hh60 else "LEADING_BREAKOUT_20")
    if result["patterns"]:
        result["status"] = "LEADERSHIP_WATCH"
    # Full date signature prevents peer comparisons across missing sessions.
    return result, tuple(dates.dt.strftime("%Y-%m-%d"))


def screen_leadership(histories, *, session_date: str, as_of: object, config=None) -> list[dict]:
    """PIT single-universe screening; peer median is not an official sector index."""
    config = config or LeadershipWatchConfig()
    cutoff = pd.to_datetime(as_of, utc=True, errors="coerce")
    parsed_session = pd.to_datetime(session_date, errors="coerce")
    if pd.isna(cutoff) or pd.isna(parsed_session):
        return [{"symbol": str(h.attrs.get("les_v6_daily_identity", {}).get("symbol", "")),
                 "version": config.version, "status": "UNAVAILABLE", "patterns": [],
                 "priority_eligible": False, "is_entry_signal": False,
                 "reasons": ["INVALID_WATCH_CUTOFF"], "metrics": {}} for h in histories]
    session = parsed_session.date().isoformat()
    observations = [_observation(history, session, cutoff, config) for history in histories]
    for item, dates in observations:
        if dates is None:
            continue
        peers = {other["symbol"]: other["metrics"]["stock_return20"] for other, other_dates in observations
                 if other_dates == dates and other["sector"] == item["sector"]
                 and other["symbol"] != item["symbol"] and other["sector"] not in {"", "UNKNOWN"}}
        item["peer_count"] = len(peers)
        item["peer_symbols"] = sorted(peers)
        if len(peers) < config.minimum_peers:
            item["reasons"].append("PEER_COMPARISON_INSUFFICIENT")
            continue
        peer_return = float(np.median(list(peers.values())))
        peer_rs = item["metrics"]["stock_return20"] - peer_return
        item["metrics"].update(peer_median_return20=peer_return, peer_excess_return20=peer_rs)
        if peer_rs < 0:
            item["reasons"].append("BELOW_PEER_MEDIAN")
        item["priority_eligible"] = item["status"] == "LEADERSHIP_WATCH" and peer_rs >= 0
    return [item for item, _ in observations]

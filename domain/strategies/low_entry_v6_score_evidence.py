"""Observable LES points, separate from admission scores and entry permission.

Dependencies follow the ordered conditions in score_setups. Unknown conditions
leave both earned points and available weight; known false conditions retain
their weight. Coverage is evidence coverage, not a probability of success.
"""
from __future__ import annotations

from domain.strategies.low_entry_v6_quality import number


EXECUTION = (("@stop",), ("@reward",), ("spread",), ("amount_ma20",))
DEPENDENCIES = {
    "REVERSAL": {
        "price_location": (("new_lows3", "drawdown60"), ("new_lows3", "position60")),
        "stabilization": (("no_new_low2",), ("higher_low",), ("close", "ema5"),
                          ("ema5_slope",), ("break_previous_high",), ("atr_stable",)),
        "demand": (("clv",), ("cmf_improving",), ("obv_slope5",),
                   ("up_volume_ratio",), ("close", "vwap20")),
        "momentum": (("rsi_cross40",), ("macd_improving",), ("macd_strict", "macd_signal_strict")),
        "relative_strength": (("rs_improving",), ("rs_slope5",), ("resilience_improving",)),
        "execution": EXECUTION,
    },
    "TREND_PULLBACK": {
        "primary_trend": (("ema10", "ema20"), ("ema20_slope",), ("ema20_extension",),
                          ("higher_low", "support_intact"), ("rs20",)),
        "pullback_quality": (("drawdown20",), ("close", "ema10", "ema20", "atr14"),
                             ("close", "vwap20", "atr14"), ("down_volume_contracts",),
                             ("atr_stable",), ("support_intact",), ("new_lows3",)),
        "relative_strength": (("rs20",), ("rs60",), ("rs_slope5",), ("resilience",)),
        "demand": (("reclaim10", "reclaim20"), ("clv",), ("cmf_improving",),
                   ("obv_slope5",), ("up_volume_ratio",)),
        "momentum": (("rsi_pullback",), ("macd_improving",), ("break_previous_high",)),
        "execution": EXECUTION,
    },
    "LEADERSHIP_CONTINUATION": {
        "relative_strength": (("rs20",), ("rs60",), ("rs_near_high",), ("resilience",)),
        "trend": (("ema10", "ema20"), ("ema20_slope",), ("close", "ema20"),
                  ("higher_high", "higher_low", "support_intact")),
        "demand": (("clv",), ("volume_ratio",), ("amount_ratio",), ("cmf20",), ("obv_slope5",)),
        "extension": (("ema20_extension",), ("breakout_extension",)),
        "execution": (("@stop",), ("spread",), ("amount_ma20",), ("restricted",)),
    },
}


def valid_price_anchor(f: dict) -> bool:
    values = [number(f.get(key)) for key in ("open", "high", "low", "close", "volume")]
    if any(value is None for value in values):
        return False
    opening, high, low, close, volume = values
    return 0 < low <= min(opening, close) <= max(opening, close) <= high and volume > 0


def score_evidence(name: str, groups: dict, f: dict, plan: dict) -> dict:
    """Quantify the same scored tests, without filling or rescaling entry gates."""
    masks = f.get("score_availability", {})
    anchor = valid_price_anchor(f)

    def known(key: str) -> bool:
        if key == "@stop":
            # A known but invalid stop is a failed test, not a missing test.
            return (number(plan.get("entry_price")) is not None
                    and number(plan.get("stop_price")) is not None
                    and known("atr14") and number(f.get("atr14")) > 0)
        if key == "@reward":
            return known("@stop") and all(known(k) for k in ("swing_high", "hh20", "hh60"))
        if key in masks:
            return str(masks[key]) == "True"
        # Older snapshots have no masks. Do not reuse stale RS booleans when
        # those snapshots explicitly report an unavailable benchmark.
        if key.startswith(("rs", "resilience")) and key not in {"rsi_cross40", "rsi_pullback", "rsi14"}:
            if str(f.get("market_ready")) != "True":
                return False
        value = f.get(key)
        return str(value) in {"True", "False"} or number(value) is not None

    components, missing = {}, set()
    dependencies = DEPENDENCIES[name]
    if groups.keys() != dependencies.keys():
        raise ValueError("LES_SCORE_EVIDENCE_GROUP_MISMATCH")
    for group, tests in groups.items():
        if len(tests) != len(dependencies[group]):
            raise ValueError("LES_SCORE_EVIDENCE_CONDITION_MISMATCH")
        earned = available = total = 0.
        for (condition, weight), fields in zip(tests, dependencies[group]):
            total += weight
            absent = [key for key in fields if not known(key)]
            if anchor and not absent and condition is not None:
                earned += float(condition) * weight
                available += weight
            else:
                missing.update(absent or (["valid_ohlcv"] if not anchor else fields))
        components[group] = {"earned_points": earned, "available_weight": available, "total_weight": total}
    earned = sum(item["earned_points"] for item in components.values())
    available = sum(item["available_weight"] for item in components.values())
    total = sum(item["total_weight"] for item in components.values())
    return {
        "estimated_score": 100 * earned / available if available else None,
        "earned_points": earned, "available_weight": available, "total_weight": total,
        "coverage_percent": 100 * available / total,
        "status": "COMPLETE" if available == total else "PARTIAL" if available else "UNAVAILABLE",
        "components": components, "missing_fields": sorted(missing),
    }

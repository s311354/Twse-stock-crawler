"""Single condition ledger for formal LES scores and observable estimates.

Dependencies follow the ordered conditions in score_setups. Unknown conditions
leave both earned points and available weight; known false conditions retain
their weight. Coverage is evidence coverage, not a probability of success.
"""
from __future__ import annotations

from domain.strategies.low_entry_v6_quality import number
from domain.strategies.low_entry_v6_correlation import excluded_conditions


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


def score_evidence(name: str, groups: dict, f: dict, plan: dict, *,
                   execution_dependencies: tuple | None = None,
                   setup_dependencies: dict | None = None,
                   excluded_channels: frozenset[str] = frozenset()) -> dict:
    """Score once: fixed-budget admission score and available-budget estimate.

    Missing conditions earn no points. Only the estimate divides by available
    weight; it is never an admission score or entry permission.
    """
    masks = f.get("score_availability", {})
    anchor = valid_price_anchor(f)
    exclusions = excluded_conditions(name, excluded_channels)

    def known(key: str) -> bool:
        if key == "@stop":
            # A known but invalid stop is a failed test, not a missing test.
            return (number(plan.get("entry_price")) is not None
                    and number(plan.get("stop_price")) is not None
                    and known("atr14") and number(f.get("atr14")) > 0)
        if key == "@reward":
            return known("@stop") and all(known(k) for k in ("swing_high", "hh20", "hh60"))
        if key == "@net_reward":
            target = number(plan.get("target_price"))
            return (known("@stop") and target is not None
                    and any(known(k) and number(f.get(k)) == target for k in ("swing_high", "hh20", "hh60")))
        # Older snapshots have no masks. Do not reuse stale RS booleans when
        # those snapshots explicitly report an unavailable benchmark.
        if key.startswith(("rs", "resilience")) and key not in {"rsi_cross40", "rsi_pullback", "rsi14"}:
            if str(f.get("market_ready")) != "True":
                return False
        if key in masks:
            return str(masks[key]) == "True"
        value = f.get(key)
        return str(value) in {"True", "False"} or number(value) is not None

    components, missing, conditions = {}, set(), []
    dependencies = dict(setup_dependencies if setup_dependencies is not None else DEPENDENCIES[name])
    if execution_dependencies is not None:
        dependencies["execution"] = execution_dependencies
    if groups.keys() != dependencies.keys():
        raise ValueError("LES_SCORE_EVIDENCE_GROUP_MISMATCH")
    for group, tests in groups.items():
        if len(tests) != len(dependencies[group]):
            raise ValueError("LES_SCORE_EVIDENCE_CONDITION_MISMATCH")
        earned = available = total = 0.
        for index, ((condition, weight), fields) in enumerate(zip(tests, dependencies[group])):
            total += weight
            absent = [key for key in fields if not known(key)]
            observed = anchor and not absent and condition is not None
            original = float(condition) if observed else None
            excluded = f"{group}:{index}" in exclusions
            if excluded and observed:
                condition = 0.
            points = float(condition) * weight if observed else 0.
            if observed:
                earned += points
                available += weight
            else:
                missing.update(absent or (["valid_ohlcv"] if not anchor else fields))
            conditions.append({"id": f"{group}:{index}", "group": group,
                               "dependencies": list(fields), "weight": weight,
                               "observed": observed, "fraction": float(condition) if observed else None,
                               "earned_points": points})
            if excluded:
                conditions[-1].update(research_excluded=True, original_fraction=original)
        components[group] = {"earned_points": earned, "available_weight": available, "total_weight": total}
    earned = sum(item["earned_points"] for item in components.values())
    available = sum(item["available_weight"] for item in components.values())
    total = sum(item["total_weight"] for item in components.values())
    return {
        "policy_version": "fixed-budget-condition-ledger/v1",
        "formal_score": 100 * earned / total if available else None,
        "estimated_score": 100 * earned / available if available else None,
        "earned_points": earned, "available_weight": available, "total_weight": total,
        "coverage_percent": 100 * available / total,
        "status": "COMPLETE" if available == total else "PARTIAL" if available else "UNAVAILABLE",
        "components": components, "conditions": conditions, "missing_fields": sorted(missing),
    }

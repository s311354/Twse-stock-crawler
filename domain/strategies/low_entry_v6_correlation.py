"""Versioned, conservative source-family pooling, not learned correlations.

The additive 100-point ledger stays intact. Repeated confirmations are removed
as explicit penalties; no freed budget is redistributed or normalized upwards.
"""
from __future__ import annotations

from math import fsum, isclose
from domain.strategies.low_entry_v6_quality import number

POLICY = "source-family-max-pooling/v1"
# EMA state and slope describe a trend axis together. MACD conditions describe
# one oscillator; reclaim/breakout conditions describe one price confirmation.
# Proximity/support/volatility conditions remain outside these pools.
CHANNELS = {
    "REVERSAL": {
        "EMA": ("stabilization:2", "stabilization:3"),
        "MACD": ("momentum:1", "momentum:2"),
        "BREAKOUT": ("stabilization:4",),
        "CLV": ("demand:0",), "CMF": ("demand:1",), "OBV": ("demand:2",),
    },
    "TREND_PULLBACK": {
        "EMA": ("primary_trend:0", "primary_trend:1"),
        "MACD": ("momentum:1",), "BREAKOUT": ("demand:0", "momentum:2"),
        "CLV": ("demand:1",), "CMF": ("demand:2",), "OBV": ("demand:3",),
    },
    "LEADERSHIP_CONTINUATION": {
        "EMA": ("trend:0", "trend:1", "trend:2"),
        "MACD": (), "BREAKOUT": ("extension:1",),
        "CLV": ("demand:0",), "CMF": ("demand:3",), "OBV": ("demand:4",),
    },
}
POOLS = {"TREND_CONFIRMATION": ("EMA", "MACD", "BREAKOUT"),
         "BUYING_FLOW": ("CLV", "CMF", "OBV")}
VARIANTS = {
    "NO_MACD": ("MACD",), "NO_EMA_CONFIRMATION": ("EMA",),
    "NO_BREAKOUT_CONFIRMATION": ("BREAKOUT",), "NO_CLV": ("CLV",),
    "NO_CMF": ("CMF",), "NO_OBV": ("OBV",),
    "ONLY_CLV_FLOW": ("CMF", "OBV"), "ONLY_CMF_FLOW": ("CLV", "OBV"),
    "ONLY_OBV_FLOW": ("CLV", "CMF"),
}


def excluded_conditions(setup: str, channels: frozenset[str]) -> frozenset[str]:
    if not channels <= CHANNELS[setup].keys():
        raise ValueError("UNKNOWN_SCORE_ABLATION_CHANNEL")
    return frozenset(key for channel in channels for key in CHANNELS[setup][channel])


def correlation_adjustment(setup: str, evidence: dict) -> dict:
    """Earn each source family only once, then each correlated pool only once.

    Unknown/false conditions supply no points. Evidence coverage is unchanged:
    this is a score policy, never a claim that absent evidence was observed.
    """
    conditions = {c["id"]: c for c in evidence["conditions"]}
    if len(conditions) != len(evidence["conditions"]):
        raise ValueError("DUPLICATE_SCORE_CONDITION")
    for condition in conditions.values():
        weight, fraction, earned = (number(condition[key]) for key in ("weight", "fraction", "earned_points"))
        if (weight is None or weight <= 0 or earned is None or earned < 0
                or (condition["observed"] and (fraction is None or not 0 <= fraction <= 1))
                or not isclose(earned, weight * fraction if condition["observed"] else 0., rel_tol=0., abs_tol=1e-12)):
            raise ValueError("INVALID_CORRELATION_CONDITION")
    channels = {}
    for channel, identifiers in CHANNELS[setup].items():
        if any(key not in conditions for key in identifiers):
            raise ValueError("CORRELATION_CONDITION_CONTRACT_MISMATCH")
        prefixes = {"EMA": ("ema",), "MACD": ("macd",),
                    "BREAKOUT": ("reclaim", "break_previous_high", "breakout_extension"),
                    "CLV": ("clv",), "CMF": ("cmf",), "OBV": ("obv",)}[channel]
        if any(not any(field.startswith(prefixes) for field in conditions[key]["dependencies"])
               for key in identifiers):
            raise ValueError("CORRELATION_CONDITION_CONTRACT_MISMATCH")
        points = [conditions[key]["earned_points"] for key in identifiers]
        channels[channel] = dict(condition_ids=list(identifiers),
            additive_points=fsum(points), retained_points=fsum(points) if channel == "EMA" else max(points, default=0.),
            aggregation="SUM_STATE_AND_DIRECTION" if channel == "EMA" else "MAX_SINGLE_CONFIRMATION",
            complete=bool(identifiers) and all(conditions[key]["observed"] for key in identifiers))
    pools, penalties = {}, []
    used = set()
    for pool, names in POOLS.items():
        identifiers = [key for name in names for key in CHANNELS[setup][name]]
        if used.intersection(identifiers):
            raise ValueError("OVERLAPPING_CORRELATION_POOLS")
        used.update(identifiers)
        additive = fsum(channels[name]["additive_points"] for name in names)
        retained = max(channels[name]["retained_points"] for name in names)
        removed = max(0., additive - retained)
        winners = [name for name in names if retained > 0 and channels[name]["retained_points"] == retained]
        pools[pool] = dict(channels=list(names), condition_ids=identifiers,
                          additive_points=additive, retained_points=retained, removed_points=removed,
                          strongest_channels=winners)
        if removed:
            penalties.append(dict(reason="CORRELATED_" + pool + "_OVERLAP", points=removed))
    return dict(policy_version=POLICY, channels=channels, pools=pools, penalties=penalties,
                removed_points=fsum(p["points"] for p in penalties),
                calibration_status="CONSERVATIVE_POLICY_NOT_EMPIRICALLY_CALIBRATED")

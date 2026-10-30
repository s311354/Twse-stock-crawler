"""One context, one fixed seed, one six-factor score and one LES decision."""
from __future__ import annotations

from math import fsum

from domain.les_v7.contracts import CONTEXTS, FACTORS, ReferencePlan, finite, ramp
from domain.les_v7.localization import Forecast, localization_gate


def rule(node: dict, features: dict, depth=0):
    """Bounded declarative rule AST, no eval/callable/source-text execution."""
    if depth > 16 or not isinstance(node, dict): raise ValueError("INVALID_CONTEXT_RULE_AST")
    op = node.get("op")
    if op in {"all", "any"}:
        args = node.get("args", ())
        if not 1 <= len(args) <= 16: raise ValueError("INVALID_RULE_ARITY")
        values = [rule(arg, features, depth + 1) for arg in args]
        if op == "all": return False if False in values else None if None in values else True
        return True if True in values else None if None in values else False
    if op == "not":
        value = rule(node["arg"], features, depth + 1)
        return None if value is None else not value
    if op not in {"gt", "ge", "lt", "le", "eq"}: raise ValueError("UNKNOWN_RULE_OPERATOR")
    field = node["field"]
    if any(name.startswith(("forecast", "score", "p_", "bottom", "weights"))
           for name in (field, node.get("other_field", ""))):
        raise ValueError("CONTEXT_CANNOT_READ_FORECAST_OR_SCORE")
    left = features.get(field)
    right = features.get(node["other_field"]) if "other_field" in node else node["value"]
    if left is None or right is None: return None
    return {"gt": lambda: left > right, "ge": lambda: left >= right,
            "lt": lambda: left < right, "le": lambda: left <= right,
            "eq": lambda: type(left) is type(right) and left == right}[op]()


def select_context(features: dict, rules: dict | None) -> tuple[str, tuple[str, ...]]:
    if rules is None: return "NO_CLEAR_CONTEXT", ("CONTEXT_RULES_NOT_PROVIDED",)
    if set(rules) != set(CONTEXTS): raise ValueError("CONTEXT_RULES_INCOMPLETE")
    def ready(*keys): return all(features.get(k) is not None for k in keys)
    guards = {
        "REVERSAL": ready("return20", "return5", "no_low2", "reclaim") and
            (features["return20"] < 0 or features["return5"] < 0) and
            (features["no_low2"] == True or features["reclaim"] == True),
        "TREND_PULLBACK": ready("prior_slope20", "slope20", "ema10", "ema20", "support_intact", "drawdown20") and
            features["prior_slope20"] > 0 and features["slope20"] > 0 and features["ema10"] > features["ema20"] and
            features["support_intact"] == True and features["drawdown20"] < 0,
        "LEADERSHIP_CONTINUATION": ready("ema10", "ema20", "slope20", "rs20", "rs_slope", "support_intact") and
            features["ema10"] > features["ema20"] and features["slope20"] > 0 and features["rs20"] > 0 and
            features["rs_slope"] >= 0 and features["support_intact"] == True,
    }
    matched = [c for c in CONTEXTS if guards[c] and rule(rules[c], features) is True]
    if "TREND_PULLBACK" in matched and "REVERSAL" not in matched:
        return "TREND_PULLBACK", tuple(matched)
    return (matched[0], tuple(matched)) if len(matched) == 1 else ("NO_CLEAR_CONTEXT", tuple(matched))


def make_plan(context, features, policy, evidence_id, *, bars=None, cutoff=None, setup_variant=None,
              session_evidence=None) -> ReferencePlan | None:
    from domain.les_v7.fr3_plan import is_fr3_reference
    if is_fr3_reference(policy):
        from domain.les_v7.fr3_plan import make_plan as fr3_plan
        return fr3_plan(context, features, policy, evidence_id, bars=bars, cutoff=cutoff,
            setup_variant=setup_variant, session_evidence=session_evidence)[0]
    from domain.les_v7.policy_r13 import PLAN_VERSION as r13_plan_version
    from domain.les_v7.policy_r14 import PLAN_VERSION as r14_plan_version
    from domain.les_v7.policy_r15 import PLAN_VERSION as r15_plan_version
    from domain.les_v7.policy_r16 import PLAN_VERSION as r16_plan_version
    from domain.les_v7.policy_r17 import PLAN_VERSION as r17_plan_version
    from domain.les_v7.policy_r18 import PLAN_VERSION as r18_plan_version
    from domain.les_v7.policy_r19 import PLAN_VERSION as r19_plan_version
    if policy and policy.get("version") == r19_plan_version:
        from domain.les_v7.plan_r19 import make_plan as bounded_plan
        return bounded_plan(context, features, policy, evidence_id, bars=bars, cutoff=cutoff,
            setup_variant=setup_variant, session_evidence=session_evidence)[0]
    if policy and policy.get("version") == r18_plan_version:
        from domain.les_v7.plan_r18 import make_plan as limit_plan
        return limit_plan(context, features, policy, evidence_id, bars=bars, cutoff=cutoff,
            setup_variant=setup_variant, session_evidence=session_evidence)[0]
    if policy and policy.get("version") == r17_plan_version:
        from domain.les_v7.plan_r17 import make_plan as unified_plan
        return unified_plan(context, features, policy, evidence_id, bars=bars, cutoff=cutoff,
            setup_variant=setup_variant, session_evidence=session_evidence)[0]
    if policy and policy.get("version") == r16_plan_version:
        from domain.les_v7.plan_r16 import make_plan as conditional_plan
        return conditional_plan(context, features, policy, evidence_id, bars=bars, cutoff=cutoff,
            setup_variant=setup_variant, session_evidence=session_evidence)[0]
    if policy and policy.get("version") in {"V7_SETUP_STRUCTURE_H20_R10", r13_plan_version, r14_plan_version, r15_plan_version}:
        from domain.les_v7.plan_r10 import make_plan as structure_plan
        return structure_plan(context, features, policy, evidence_id, bars=bars, cutoff=cutoff,
            **(dict(setup_variant=setup_variant) if setup_variant is not None else {}))[0]
    if policy is None or context not in CONTEXTS: return None
    if (policy.get("policy_id") != "REFERENCE_H20" or not policy.get("version")
            or policy.get("entry_rule") != "CUTOFF_CLOSE_NEXT_SESSION_BAND"
            or policy.get("support_rule") != "STRICT_PIVOT_ELSE_PRIOR_LL20"):
        raise ValueError("UNSUPPORTED_OR_INCOMPLETE_REFERENCE_POLICY")
    values = [features.get(k) for k in ("close", "support", "atr")]
    if any(v is None or v <= 0 for v in values): return None
    entry, support, atr = values
    stop = support - .3 * atr
    if not 0 < stop < entry: return None
    shares = policy["reference_shares"]
    if type(shares) is not int or shares <= 0: raise ValueError("REFERENCE_SHARES_REQUIRED")
    band_low, band_high = policy["entry_band_atr"]
    finite(band_low); finite(band_high)
    if band_low > band_high: raise ValueError("INVALID_ENTRY_BAND")
    # Prior resistance, never a made-up +2R target or a score-driven anchor.
    target = features.get("hh20")
    if target is not None and target <= entry: target = None
    if "cost_path_policy" in policy:
        from domain.les_v7.reference_path import estimate_reference_costs
        estimated = estimate_reference_costs(entry, stop, target, shares, policy["cost_path_policy"])
        cost_up, cost_stop = estimated["cost_up_per_share"], estimated["cost_stop_per_share"]
    else:
        cost_up = finite(policy["cost_up_per_share"], "cost_up", 0)
        cost_stop = finite(policy["cost_stop_per_share"], "cost_stop", 0)
    return ReferencePlan("REFERENCE_H20", policy["version"], evidence_id, context, entry, stop, target, atr,
        (entry + band_low * atr, entry + band_high * atr), shares, cost_up, cost_stop)


def triggers(context: str, f: dict, rules: dict | None) -> dict:
    if rules is None:
        return dict(starter=None, confirmed=None, evidence_families=[], reasons=["TRIGGER_RULES_NOT_PROVIDED"])
    def known(*names): return all(f.get(k) is not None for k in names)
    families = []
    if known("clv") and f["clv"] > .5: families.append("CLV")
    for key, family in (("cmf_improved", "CMF"), ("obv_improved", "OBV"), ("rsi_cross40", "RSI"), ("hist_improved", "MACD")):
        if f.get(key) is True: families.append(family)
    if known("rs_slope") and f["rs_slope"] > 0: families.append("RS_LINE")
    count = len(set(families))
    if context == "REVERSAL" and known("no_low2", "slope5", "close", "ema5", "reclaim", "clv", "vr", "previous_high"):
        stable = f["no_low2"] == True and f["slope5"] > 0 and f["close"] > f["ema5"]
        quick = f["reclaim"] == True and f["clv"] > .5 and f["vr"] > 1.2
        starter = (stable or quick) and count >= 2
        confirmed_structure = (True if f.get("higher_low") == True or f["reclaim"] == True else
                               None if f.get("higher_low") is None else False)
        confirmed = (None if confirmed_structure is None else confirmed_structure
                     and f["close"] > f["previous_high"] and f["slope5"] > 0 and count >= 2)
    elif context == "TREND_PULLBACK" and known("close", "ema10", "support_intact", "down_volume_ratio", "previous_high", "rs20_change5"):
        near = rule(rules["near_pullback_support"], f)
        starter = None if near is None else near and f["down_volume_ratio"] < 1 and f["support_intact"] == True and count >= 1
        confirmed = (f["close"] > f["ema10"] and f["close"] > f["previous_high"]
                     and f["support_intact"] == True and count >= 1 and f["rs20_change5"] >= 0)
    elif context == "LEADERSHIP_CONTINUATION" and known("clv", "vr", "breakout", "reclaim", "previous_high", "close", "extension", "breakout_extension"):
        base = rule(rules["tight_base_or_retest"], f)
        starter = None if base is None else base and f["clv"] > .3 and count >= 1
        confirmed = ((f["breakout"] == True or f["reclaim"] == True and f["close"] > f["previous_high"])
                     and f["vr"] >= 1.1 and f["clv"] > .4 and count >= 2)
        extended = f["extension"] > 2 or f["breakout"] == True and f["breakout_extension"] > 1.25
        if extended:
            return dict(starter=False, confirmed=False, evidence_families=families, reasons=["WAIT_RETEST"])
    else:
        return dict(starter=None, confirmed=None, evidence_families=families, reasons=["TRIGGER_FEATURES_UNKNOWN"])
    return dict(starter=bool(starter) if starter is not None else None,
                confirmed=bool(confirmed) if confirmed is not None else None,
                evidence_families=families, reasons=[] if starter is not None and confirmed is not None else ["TRIGGER_FEATURES_UNKNOWN"])


def vetoes(f: dict) -> tuple[str, ...]:
    reasons = []
    def known(*names): return all(f.get(k) is not None for k in names)
    if known("support", "close") and f["close"] < f["support"]: reasons.append("SUPPORT_FAILURE")
    if known("open", "close", "support") and f["open"] < f["support"] and f["close"] < f["support"]: reasons.append("GAP_FAILURE")
    if known("clv", "vr", "close", "previous_close", "support") and f["close"] < f["previous_close"] and f["clv"] < -.5 and f["vr"] > 1.5 and f["close"] < f["support"]:
        reasons.append("DISTRIBUTION_BREAK")
    if known("slope5", "slope20", "atr_expansion", "return5", "atr", "close", "reclaim"):
        if f["slope5"] < 0 and f["slope20"] < 0 and f["atr_expansion"] > 1.2 and f["return5"] < -1.5 * f["atr"] / f["close"] and f["reclaim"] != True:
            reasons.append("ACCELERATING_DECLINE")
    if known("new_lows3", "close", "support", "clv") and f["new_lows3"] >= 2 and f["close"] < f["support"] and f["clv"] < 0:
        reasons.append("PERSISTENT_LOW_FAILURE")
    return tuple(reasons)


def factor_vector(context: str, f: dict, forecast: Forecast, plan: ReferencePlan,
                  *, leadership_rr_min=None, price_discovery=False) -> tuple[tuple | None, dict]:
    """BL_MAPPING_B1. Missing subslots never get renormalized or guessed neutral."""
    def blend(parts): return fsum(w * v for w, v in parts) if all(v is not None for w, v in parts) else None
    def ind(key): return float(f[key]) if f.get(key) is not None else None
    def reverse(v): return 1 - v if v is not None else None
    intact, higher, reclaim = ind("support_intact"), ind("higher_low"), ind("reclaim")
    contraction = reverse(ramp(f.get("atr_expansion"), .8, 1.2))
    if context == "REVERSAL":
        recovered = (1. if higher == 1. or reclaim == 1. else
                     0. if higher is not None and reclaim is not None else None)
        structure = blend(((.4, intact), (.3, recovered), (.3, contraction)))
        trend = blend(((.6, ramp(f.get("slope5"), -.05, .05)), (.4, float(f["close"] > f["ema5"]) if f.get("ema5") is not None else None)))
        rs20 = blend(((.5, ramp(f.get("rs20"), -.1, .05)), (.5, ramp(f.get("rs20_change5"), -.03, .03))))
        momentum = blend(((.5, ramp(f.get("rsi_delta3"), 0., 10.)), (.5, ramp(f.get("hist_delta3_atr"), 0., .05))))
    elif context in {"TREND_PULLBACK", "LEADERSHIP_CONTINUATION"}:
        if context == "TREND_PULLBACK":
            structure = blend(((.4, intact), (.3, higher), (.3, reverse(ramp(f.get("down_volume_ratio"), .6, 1.2)))))
            trend = blend(((.6, ramp(f.get("slope20"), 0., .03)), (.4, float(f["ema10"] > f["ema20"]) if f.get("ema10") is not None and f.get("ema20") is not None else None)))
        else:
            retest = None if f.get("clv") is None or f.get("breakout") is None or reclaim is None else float(reclaim == 1 and f["clv"] > .3 or f["breakout"] == True)
            structure = blend(((.4, intact), (.3, contraction), (.3, retest)))
            trend = blend(((.5, ramp(f.get("slope20"), 0., .03)), (.25, float(f["ema10"] > f["ema20"]) if f.get("ema10") is not None and f.get("ema20") is not None else None),
                           (.25, float(f["close"] > f["ema20"]) if f.get("ema20") is not None else None)))
        rs20 = ramp(f.get("rs20"), -.05, .1)
        momentum = blend(((.5, ramp(f.get("rsi"), 40., 60.)), (.5, ramp(f.get("hist_delta3_atr"), -.02, .05))))
    else:
        return None, dict(reason="NO_GLOBAL_MAPPING_ARTIFACT")
    location = reverse(ramp((plan.entry - forecast.quantiles[1]) / plan.atr, .25, 2.)) if forecast.status == "VALID" else None
    stop = reverse(ramp(plan.stop_atr, .5, 1.5))
    rr_min = 2. if context == "REVERSAL" else 1.8 if context == "TREND_PULLBACK" else leadership_rr_min
    rr = (0. if price_discovery else ramp(plan.net_rr, rr_min, 3.)
          if rr_min is not None and plan.net_rr is not None else None)
    participation = plan.reference_shares * plan.entry / f["adv"] if f.get("adv") is not None and f["adv"] > 0 else None
    demand = blend(((.35, ramp(f.get("clv"), 0., .6)), (.25, ramp(f.get("cmf"), -.05, .1)),
                    (.2, ramp(f.get("obv_change5"), -.3, .3)),
                    (.2, float(f["close"] > f["previous_close"]) * ramp(f["vr"], .8, 1.5) if f.get("previous_close") is not None and f.get("vr") is not None else None)))
    vector = (blend(((.5, structure), (.5, location))), trend,
              blend(((.4, rs20), (.3, ramp(f.get("rs_slope"), -.003, .003)), (.3, ramp(f.get("resilience"), -.05, .05)))),
              demand, blend(((.75, momentum), (.25, forecast.p_soon))),
              blend(((.4, stop), (.3, rr), (.2, reverse(ramp(participation, .002, .01))), (.1, reverse(ramp(f.get("spread"), .001, .005))))))
    return (tuple(vector) if all(v is not None for v in vector) else None,
            dict(zip(FACTORS, vector), reference_participation=participation, rr_net=plan.net_rr))


def score_once(factors, weights, penalties: dict, caps: dict) -> dict:
    if factors is None: return dict(raw_score=None, adjusted_score=None, final_score=None, total_penalty=None, effective_cap=None)
    if len(factors) != 6 or len(weights) != 6: raise ValueError("SIX_FACTORS_AND_WEIGHTS_REQUIRED")
    x = [finite(v, "factor", 0, 1) for v in factors]
    w = [finite(v, "weight", 0, 1) for v in weights]
    if abs(fsum(w) - 1.) > 1e-9: raise ValueError("WEIGHTS_MUST_SUM_ONE_NO_RENORMALIZATION")
    raw = 100 * fsum(a * b for a, b in zip(x, w))
    penalty = fsum(finite(v, "penalty", 0) for v in penalties.values())
    cap = min((finite(v, "cap", 0, 100) for v in caps.values()), default=100.)
    adjusted = max(0., raw - penalty)
    return dict(raw_score=raw, adjusted_score=adjusted, final_score=min(cap, adjusted), total_penalty=penalty, effective_cap=cap)


def decide(*, quality: dict, valuation: str, context: str, forecast: Forecast, final: float | None,
           hard_veto=False, required_valid=True, starter_trigger=False, confirmed_trigger=False,
           starter_gate=False, confirmed_gate=False) -> str:
    if hard_veto or quality["status"] == "REJECT" or valuation == "FAIL": return "NO_BUY"
    if (required_valid is not True or forecast.status in {"NOT_READY", "INVALID"}
            or quality["status"] in {"UNKNOWN", "CONFIG_INCOMPLETE", "DATA_INSUFFICIENT"}
            or valuation != "PASS"):
        return "DATA_INSUFFICIENT"
    if quality["status"] != "PASS" or forecast.status == "OOD" or context == "NO_CLEAR_CONTEXT": return "WATCH_ONLY"
    if final is None: return "DATA_INSUFFICIENT"
    value = finite(final, "final", 0, 100)
    if value < 55: return "AVOID"
    if value < 65: return "WATCH"
    if value >= 80 and quality["quality_score"] >= 70 and confirmed_trigger is True and confirmed_gate is True: return "BUY"
    if value >= 72 and quality["quality_score"] >= 75 and starter_trigger is True and starter_gate is True: return "BUY_SMALL"
    return "ARMED"

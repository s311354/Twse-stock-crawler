"""R16 causal structures and explicit next-open conditional plans.

This module selects anchors before score/RR. It does not certify source
authenticity, grant live authority, move a failed stop, or simulate a fill.
R15 remains the fallback and its arithmetic is reused unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from decimal import Decimal
from math import isfinite

import pandas as pd

from domain.les_v7.contracts import CONTEXTS, ReferencePlan, finite, identity, timestamp
from domain.les_v7.features import select_bars, wilder
from domain.les_v7.reference_path import _legal, _round_price, estimate_reference_costs
from domain.strategies.low_entry_v6_hybrid import ContextSelection, NO_CLEAR_CONTEXT, known_features

PLAN_VERSION = R16_VERSION = "V7_SETUP_STRUCTURE_H20_R16"
A_VARIANT = "EARLY_PULLBACK_FLOOR_R16"
C_VARIANT = "PRICE_DISCOVERY_MEASURED_TARGET_R16"
NEW_VARIANTS = frozenset({A_VARIANT, C_VARIANT})
SCHEMA = "les-v7-r16-structure-selection/v1"
PLAN_SCHEMA = "les-v7-setup-plan-evidence/v2"
ROUTE_SCHEMA = "les-v7-reference-entry-route/v2"
A_ENTRY_BASIS = "STRUCTURAL_CONDITIONAL_BAND"
CLOSE_ENTRY_BASIS = "CUTOFF_CLOSE"
OBSERVED_TARGET = "CONFIRMED_UNBROKEN_OVERHEAD_RESISTANCE"
PROJECTED_TARGET = "MODEL_PROJECTED_TARGET"


def structural_policy():
    return dict(version=PLAN_VERSION, selection_order="ORIGINAL_ANCHORS_ELSE_EXACT_PULLBACK_GAP_ELSE_VERIFIED_PRICE_DISCOVERY_ELSE_R15",
        early_pullback=dict(version=A_VARIANT, prior_windows=[5, 10], width_atr=[3.5, 4.],
            session_coverage="VERIFIED_OFFICIAL_PRIOR20_AND_SIGNAL_EXACT_OPEN_SEQUENCE",
            atr_basis="PRIOR_WILDER14", minimum_consecutive_prior_closes=2,
            confirmation_atr=.5, minimum_volume_ratio=1.2, minimum_amount_ratio=1.2,
            demand_conditions=2, entry_band_atr=[.5, 1.], maximum_total_weight=.005,
            maximum_nav_risk_fraction=.0005),
        price_discovery=dict(version=C_VARIANT, prior_sessions=120, base_sessions=20,
            session_coverage="VERIFIED_OFFICIAL_PRIOR120_AND_SIGNAL_EXACT_OPEN_SEQUENCE",
            width_atr=[1., 4.], maximum_extension_atr=.75, minimum_clv=.6,
            clv_scale="[-1,1]", minimum_volume_ratio=1.5, minimum_amount_ratio=1.5,
            demand_conditions=2, target_multiple=.5, target_method=PROJECTED_TARGET,
            entry_band_atr=[0., .25], maximum_total_weight=.0025,
            maximum_nav_risk_fraction=.00025),
        entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", expiry="NEXT_SESSION_CLOSE",
        stop_buffer_atr=.3, new_routes="BUY_SMALL_ONLY", performance_validated=False)


def policy(reference_policy, *, available_at):
    from domain.les_v7.plan_r10 import policy as previous, R15_VERSION
    from domain.les_v7.policy_r16 import AVAILABLE_AT
    if timestamp(available_at) < timestamp(AVAILABLE_AT):
        raise ValueError("R16_REFERENCE_POLICY_BEFORE_RELEASE")
    result = previous(reference_policy, available_at=available_at, version=R15_VERSION)
    result.update(version=PLAN_VERSION, support_rule="CAUSAL_FIXED_PRIORITY_STRUCTURES_R16",
        conditional_entry="NEXT_OFFICIAL_SESSION_OPEN_IN_STRUCTURAL_BAND_R16")
    result["structural_rules"]["r16"] = structural_policy()
    return result


def _as_r15_policy(reference_policy):
    from domain.les_v7.plan_r10 import R15_VERSION, R15_BAND_POLICY
    result = deepcopy(dict(reference_policy))
    result.update(version=R15_VERSION, support_rule="SETUP_BOUND_CAUSAL_PIVOTS_R10",
        conditional_entry="UNFILLED_DIAGNOSTIC_ONLY")
    result["structural_rules"].pop("r16", None)
    result["structural_rules"]["entry_band_policy"] = R15_BAND_POLICY
    return result


def require_policy(reference_policy, cutoff=None):
    from domain.les_v7.plan_r10 import _require_policy
    from domain.les_v7.policy_r16 import AVAILABLE_AT
    if (reference_policy.get("version") != PLAN_VERSION
            or reference_policy.get("support_rule") != "CAUSAL_FIXED_PRIORITY_STRUCTURES_R16"
            or reference_policy.get("conditional_entry") != "NEXT_OFFICIAL_SESSION_OPEN_IN_STRUCTURAL_BAND_R16"
            or reference_policy.get("structural_rules", {}).get("r16") != structural_policy()):
        raise ValueError("R16_REFERENCE_POLICY_MISMATCH")
    if timestamp(reference_policy["available_at"]) < timestamp(AVAILABLE_AT):
        raise ValueError("R16_REFERENCE_POLICY_BEFORE_RELEASE")
    _require_policy(_as_r15_policy(reference_policy), cutoff)
    return reference_policy


@dataclass(frozen=True)
class R16ReferencePlan(ReferencePlan):
    entry_basis: str = CLOSE_ENTRY_BASIS
    signal_close: float = 0.
    setup_variant: str | None = None
    target_method: str = OBSERVED_TARGET
    structure_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        from domain.les_v7.context_r15 import VARIANT as R15_VARIANT
        if self.policy_version != PLAN_VERSION or finite(self.signal_close) <= 0:
            raise ValueError("R16_PLAN_VERSION_AND_SIGNAL_CLOSE_REQUIRED")
        if (not isinstance(self.structure_id, str) or len(self.structure_id) != 64
                or any(char not in "0123456789abcdef" for char in self.structure_id)):
            raise ValueError("R16_STRUCTURE_ID_REQUIRED")
        if self.setup_variant not in {None, R15_VARIANT, A_VARIANT, C_VARIANT}:
            raise ValueError("R16_SETUP_VARIANT_NOT_RECOGNIZED")
        if self.setup_variant == A_VARIANT:
            if (self.entry_basis != A_ENTRY_BASIS or self.entry != self.band[1]
                    or self.target_method != OBSERVED_TARGET):
                raise ValueError("R16_EARLY_PULLBACK_ENTRY_BASIS_INVALID")
        elif self.setup_variant == C_VARIANT:
            if (self.entry_basis != CLOSE_ENTRY_BASIS or self.entry != self.signal_close
                    or self.band[0] != self.signal_close or self.target_method != PROJECTED_TARGET):
                raise ValueError("R16_PRICE_DISCOVERY_BASIS_INVALID")
        elif (self.entry_basis != CLOSE_ENTRY_BASIS or self.entry != self.signal_close
                or self.band[0] != self.signal_close
                or self.target_method != OBSERVED_TARGET):
            raise ValueError("R16_ORDINARY_CLOSE_BASIS_INVALID")


def plan_from_record(record):
    if record is None:
        return None
    values = dict(record, band=tuple(record["band"]))
    return (R16ReferencePlan if values.get("policy_version") == PLAN_VERSION else ReferencePlan)(**values)


def _plain_plan(plan):
    return ReferencePlan(**{field.name: getattr(plan, field.name) for field in fields(ReferencePlan)
        if field.name != "policy_version"}, policy_version="V7_SETUP_STRUCTURE_H20_R15")


def evaluate_entry(plan, observed_entry, reference_policy, *, starter=False):
    from domain.les_v7.plan_r10 import evaluate_entry as previous
    require_policy(reference_policy)
    if plan is None:
        return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    if not isinstance(plan, R16ReferencePlan):
        raise ValueError("R16_SOURCE_BOUND_PLAN_REQUIRED")
    return previous(_plain_plan(plan), observed_entry, _as_r15_policy(reference_policy), starter=starter)


def validate_band(plan, reference_policy, *, starter=False):
    require_policy(reference_policy)
    if plan is None:
        return dict(state="UNKNOWN", reasons=["REFERENCE_PLAN_MISSING"])
    low = _round_price(Decimal(str(plan.band[0])), "up")
    high = _round_price(Decimal(str(plan.band[1])), "down")
    if low > high or low <= Decimal(str(plan.stop)):
        return dict(state="FAIL", reasons=["NO_VALID_EXECUTABLE_ENTRY_BAND"])
    result = evaluate_entry(plan, float(high), reference_policy, starter=starter)
    result.update(band=list(plan.band), executable_band=[float(low), float(high)],
        evaluation_basis="HIGHEST_PERMITTED_LEGAL_ENTRY_TICK")
    return result


def _num(value):
    if isinstance(value, bool): return None
    try: result = float(value)
    except (TypeError, ValueError, OverflowError): return None
    return result if isfinite(result) else None


def _ready_number(features, name):
    return _num(known_features(features).get(name))


def _rows(bars, cutoff):
    selected = select_bars(bars, cutoff)
    columns = [name for name in ("Date", "Open", "High", "Low", "Close", "Volume", "Amount") if name in selected]
    rows = selected[columns].to_dict("records")
    for row in rows:
        row["Date"] = str(row["Date"])[:10]
        for key in columns[1:]: row[key] = _num(row[key])
    return selected, rows


def _atr(selected):
    previous = selected["Close"].shift(1)
    tr = pd.concat((selected["High"] - selected["Low"], (selected["High"] - previous).abs(),
        (selected["Low"] - previous).abs()), axis=1).max(axis=1)
    series = wilder(tr.where(previous.notna()))
    return (_num(series.iloc[-2]), _num(series.iloc[-1])) if len(series) >= 2 else (None, None)


def _demand(features):
    f = known_features(features)
    number = lambda name, test: test(_num(f[name])) if _num(f.get(name)) is not None else None
    checks = dict(CLV=number("clv", lambda x: x > .4),
        CMF_IMPROVING=f.get("cmf_improving") if isinstance(f.get("cmf_improving"), bool) else None,
        OBV=number("obv_slope5", lambda x: x > 0), UP_VOLUME=number("up_volume_ratio", lambda x: x > 1),
        CLOSE_ABOVE_VWAP=(f["close"] > f["vwap20"] if _num(f.get("close")) is not None and _num(f.get("vwap20")) is not None else None))
    passed = sum(value is True for value in checks.values())
    return (True if passed >= 2 else None if any(value is None for value in checks.values()) else False), checks


def _audit(variant, cutoff):
    return dict(schema=SCHEMA, policy_id=identity(structural_policy()), cutoff=timestamp(cutoff).isoformat(),
        state="UNKNOWN", setup_variant=None, candidate_variant=variant, checks={}, metrics={},
        support=None, bars_id=None, reasons=[],
        source_verification="INHERITED_NATIVE_EVIDENCE_BINDING_NOT_CERTIFIED_HERE")


def _finish(audit, checks, **values):
    reasons = [name for name, value in checks.items() if value is not True]
    state = "UNKNOWN" if any(value is None for value in checks.values()) else "PASS" if not reasons else "FAIL"
    audit.update(checks=checks, reasons=reasons, state=state,
        setup_variant=audit["candidate_variant"] if state == "PASS" else None, **values)
    return audit


def _participation(rows, multiplier):
    prior, last = rows[-21:-1], rows[-1]
    result, metrics = {}, {}
    for column in ("Volume", "Amount"):
        values = [row.get(column) for row in prior]
        mean = sum(values) / 20 if len(values) == 20 and all(v is not None and v >= 0 for v in values) else None
        value = last.get(column)
        ratio = value / mean if value is not None and mean is not None and mean > 0 else None
        result[column.upper() + "_PARTICIPATION"] = ratio >= multiplier if ratio is not None else None
        metrics[column.lower() + "_prior20_mean"] = mean
        metrics[column.lower() + "_ratio"] = ratio
    return result, metrics


def assess_early_pullback(bars, features, *, cutoff, impulse_high, session_evidence=None, market=None):
    audit = _audit(A_VARIANT, cutoff)
    selected, rows = _rows(bars, cutoff)
    if len(rows) < 21:
        audit["reasons"] = ["R16_PRIOR_20_SESSION_HISTORY_MISSING"]
        return audit
    prior_atr, signal_atr = _atr(selected)
    audit["bars_id"] = identity(rows[-124:])
    if prior_atr is None or signal_atr is None or min(prior_atr, signal_atr) <= 0:
        audit["reasons"] = ["R16_PRIOR_ATR_UNAVAILABLE"]
        return audit
    prior5, prior10, last = rows[-6:-1], rows[-11:-1], rows[-1]
    floor = min(row["Low"] for row in prior5)
    first = next(index for index, row in enumerate(prior5) if row["Low"] == floor)
    formed, threshold = prior5[first]["Date"], floor + .5 * prior_atr
    runs, current = [], []
    for row in prior5[first + 1:]:
        if row["Close"] >= threshold:
            current.append(row["Date"])
            if len(current) == 2: runs = current[:]; break
        else: current = []
    width5 = max(row["High"] for row in prior5) - floor
    width10 = max(row["High"] for row in prior10) - min(row["Low"] for row in prior10)
    f = known_features(features)
    ema5, slope = _num(f.get("ema5")), _num(f.get("ema5_slope"))
    demand, conditions = _demand(features)
    participation, metrics = _participation(rows, 1.2)
    from domain.les_v7.session_coverage_r16 import verify_window
    coverage = verify_window(session_evidence, market=market, cutoff=cutoff,
        observed_sessions=[row["Date"] for row in rows[-21:]], prior_sessions=20)
    checks = dict(HISTORY_READY=f.get("history_ready") if isinstance(f.get("history_ready"), bool) else None,
        MARKET_READY=f.get("market_ready") if isinstance(f.get("market_ready"), bool) else None,
        FULL_PRIOR_20=(True if coverage["state"] == "PASS" else False if coverage["state"] == "FAIL" else None),
        FLOOR_AFTER_IMPULSE=(formed > impulse_high["session"] if impulse_high else None),
        PRIOR_5_WIDTH=width5 <= 3.5 * prior_atr, PRIOR_10_WIDTH=width10 <= 4 * prior_atr,
        TWO_CONSECUTIVE_PRIOR_CLOSES=len(runs) == 2, SIGNAL_HOLDS_FLOOR=last["Low"] >= floor,
        CLOSE_ABOVE_FLOOR=last["Close"] > floor,
        CLOSE_AT_OR_ABOVE_EMA5=last["Close"] >= ema5 if ema5 is not None else None,
        POSITIVE_EMA5_SLOPE=slope > 0 if slope is not None else None,
        DEMAND_AT_LEAST_TWO=demand, **participation)
    support = dict(value=floor, session=formed, formation_session=formed,
        confirmed_at_session=runs[-1] if runs else None, confirmation_sessions=runs,
        confirmation_level=threshold, method=A_VARIANT, impulse_high=deepcopy(impulse_high),
        window_start=prior5[0]["Date"], window_end=prior5[-1]["Date"],
        source_rows_sha256=identity(prior5))
    metrics.update(prior_atr=prior_atr, signal_atr=signal_atr, prior5_width=width5,
        prior10_width=width10, signal_close=last["Close"], demand_conditions=conditions,
        ema5=ema5, ema5_slope=slope)
    return _finish(audit, checks, support=support, metrics=metrics, session_coverage=coverage,
        operands_id=identity(dict(rows=rows[-21:], session_coverage_id=identity(coverage),
        prior_atr=prior_atr, signal_atr=signal_atr, impulse_high=impulse_high, features={key: f.get(key) for key in
        ("ema5", "ema5_slope", "clv", "cmf_improving", "obv_slope5", "up_volume_ratio", "close", "vwap20", "history_ready", "market_ready")})))


def assess_price_discovery(bars, features, *, cutoff, support=None, support_method=None,
                           session_evidence=None, market=None):
    from domain.les_v7.context_r15 import assess_short_base, SUPPORT_METHOD
    audit = _audit(C_VARIANT, cutoff)
    selected, rows = _rows(bars, cutoff)
    if len(rows) < 121:
        audit["reasons"] = ["R16_VERIFIED_PRIOR_120_SESSION_HISTORY_MISSING"]
        return audit
    prior_atr, signal_atr = _atr(selected)
    audit["bars_id"] = identity(rows[-124:])
    if prior_atr is None or signal_atr is None or min(prior_atr, signal_atr) <= 0:
        audit["reasons"] = ["R16_PRIOR_ATR_UNAVAILABLE"]
        return audit
    prior20, prior120, last = rows[-21:-1], rows[-121:-1], rows[-1]
    from domain.les_v7.session_coverage_r16 import verify_window
    session_coverage = verify_window(session_evidence, market=market, cutoff=cutoff,
        observed_sessions=[row["Date"] for row in rows[-121:]], prior_sessions=120)
    high20, low20 = max(row["High"] for row in prior20), min(row["Low"] for row in prior20)
    high120, width = max(row["High"] for row in prior120), high20 - low20
    clv = (2 * last["Close"] - last["High"] - last["Low"]) / (last["High"] - last["Low"]) if last["High"] > last["Low"] else 0.
    f = known_features(features)
    demand, conditions = _demand(features)
    participation, metrics = _participation(rows, 1.5)
    required = {name: _num(f.get(name)) for name in ("ema10", "ema20", "ema20_slope", "rs20", "rs_slope5")}
    compare = lambda name: required[name] > 0 if required[name] is not None else None
    support_audit = None
    if support is None:
        support_audit = assess_short_base(bars, features, cutoff=cutoff)
        if support_audit["state"] == "PASS":
            support, support_method = deepcopy(support_audit["support"]), SUPPORT_METHOD
    checks = dict(HISTORY_READY=f.get("history_ready") if isinstance(f.get("history_ready"), bool) else None,
        MARKET_READY=f.get("market_ready") if isinstance(f.get("market_ready"), bool) else None,
        FULL_PRIOR_120=(True if session_coverage["state"] == "PASS" else
            False if session_coverage["state"] == "FAIL" else None),
        ABOVE_PRIOR_120_HIGH=last["Close"] > high120,
        PRIOR_20_WIDTH=1 * prior_atr <= width <= 4 * prior_atr,
        ABOVE_PRIOR_20_HIGH=last["Close"] > high20,
        BREAKOUT_EXTENSION=0 < last["Close"] - high20 <= .75 * prior_atr,
        CLV=clv >= .6, DEMAND_AT_LEAST_TWO=demand,
        EMA_UPTREND=(required["ema10"] > required["ema20"] if required["ema10"] is not None and required["ema20"] is not None else None),
        POSITIVE_EMA20_SLOPE=compare("ema20_slope"), POSITIVE_RS20=compare("rs20"),
        POSITIVE_RS_SLOPE5=compare("rs_slope5"), SUPPORT_READY=(True if support is not None else
            None if support_audit and support_audit["state"] == "UNKNOWN" else False), **participation)
    metrics.update(prior_atr=prior_atr, signal_atr=signal_atr, prior20_high=high20,
        prior20_low=low20, prior120_high=high120, prior20_width=width, clv=clv,
        breakout_extension_atr=(last["Close"] - high20) / prior_atr,
        target=high20 + .5 * width, target_multiple=.5, target_method=PROJECTED_TARGET,
        signal_close=last["Close"], demand_conditions=conditions, **required)
    return _finish(audit, checks, support=deepcopy(support), support_method=support_method,
        support_audit=support_audit, session_coverage=session_coverage, metrics=metrics,
        operands_id=identity(dict(rows=rows[-121:], session_coverage=session_coverage,
            prior_atr=prior_atr, signal_atr=signal_atr, features={key: f.get(key) for key in
            ("ema10", "ema20", "ema20_slope", "rs20", "rs_slope5", "clv", "cmf_improving", "obv_slope5", "up_volume_ratio", "close", "vwap20", "history_ready", "market_ready")})))


def prepare_selection(initial, features, *, bars, cutoff, session_evidence=None, market=None):
    """Fixed causal priority; no score/RR input, no A-to-C rescue."""
    from domain.les_v7.plan_r10 import _pivots, _support, _resistance
    from domain.les_v7.context_r15 import select_context as fallback
    if not isinstance(initial, ContextSelection): raise ValueError("R16_ORIGINAL_CONTEXT_SELECTION_REQUIRED")
    audit = _audit(None, cutoff)
    audit.update(original_selection=initial.record(), priority="ANCHORS_THEN_EXACT_A_THEN_C_THEN_R15")
    if bars is None or bars.empty:
        audit["reasons"] = ["R16_COMPLETED_BARS_MISSING"]
        return initial, audit
    selected, all_rows = _rows(bars, cutoff)
    rows = all_rows[-124:]
    if len(rows) < 5:
        audit["reasons"] = ["R16_CONFIRMED_STRUCTURE_HISTORY_MISSING"]
        return initial, audit
    atr = _num(known_features(features).get("atr14"))
    close = _num(known_features(features).get("close"))
    if atr is None or atr <= 0 or close is None or close != rows[-1]["Close"]:
        audit["reasons"] = ["R16_SOURCE_CLOSE_OR_ATR_MISSING_OR_CONFLICTING"]
        return initial, audit
    lows, highs = _pivots(rows, "Low", low=True), _pivots(rows, "High", low=False)
    support, missing, method = _support(rows, lows, highs, initial.context, atr) if initial.context in CONTEXTS else (None, "NO_SELECTED_SETUP", None)
    overhead = _resistance(rows, highs, close, max(0, len(rows) - 120))
    audit.update(bars_id=identity(rows), original_support=deepcopy(support), original_support_missing=missing,
        original_resistance=deepcopy(overhead))
    if support is not None and overhead is not None:
        audit.update(state="NOT_APPLICABLE", reasons=["ORIGINAL_SOURCE_ANCHORS_PRESERVED"])
        return initial, audit
    if initial.context == "TREND_PULLBACK" and missing == "SAME_PULLBACK_CONFIRMED_LOW_MISSING":
        impulse = [point for point in highs if point["index"] >= max(0, len(rows) - 60)][-1]
        result = assess_early_pullback(bars, features, cutoff=cutoff, impulse_high=impulse,
            session_evidence=session_evidence, market=market)
        result.update(original_selection=initial.record(), original_support_missing=missing,
            original_resistance=deepcopy(overhead), priority=audit["priority"])
        return initial, result
    # An absent target is not enough. Full120 and a strict high breakout must
    # establish a verified no-overhead setting, including non-pivot highs.
    if overhead is None:
        # NO_CLEAR_CONTEXT is not permission to ignore a strict leadership
        # support. Select it first and never replace it after a risk failure.
        if initial.context == NO_CLEAR_CONTEXT:
            support, _, method = _support(rows, lows, highs, "LEADERSHIP_CONTINUATION", atr)
        result = assess_price_discovery(bars, features, cutoff=cutoff, support=support, support_method=method,
            session_evidence=session_evidence, market=market)
        result.update(original_selection=initial.record(), original_support_missing=missing,
            original_resistance=None, priority=audit["priority"])
        if result["state"] == "PASS":
            if initial.context == NO_CLEAR_CONTEXT:
                checks = tuple((name, tuple(result["checks"].items()) if name == "LEADERSHIP_CONTINUATION" else items)
                    for name, items in initial.evidence)
                return ContextSelection("LEADERSHIP_CONTINUATION", checks), result
            return initial, result
        # Unknown source dependencies may not be converted into a normal pass.
        # A known non-breakout can still use the exact existing R15 behavior.
        if result["state"] == "UNKNOWN": return initial, result
    selected_context, old = fallback(initial, features, bars=bars, cutoff=cutoff)
    old = deepcopy(old)
    old.update(r16_fallback=True, r16_selection_audit=audit)
    return selected_context, old


def _native_selection(bars, reference_policy, cutoff, *, session_evidence=None):
    from domain.les_v7.rule_baseline import prepare_baseline, select_baseline_context, RELAXED_BASELINE_POLICY
    prepared = prepare_baseline(select_bars(bars, cutoff), market=reference_policy["cost_path_policy"]["market"],
        cutoff=cutoff, policy_version=RELAXED_BASELINE_POLICY)
    initial = select_baseline_context(prepared["features"], policy_version=RELAXED_BASELINE_POLICY)
    return prepare_selection(initial, prepared["features"], bars=bars, cutoff=cutoff,
        session_evidence=session_evidence, market=reference_policy["cost_path_policy"]["market"])


def make_plan(context, features, reference_policy, evidence_id, *, bars=None, cutoff, setup_variant=None,
              session_evidence=None):
    """Rebuild the selected route from completed source bars, then price it."""
    from domain.les_v7.plan_r10 import make_plan as previous, _pivots, _resistance
    require_policy(reference_policy, cutoff)
    selection, audit = _native_selection(bars, reference_policy, cutoff, session_evidence=session_evidence) if bars is not None and not bars.empty else (None, _audit(None, cutoff))
    variant = audit.get("setup_variant")
    if selection is not None and (selection.context != context or setup_variant is not None and setup_variant != variant):
        raise ValueError("R16_SOURCE_BOUND_CONTEXT_VARIANT_MISMATCH")
    sidecar = dict(schema=PLAN_SCHEMA, policy_id=identity(reference_policy), evidence_id=evidence_id,
        cutoff=timestamp(cutoff).isoformat(), context=context, state="UNKNOWN", reasons=[], anchors=None,
        bars_id=audit.get("bars_id"), setup_variant=variant, short_base_audit=audit,
        conditional_entry=None, source_verification="INHERITED_NATIVE_EVIDENCE_BINDING_NOT_CERTIFIED_HERE")
    if context not in CONTEXTS:
        sidecar["reasons"] = ["NO_SELECTED_SETUP"]
        return None, sidecar
    if bars is None or bars.empty:
        sidecar["reasons"] = ["SOURCE_BOUND_COMPLETED_BARS_MISSING"]
        return None, sidecar
    if audit.get("candidate_variant") == A_VARIANT and audit["state"] != "PASS":
        sidecar["reasons"] = audit["reasons"]
        return None, sidecar
    if audit["state"] == "UNKNOWN":
        sidecar["reasons"] = audit["reasons"]
        return None, sidecar
    if variant not in NEW_VARIANTS:
        old_plan, old_sidecar = previous(context, features, _as_r15_policy(reference_policy), evidence_id,
            bars=bars, cutoff=cutoff, setup_variant=variant)
        old_sidecar.update(schema=PLAN_SCHEMA, policy_id=identity(reference_policy), r16_selection=audit)
        if old_plan is None: return None, old_sidecar
        structure_id = identity(dict(selection=audit, anchors=old_sidecar["anchors"], entry_basis=CLOSE_ENTRY_BASIS))
        values = asdict(old_plan); values.update(policy_version=PLAN_VERSION, entry_basis=CLOSE_ENTRY_BASIS,
            signal_close=old_plan.entry, setup_variant=variant, target_method=OBSERVED_TARGET, structure_id=structure_id)
        plan = R16ReferencePlan(**values)
        old_sidecar.update(plan_id=plan.plan_id, entry_basis=CLOSE_ENTRY_BASIS, signal_close=plan.signal_close,
            target_method=OBSERVED_TARGET, structure_id=structure_id)
        return plan, old_sidecar
    selected, all_rows = _rows(bars, cutoff)
    rows, last = all_rows[-124:], all_rows[-1]
    atr, close = _num(features.get("atr")), _num(features.get("close"))
    if atr is None or close is None or close != last["Close"] or atr != audit["metrics"]["signal_atr"]:
        raise ValueError("R16_PLAN_SOURCE_CLOSE_OR_ATR_MISMATCH")
    support = deepcopy(audit["support"])
    support["index"] = next(index for index, row in enumerate(rows) if row["Date"] == support["session"])
    overhead = _resistance(rows, _pivots(rows, "High", low=False), close, max(0, len(rows) - 120))
    stop = support["value"] - .3 * atr
    if not 0 < stop < close or last["Low"] < support["value"]:
        sidecar["reasons"] = ["SELECTED_STRUCTURE_BROKEN_OR_STOP_INVALID"]
        return None, sidecar
    if any(row["Low"] <= stop for row in rows[support["index"] + 1:]):
        sidecar["reasons"] = ["SELECTED_STRUCTURE_STOP_ALREADY_VIOLATED"]
        return None, sidecar
    if variant == A_VARIANT:
        lower, upper = (support["value"] + multiplier * atr for multiplier in (.5, 1.))
        target, entry_basis, target_method = (overhead["value"] if overhead else None), A_ENTRY_BASIS, OBSERVED_TARGET
    else:
        lower, upper = close, close + .25 * atr
        target, entry_basis, target_method = audit["metrics"]["target"], CLOSE_ENTRY_BASIS, PROJECTED_TARGET
    band = (float(_round_price(Decimal(str(lower)), "up")), float(_round_price(Decimal(str(upper)), "down")))
    entry = band[1] if variant == A_VARIANT else close
    sidecar.update(entry_basis=entry_basis, signal_close=close, target_method=target_method,
        anchors=dict(support=support, support_method=audit.get("support_method", A_VARIANT),
            resistance=deepcopy(overhead), target_projection=(dict(method=PROJECTED_TARGET,
                high20=audit["metrics"]["prior20_high"], low20=audit["metrics"]["prior20_low"],
                multiple=.5, value=target) if variant == C_VARIANT else None),
            stop_buffer_atr=.3, selection_order="SETUP_ANCHORS_BEFORE_COSTS_OR_RR"))
    if band[0] > band[1] or band[0] <= stop:
        sidecar["reasons"] = ["NO_VALID_EXECUTABLE_ENTRY_BAND"]
        return None, sidecar
    if target is not None and target <= entry:
        sidecar["reasons"] = ["FIXED_TARGET_NOT_ABOVE_ENTRY"]
        return None, sidecar
    structure_id = identity(dict(selection=audit, anchors=sidecar["anchors"], entry_basis=entry_basis,
        signal_close=close, band=list(band)))
    shares = reference_policy["reference_shares"]
    costs = estimate_reference_costs(entry, stop, target, shares, reference_policy["cost_path_policy"])
    plan = R16ReferencePlan("REFERENCE_H20", PLAN_VERSION, evidence_id, context, entry, stop, target, atr,
        band, shares, costs["cost_up_per_share"], costs["cost_stop_per_share"],
        entry_basis=entry_basis, signal_close=close, setup_variant=variant,
        target_method=target_method, structure_id=structure_id)
    core, tactical = evaluate_entry(plan, entry, reference_policy), evaluate_entry(plan, entry, reference_policy, starter=True)
    sidecar.update(state="VERIFIED_ANCHORS" if target is not None else "PARTIAL",
        reasons=[] if target is not None else ["EVIDENCE_DERIVED_RESISTANCE_MISSING"],
        plan_id=plan.plan_id, structure_id=structure_id, formal_entry_basis=entry_basis,
        close_evaluation=core, band_evaluation=validate_band(plan, reference_policy),
        starter_band_evaluation=validate_band(plan, reference_policy, starter=True),
        r15_risk=dict(core=core, tactical=tactical, band_route="TACTICAL",
            core_band=validate_band(plan, reference_policy), tactical_band=validate_band(plan, reference_policy, starter=True)),
        conditional_entry=dict(state="UNFILLED", entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND",
            expiry="NEXT_SESSION_CLOSE", band=list(band), entry_basis=entry_basis,
            signal_close=close, live_order_authorized=False))
    return plan, sidecar


def bind_entry_route(plan, reference_policy, *, strategy_decision, risk_route, setup_variant=None):
    require_policy(reference_policy)
    if not isinstance(plan, R16ReferencePlan) or reference_policy.get("reference_shares") != plan.reference_shares:
        raise ValueError("R16_SOURCE_BOUND_PLAN_AND_REFERENCE_SIZE_REQUIRED")
    # Re-run the dataclass invariants as a public-boundary defense, even if an
    # internal caller bypassed frozen assignment via deserialization tricks.
    plan_from_record(asdict(plan))
    if not all(_legal(Decimal(str(value))) for value in (*plan.band, plan.entry)):
        raise ValueError("R16_ENTRY_BAND_LEGAL_TICKS_REQUIRED")
    if setup_variant is not None and setup_variant != plan.setup_variant:
        raise ValueError("R16_ENTRY_ROUTE_VARIANT_MISMATCH")
    if strategy_decision not in {"BUY", "BUY_SMALL"} or risk_route not in {"CORE", "TACTICAL"}:
        raise ValueError("R16_ENTRY_ROUTE_DECISION_OR_ROUTE_INVALID")
    if plan.setup_variant is not None and (strategy_decision != "BUY_SMALL" or risk_route != "TACTICAL"):
        raise ValueError("R16_NEW_VARIANT_REQUIRES_TACTICAL_SMALL")
    if strategy_decision == "BUY" and risk_route != "CORE":
        raise ValueError("R16_BUY_REQUIRES_CORE")
    if (evaluate_entry(plan, plan.entry, reference_policy, starter=risk_route == "TACTICAL")["state"] != "PASS"
            or validate_band(plan, reference_policy, starter=risk_route == "TACTICAL")["state"] != "PASS"):
        raise ValueError("R16_ENTRY_ROUTE_NATIVE_RISK_NOT_PASS")
    result = dict(schema=ROUTE_SCHEMA, plan_id=plan.plan_id, reference_policy_id=identity(reference_policy),
        reference_policy=deepcopy(reference_policy), risk_route=risk_route, strategy_decision=strategy_decision,
        setup_variant=plan.setup_variant, entry_basis=plan.entry_basis, signal_close=plan.signal_close,
        target_method=plan.target_method, structure_id=plan.structure_id,
        entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", expiry="NEXT_SESSION_CLOSE",
        authority="REFERENCE_RESEARCH_ROUTE_NOT_LIVE_ORDER_AUTHORITY")
    return dict(result, route_id=identity(result))

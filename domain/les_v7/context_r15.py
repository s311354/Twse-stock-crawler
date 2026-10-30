"""R15 early-base confirmation, causal and separate from the old R14 rule."""
from __future__ import annotations

from domain.les_v7.contracts import identity
from domain.les_v7.context_r14 import assess_short_base as _previous_assessment, policy as _previous_policy
from domain.strategies.low_entry_v6_hybrid import ContextSelection, NO_CLEAR_CONTEXT


VARIANT = "EARLY_SHORT_BASE_BREAKOUT_R15"
SCHEMA = "les-v7-early-short-base-context/v2"
SUPPORT_METHOD = "FROZEN_PRIOR5_FLOOR_TWO_SUBSEQUENT_COMPLETED_CLOSES_R15"


def policy():
    result = _previous_policy()
    for key in ("touch_zone_atr", "minimum_touch_episodes", "episode_separator"):
        result.pop(key)
    result.update(version=VARIANT, base_width_atr=[3.5, 4.],
        minimum_volume_ratio=1.5, minimum_amount_ratio=1.5,
        support_method=SUPPORT_METHOD, confirmation_close_atr=.5,
        minimum_subsequent_completed_prior_closes=2, confirmation_excludes_signal=True,
        maximum_total_weight=.01, maximum_nav_risk_fraction=.0005)
    return result


context_policy = policy


def assess_short_base(bars, features, *, cutoff):
    # Reuse the identical bar/availability/ATR preparation and unchanged
    # operands. Only declared R15 conditions are replaced; R14 stays immutable.
    audit = _previous_assessment(bars, features, cutoff=cutoff)
    audit.update(schema=SCHEMA, policy_id=identity(policy()), setup_variant=None)
    metrics = audit.get("metrics", {})
    if not metrics or not audit.get("support"):
        audit["reasons"] = [reason.replace("R14_", "R15_") for reason in audit["reasons"]]
        return audit
    from domain.les_v7.features import select_bars
    selected = select_bars(bars, cutoff)
    prior = selected.iloc[-6:-1]
    floor = float(prior["Low"].min())
    first = next(i for i, value in enumerate(prior["Low"]) if float(value) == floor)
    formed = str(prior.iloc[first]["Date"])[:10]
    upper = floor + .5 * metrics["prior_atr"]
    confirmations = [str(row["Date"])[:10] for row in prior.iloc[first + 1:].to_dict("records")
                     if float(row["Close"]) > upper]
    support = dict(value=floor, session=formed, formation_session=formed,
        confirmed_at_session=confirmations[1] if len(confirmations) >= 2 else None,
        method=SUPPORT_METHOD, window_start=str(prior.iloc[0]["Date"])[:10],
        window_end=str(prior.iloc[-1]["Date"])[:10], prior_atr=metrics["prior_atr"],
        confirmation_level=upper, subsequent_close_sessions=confirmations,
        subsequent_close_count=len(confirmations), signal_low=audit["support"]["signal_low"],
        source_rows_sha256=audit["support"]["source_rows_sha256"])
    checks = dict(audit["checks"])
    checks.pop("TWO_SEPARATED_SUPPORT_TESTS")
    checks.update(PRIOR_5_WIDTH=metrics["prior5_width"] <= 3.5 * metrics["prior_atr"],
        BREAKOUT_VOLUME=(metrics["breakout_volume_ratio"] >= 1.5
                         if metrics["breakout_volume_ratio"] is not None else None),
        BREAKOUT_AMOUNT=(metrics["breakout_amount_ratio"] >= 1.5
                         if metrics["breakout_amount_ratio"] is not None else None),
        TWO_SUBSEQUENT_COMPLETED_PRIOR_CLOSES=len(confirmations) >= 2)
    reasons = [key for key, value in checks.items() if value is not True]
    state = "UNKNOWN" if any(value is None for value in checks.values()) else "PASS" if not reasons else "FAIL"
    audit.update(checks=checks, reasons=reasons, state=state, support=support,
        setup_variant=VARIANT if state == "PASS" else None)
    return audit


def select_context(selection, features, *, bars, cutoff):
    if not isinstance(selection, ContextSelection):
        raise ValueError("R15_ORIGINAL_CONTEXT_SELECTION_REQUIRED")
    if selection.context != NO_CLEAR_CONTEXT:
        return selection, dict(schema=SCHEMA, policy_id=identity(policy()), state="NOT_APPLICABLE",
            setup_variant=None, reasons=["ORIGINAL_SETUP_PRESERVED"], original_selection=selection.record())
    audit = assess_short_base(bars, features, cutoff=cutoff)
    audit["original_selection"] = selection.record()
    if audit["state"] != "PASS":
        return selection, audit
    evidence = tuple((name, tuple(audit["checks"].items()) if name == "LEADERSHIP_CONTINUATION" else checks)
                     for name, checks in selection.evidence)
    return ContextSelection("LEADERSHIP_CONTINUATION", evidence), audit

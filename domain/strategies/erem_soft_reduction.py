"""Opt-in, unvalidated soft-reduction research policy; no account/order logic."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import pandas as pd


VERSION = "EREM_SOFT_REDUCE_CONFIRM_COOLDOWN_R1_RESEARCH"
RESEARCH_DECLARED_AT = "2026-10-25T14:40:13Z"
FORMAL_VERSION = "EREM_SOFT_REDUCE_CONFIRM_COOLDOWN_B1"


@dataclass(frozen=True)
class SoftReductionResearchPolicy:
    version: str = VERSION
    confirmation_observations: int = 2
    cooldown_observations: int = 2
    research_only: bool = True

    def __post_init__(self):
        if (self.version != VERSION or type(self.confirmation_observations) is not int
                or self.confirmation_observations != 2 or type(self.cooldown_observations) is not int
                or self.cooldown_observations != 2 or self.research_only is not True):
            raise ValueError("EREM_RESEARCH_POLICY_VERSION_OR_PARAMETERS_MISMATCH")

    def record(self):
        return dict(asdict(self), research_declared_at=RESEARCH_DECLARED_AT,
            historical_cases_before_declaration="RETROSPECTIVE_NOT_OOS_OR_HISTORICAL_DEPLOYMENT",
            score_changed=False, native_exit_delayed=False,
            observation_unit="CONSECUTIVE_VALID_INPUT_ROWS_NOT_INFERRED_OFFICIAL_SESSIONS",
            performance_validated=False, full_cost_return_validated=False)


@dataclass(frozen=True)
class SoftReductionFormalPolicy:
    """User-adopted B; identical R1 arithmetic, not a performance certification."""
    version: str = FORMAL_VERSION
    confirmation_observations: int = 2
    cooldown_observations: int = 2
    research_only: bool = False

    def __post_init__(self):
        if (self.version != FORMAL_VERSION or type(self.confirmation_observations) is not int
                or self.confirmation_observations != 2 or type(self.cooldown_observations) is not int
                or self.cooldown_observations != 2 or self.research_only is not False):
            raise ValueError("EREM_FORMAL_B_POLICY_VERSION_OR_PARAMETERS_MISMATCH")

    def record(self):
        return dict(asdict(self), adopted_on="2026-10-27", adoption_basis="EXPLICIT_USER_SELECTION_B",
            predecessor=VERSION, residual_full_exit=True, score_changed=False, native_exit_delayed=False,
            observation_unit="CONSECUTIVE_VALID_INPUT_ROWS_NOT_INFERRED_OFFICIAL_SESSIONS",
            performance_validated=False, full_cost_return_validated=False,
            comparison_audit_id="e9ed9135ac6e38069f22c91719bbf68c8195bf247c1695882ce2ed4182ea3777")


def apply_soft_reduction_policy(decisions, *, entry_les, holding_days, entry_decision,
                                entry_grace, policy: SoftReductionResearchPolicy):
    """Causal full-prefix transform, scoped to one verified position history.

    Do not share state across symbols/accounts. Missing position/source input,
    grace, a changed entry anchor or a falling holding age reset confirmation.
    The original EXIT is always preserved. Hard stops/financial exits are
    outside EREM and must never be routed through this soft-reduction filter.
    """
    if type(policy) not in (SoftReductionResearchPolicy, SoftReductionFormalPolicy):
        raise ValueError("EREM_RESEARCH_POLICY_REQUIRED")
    series = (decisions, entry_les, holding_days, entry_decision, entry_grace)
    if any(not isinstance(value, pd.Series) or not value.index.equals(decisions.index) for value in series):
        raise ValueError("EREM_RESEARCH_INPUT_INDEX_MISMATCH")
    output, statuses, streaks, cooldowns = [], [], [], []
    warning = cooldown = 0
    previous_age = previous_anchor = previous_kind = None
    for base, les, age, kind, grace in zip(*(value.tolist() for value in series), strict=True):
        valid = (pd.notna(les) and pd.notna(age) and pd.notna(kind)
                 and kind in {"BUY", "BUY_SMALL", "EARLY_BUY"}
                 and type(les) in (int, float) and 0 <= les <= 100
                 and type(age) in (int, float) and 0 <= age < float("inf"))
        if not valid:
            warning = cooldown = 0
            previous_age = previous_anchor = previous_kind = None
        elif previous_age is not None and (age < previous_age or les != previous_anchor or kind != previous_kind):
            warning = cooldown = 0
        if valid:
            previous_age, previous_anchor, previous_kind = age, les, kind
        base = base if isinstance(base, str) else "UNKNOWN"
        reduction = re.fullmatch(r"REDUCE\s+(\d+(?:\.\d+)?)%", base)
        action = base
        if base == "EXIT":
            warning = cooldown = 0
            status = "NATIVE_EXIT_IMMEDIATE"
        elif not valid:
            action = "HOLD_REVIEW" if reduction or base == "UNKNOWN" else base
            status = "NO_VERIFIED_POSITION_CONTEXT"
        elif type(grace) is not bool:
            action, status = "HOLD_REVIEW", "GRACE_INPUT_UNKNOWN"
            warning = cooldown = 0
        elif grace:
            action, status = base, "NATIVE_ENTRY_GRACE"
            warning = cooldown = 0
        elif reduction and not 0 < float(reduction[1]) <= 100:
            action, status = "HOLD_REVIEW", "INVALID_REDUCTION_COMMAND"
            warning = cooldown = 0
        elif not reduction and base not in {"HOLD", "STRONG HOLD", "HOLD_REVIEW"}:
            action, status = "HOLD_REVIEW", "UNKNOWN_NATIVE_COMMAND"
            warning = cooldown = 0
        elif cooldown > 0:
            cooldown -= 1
            warning = 0
            action = "HOLD_REVIEW" if reduction else base
            status = "SOFT_REDUCTION_COOLDOWN"
        elif reduction:
            warning += 1
            if warning >= policy.confirmation_observations:
                cooldown = policy.cooldown_observations
                status = "SOFT_REDUCTION_CONFIRMED"
                warning = 0
            else:
                action, status = "HOLD_REVIEW", "SOFT_REDUCTION_PENDING_CONFIRMATION"
        else:
            warning = 0
            action = base if base in {"HOLD", "STRONG HOLD", "HOLD_REVIEW"} else "HOLD_REVIEW"
            status = "NATIVE_NON_REDUCTION" if action == base else "UNKNOWN_NATIVE_COMMAND"
        output.append(action)
        statuses.append(status)
        streaks.append(warning)
        cooldowns.append(cooldown)
    return pd.DataFrame({"decision": output, "status": statuses,
        "pending_warning_count": streaks, "cooldown_remaining": cooldowns}, index=decisions.index)

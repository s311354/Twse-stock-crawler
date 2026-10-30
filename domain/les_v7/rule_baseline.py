"""Explicit uncalibrated v7 baseline, reusing v6's fixed Hybrid formulas.

This is not a fitted localization model and produces no success probabilities.
The causal H20 close plan remains owned by v7. R4 retains the first ninety v6
factor points and scores execution from that plan's bound costs; earlier policy
versions retain their original formulas. This is not v6 signal-high plan parity.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from functools import lru_cache
import json

import pandas as pd

from domain.features import SharedFeaturePipeline
from domain.les_v7.contracts import FACTORS, canonical, identity, timestamp
from domain.les_v7.reference_path import PATH_SCHEMA, PATH_VERSION, GRID_VERSION, COORDINATE_POLICY
from domain.strategies.low_entry_score_v6 import _clean
from domain.strategies.low_entry_v6_features import build_features
from domain.strategies.low_entry_v6_hybrid import (ContextSelection, NO_CLEAR_CONTEXT, known_features,
    model_spec, select_context, score_factors)
from domain.strategies.low_entry_v6_quality import number
from domain.strategies.low_entry_v6_decision import compose_score
from domain.strategies.low_entry_v6_signals import relaxed_hybrid_config, hard_vetoes, veto_evidence_complete, triggers


LEGACY_BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R1"
LEGACY_DECISION_POLICY = "RULE_BASED_V6_HYBRID_R1"
LEGACY_AVAILABLE_AT = "2026-10-13T14:23:16Z"
R2_BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R2"
R2_DECISION_POLICY = "RULE_BASED_V6_HYBRID_R2"
R2_AVAILABLE_AT = "2026-10-14T00:25:04Z"
R3_BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R3"
R3_DECISION_POLICY = "RULE_BASED_V6_HYBRID_R3"
R3_AVAILABLE_AT = "2026-10-14T00:56:31Z"
R4_BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R4"
R4_DECISION_POLICY = "RULE_BASED_V6_HYBRID_R4"
R4_AVAILABLE_AT = "2026-10-14T03:31:16Z"
BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R5"
DECISION_POLICY = "RULE_BASED_V6_HYBRID_R5"
BASELINE_SCHEMA = "les-v7-rule-baseline/v1"
ENVELOPE_SCHEMA = "LES_V7_RULE_BASELINE_ENVELOPE_1"
AVAILABLE_AT = "2026-10-14T04:05:10Z"
SOURCE_BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R6"
SOURCE_DECISION_POLICY = "RULE_BASED_V6_HYBRID_R6"
SOURCE_AVAILABLE_AT = "2026-10-14T06:38:00Z"
SIGNED_BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R7"
SIGNED_DECISION_POLICY = "RULE_BASED_V6_HYBRID_R7"
from domain.les_v7.signed_audit import AVAILABLE_AT as SIGNED_AVAILABLE_AT
PENALTY_BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R8"
PENALTY_DECISION_POLICY = "RULE_BASED_V6_HYBRID_R8"
# Actual publication time after the initial offline policy/compatibility checks.
PENALTY_AVAILABLE_AT = "2026-10-14T12:01:20Z"
GRADED_BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R9"
GRADED_DECISION_POLICY = "RULE_BASED_V6_HYBRID_R9"
# Frozen to the actual release clock after offline verification.
GRADED_AVAILABLE_AT = "2026-10-14T12:44:41Z"
RECOVERY_BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R10"
RECOVERY_DECISION_POLICY = "RULE_BASED_V6_HYBRID_R10"
from domain.les_v7.valuation_recovery_r10 import AVAILABLE_AT as RECOVERY_AVAILABLE_AT
INDUSTRY_BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R11"
INDUSTRY_DECISION_POLICY = "RULE_BASED_V6_HYBRID_R11"
# Recorded release clock after the initial 727 offline regression checks.
INDUSTRY_AVAILABLE_AT = "2026-10-15T15:12:56Z"
CLOSE_BASELINE_POLICY = "V7_RULE_BASED_BASELINE_R12"
CLOSE_DECISION_POLICY = "RULE_BASED_V6_HYBRID_R12"
# Final release clock is frozen after the initial close-support checks.
CLOSE_AVAILABLE_AT = "2026-10-15T15:51:47Z"
from domain.les_v7.policy_r13 import (BASELINE_VERSION as RELAXED_BASELINE_POLICY,
    DECISION_VERSION as RELAXED_DECISION_POLICY, AVAILABLE_AT as RELAXED_AVAILABLE_AT)
from domain.les_v7.policy_r14 import (BASELINE_VERSION as ADAPTIVE_BASELINE_POLICY,
    DECISION_VERSION as ADAPTIVE_DECISION_POLICY, AVAILABLE_AT as ADAPTIVE_AVAILABLE_AT)
from domain.les_v7.policy_r15 import (BASELINE_VERSION as TACTICAL_BASELINE_POLICY,
    DECISION_VERSION as TACTICAL_DECISION_POLICY, AVAILABLE_AT as TACTICAL_AVAILABLE_AT)
from domain.les_v7.policy_r16 import (BASELINE_VERSION as CONDITIONAL_BASELINE_POLICY,
    DECISION_VERSION as CONDITIONAL_DECISION_POLICY, AVAILABLE_AT as CONDITIONAL_AVAILABLE_AT)
from domain.les_v7.policy_r17 import (BASELINE_VERSION as UNIFIED_BASELINE_POLICY,
    DECISION_VERSION as UNIFIED_DECISION_POLICY, AVAILABLE_AT as UNIFIED_AVAILABLE_AT)
from domain.les_v7.policy_r18 import (BASELINE_VERSION as LIMIT_BASELINE_POLICY,
    DECISION_VERSION as LIMIT_DECISION_POLICY, AVAILABLE_AT as LIMIT_AVAILABLE_AT)
from domain.les_v7.policy_r19 import (BASELINE_VERSION as RISK_BOUNDED_BASELINE_POLICY,
    DECISION_VERSION as RISK_BOUNDED_DECISION_POLICY, AVAILABLE_AT as RISK_BOUNDED_AVAILABLE_AT)
from domain.les_v7.policy_r20 import (BASELINE_VERSION as STABILIZATION_BASELINE_POLICY,
    DECISION_VERSION as STABILIZATION_DECISION_POLICY, AVAILABLE_AT as STABILIZATION_AVAILABLE_AT)
from domain.les_v7.policy_r21 import (BASELINE_VERSION as OBSERVATION_BASELINE_POLICY,
    DECISION_VERSION as OBSERVATION_DECISION_POLICY, AVAILABLE_AT as OBSERVATION_AVAILABLE_AT)
from domain.les_v7.policy_r22 import (BASELINE_VERSION as PROGRESSIVE_BASELINE_POLICY,
    DECISION_VERSION as PROGRESSIVE_DECISION_POLICY, AVAILABLE_AT as PROGRESSIVE_AVAILABLE_AT)
from domain.les_v7.policy_r23 import (BASELINE_VERSION as SCOPED_SOURCE_BASELINE_POLICY,
    DECISION_VERSION as SCOPED_SOURCE_DECISION_POLICY, AVAILABLE_AT as SCOPED_SOURCE_AVAILABLE_AT)
from domain.les_v7.policy_r24 import (BASELINE_VERSION as AUDIT_SOURCE_BASELINE_POLICY,
    DECISION_VERSION as AUDIT_SOURCE_DECISION_POLICY, AVAILABLE_AT as AUDIT_SOURCE_AVAILABLE_AT)
from domain.les_v7.policy_r25 import (BASELINE_VERSION as TURNOVER_BASELINE_POLICY,
    DECISION_VERSION as TURNOVER_DECISION_POLICY, AVAILABLE_AT as TURNOVER_AVAILABLE_AT)
RISK_BOUNDED_POLICIES = frozenset({RISK_BOUNDED_BASELINE_POLICY, STABILIZATION_BASELINE_POLICY,
    OBSERVATION_BASELINE_POLICY, PROGRESSIVE_BASELINE_POLICY, SCOPED_SOURCE_BASELINE_POLICY,
    AUDIT_SOURCE_BASELINE_POLICY, TURNOVER_BASELINE_POLICY})
LIMIT_POLICIES = frozenset({LIMIT_BASELINE_POLICY, *RISK_BOUNDED_POLICIES})
CONDITIONAL_POLICIES = frozenset({CONDITIONAL_BASELINE_POLICY, UNIFIED_BASELINE_POLICY, *LIMIT_POLICIES})
TACTICAL_POLICIES = frozenset({TACTICAL_BASELINE_POLICY, *CONDITIONAL_POLICIES})
CLOSE_POLICIES = frozenset({CLOSE_BASELINE_POLICY, RELAXED_BASELINE_POLICY, ADAPTIVE_BASELINE_POLICY, *TACTICAL_POLICIES})
INDUSTRY_POLICIES = frozenset({INDUSTRY_BASELINE_POLICY, *CLOSE_POLICIES})
RECOVERY_POLICIES = frozenset({RECOVERY_BASELINE_POLICY, *INDUSTRY_POLICIES})
GRADED_POLICIES = frozenset({GRADED_BASELINE_POLICY, *RECOVERY_POLICIES})
CURRENT_SOURCE_BASELINE_POLICY = TURNOVER_BASELINE_POLICY
SOURCE_POLICIES = frozenset({SOURCE_BASELINE_POLICY, SIGNED_BASELINE_POLICY, PENALTY_BASELINE_POLICY, *GRADED_POLICIES})
BOUND_SCORE_POLICIES = frozenset({R4_BASELINE_POLICY, BASELINE_POLICY, *SOURCE_POLICIES})
CONFIG = relaxed_hybrid_config()


def current_source_baseline_version(cutoff):
    """Select the deployment vintage; never borrow a future scoring policy."""
    from domain.les_v7.policy_r25 import DAILY_ACTIVATED_AT as R25_ACTIVATED_AT
    if R25_ACTIVATED_AT is not None and timestamp(cutoff) >= timestamp(R25_ACTIVATED_AT):
        return TURNOVER_BASELINE_POLICY
    from domain.les_v7.policy_r24 import DAILY_ACTIVATED_AT as R24_ACTIVATED_AT
    if R24_ACTIVATED_AT is not None and timestamp(cutoff) >= timestamp(R24_ACTIVATED_AT):
        return AUDIT_SOURCE_BASELINE_POLICY
    from domain.les_v7.policy_r23 import DAILY_ACTIVATED_AT as R23_ACTIVATED_AT
    if R23_ACTIVATED_AT is not None and timestamp(cutoff) >= timestamp(R23_ACTIVATED_AT):
        return SCOPED_SOURCE_BASELINE_POLICY
    from domain.les_v7.policy_r22 import DAILY_ACTIVATED_AT as R22_ACTIVATED_AT
    if R22_ACTIVATED_AT is not None and timestamp(cutoff) >= timestamp(R22_ACTIVATED_AT):
        return PROGRESSIVE_BASELINE_POLICY
    from domain.les_v7.policy_r21 import DAILY_ACTIVATED_AT as R21_ACTIVATED_AT
    if R21_ACTIVATED_AT is not None and timestamp(cutoff) >= timestamp(R21_ACTIVATED_AT):
        return OBSERVATION_BASELINE_POLICY
    from domain.les_v7.policy_r20 import DAILY_ACTIVATED_AT as R20_ACTIVATED_AT
    if R20_ACTIVATED_AT is not None and timestamp(cutoff) >= timestamp(R20_ACTIVATED_AT):
        return STABILIZATION_BASELINE_POLICY
    from domain.les_v7.policy_r19 import DAILY_ACTIVATED_AT as R19_ACTIVATED_AT
    if R19_ACTIVATED_AT is not None and timestamp(cutoff) >= timestamp(R19_ACTIVATED_AT):
        return RISK_BOUNDED_BASELINE_POLICY
    from domain.les_v7.policy_r18 import DAILY_ACTIVATED_AT
    if DAILY_ACTIVATED_AT is not None and timestamp(cutoff) >= timestamp(DAILY_ACTIVATED_AT):
        return LIMIT_BASELINE_POLICY
    if timestamp(cutoff) >= timestamp(UNIFIED_AVAILABLE_AT): return UNIFIED_BASELINE_POLICY
    if timestamp(cutoff) >= timestamp(CONDITIONAL_AVAILABLE_AT): return CONDITIONAL_BASELINE_POLICY
    if timestamp(cutoff) >= timestamp(TACTICAL_AVAILABLE_AT): return TACTICAL_BASELINE_POLICY
    if timestamp(cutoff) >= timestamp(ADAPTIVE_AVAILABLE_AT): return ADAPTIVE_BASELINE_POLICY
    if timestamp(cutoff) >= timestamp(RELAXED_AVAILABLE_AT): return RELAXED_BASELINE_POLICY
    if timestamp(cutoff) >= timestamp(CLOSE_AVAILABLE_AT): return CLOSE_BASELINE_POLICY
    if timestamp(cutoff) >= timestamp(INDUSTRY_AVAILABLE_AT): return INDUSTRY_BASELINE_POLICY
    if timestamp(cutoff) >= timestamp(RECOVERY_AVAILABLE_AT): return RECOVERY_BASELINE_POLICY
    if timestamp(cutoff) >= timestamp(GRADED_AVAILABLE_AT): return GRADED_BASELINE_POLICY
    if timestamp(cutoff) >= timestamp(PENALTY_AVAILABLE_AT): return PENALTY_BASELINE_POLICY
    return SIGNED_BASELINE_POLICY if timestamp(cutoff) >= timestamp(SIGNED_AVAILABLE_AT) else SOURCE_BASELINE_POLICY


def baseline_policy(market="TWSE", *, version=BASELINE_POLICY) -> dict:
    if market not in {"TWSE", "TPEX"}: raise ValueError("BASELINE_MARKET_REQUIRED")
    if version == TURNOVER_BASELINE_POLICY:
        from domain.les_v7.policy_r25 import technical_policy
        from domain.les_v7.rule_score_r25 import score_policy
        result = baseline_policy(market, version=AUDIT_SOURCE_BASELINE_POLICY)
        predecessor = identity(result)
        spec = dict(result["model_spec"], curves=dict(result["model_spec"]["curves"]))
        spec["curves"]["amount_ratio"] = [.5, 1.5]
        spec.update(policy_version="context-aware-unified-score/native-r25",
            threshold_basis="EFFECTIVE_NATIVE_BUY_70_SMALL_55_UNCALIBRATED_POLICY")
        result.update(policy_version=version, available_at=TURNOVER_AVAILABLE_AT,
            mapping_version="V7_R24_NORMAL_TURNOVER_AND_SMALL_THRESHOLD_R25",
            predecessor_policy_id=predecessor, model_spec=spec, score_policy=score_policy(),
            unified_technical_policy=technical_policy(result["unified_technical_policy"]))
        return result
    if version == AUDIT_SOURCE_BASELINE_POLICY:
        from domain.les_v7.quality_policy import r7_quality_policy
        from domain.les_v7.source_policy import source_policy, AUDIT_SOURCE_VERSION
        result = baseline_policy(market, version=SCOPED_SOURCE_BASELINE_POLICY)
        predecessor = identity(result)
        result.update(policy_version=version, available_at=AUDIT_SOURCE_AVAILABLE_AT,
            mapping_version="V7_R23_SCORING_WITH_SOURCE_BOUND_AUDIT_R24",
            predecessor_policy_id=predecessor, quality_policy=r7_quality_policy(),
            source_policy=source_policy(version=AUDIT_SOURCE_VERSION))
        return result
    if version == SCOPED_SOURCE_BASELINE_POLICY:
        from domain.les_v7.source_policy import source_policy, SCOPED_VERSION
        result = baseline_policy(market, version=PROGRESSIVE_BASELINE_POLICY)
        predecessor = identity(result)
        result.update(policy_version=version, available_at=SCOPED_SOURCE_AVAILABLE_AT,
            mapping_version="V7_R22_SCORING_DEPENDENCY_SCOPED_PE_SOURCE_R23",
            predecessor_policy_id=predecessor, source_policy=source_policy(version=SCOPED_VERSION))
        return result
    if not isinstance(version, str) or version not in {LEGACY_BASELINE_POLICY, R2_BASELINE_POLICY, R3_BASELINE_POLICY, *BOUND_SCORE_POLICIES}:
        raise ValueError("V7_BASELINE_POLICY_VERSION_UNKNOWN")
    available = {LEGACY_BASELINE_POLICY: LEGACY_AVAILABLE_AT, R2_BASELINE_POLICY: R2_AVAILABLE_AT,
                 R3_BASELINE_POLICY: R3_AVAILABLE_AT, R4_BASELINE_POLICY: R4_AVAILABLE_AT,
                 BASELINE_POLICY: AVAILABLE_AT, SOURCE_BASELINE_POLICY: SOURCE_AVAILABLE_AT,
                 SIGNED_BASELINE_POLICY: SIGNED_AVAILABLE_AT,
                 PENALTY_BASELINE_POLICY: PENALTY_AVAILABLE_AT,
                 GRADED_BASELINE_POLICY: GRADED_AVAILABLE_AT,
                 RECOVERY_BASELINE_POLICY: RECOVERY_AVAILABLE_AT,
                 INDUSTRY_BASELINE_POLICY: INDUSTRY_AVAILABLE_AT,
                 CLOSE_BASELINE_POLICY: CLOSE_AVAILABLE_AT,
                 RELAXED_BASELINE_POLICY: RELAXED_AVAILABLE_AT,
                 ADAPTIVE_BASELINE_POLICY: ADAPTIVE_AVAILABLE_AT,
                 TACTICAL_BASELINE_POLICY: TACTICAL_AVAILABLE_AT,
                 CONDITIONAL_BASELINE_POLICY: CONDITIONAL_AVAILABLE_AT,
                 LIMIT_BASELINE_POLICY: LIMIT_AVAILABLE_AT,
                 RISK_BOUNDED_BASELINE_POLICY: RISK_BOUNDED_AVAILABLE_AT,
                 STABILIZATION_BASELINE_POLICY: STABILIZATION_AVAILABLE_AT,
                 OBSERVATION_BASELINE_POLICY: OBSERVATION_AVAILABLE_AT,
                 PROGRESSIVE_BASELINE_POLICY: PROGRESSIVE_AVAILABLE_AT,
                 UNIFIED_BASELINE_POLICY: UNIFIED_AVAILABLE_AT}[version]
    cost = dict(version="V6_FIXED_FEE_ASSUMPTION_WITH_V7_LEGAL_GRID_R1", rounding="ROUND_HALF_UP",
        evidence_ids=["LES_V6_FIXED_CONFIG_FEE_ASSUMPTION"], available_at=available,
        currency_quantum=".01", buy_rate=CONFIG.buy_fee_rate, sell_rate=CONFIG.sell_cost_rate,
        minimum_buy_fee=0., minimum_sell_fee=0., effective_start="2026-10-13", effective_end="9999-12-31")
    path = dict(schema=PATH_SCHEMA, version=PATH_VERSION, instrument_type="ORDINARY_STOCK", market=market,
        entry_rule="NEXT_OFFICIAL_SESSION_OPEN_IN_BAND", exit_rule="FIRST_STOP_OR_NET_2R_ELSE_H20_CLOSE",
        exit_activation="NEXT_SESSION_AFTER_FILL", shares_rule="NATIVE_REFERENCE_SHARES", price_grid=GRID_VERSION,
        coordinate_policy=COORDINATE_POLICY, source_evidence_ids=["V7_RULE_BASELINE_DECLARED_REFERENCE_POLICY"],
        available_at=available, slippage_bps=0., cost_policy=cost)
    from domain.les_v7.quality_policy import default_quality_policy
    policy = dict(schema=BASELINE_SCHEMA, policy_version=version, available_at=available,
        mode="RULE_BASED_BASELINE", market=market, config=asdict(CONFIG), model_spec=model_spec(CONFIG),
        quality_policy=default_quality_policy(), mapping_version="V6_HYBRID_FIXED_FACTORS_WITH_V7_H20_R1",
        config_usage="FACTOR_SCORING_AND_CONTEXT_ONLY_NOT_POSITION_POLICY",
        native_risk_limits=dict(maximum_spread=.005, maximum_reference_order_adv=.01, maximum_stop_atr=1.5,
            starter_stop_atr=dict(REVERSAL=1.2, TREND_PULLBACK=1.3, LEADERSHIP_CONTINUATION=1.3),
            minimum_net_rr=dict(REVERSAL=2., TREND_PULLBACK=1.8, LEADERSHIP_CONTINUATION=1.8),
            source_bound_hard_gates_required=True, buy_quality_minimum=70., small_quality_minimum=75.),
        weights_kind="FIXED_RULE_BUDGETS_NOT_FITTED", prediction_status="NOT_APPLICABLE",
        calibration_status="NOT_APPLICABLE", performance_validated=False,
        reference_policy=dict(policy_id="REFERENCE_H20", version="V7_RULE_BASELINE_H20_CLOSE_R1",
            entry_rule="CUTOFF_CLOSE_NEXT_SESSION_BAND", support_rule="STRICT_PIVOT_ELSE_PRIOR_LL20",
            reference_shares=1, entry_band_atr=[0., .25], leadership_rr_min=1.8, cost_path_policy=path),
        plan_basis="V7_H20_CLOSE_PLAN_NOT_V6_SIGNAL_HIGH_PLAN",
        score_rr_basis="V6_FEE_FORMULA_SAME_BOUND_PRICE_OPERANDS",
        hard_risk_rr_basis="V7_BOUND_PLAN_COSTS_AND_LEGAL_GRID")
    if version != LEGACY_BASELINE_POLICY:
        policy.update(mapping_version="V6_HYBRID_FIXED_FACTORS_WITH_V7_H20_R2",
            context_policy=dict(version="causal-structure-context/v7-baseline-r2",
                leadership_rule="VERIFIED_DOWNSIDE_RESILIENCE_OR_MEASURED_RELATIVE_LEADERSHIP",
                relative_leadership=dict(positive=["rs20", "rs60", "rs_slope5"],
                    required_true=["rs_near_high", "support_intact", "no_new_low2", "history_ready", "market_ready"]),
                other_contexts="UNCHANGED_V6_CAUSAL_STRUCTURE_V1", hard_gates="UNCHANGED"))
    if version in {R3_BASELINE_POLICY, *BOUND_SCORE_POLICIES}:
        from domain.les_v7.valuation_price import policy as price_valuation_policy
        policy.update(mapping_version="V6_HYBRID_FIXED_FACTORS_WITH_V7_H20_R3",
                      valuation_policy=price_valuation_policy())
    if version in BOUND_SCORE_POLICIES:
        from domain.les_v7.rule_score_r4 import score_policy
        policy.update(mapping_version="V6_HYBRID_FIXED_FACTORS_WITH_V7_H20_R4",
            score_policy=score_policy(), score_rr_basis="V7_BOUND_PLAN_COSTS_AND_LEGAL_GRID",
            veto_classification="TECHNICAL_VETO_SEPARATE_FROM_GATE_AND_EXECUTION_FAILURE_R1")
        policy["context_policy"].update(version="causal-structure-context/v7-baseline-r4",
            other_contexts="REVERSAL_UNCHANGED_PULLBACK_BOUNDED_ATR_TOLERANCE_R1",
            pullback_rule=dict(ema10_below_ema20_max_atr=.25, close_below_ema20_max_atr=.75,
                relaxed_branch_requires_positive_ema20_slope=True,
                required_unchanged=["HELD_SUPPORT", "HELD_LOW", "PULLBACK"],
                selection_order="UNCHANGED_PRE_SCORE_V6_PRIORITY"))
    if version in {BASELINE_POLICY, *SOURCE_POLICIES}:
        from domain.les_v7.valuation_input_policy import input_policy
        policy.update(mapping_version="V6_HYBRID_FIXED_FACTORS_WITH_V7_H20_R5",
                      valuation_input_policy=input_policy())
    if version in SOURCE_POLICIES:
        from domain.les_v7.quality_policy import r5_quality_policy
        from domain.les_v7.source_policy import source_policy
        policy.update(mapping_version="V6_HYBRID_FIXED_FACTORS_WITH_V7_H20_R6",
                      quality_policy=r5_quality_policy(), source_policy=source_policy())
    if version in {SIGNED_BASELINE_POLICY, PENALTY_BASELINE_POLICY, *GRADED_POLICIES}:
        from domain.les_v7.quality_policy import r6_quality_policy
        from domain.les_v7.source_policy import SIGNED_VERSION
        policy.update(mapping_version="V6_HYBRID_FIXED_FACTORS_WITH_V7_H20_R7",
                      quality_policy=r6_quality_policy(), source_policy=source_policy(version=SIGNED_VERSION),
                      signed_report_policy="SOURCE_BOUND_SIGNED_REPORT_PRECEDENCE_R1")
    if version in {PENALTY_BASELINE_POLICY, *GRADED_POLICIES}:
        from domain.les_v7.rule_score_r8 import adjustment_policy
        policy.update(mapping_version="V6_HYBRID_FIXED_FACTORS_WITH_V7_H20_R8",
                      score_adjustment_policy=adjustment_policy())
    if version in GRADED_POLICIES:
        from domain.les_v7.entry_policy_r9 import policy as entry_policy
        policy.update(mapping_version="V6_HYBRID_FIXED_FACTORS_WITH_V7_H20_R9",
                      entry_admission_policy=entry_policy())
        policy["native_risk_limits"].update(buy_quality_minimum=65., small_quality_minimum=60.)
    if version in RECOVERY_POLICIES:
        from domain.les_v7.valuation_recovery_r10 import policy as recovery_policy
        from domain.les_v7.plan_r10 import policy as structure_policy
        from domain.les_v7.valuation_input_policy import input_policy, RECOVERY_VERSION
        policy.update(mapping_version="V6_HYBRID_FIXED_FACTORS_WITH_V7_STRUCTURE_R10",
            valuation_policy=recovery_policy(),
            valuation_input_policy=input_policy(version=RECOVERY_VERSION),
            reference_policy=structure_policy(policy["reference_policy"], available_at=available),
            plan_basis="SETUP_BOUND_STRUCTURE_NOT_RR_BACKSOLVED_ANCHORS")
    if version in INDUSTRY_POLICIES:
        from domain.les_v7.industry_score import industry_score_policy
        policy.update(mapping_version="V6_HYBRID_SOURCE_BOUND_INDUSTRY_R11",
                      industry_score_policy=industry_score_policy())
    if version in CLOSE_POLICIES:
        from domain.les_v7.features import CLOSE_SUPPORT_FEATURE_VERSION, close_support_policy
        policy.update(mapping_version="V6_HYBRID_CONFIRMED_CLOSE_SUPPORT_R12",
            native_feature_version=CLOSE_SUPPORT_FEATURE_VERSION,
            support_confirmation_policy=dict(close_support_policy(), frozen_model_path="UNCHANGED_R1"))
    if version == RELAXED_BASELINE_POLICY:
        from domain.les_v7.valuation_relaxed_r13 import policy as relaxed_valuation_policy
        from domain.les_v7.valuation_input_policy import RELAXED_VERSION
        from domain.les_v7.entry_policy_r13 import policy as relaxed_admission_policy
        from domain.les_v7.policy_r13 import PLAN_VERSION, risk_limits
        policy.update(mapping_version="V6_HYBRID_GRADED_VALUATION_STRUCTURE_R13",
            valuation_policy=relaxed_valuation_policy(),
            valuation_input_policy=input_policy(version=RELAXED_VERSION),
            entry_admission_policy=relaxed_admission_policy(), native_risk_limits=risk_limits(),
            reference_policy=structure_policy(policy["reference_policy"], available_at=available, version=PLAN_VERSION))
    if version == ADAPTIVE_BASELINE_POLICY:
        from domain.les_v7.valuation_recovery_r14 import policy as adaptive_valuation_policy
        from domain.les_v7.valuation_input_policy import input_policy
        from domain.les_v7.entry_policy_r14 import policy as adaptive_admission_policy
        from domain.les_v7.policy_r14 import INPUT_VERSION, PLAN_VERSION, risk_limits
        from domain.les_v7.context_r14 import context_policy
        policy.update(mapping_version="V6_HYBRID_SHORT_BASE_RECOVERY_COMPONENTS_R14",
            valuation_policy=adaptive_valuation_policy(),
            valuation_input_policy=input_policy(version=INPUT_VERSION),
            entry_admission_policy=adaptive_admission_policy(), native_risk_limits=risk_limits(),
            short_base_policy=context_policy(),
            reference_policy=structure_policy(policy["reference_policy"], available_at=available, version=PLAN_VERSION))
    if version in TACTICAL_POLICIES:
        from domain.les_v7.valuation_recovery_r15 import policy as tactical_valuation_policy
        from domain.les_v7.valuation_input_policy import input_policy
        from domain.les_v7.entry_policy_r15 import policy as tactical_admission_policy
        from domain.les_v7.policy_r15 import INPUT_VERSION, PLAN_VERSION, risk_limits, tactical_risk_limits
        from domain.les_v7.context_r15 import context_policy
        policy.update(mapping_version="V6_HYBRID_CORE_TACTICAL_ROLE_BOUND_R15",
            valuation_policy=tactical_valuation_policy(),
            valuation_input_policy=input_policy(version=INPUT_VERSION),
            entry_admission_policy=tactical_admission_policy(), native_risk_limits=risk_limits(),
            tactical_risk_limits=tactical_risk_limits(), short_base_policy=context_policy(),
            reference_policy=structure_policy(policy["reference_policy"], available_at=available, version=PLAN_VERSION))
        if version == CONDITIONAL_BASELINE_POLICY:
            from domain.les_v7.entry_policy_r16 import policy as conditional_admission_policy
            from domain.les_v7.plan_r16 import policy as conditional_plan_policy
            policy.update(mapping_version="V6_HYBRID_CAUSAL_CONDITIONAL_STRUCTURE_R16",
                entry_admission_policy=conditional_admission_policy(),
                reference_policy=conditional_plan_policy(policy["reference_policy"], available_at=available))
        if version in {UNIFIED_BASELINE_POLICY, *LIMIT_POLICIES}:
            from domain.les_v7.entry_policy_r17 import policy as unified_admission_policy
            from domain.les_v7.plan_r17 import policy as unified_plan_policy
            from domain.les_v7.policy_r17 import technical_policy
            from domain.les_v7.context_r17 import policy as unified_context_policy
            policy.update(mapping_version="V6_HYBRID_UNIFIED_SOFT_TECHNICAL_R17",
                entry_admission_policy=unified_admission_policy(),
                unified_technical_policy=technical_policy(),
                context_policy=unified_context_policy(),
                reference_policy=unified_plan_policy(policy["reference_policy"], available_at=available))
            policy["score_adjustment_policy"] = dict(policy["score_adjustment_policy"],
                version="R17_EXISTING_FACTOR_POINTS_NO_DUPLICATE_TRIGGER_DEDUCTIONS",
                deductions=dict(no_clear_context=0., trigger_not_ready=0., unconfirmed_starter=0.))
        if version in LIMIT_POLICIES:
            from domain.les_v7.entry_policy_r18 import policy as limit_admission_policy
            from domain.les_v7.plan_r18 import policy as limit_plan_policy
            from domain.les_v7.policy_r18 import technical_policy, tactical_risk_limits
            policy.update(mapping_version="V6_HYBRID_SUPPORT_RESISTANCE_CONDITIONAL_LIMIT_R18",
                entry_admission_policy=limit_admission_policy(),
                unified_technical_policy=technical_policy(), tactical_risk_limits=tactical_risk_limits(),
                reference_policy=limit_plan_policy(policy["reference_policy"], available_at=available))
            if version in RISK_BOUNDED_POLICIES:
                from domain.les_v7.plan_r19 import policy as bounded_plan_policy
                from domain.les_v7.policy_r19 import technical_policy as bounded_technical_policy
                policy.update(mapping_version="V6_HYBRID_RISK_BOUNDED_ENTRY_R19",
                    unified_technical_policy=bounded_technical_policy(),
                    reference_policy=bounded_plan_policy(policy["reference_policy"], available_at=available))
            if version in {STABILIZATION_BASELINE_POLICY, OBSERVATION_BASELINE_POLICY, PROGRESSIVE_BASELINE_POLICY}:
                from domain.les_v7.rule_score_r20 import score_policy as stabilization_score_policy
                from domain.les_v7.policy_r20 import technical_policy as stabilization_technical_policy
                policy.update(mapping_version="V6_HYBRID_EARLY_STABILIZATION_SCORE_R20",
                    score_policy=stabilization_score_policy(),
                    unified_technical_policy=stabilization_technical_policy())
            if version in {OBSERVATION_BASELINE_POLICY, PROGRESSIVE_BASELINE_POLICY}:
                from domain.les_v7.rule_score_r21 import score_policy as observation_score_policy
                from domain.les_v7.policy_r21 import technical_policy as observation_technical_policy
                from domain.les_v7.industry_score import HORIZON_VERSION, industry_score_policy
                policy.update(mapping_version="V7_LOW_ZONE_OBSERVATION_AND_HORIZON_SOURCE_R21",
                    score_policy=observation_score_policy(), unified_technical_policy=observation_technical_policy(),
                    industry_score_policy=industry_score_policy(version=HORIZON_VERSION))
            if version == PROGRESSIVE_BASELINE_POLICY:
                from domain.les_v7.rule_score_r22 import score_policy as progressive_score_policy
                from domain.les_v7.policy_r22 import SOURCE_SELECTION_VERSION, technical_policy as progressive_technical_policy
                policy.update(mapping_version="V7_PROGRESSIVE_CONFIRMATION_SOURCE_SEMANTICS_R22",
                    score_policy=progressive_score_policy(), unified_technical_policy=progressive_technical_policy(),
                    source_selection_policy=SOURCE_SELECTION_VERSION)
                policy["reference_policy"]["source_selection_policy"] = SOURCE_SELECTION_VERSION
    return policy


def baseline_decision_policy(policy):
    version = policy.get("policy_version")
    if version == LEGACY_BASELINE_POLICY: return LEGACY_DECISION_POLICY
    if version == R2_BASELINE_POLICY: return R2_DECISION_POLICY
    if version == R3_BASELINE_POLICY: return R3_DECISION_POLICY
    if version == R4_BASELINE_POLICY: return R4_DECISION_POLICY
    if version == BASELINE_POLICY: return DECISION_POLICY
    if version == SOURCE_BASELINE_POLICY: return SOURCE_DECISION_POLICY
    if version == SIGNED_BASELINE_POLICY: return SIGNED_DECISION_POLICY
    if version == PENALTY_BASELINE_POLICY: return PENALTY_DECISION_POLICY
    if version == GRADED_BASELINE_POLICY: return GRADED_DECISION_POLICY
    if version == RECOVERY_BASELINE_POLICY: return RECOVERY_DECISION_POLICY
    if version == INDUSTRY_BASELINE_POLICY: return INDUSTRY_DECISION_POLICY
    if version == CLOSE_BASELINE_POLICY: return CLOSE_DECISION_POLICY
    if version == RELAXED_BASELINE_POLICY: return RELAXED_DECISION_POLICY
    if version == ADAPTIVE_BASELINE_POLICY: return ADAPTIVE_DECISION_POLICY
    if version == TACTICAL_BASELINE_POLICY: return TACTICAL_DECISION_POLICY
    if version == CONDITIONAL_BASELINE_POLICY: return CONDITIONAL_DECISION_POLICY
    if version == UNIFIED_BASELINE_POLICY: return UNIFIED_DECISION_POLICY
    if version == LIMIT_BASELINE_POLICY: return LIMIT_DECISION_POLICY
    if version == RISK_BOUNDED_BASELINE_POLICY: return RISK_BOUNDED_DECISION_POLICY
    if version == STABILIZATION_BASELINE_POLICY: return STABILIZATION_DECISION_POLICY
    if version == OBSERVATION_BASELINE_POLICY: return OBSERVATION_DECISION_POLICY
    if version == PROGRESSIVE_BASELINE_POLICY: return PROGRESSIVE_DECISION_POLICY
    if version == SCOPED_SOURCE_BASELINE_POLICY: return SCOPED_SOURCE_DECISION_POLICY
    if version == AUDIT_SOURCE_BASELINE_POLICY: return AUDIT_SOURCE_DECISION_POLICY
    if version == TURNOVER_BASELINE_POLICY: return TURNOVER_DECISION_POLICY
    raise ValueError("V7_BASELINE_POLICY_VERSION_UNKNOWN")


def require_baseline_policy(policy, *, market, cutoff):
    if not isinstance(policy, Mapping): raise ValueError("V7_BASELINE_POLICY_MISMATCH")
    expected = baseline_policy(market, version=policy.get("policy_version"))
    if identity(policy) != identity(expected): raise ValueError("V7_BASELINE_POLICY_MISMATCH")
    if timestamp(expected["available_at"]) > timestamp(cutoff): raise ValueError("V7_BASELINE_NOT_AVAILABLE_AT_CUTOFF")
    return expected


def select_baseline_context(features, *, policy_version=BASELINE_POLICY):
    """R2 adds relative leadership; R4 adds bounded support-holding pullbacks.

    No rank, score, Quality, quote or plan participates in context selection.
    Missing/stale market operands never supply relative-leadership evidence.
    The shared v6 selector and loaded v7 models retain their original rules.
    """
    if policy_version in {UNIFIED_BASELINE_POLICY, *LIMIT_POLICIES}:
        from domain.les_v7.context_r17 import select_context as unified_context
        return unified_context(features)
    selection = select_context(features)
    if policy_version == LEGACY_BASELINE_POLICY: return selection
    if policy_version not in {R2_BASELINE_POLICY, R3_BASELINE_POLICY, *BOUND_SCORE_POLICIES}: raise ValueError("V7_BASELINE_POLICY_VERSION_UNKNOWN")
    f = known_features(features)
    positive = [number(f.get(key)) for key in ("rs20", "rs60", "rs_slope5")]
    required = [f.get(key) for key in ("rs_near_high", "support_intact", "no_new_low2", "history_ready", "market_ready")]
    relative = (all(value is not None and value > 0 for value in positive)
                and all(value is True for value in required))
    checks = tuple((name, tuple(("RESILIENCE_OR_RELATIVE_LEADERSHIP",
        True if value is True or relative else value) if name == "LEADERSHIP_CONTINUATION" and key == "RESILIENCE"
        else (key, value) for key, value in items)) for name, items in selection.evidence)
    if policy_version in BOUND_SCORE_POLICIES:
        atr, short, medium, slope, close = (number(f.get(key)) for key in
            ("atr14", "ema10", "ema20", "ema20_slope", "close"))
        early_uptrend = (all(value is not None for value in (atr, short, medium, slope))
                         and atr > 0 and slope > 0 and short >= medium - .25 * atr)
        near_trend = (all(value is not None for value in (atr, close, medium, slope))
                      and atr > 0 and slope > 0 and close >= medium - .75 * atr)
        checks = tuple((name, tuple((key, True if value is True or
            (key == "UPTREND" and early_uptrend) or (key == "NEAR_TREND" and near_trend) else value)
            if name == "TREND_PULLBACK" and key in {"UPTREND", "NEAR_TREND"} else (key, value)
            for key, value in items)) for name, items in checks)
    selected = next((name for name, items in checks if all(value is True for _, value in items)), NO_CLEAR_CONTEXT)
    return ContextSelection(selected, checks)


def _selection_record(selection, policy):
    record = selection.record()
    if policy["policy_version"] != LEGACY_BASELINE_POLICY:
        record["policy_version"] = policy["context_policy"]["version"]
    return record


@lru_cache(maxsize=16)
def _features(rows_json):
    data = pd.DataFrame(json.loads(rows_json))
    if data.empty: return canonical({})
    frame = build_features(data, SharedFeaturePipeline().build(data))
    features = frame.iloc[-1].to_dict()
    current = features.get("rsi14")
    previous = frame["rsi14"].iloc[-2] if len(frame) > 1 else None
    features["rsi_change"] = current - previous if pd.notna(current) and previous is not None and pd.notna(previous) else None
    features["score_availability"] = dict(features.get("score_availability", {}))
    features["score_availability"].update(rsi_change=features["rsi_change"] is not None,
        rsi14=pd.notna(current), volume_ma20_strict=features["score_availability"].get("volume_ratio", False))
    features.update(industry_rs20=None, industry_rs60=None, industry_status="UNKNOWN")
    return canonical(_clean(features))


def prepare_baseline(bars, *, market, cutoff, policy_version=BASELINE_POLICY,
                     industry_input=None, session_evidence=None):
    policy = require_baseline_policy(baseline_policy(market, version=policy_version), market=market, cutoff=cutoff)
    rows = bars.astype(object).where(pd.notna(bars), None).to_dict("records")
    for row in rows: row["Date"] = pd.Timestamp(row["Date"]).date().isoformat()
    features = json.loads(_features(canonical(rows)))
    industry = None
    if policy_version in INDUSTRY_POLICIES:
        from domain.les_v7.industry_score import apply_industry_features
        source = industry_input or {}
        features, industry = apply_industry_features(features, bars, source.get("benchmark"),
            symbol=source.get("symbol", ""), market=market, sector=source.get("sector", ""), cutoff=cutoff,
            version=policy["industry_score_policy"]["version"])
    if policy_version in {UNIFIED_BASELINE_POLICY, *LIMIT_POLICIES}:
        from domain.les_v7.context_r17 import price_structure_features
        features = price_structure_features(bars, features, cutoff=cutoff)
    selected = select_baseline_context(features, policy_version=policy_version)
    short_base = None
    if policy_version in {ADAPTIVE_BASELINE_POLICY, *TACTICAL_POLICIES}:
        if policy_version in LIMIT_POLICIES:
            from domain.les_v7.plan_r18 import prepare_selection as select_short_base
        elif policy_version == UNIFIED_BASELINE_POLICY:
            from domain.les_v7.plan_r17 import prepare_selection as select_short_base
        elif policy_version == CONDITIONAL_BASELINE_POLICY:
            from domain.les_v7.plan_r16 import prepare_selection as select_short_base
        elif policy_version == TACTICAL_BASELINE_POLICY:
            from domain.les_v7.context_r15 import select_context as select_short_base
        else:
            from domain.les_v7.context_r14 import select_context as select_short_base
        selected, short_base = select_short_base(selected, features, bars=bars, cutoff=cutoff,
            **(dict(session_evidence=session_evidence, market=market)
               if policy_version in CONDITIONAL_POLICIES else {}),
            **(dict(selection_policy=policy["source_selection_policy"])
               if "source_selection_policy" in policy else {}))
    result = dict(schema=BASELINE_SCHEMA, policy=policy, policy_id=identity(policy), market_rows=rows,
        features=features, selection=_selection_record(selected, policy))
    if policy_version in {UNIFIED_BASELINE_POLICY, *LIMIT_POLICIES}:
        from domain.les_v7.context_r17 import audit as interpretation_audit
        result["selection"]["interpretation_audit"] = interpretation_audit(features)
    if short_base is not None:
        result["selection"].update(setup_variant=short_base.get("setup_variant"), short_base_audit=short_base)
    if policy_version in INDUSTRY_POLICIES:
        result["industry_score"] = industry
    if policy_version in CONDITIONAL_POLICIES:
        result["session_evidence"] = session_evidence
    return result


def baseline_triggers(prepared):
    features = known_features(prepared["features"])
    selected = prepared["selection"]["selected_setup"]
    if prepared.get("policy", {}).get("policy_version") in {UNIFIED_BASELINE_POLICY, *LIMIT_POLICIES}:
        # Strength confirmation is already scored in the fixed six-factor
        # ledger. These flags describe causal setup readiness, not a second
        # volume/EMA/demand admission test or a filled conditional order.
        from domain.les_v7.policy_r18 import SUPPORT_RESISTANCE_LIMIT
        neutral = prepared["selection"].get("setup_variant") == SUPPORT_RESISTANCE_LIMIT
        ready = ((selected != NO_CLEAR_CONTEXT or neutral) and features.get("history_ready") is True
                 and features.get("market_ready") is True)
        return dict(starter=True if ready else None,
            confirmed=(not bool(prepared["selection"].get("setup_variant"))) if ready else None,
            evidence_families=[("R19" if prepared["policy"]["policy_version"] in RISK_BOUNDED_POLICIES else
                "R18" if prepared["policy"]["policy_version"] == LIMIT_BASELINE_POLICY else "R17") + "_CAUSAL_STRUCTURE_ONLY_TECHNICAL_STRENGTH_SCORED_ONCE"],
            reasons=[] if ready else [("R19" if prepared["policy"]["policy_version"] in RISK_BOUNDED_POLICIES else
                "R18" if prepared["policy"]["policy_version"] == LIMIT_BASELINE_POLICY else "R17") + "_CAUSAL_STRUCTURE_NOT_READY"])
    if (prepared.get("policy", {}).get("policy_version") in {ADAPTIVE_BASELINE_POLICY, *TACTICAL_POLICIES}
            and prepared["selection"].get("setup_variant") is not None):
        # The source-bound new subtype is explicitly starter-only; it cannot
        # borrow a confirmed trigger from a different, failed older context.
        return dict(starter=True, confirmed=False,
            evidence_families=[prepared["selection"]["setup_variant"]], reasons=[])
    if selected == "NO_CLEAR_CONTEXT" or not features.get("history_ready") or not features.get("market_ready"):
        return dict(starter=None, confirmed=None, evidence_families=[], reasons=["BASELINE_TRIGGER_INPUTS_UNKNOWN"])
    starter, confirmed, demand = triggers(selected, features)
    return dict(starter=bool(starter), confirmed=bool(confirmed),
                evidence_families=["V6_DEMAND_CONDITIONS:" + str(int(demand))], reasons=[])


def localization_record():
    return dict(status="NOT_APPLICABLE", quantiles=None, timing=None, p_fill=None, p_rebound=None, p_entry=None,
        raw_log_quantiles=None, raw_probabilities=None, repairs=[], reasons=["UNCALIBRATED_RULE_BASELINE_NO_PREDICTED_ODDS"],
        calibration_version=None, p_bottom_soon=None, p_bottom_now=None, timing_median=None, timing_entropy=None)


def score_baseline(prepared, *, plan, quality, gates, native_features, risks, standard,
                   starter_gate, confirmed_gate, valuation_receipt=None, entry_binding=None,
                   execution_assessment=None, legacy_resilience=False):
    if legacy_resilience and prepared["policy"]["policy_version"] != LIMIT_BASELINE_POLICY:
        raise ValueError("LEGACY_RESILIENCE_ONLY_FOR_ARCHIVED_R18")
    features = known_features(prepared["features"])
    if prepared["policy"]["policy_version"] in LIMIT_POLICIES:
        saved = prepared["selection"]
        selection = ContextSelection(saved["selected_setup"], tuple((name, tuple(items.items()))
            for name, items in saved["evidence"].items()))
    else:
        selection = select_baseline_context(features, policy_version=prepared["policy"]["policy_version"])
    if prepared["policy"]["policy_version"] in {ADAPTIVE_BASELINE_POLICY, *TACTICAL_POLICIES} - LIMIT_POLICIES:
        if prepared["policy"]["policy_version"] == UNIFIED_BASELINE_POLICY:
            from domain.les_v7.plan_r17 import prepare_selection as select_short_base
        elif prepared["policy"]["policy_version"] == CONDITIONAL_BASELINE_POLICY:
            from domain.les_v7.plan_r16 import prepare_selection as select_short_base
        elif prepared["policy"]["policy_version"] == TACTICAL_BASELINE_POLICY:
            from domain.les_v7.context_r15 import select_context as select_short_base
        else:
            from domain.les_v7.context_r14 import select_context as select_short_base
        selection, _ = select_short_base(selection, features,
            bars=pd.DataFrame(prepared["market_rows"]), cutoff=entry_binding["cutoff"],
            **(dict(session_evidence=prepared.get("session_evidence"), market=prepared["policy"]["market"])
               if prepared["policy"]["policy_version"] in CONDITIONAL_POLICIES else {}))
    states = {gate.gate_id: gate.state for gate in gates}
    features["restricted"] = False if states.get("TRADING_STATUS") == "PASS" else True if states.get("TRADING_STATUS") == "FAIL" else None
    features["score_availability"] = dict(features.get("score_availability", {}),
        restricted=states.get("TRADING_STATUS") in {"PASS", "FAIL"})
    alias = dict(entry_price=plan.entry if plan else None, stop_price=plan.stop if plan else None,
        target_price=plan.target if plan else None, stop_source="V7_BOUND_H20_STRUCTURE")
    atr = features.get("atr14")
    alias["stop_distance_atr"] = (plan.entry - plan.stop) / atr if plan and atr is not None and atr > 0 else None
    alias["stop_distance_fraction"] = (plan.entry - plan.stop) / plan.entry if plan else None
    is_r4 = prepared["policy"]["policy_version"] in BOUND_SCORE_POLICIES
    if is_r4:
        from domain.les_v7.rule_score_r4 import score_factors_r4
        if prepared["policy"]["policy_version"] == STABILIZATION_BASELINE_POLICY:
            from domain.les_v7.rule_score_r20 import score_factors_r20 as score_factors_r4
        if prepared["policy"]["policy_version"] == OBSERVATION_BASELINE_POLICY:
            from domain.les_v7.rule_score_r21 import score_factors_r21 as score_factors_r4
        if prepared["policy"]["policy_version"] in {PROGRESSIVE_BASELINE_POLICY, SCOPED_SOURCE_BASELINE_POLICY, AUDIT_SOURCE_BASELINE_POLICY}:
            from domain.les_v7.rule_score_r22 import score_factors_r22 as score_factors_r4
        if prepared["policy"]["policy_version"] == TURNOVER_BASELINE_POLICY:
            from domain.les_v7.rule_score_r25 import score_factors_r25 as score_factors_r4
        scored = score_factors_r4(features, selection, plan, native_features, config=CONFIG,
                                 policy=prepared["policy"], legacy_resilience=legacy_resilience)
    else:
        scored = score_factors(features, selection, alias, config=CONFIG)
    ledger = scored["score_evidence"]
    trigger = baseline_triggers(prepared)
    if prepared["policy"]["policy_version"] in TACTICAL_POLICIES:
        from domain.les_v7.routing_r15 import score_routes
        if prepared["policy"]["policy_version"] == UNIFIED_BASELINE_POLICY:
            from domain.les_v7.routing_r17 import score_routes
        if prepared["policy"]["policy_version"] in LIMIT_POLICIES:
            from domain.les_v7.routing_r18 import score_routes
            if prepared["policy"]["policy_version"] in RISK_BOUNDED_POLICIES:
                from domain.les_v7.routing_r19 import score_routes
        return score_routes(prepared, selection=selection, features=features, scored=scored,
            trigger=trigger, plan=plan, quality=quality, gates=gates, native_features=native_features,
            risks=risks, starter_gate=starter_gate, confirmed_gate=confirmed_gate,
            valuation_receipt=valuation_receipt, entry_binding=entry_binding, config=CONFIG,
            **(dict(execution_assessment=execution_assessment)
               if prepared["policy"]["policy_version"] in LIMIT_POLICIES else {}))
    caps = {}
    def cap(reason, value=61.): caps[reason] = value
    if quality.get("status") != "PASS": cap("BASELINE_QUALITY_NOT_PASS", 0. if quality.get("status") == "REJECT" else 61.)
    if quality.get("quality_score") is None or quality.get("quality_score", 0) < 70: cap("BASELINE_QUALITY_TIER_NOT_READY")
    for gate, state in states.items():
        if state != "PASS": cap(gate + "_" + state, 0. if state == "FAIL" else 61.)
    if not plan or not standard: cap("BASELINE_REFERENCE_RISK_NOT_PASS")
    if selection.context == "NO_CLEAR_CONTEXT": cap("BASELINE_NO_CLEAR_CONTEXT")
    if not features.get("history_ready") or not features.get("market_ready"): cap("BASELINE_ALIGNED_HISTORY_UNKNOWN")
    if not veto_evidence_complete(features): cap("BASELINE_VETO_EVIDENCE_UNKNOWN")
    extra_vetoes = list(hard_vetoes(features))
    technical_vetoes = []
    if is_r4:
        from domain.les_v7.decision import vetoes
        technical_vetoes = sorted(set(vetoes(native_features)) | set(extra_vetoes))
        if technical_vetoes: cap("BASELINE_HARD_VETO", 0.)
        if "EXECUTION_FAIL" in risks: cap("BASELINE_EXECUTION_FAIL", 0.)
    elif risks or extra_vetoes: cap("BASELINE_HARD_VETO", 0.)
    if trigger["confirmed"] is not True: cap("BASELINE_NOT_CONFIRMED", 71.)
    if trigger["starter"] is not True and trigger["confirmed"] is not True: cap("BASELINE_TRIGGER_NOT_READY")
    if not confirmed_gate: cap("BASELINE_CONFIRMED_RISK_NOT_READY", 71.)
    if not starter_gate and not confirmed_gate: cap("BASELINE_STARTER_RISK_NOT_READY")
    penalties = {"RISK_OFF": CONFIG.risk_off_score_increment} if features.get("market_regime") == "RISK_OFF" else {}
    qualification_reasons = []
    admission = None
    is_r9 = prepared["policy"]["policy_version"] in GRADED_POLICIES
    if is_r9:
        from domain.les_v7.entry_policy_r9 import assess
        if prepared["policy"]["policy_version"] == RELAXED_BASELINE_POLICY:
            from domain.les_v7.entry_policy_r13 import assess
        elif prepared["policy"]["policy_version"] == ADAPTIVE_BASELINE_POLICY:
            from domain.les_v7.entry_policy_r14 import assess
        if not isinstance(entry_binding, Mapping): raise ValueError("R9_SOURCE_BINDING_REQUIRED")
        admission = assess(quality, gates, valuation_receipt, symbol=entry_binding["symbol"],
            cutoff=entry_binding["cutoff"], current_price=native_features.get("close"),
            **(dict(setup_variant=prepared["selection"].get("setup_variant"))
               if prepared["policy"]["policy_version"] == ADAPTIVE_BASELINE_POLICY else {}))
        if admission["valuation"]["state"] == "ALLOWED_WITH_WARNING" and admission["valuation"]["points"] > 0:
            penalties["VALUATION_PRICE_WARNING"] = admission["valuation"]["points"]
    if prepared["policy"]["policy_version"] in {PENALTY_BASELINE_POLICY, *GRADED_POLICIES}:
        from domain.les_v7.rule_score_r8 import compose_penalty_score
        # Gates remain non-compensable in the decision below, not numerical caps.
        qualification_reasons = list(caps)
        caps = {}
        deductions = prepared["policy"]["score_adjustment_policy"]["deductions"]
        if selection.context == "NO_CLEAR_CONTEXT":
            penalties["BASELINE_NO_CLEAR_CONTEXT"] = deductions["no_clear_context"]
        elif trigger["starter"] is not True and trigger["confirmed"] is not True:
            penalties["BASELINE_TRIGGER_NOT_READY"] = deductions["trigger_not_ready"]
        elif trigger["confirmed"] is not True:
            penalties["BASELINE_NOT_CONFIRMED"] = deductions["unconfirmed_starter"]
        outcome = compose_penalty_score(ledger, [dict(reason=k, points=v) for k, v in penalties.items()])
    else:
        outcome = compose_score(ledger, [dict(reason=key, points=value) for key, value in penalties.items()],
            [dict(reason=key, maximum=value) for key, value in caps.items()])
    values = tuple(ledger["components"][name]["earned_points"] / ledger["components"][name]["total_weight"]
                   for name in FACTORS) if outcome["raw_score"] is not None else None
    weights = tuple(prepared["policy"]["model_spec"]["factor_budgets"][name] / 100 for name in FACTORS)
    contract = dict(raw_score=outcome["raw_score"], adjusted_score=outcome["adjusted_score"],
        final_score=outcome["final_score"], total_penalty=outcome["penalty"], effective_cap=outcome["safety_cap"])
    final, qs = contract["final_score"], quality.get("quality_score")
    if is_r9:
        # Only independently verified economic warnings may be graded; R13
        # adds its explicit small-only bear-scenario warning, not source PASS.
        effective_risks = [r for r in risks if r != "VALUATION_FAIL"]
        if effective_risks or extra_vetoes or technical_vetoes or admission["state"] == "BLOCKED":
            decision = "NO_BUY"
        elif admission["state"] == "UNKNOWN": decision = "DATA_INSUFFICIENT"
        elif not features.get("history_ready") or not features.get("market_ready") or not veto_evidence_complete(features) or final is None:
            decision = "DATA_INSUFFICIENT"
        elif selection.context == "NO_CLEAR_CONTEXT": decision = "WATCH_ONLY"
        elif not plan: decision = "DATA_INSUFFICIENT"
        elif final >= 72 and admission["buy_allowed"] and trigger["confirmed"] is True and confirmed_gate: decision = "BUY"
        elif final >= 62 and admission["small_allowed"] and trigger["starter"] is True and starter_gate: decision = "BUY_SMALL"
        elif final < 55: decision = "AVOID"
        elif final < 62: decision = "WATCH"
        else: decision = "ARMED"
    elif risks or extra_vetoes or technical_vetoes or quality.get("status") == "REJECT" or any(state == "FAIL" for state in states.values()):
        decision = "NO_BUY"
    elif (selection.context == "NO_CLEAR_CONTEXT" and quality.get("status") == "PASS"
            and all(states.get(gate) == "PASS" for gate in ("DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION"))
            and features.get("history_ready") and features.get("market_ready") and veto_evidence_complete(features)):
        decision = "WATCH_ONLY"  # No plan is not evidence that a nonexistent risk plan passed.
    elif (any(state != "PASS" for state in states.values()) or quality.get("status") != "PASS"
            or not features.get("history_ready") or not features.get("market_ready") or not veto_evidence_complete(features)
            or not plan or final is None):
        decision = "DATA_INSUFFICIENT"
    elif selection.context == "NO_CLEAR_CONTEXT": decision = "WATCH_ONLY"
    elif final >= 72 and qs is not None and qs >= 70 and trigger["confirmed"] is True and confirmed_gate: decision = "BUY"
    elif final >= 62 and qs is not None and qs >= 75 and trigger["starter"] is True and starter_gate: decision = "BUY_SMALL"
    elif final < 55: decision = "AVOID"
    elif final < 62: decision = "WATCH"
    else: decision = "ARMED"
    result = dict(score=contract, factors=values, weights=weights, penalties=penalties, caps=caps, decision=decision,
        score_evidence=ledger, factor_ledger=scored["factor_budgets"], trigger=trigger, hard_vetoes=extra_vetoes,
        reasons=qualification_reasons + (["NO_CLEAR_CONTEXT_WATCH_ONLY_NOT_ENTRY"] if decision == "WATCH_ONLY" else []))
    if is_r4: result["technical_vetoes"] = technical_vetoes
    if is_r9:
        result["entry_admission"] = admission
        result["reasons"] = ["R9_QUALITY_TIER_" + str(admission["quality"]["tier"]), admission["valuation"]["reason"]]
    return result


def validate_rule_baseline_report(report):
    """Recalculate the fixed baseline; rehashed scores/default odds cannot pass."""
    from domain.les_v7.contracts import GateReceipt, ReferencePlan
    from domain.les_v7.features import select_bars
    from domain.les_v7.native_evidence import _validated_features
    from domain.les_v7.contracts import FEATURE_VERSION
    from domain.les_v7.decision import make_plan, vetoes
    d, e = report["diagnostics"], report["envelope"]
    saved = d["rule_baseline"]
    binding = d["gate_bindings"]["SAFETY"]
    market, cutoff = binding["market"], e["event"]["cutoff_at"]
    require_baseline_policy(saved["policy"], market=market, cutoff=cutoff)
    if (saved.get("schema") != BASELINE_SCHEMA or saved.get("policy_id") != identity(saved["policy"])
            or e["model"] != dict(model_status="RULE_BASED_BASELINE", status_basis="HASH_BOUND_RUNTIME_FACT_NOT_DOCUMENT_DECLARATION",
                bundle_sha256=None, trained_artifacts_status="NOT_APPLICABLE", validation_evidence_status="NOT_APPLICABLE")
            or e["schema_version"] != ENVELOPE_SCHEMA):
        raise ValueError("V7_BASELINE_MODE_BINDING_MISMATCH")
    bars = select_bars(pd.DataFrame(saved["market_rows"]), cutoff)
    version = saved["policy"]["policy_version"]
    source = d.get("native_evidence", {}).get("source_packet", {})
    if version in INDUSTRY_POLICIES and source.get("industry_input", {}).get("symbol") != report["symbol"]:
        raise ValueError("V7_INDUSTRY_SECURITY_BINDING_MISMATCH")
    prepared = prepare_baseline(bars, market=market, cutoff=cutoff, policy_version=version,
        industry_input=source.get("industry_input") if version in INDUSTRY_POLICIES else None,
        session_evidence=source.get("calendar_evidence") if version in CONDITIONAL_POLICIES else None)
    for key in ("schema", "policy", "policy_id", "market_rows", "features", "selection"):
        if identity(saved.get(key)) != identity(prepared[key]):
            raise ValueError("V7_BASELINE_SOURCE_OR_FEATURE_RECOMPUTATION_MISMATCH")
    if version in CONDITIONAL_POLICIES and identity(saved.get("session_evidence")) != identity(prepared["session_evidence"]):
        raise ValueError("V7_R16_SESSION_SOURCE_RECOMPUTATION_MISMATCH")
    if version in INDUSTRY_POLICIES and identity(saved.get("industry_score")) != identity(prepared["industry_score"]):
        raise ValueError("V7_INDUSTRY_SCORE_RECOMPUTATION_MISMATCH")
    if identity(prepared["market_rows"]) != binding["bars_hash"]:
        raise ValueError("V7_BASELINE_BAR_IDENTITY_MISMATCH")
    f = json.loads(_validated_features(canonical(prepared["market_rows"]), cutoff,
        prepared["policy"].get("native_feature_version", FEATURE_VERSION)))
    session = prepared["market_rows"][-1]["Date"] if prepared["market_rows"] else None
    if identity(f) != identity(d["features"]) or session != report["signal_date"]:
        raise ValueError("V7_BASELINE_NATIVE_FEATURE_MISMATCH")
    selected = prepared["selection"]["selected_setup"]
    structure = None
    if version in LIMIT_POLICIES:
        from domain.les_v7.plan_r18 import make_plan as limit_plan
        if version in RISK_BOUNDED_POLICIES:
            from domain.les_v7.plan_r19 import make_plan as limit_plan
        plan, structure = limit_plan(selected, f, prepared["policy"]["reference_policy"],
            e["event"]["evidence_id"], bars=bars, cutoff=cutoff, session_evidence=source.get("calendar_evidence"))
    else:
        plan = make_plan(selected, f, prepared["policy"]["reference_policy"], e["event"]["evidence_id"],
            **(dict(bars=bars, cutoff=cutoff) if version in RECOVERY_POLICIES else {}),
            **(dict(session_evidence=source.get("calendar_evidence")) if version in CONDITIONAL_POLICIES else {}))
    if (e["event"]["context"] != selected or identity(d["context_matches"]) != identity(prepared["selection"]["matches"])
            or identity(d["reference_plan"]) != identity(asdict(plan) if plan else None)):
        raise ValueError("V7_BASELINE_PLAN_RECOMPUTATION_MISMATCH")
    gates = [GateReceipt.bind(dict(gate_id=row["gate_id"], gate_version=row["gate_version"], state=row["state"],
            input_hash=row["input_hash"], source_available_at=row["source_available_at"], reason_codes=row["reasons"],
            proof_record_ids=row["proof_record_ids"], input_ids=row["input_ids"], checked_at=row["checked_at"]),
            gate_id=row["gate_id"], inputs=d["gate_bindings"][row["gate_id"]], cutoff=cutoff)
        if row["gate_version"] != "UNAVAILABLE" else GateReceipt.bind(None, gate_id=row["gate_id"],
            inputs=d["gate_bindings"][row["gate_id"]], cutoff=cutoff) for row in d["gate_receipts"]]
    if len(gates) != 5 or {g.gate_id for g in gates} != {"DATA_INTEGRITY", "TRADING_STATUS", "SAFETY", "VALUATION", "STRESS_RISK"}:
        raise ValueError("V7_BASELINE_REQUIRED_GATE_SET_INVALID")
    if identity([asdict(gate) for gate in gates]) != identity(d["gate_receipts"]):
        raise ValueError("V7_BASELINE_GATE_PROJECTION_MISMATCH")
    risks = list(vetoes(f))
    risks.extend(g.gate_id + "_FAIL" for g in gates if g.state == "FAIL")
    participation = plan.entry * plan.reference_shares / f["adv"] if plan and f.get("adv") and f["adv"] > 0 else None
    known = f.get("spread") is not None and participation is not None
    liquidity = known and f["spread"] <= .005 and participation <= .01
    if known and not liquidity: risks.append("EXECUTION_FAIL")
    rr_min = 2. if selected == "REVERSAL" else 1.8
    if plan and (plan.stop_atr > 1.5 or plan.net_rr is not None and plan.net_rr < rr_min): risks.append("EXECUTION_FAIL")
    standard = bool(plan and liquidity and next(g for g in gates if g.gate_id == "STRESS_RISK").state == "PASS"
                    and plan.net_rr is not None and plan.net_rr >= rr_min and plan.stop_atr <= 1.5)
    starter_gate = bool(standard and plan.stop_atr <= (1.2 if selected == "REVERSAL" else 1.3))
    confirmed_gate = bool(standard)
    execution_assessment = None
    if version in LIMIT_POLICIES:
        from domain.les_v7.risk_r18 import assess_execution, require_assessment
        if version in RISK_BOUNDED_POLICIES:
            from domain.les_v7.risk_r19 import assess_execution, require_assessment
        receipt = d["native_evidence"]["produced_receipts"].get("STRESS_RISK") or {}
        execution_assessment = receipt.get("execution_assessment")
        if execution_assessment is None:
            if plan is not None and next(g for g in gates if g.gate_id == "STRESS_RISK").state == "PASS":
                raise ValueError("R18_EXECUTION_ASSESSMENT_REQUIRED")
            execution_assessment = assess_execution(plan, prepared["policy"]["reference_policy"], f)
        execution_assessment = require_assessment(execution_assessment, plan, prepared["policy"]["reference_policy"], f)
        if identity(d.get("execution_assessment")) != identity(execution_assessment):
            raise ValueError("R18_EXECUTION_ASSESSMENT_RECOMPUTATION_MISMATCH")
        risks = [reason for reason in risks if reason != "EXECUTION_FAIL"]
        if execution_assessment["state"] == "FAIL": risks.append("EXECUTION_FAIL")
        standard = bool(plan and execution_assessment["state"] == "PASS"
            and next(g for g in gates if g.gate_id == "STRESS_RISK").state == "PASS")
        starter_gate = bool(standard and execution_assessment["tactical"]["state"] == "PASS")
        confirmed_gate = bool(standard and execution_assessment["core"]["state"] == "PASS")
        if identity(d.get("structure_plan")) != identity(structure):
            raise ValueError("V7_R10_STRUCTURE_RECOMPUTATION_MISMATCH")
    elif version in {RELAXED_BASELINE_POLICY, ADAPTIVE_BASELINE_POLICY, *TACTICAL_POLICIES}:
        from domain.les_v7.plan_r10 import evaluate_entry
        # Compare original decimal operands, not a rounded/epsilon-adjusted
        # stop_atr float. The plan and native producer use this same contract.
        entry_check = evaluate_entry(plan, plan.entry, prepared["policy"]["reference_policy"],
            **(dict(starter=True) if version in TACTICAL_POLICIES else {})) if plan else None
        risks = [reason for reason in risks if reason != "EXECUTION_FAIL"]
        if (known and not liquidity) or (entry_check and entry_check["state"] == "FAIL"):
            risks.append("EXECUTION_FAIL")
        standard = bool(plan and liquidity and entry_check["state"] == "PASS"
            and next(g for g in gates if g.gate_id == "STRESS_RISK").state == "PASS")
        starter_gate = confirmed_gate = standard
    if version in RECOVERY_POLICIES and version not in LIMIT_POLICIES:
        from domain.les_v7.plan_r10 import make_plan as structure_plan, validate_band
        _, structure = structure_plan(selected, f, prepared["policy"]["reference_policy"],
                                      e["event"]["evidence_id"], bars=bars, cutoff=cutoff,
            **(dict(session_evidence=source.get("calendar_evidence")) if version in CONDITIONAL_POLICIES else {}))
        if identity(d.get("structure_plan")) != identity(structure):
            raise ValueError("V7_R10_STRUCTURE_RECOMPUTATION_MISMATCH")
        confirmed_gate = bool(standard and validate_band(plan, prepared["policy"]["reference_policy"])["state"] == "PASS") if plan else False
        starter_gate = bool(standard and validate_band(plan, prepared["policy"]["reference_policy"], starter=True)["state"] == "PASS") if plan else False
    # R18 was archived before the Hybrid observed-zero correction. Its exact
    # ledger shape identifies the old formula, not permission to skip checks.
    # Recompute every score, condition, gate and plan under that formula; new
    # generation continues to use the corrected default. R19 is not downgraded.
    resilience_rows = [c for c in saved.get("score_evidence", {}).get("conditions", [])
                       if c.get("id") == "relative_strength:correction_resilience"]
    legacy_resilience = bool(version == LIMIT_BASELINE_POLICY and len(resilience_rows) == 1
        and resilience_rows[0].get("dependencies") ==
        ["market_correction20", "downside_capture20", "correction_support", "low"])
    expected = score_baseline(prepared, plan=plan, quality=d["quality"], gates=gates, native_features=f,
        risks=risks, standard=standard, starter_gate=starter_gate, confirmed_gate=confirmed_gate,
        legacy_resilience=legacy_resilience,
        **(dict(valuation_receipt=d["native_evidence"]["produced_receipts"].get("VALUATION"), entry_binding=binding)
           if prepared["policy"]["policy_version"] in GRADED_POLICIES else {}),
        **(dict(execution_assessment=execution_assessment) if version in LIMIT_POLICIES else {}))
    if (identity(d.get("entry_admission")) != identity(expected.get("entry_admission"))):
        raise ValueError("V7_ENTRY_ADMISSION_RECOMPUTATION_MISMATCH")
    if version in TACTICAL_POLICIES:
        route_key = ("r19_route_evaluation" if version in RISK_BOUNDED_POLICIES else
                     "r18_route_evaluation" if version == LIMIT_BASELINE_POLICY else
                     "r17_route_evaluation" if version == UNIFIED_BASELINE_POLICY else
                     "r16_route_evaluation" if version == CONDITIONAL_BASELINE_POLICY else "r15_route_evaluation")
        if identity(d.get(route_key)) != identity(expected[route_key]):
            raise ValueError("V7_TACTICAL_ROUTE_RECOMPUTATION_MISMATCH")
    if (prepared["policy"]["policy_version"] in BOUND_SCORE_POLICIES
            and identity(saved.get("technical_vetoes")) != identity(expected["technical_vetoes"])):
        raise ValueError("V7_BASELINE_TECHNICAL_VETO_RECOMPUTATION_MISMATCH")
    scope = d.get("run_scope")
    diagnostic = scope is not None
    final = None if diagnostic else expected["score"]["final_score"]
    decision = expected["decision"]
    if diagnostic and not risks and not expected["hard_vetoes"] and d["quality"]["status"] == "PASS" and all(g.state == "PASS" for g in gates):
        decision = "WATCH_ONLY"
    errors = d.get("daily_input_errors", [])
    if not isinstance(errors, list) or any(not isinstance(error, str) or not error for error in errors):
        raise ValueError("V7_BASELINE_DAILY_INPUT_ERRORS_INVALID")
    if errors:
        final = None
        if decision != "NO_BUY": decision = "DATA_INSUFFICIENT"
    if version in TACTICAL_POLICIES:
        from domain.les_v7.plan_r10 import bind_entry_route
        entry_route = (bind_entry_route(plan, prepared["policy"]["reference_policy"],
            strategy_decision=decision, risk_route=expected[route_key]["selected_route"],
            setup_variant=prepared["selection"].get("setup_variant"))
            if decision in {"BUY", "BUY_SMALL"} else None)
        if identity(d.get("reference_entry_route")) != identity(entry_route):
            raise ValueError("V7_R15_REFERENCE_ENTRY_ROUTE_MISMATCH")
    expected_score = dict(factor_order=FACTORS, factor_values=expected["factors"], applicable_weights=expected["weights"],
                          **dict(expected["score"], final_score=final))
    if (identity(expected_score) != identity(e["score"])
            or identity(expected["score"]) != identity(d["diagnostic_score"])
            or identity(expected["score_evidence"]) != identity(saved["score_evidence"])
            or identity(expected["hard_vetoes"]) != identity(saved["hard_vetoes"])
            or identity(expected["factor_ledger"]) != identity(d["factor_ledger"])
            or identity(expected["caps"]) != identity(d["safety_caps"])
            or identity(expected["penalties"]) != identity(d["penalties"])
            or decision != e["strategy"]["strategy_decision"]
            or any(d[key] != expected["trigger"][key] for key in ("starter", "confirmed", "evidence_families"))
            or d["starter_gate"] != starter_gate or d["confirmed_gate"] != confirmed_gate):
        raise ValueError("V7_BASELINE_SCORE_OR_DECISION_RECOMPUTATION_MISMATCH")
    prediction = localization_record()
    envelope_prediction = dict(status="NOT_APPLICABLE", bottom_price_quantiles=None, timing_pmf=None, p_fill=None,
        p_rebound_given_fill=None, p_entry_given_fill=None, p_bottom_soon=None, forecast_result_id=identity(prediction))
    if identity(d["localization"]) != identity(prediction) or identity(e["localization"]) != identity(envelope_prediction):
        raise ValueError("V7_BASELINE_CANNOT_FABRICATE_CALIBRATED_PREDICTIONS")

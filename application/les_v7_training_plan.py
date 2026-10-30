"""Preregistered real-research settings, not fitted parameters or readiness.

The dates and tolerances below are initial research policy choices fixed before
outcome inspection. They neither certify historical availability nor change any
Quality, path, price-coordinate or execution gate. No fitting/HTTP/account port.
"""
from copy import deepcopy
from math import log

from application.les_v7_dataset import validate_build_config
from domain.les_v7.contracts import identity
from domain.les_v7.quality_policy import default_quality_policy
from domain.les_v7.training import reference_config_id, validate_training_config

PLAN_SCHEMA = "les-v7-fixed-research-plan/v1"
POLICY = "ELEC_H20_DFA512_CONTEXT_SUPPORT_R2"
EVALUATION = "2026-08-07T13:30:00+08:00"
SEED_FIELDS = {"feature_names", "feature_bounds", "context_rules", "reference_policy"}
R3_FEATURES = ["slope5", "slope20", "return5", "return20", "drawdown20", "extension",
               "atr_fraction", "atr_expansion", "clv", "cmf", "dfa512", "delta_dfa512_5",
               "higuchi256", "higuchi512", "context_REVERSAL", "context_TREND_PULLBACK",
               "context_LEADERSHIP_CONTINUATION"]
R3_BOUNDS = [[-10, 10], [-10, 10], [-.99, 5], [-.99, 5], [-.99, 0], [-50, 50],
             [0, 1], [0, 20], [-1, 1], [-1, 1], [0, 3], [-3, 3], [0, 3], [0, 3],
             [0, 1], [0, 1], [0, 1]]


def _reject_synthetic(value):
    if isinstance(value, dict):
        return any(_reject_synthetic(key) or _reject_synthetic(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_reject_synthetic(item) for item in value)
    return isinstance(value, str) and "SYNTHETIC" in value.upper()


def generate_fixed_research_plan(seed, *, quality_policy=None):
    """Return matched build/training settings without inspecting any outcomes.

    Pass the exact four-field native reference seed, not a full (especially
    synthetic example) fit config. Missing older history or official evidence
    remains a prerequisite; generating settings does not declare it repaired.
    """
    if not isinstance(seed, dict) or set(seed) != SEED_FIELDS:
        raise ValueError("EXACT_NATIVE_REFERENCE_SEED_REQUIRED_NOT_FIT_CONFIG")
    if _reject_synthetic(seed) or _reject_synthetic(quality_policy):
        raise ValueError("SYNTHETIC_SETTINGS_CANNOT_FREEZE_REAL_RESEARCH_PLAN")
    seed = deepcopy(seed)
    enhanced = seed["feature_names"] == R3_FEATURES
    if seed["feature_names"] != ["slope5", "dfa512"] and not enhanced:
        raise ValueError("FIXED_RESEARCH_PLAN_REQUIRES_REVIEWED_SLOPE5_DFA512_SEED")
    if enhanced and seed["feature_bounds"] != R3_BOUNDS:
        raise ValueError("FIXED_RESEARCH_PLAN_REQUIRES_REVIEWED_R3_SUPPORT_BOUNDS")
    path = seed["reference_policy"].get("cost_path_policy")
    if not isinstance(path, dict) or path.get("market") != "TWSE":
        raise ValueError("SOURCE_BOUND_TWSE_REFERENCE_PATH_SEED_REQUIRED")
    build = dict(schema="les-v7-dataset-build-config/v1", seed=deepcopy(seed),
        event_start="2026-01-14", event_end="2026-07-09", evaluation_cutoff=EVALUATION,
        event_sampling="ALL_FIXED_UNIVERSE_OFFICIAL_SESSIONS", max_events=4096,
        path_policy=deepcopy(path), coordinate_coverages=[],
        policy_scope="INITIAL_REAL_RESEARCH_POLICY_NOT_OPTIMIZED_OR_DEPLOYMENT_ACCEPTANCE")
    training = dict(schema="les-v7-training-config/v1", **deepcopy(seed),
        trigger_rules=dict(
            # EMA20 proximity and intact support, not a claim of a strict-pivot retest.
            near_pullback_support=dict(op="all", args=[
                dict(field="extension", op="le", value=1.),
                dict(field="support_intact", op="eq", value=True)]),
            # Compression/reclaim proxy: an initial causal policy, not an optimized base classifier.
            tight_base_or_retest=dict(op="any", args=[
                dict(field="atr_expansion", op="le", value=1.),
                dict(field="reclaim", op="eq", value=True)])),
        quality_policy=deepcopy(default_quality_policy() if quality_policy is None else quality_policy),
        fit_cutoff="2026-06-30T13:30:00+08:00",
        validation_start="2026-07-01T13:30:00+08:00",
        validation_end="2026-07-09T13:30:00+08:00", evaluation_cutoff=EVALUATION,
        folds=[dict(fold_id="past-only-2026-march", fit_cutoff="2026-03-02T13:30:00+08:00",
                    prediction_start="2026-03-03T13:30:00+08:00", prediction_end="2026-03-13T13:30:00+08:00"),
               dict(fold_id="past-only-2026-april", fit_cutoff="2026-04-20T13:30:00+08:00",
                    prediction_start="2026-04-21T13:30:00+08:00", prediction_end="2026-04-30T13:30:00+08:00"),
               dict(fold_id="past-only-2026-may", fit_cutoff="2026-04-30T13:30:00+08:00",
                    prediction_start="2026-05-01T13:30:00+08:00", prediction_end="2026-05-29T13:30:00+08:00")],
        min_head_rows=500, min_calibration_rows=250, min_weight_rows=500,
        min_context_weight_rows=100, min_holdout_rows=500,
        head_iterations=1000, calibration_iterations=500, regularization=.01,
        null_seed=19, null_replicates=64,
        weight_solver=dict(max_iterations=1000, tolerance=1e-6),
        acceptance=dict(max_brier=.25, max_timing_nll=log(21), max_pinball_loss=.05),
        policy_scope="INITIAL_REAL_RESEARCH_POLICY_NOT_OPTIMIZED_OR_DEPLOYMENT_ACCEPTANCE")
    if path.get("coordinate_policy") == "OFFICIAL_FINAL_ACTIONS_RECONSTRUCTION_R1":
        # Explicit opt-in only; the historical-observation default remains unchanged.
        build["reference_evidence_policy"] = "OFFICIAL_FINAL_REFERENCE_RECONSTRUCTION_R1"
        training["reference_evidence_policy"] = "OFFICIAL_FINAL_REFERENCE_RECONSTRUCTION_R1"
    if enhanced:
        # July R2 holdout was already inspected. The new independent September
        # partition is fixed before this profile is fitted or scored; no date
        # search, threshold changes, winner selection or holdout tuning.
        evaluation = "2026-10-12T13:30:00+08:00"
        build.update(event_end="2026-09-07", evaluation_cutoff=evaluation,
                     fixed_universe_date="2026-08-07")
        training.update(fit_cutoff="2026-08-28T13:30:00+08:00",
                        validation_start="2026-09-01T13:30:00+08:00",
                        validation_end="2026-09-07T13:30:00+08:00", evaluation_cutoff=evaluation,
                        head_solver="STANDARDIZED_CONVEX_R2", head_iterations=2000)
        training["folds"].extend([
            dict(fold_id="past-only-2026-june", fit_cutoff="2026-05-29T13:30:00+08:00",
                 prediction_start="2026-06-01T13:30:00+08:00", prediction_end="2026-06-30T13:30:00+08:00"),
            dict(fold_id="past-only-2026-july-development", fit_cutoff="2026-06-30T13:30:00+08:00",
                 prediction_start="2026-07-01T13:30:00+08:00", prediction_end="2026-07-31T13:30:00+08:00")])
    validate_build_config(build)
    validate_training_config(training)
    return dict(schema=PLAN_SCHEMA, policy_id="ELEC_H20_CAUSAL_GEOMETRY_CONVEX_R3" if enhanced else POLICY,
        history_start="2023-07-03" if enhanced else "2023-10-01", history_end="2026-10-12" if enhanced else "2026-08-07", universe_date="2026-08-07",
        universe_policy="ALL_FIXED_20260807_ELEC_FINAL_SNAPSHOT_NOT_CANDIDATE_PIT_MEMBERSHIP",
        original_seed_sha256=identity(seed), reference_config_id=reference_config_id(seed),
        build_config=build, training_config=training,
        readiness=dict(status="NOT_ASSESSED_SETTINGS_ONLY", required_evidence=[
            "OLDER_OFFICIAL_BARS_FOR_513_VALID_CLOSES_BY_20260105",
            "VERIFIED_OFFICIAL_CALENDAR_AND_H20_MATURITY",
            "SOURCE_BOUND_COMPLETE_CORPORATE_ACTION_COVERAGE",
            "BOTH_CLASSES_PER_BINARY_HEAD_AND_CALIBRATOR",
            "MATURE_PAST_CALIBRATED_OOF_CONTEXT_SAMPLES",
            "INDEPENDENT_HOLDOUT_SUPPORT"],
            no_missing_evidence_override=True, data_gaps_remain_exclusions=True),
        interpretation=dict(
            support_floors="PROCESSING_SUPPORT_NOT_INDEPENDENT_SAMPLE_COUNTS",
            context_support="R2_EXTENDS_PAST_ONLY_OOF_AND_MATURE_HOLDOUT_COVERAGE_TO_ADDRESS_R1_CONTEXT_FLOOR_GAP_WITHOUT_LOWERING_NATIVE_MINIMUMS",
            **(dict(model_profile="R3_MULTISCALE_FRACTAL_TREND_VOLATILITY_AND_CONTEXT_NOT_HOLDOUT_SELECTED",
                    development_only_ranges=["2026-07-01/2026-07-09"],
                    independent_holdout="FIXED_SEPTEMBER_PARTITION_NEVER_USED_FOR_PARAMETER_SELECTION",
                    optimizer="TRAINING_ONLY_STANDARDIZATION_RAW_RUNTIME_COEFFICIENTS_CONVEX_R2") if enhanced else {}),
            brier_limit="UNINFORMATIVE_0_5_REFERENCE_CEILING_NOT_SKILL_CERTIFICATION",
            timing_limit="UNIFORM_21_BUCKET_REFERENCE_CEILING_NOT_SKILL_CERTIFICATION",
            pinball_limit="UNWEIGHTED_MEAN_PRICE_ERROR_DIVIDED_BY_CUTOFF_CLOSE_ACROSS_THREE_QUANTILES",
            trigger_rules="CAUSAL_INITIAL_PROXIMITY_COMPRESSION_RECLAIM_POLICY_NOT_OPTIMIZED",
            ablation="SAME_EVENT_PAST_ONLY_BASELINE_COMPARISON_STILL_REQUIRED",
            overlapping_labels="ADJACENT_H20_WINDOWS_ARE_DEPENDENT",
            availability="RETROSPECTIVE_RECONSTRUCTION_IS_NOT_ORIGINAL_OBSERVATION_OR_DEPLOYMENT"),
        research_only=True, approved_for_live=False, fitting_performed=False, execution_action="NO_ACTION")


def _validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("FIXED_RESEARCH_PLAN_SCHEMA_REQUIRED")
    build, training = plan["build_config"], plan["training_config"]
    if (plan.get("original_seed_sha256") != identity(build["seed"])
            or plan.get("reference_config_id") != reference_config_id(build["seed"])
            or reference_config_id(training) != reference_config_id(build["seed"])):
        raise ValueError("RESEARCH_PLAN_REFERENCE_SEED_MISMATCH")
    expected = generate_fixed_research_plan(build["seed"], quality_policy=training["quality_policy"])
    if plan != expected:
        raise ValueError("FIXED_RESEARCH_PLAN_POLICY_OR_CONFIG_MISMATCH")

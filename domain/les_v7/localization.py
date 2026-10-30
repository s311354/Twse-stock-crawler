"""Hash-bound frozen five-head inference. No fitted-artifact/prior fallback."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import numpy as np

from domain.les_v7.contracts import (CONTEXTS, FEATURE_VERSION, MAPPING_VERSION,
                                     ReferencePlan, canonical, finite, identity, timestamp)
from domain.les_v7.fractals import is_active_fractal_feature
from domain.les_v7.provenance import RUNTIME_SCHEMA, validate_runtime_provenance


@dataclass(frozen=True)
class FrozenBundle:
    payload_json: str
    sha256: str

    @classmethod
    def load(cls, record: dict) -> FrozenBundle:
        payload_json = canonical(record["payload"])
        if record["sha256"] != hashlib.sha256(payload_json.encode("utf-8")).hexdigest():
            raise ValueError("MODEL_BUNDLE_HASH_MISMATCH")
        # Daily evidence freezes JSON arrays as tuples. Decode only this model
        # boundary, after hashing the exact canonical content, so all existing
        # profile/provenance/calibration validators see standard JSON types.
        # Never thaw the surrounding financial evidence or weaken coordinates.
        payload = json.loads(payload_json)
        required = ("bundle_id", "trained_until", "last_label_available_at", "fit_cutoff", "feature_names",
                    "feature_bounds", "imputation", "supported_masks", "heads", "weights", "calibration",
                    "context_rules", "trigger_rules", "reference_policy", "quality_policy", "training_manifest")
        if any(k not in payload for k in required):
            raise ValueError("MODEL_BUNDLE_DEPENDENCIES_INCOMPLETE")
        if payload.get("feature_profile") != FEATURE_VERSION or payload.get("mapping_version") != MAPPING_VERSION:
            raise ValueError("INCOMPATIBLE_V7_FEATURE_OR_MAPPING_VERSION")
        if payload.get("trained_until") is None:
            raise ValueError("MODEL_BUNDLE_NOT_PUBLISHED")
        if not timestamp(payload["last_label_available_at"]) <= timestamp(payload["fit_cutoff"]) <= timestamp(payload["trained_until"]):
            raise ValueError("TRAINING_LABEL_TIME_LEAKAGE")
        manifest = payload["training_manifest"]
        runtime_schema = payload.get("runtime_provenance_schema")
        if runtime_schema not in {None, RUNTIME_SCHEMA}:
            raise ValueError("UNKNOWN_RUNTIME_PROVENANCE_SCHEMA")
        if runtime_schema == RUNTIME_SCHEMA:
            if len(canonical(record).encode("utf-8")) > 1024 * 1024:
                raise ValueError("FROZEN_MODEL_EXCEEDS_RUNTIME_SIZE_LIMIT")
            validate_runtime_provenance(payload)
        if (manifest.get("upstream_predictions") != "STRICTLY_PAST_OOF"
                or not manifest.get("dataset_id") or not manifest.get("oof_prediction_ids")):
            raise ValueError("STRICTLY_PAST_OOF_MANIFEST_REQUIRED")
        for fold in manifest.get("folds", ()):
            if not timestamp(fold["last_training_label_available_at"]) <= timestamp(fold["fit_cutoff"]) < timestamp(fold["prediction_at"]):
                raise ValueError("OOF_TIME_LEAKAGE")
        if not manifest.get("folds"):
            raise ValueError("OOF_FOLDS_REQUIRED")
        for context in (*CONTEXTS, "GLOBAL"):
            vector = payload["weights"].get(context)
            if vector is None or len(vector) != 6 or any(finite(v, "weight", 0, 1) < 0 for v in vector):
                raise ValueError("ESTIMATED_CONTEXT_WEIGHTS_REQUIRED")
            if abs(math.fsum(vector) - 1) > 1e-9:
                raise ValueError("WEIGHTS_MUST_SUM_ONE_NO_RENORMALIZATION")
        names = payload["feature_names"]
        n = len(names)
        if (not 1 <= n <= 64 or any(not isinstance(name, str) or not 1 <= len(name) <= 80 for name in names)
                or len(set(names)) != n):
            raise ValueError("BOUNDED_UNIQUE_FEATURE_NAMES_REQUIRED")
        has_fractals = any(is_active_fractal_feature(name) for name in names)
        if has_fractals and payload.get("fractal_mode") == "V7_NO_FRACTAL_BASELINE":
            raise ValueError("FRACTAL_FEATURES_IN_NO_FRACTAL_BASELINE")
        if not has_fractals and (payload.get("fractal_mode") != "V7_NO_FRACTAL_BASELINE"
                                 or not payload.get("baseline_validation_report_id")):
            raise ValueError("EXPLICIT_VALIDATED_NO_FRACTAL_BASELINE_REQUIRED")
        if any(name.startswith(("lppl", "mfdfa")) for name in names):
            raise ValueError("SHADOW_FEATURES_REQUIRE_SEPARATE_ABLATION_PROFILE")
        for key in ("feature_bounds", "imputation"):
            if len(payload[key]) != n: raise ValueError("FEATURE_PROFILE_DIMENSION_MISMATCH")
        for bounds, imputed in zip(payload["feature_bounds"], payload["imputation"]):
            if len(bounds) != 2 or finite(bounds[0]) >= finite(bounds[1]):
                raise ValueError("INVALID_FEATURE_SUPPORT_BOUNDS")
            if imputed is not None: finite(imputed, "imputation", *bounds)
        if not payload["supported_masks"] or any(len(mask) != n or any(type(v) is not bool for v in mask) for mask in payload["supported_masks"]):
            raise ValueError("VALIDATED_FEATURE_MASKS_REQUIRED")
        shapes = dict(price=(3, n + 1), timing=(21, n + 1), fill=(n + 1,), rebound=(n + 1,), joint=(n + 1,))
        for key, shape in shapes.items():
            matrix = np.asarray(payload["heads"].get(key), dtype=object)
            if matrix.shape != shape:
                raise ValueError("INVALID_HEAD_COEFFICIENTS:" + key)
            for value in matrix.flat: finite(value, "coefficient")
        from domain.les_v7.execution_profile import validate_execution_profile
        partition = validate_execution_profile(payload)
        protected = payload["heads"].get("fill_only_base_count")
        if partition is None:
            if protected is not None:
                raise ValueError("FILL_ONLY_HEAD_PROFILE_REQUIRED")
        elif (type(protected) is not int or protected != len(partition[0])
                or any(np.any(np.asarray(payload["heads"][name])[..., protected + 1:] != 0.)
                       for name in ("price", "timing", "rebound", "joint"))):
            raise ValueError("FILL_ONLY_PROTECTED_HEAD_COORDINATES_MISMATCH")
        if "fill_calibration_profile" in payload:
            from domain.les_v7.fill_calibration import validate_profile
            validate_profile(payload["fill_calibration_profile"])
        calibration = payload["calibration"]
        if (not calibration.get("version") or calibration.get("method") != "POST_PROJECTION_PLATT_TEMPERATURE_R1"
                or not calibration.get("validation_report_id") or calibration.get("projection_validated") is not True
                or finite(calibration["timing_temperature"]) <= 0):
            raise ValueError("CALIBRATION_ARTIFACT_REQUIRED")
        for key in ("fill", "rebound", "joint"):
            if len(calibration[key]) != 2:
                raise ValueError("INVALID_PLATT_CALIBRATION")
            for value in calibration[key]: finite(value, "calibration")
        if payload["reference_policy"].get("policy_id") != "REFERENCE_H20":
            raise ValueError("UNSUPPORTED_LABEL_EXIT_POLICY")
        reference = payload["reference_policy"]
        if str(reference.get("version", "")).startswith("FR3_"):
            from domain.les_v7.fr3_plan import is_fr3_reference, validate_policy
            if not is_fr3_reference(reference):
                raise ValueError("UNSUPPORTED_FR3_REFERENCE_POLICY")
            validate_policy(reference)
        if payload.get("training_pipeline_schema") == "LES_V7_OFFLINE_TRAINING_R1":
            reports = payload.get("validation_reports", {})
            report_ids = payload.get("validation_report_ids", {})
            if (payload.get("research_only") is not True or payload.get("approved_for_live") is not False
                    or set(report_ids) != {"calibration", "holdout", "fractal_null", "projection", "ablation"}
                    or any(ref not in reports or identity(reports[ref]) != ref for ref in report_ids.values())
                    or payload["calibration"]["validation_report_id"] != report_ids["calibration"]
                    or payload.get("fractal_null_validation_report_id") != report_ids["fractal_null"]):
                raise ValueError("TRAINING_VALIDATION_REPORT_CONTENT_REQUIRED")
            if timestamp(payload["validation_available_at"]) > timestamp(payload["trained_until"]):
                raise ValueError("TRAINING_VALIDATION_NOT_AVAILABLE_AT_PUBLICATION")
        if "fill_calibration_profile" in payload:
            report = payload.get("validation_reports", {}).get(calibration["validation_report_id"], {})
            if report.get("schema") == "les-v7-compact-validation-report/v2":
                report = report.get("report", {})
            component = report.get("fill_calibration_report", {})
            original = report.get("legacy_calibration_report", {}).get("fitted_parameters", {})
            if (report.get("fill_calibration_profile") != payload["fill_calibration_profile"]
                    or component.get("profile") != payload["fill_calibration_profile"]
                    or component.get("fitted_pair") != calibration["fill"]
                    or component.get("converged") is not True
                    or any(original.get(key) != calibration[key] for key in ("rebound", "joint", "timing_temperature"))
                    or calibration["version"] != str(original.get("version")) + "+" + payload["fill_calibration_profile"]["version"]):
                raise ValueError("FILL_CALIBRATION_COMPONENT_BINDING_MISMATCH")
            if payload["fill_calibration_profile"].get("schema") == "les-v7-fill-calibration-profile/v3":
                from domain.les_v7.fill_slope_calibration import verify_component
                verify_component(component)
            if "alpha" in payload["fill_calibration_profile"]:
                from domain.les_v7.fill_calibration import profile, deploy_fill_pair
                optimizer = component.get("optimizer_fit_report", {})
                if (optimizer.get("profile") != profile() or optimizer.get("converged") is not True
                        or component.get("optimizer_fitted_pair") != optimizer.get("fitted_pair")
                        or component["fitted_pair"] != deploy_fill_pair(optimizer.get("fitted_pair"),
                            payload["fill_calibration_profile"]["alpha"])
                        or any(component.get(key) != optimizer.get(key) for key in
                               ("fit_cutoff", "last_label_available_at", "data_sha256", "row_ids", "row_count"))):
                    raise ValueError("FILL_CALIBRATION_SHRINK_COMPONENT_BINDING_MISMATCH")
        return cls(payload_json, record["sha256"])

    def record(self) -> dict:
        return json.loads(self.payload_json)  # Detached, not a mutable model cache.


@dataclass(frozen=True)
class Forecast:
    status: str
    quantiles: tuple[float, float, float] | None = None
    timing: tuple[float, ...] | None = None
    p_fill: float | None = None
    p_rebound: float | None = None
    p_entry: float | None = None
    raw_log_quantiles: tuple[float, ...] | None = None
    raw_probabilities: tuple[float, ...] | None = None
    repairs: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    calibration_version: str | None = None

    def __post_init__(self):
        if self.status not in {"VALID", "NOT_READY", "INVALID", "OOD"}:
            raise ValueError("INVALID_FORECAST_STATUS")
        if self.status != "VALID": return
        if (self.quantiles is None or len(self.quantiles) != 3 or self.timing is None
                or len(self.timing) != 21 or not self.calibration_version):
            raise ValueError("COMPLETE_CALIBRATED_FORECAST_REQUIRED")
        prices = [finite(v, "bottom_price", 0) for v in self.quantiles]
        if prices[0] <= 0 or prices != sorted(prices): raise ValueError("ORDERED_POSITIVE_QUANTILES_REQUIRED")
        probabilities = [finite(v, "timing_probability", 0, 1) for v in self.timing]
        if abs(math.fsum(probabilities) - 1.) > 1e-9: raise ValueError("TIMING_PMF_MUST_SUM_ONE")
        for value in (self.p_fill, self.p_rebound, self.p_entry): finite(value, "probability", 0, 1)
        if self.p_entry > self.p_rebound: raise ValueError("JOINT_EXCEEDS_REBOUND")

    @property
    def p_soon(self):
        return math.fsum(self.timing[:4]) if self.timing is not None else None

    def record(self):
        return dict(asdict(self), p_bottom_soon=self.p_soon,
                    p_bottom_now=self.timing[0] if self.timing else None,
                    timing_median=int(np.searchsorted(np.cumsum(self.timing), .5)) if self.timing else None,
                    timing_entropy=-math.fsum(p * math.log(p) for p in self.timing if p > 0) if self.timing else None)


def sigmoid(value):
    return float(np.exp(-np.logaddexp(0., -value)))


def predict(bundle: FrozenBundle | None, features: dict, plan: ReferencePlan | None, *, cutoff: str) -> Forecast:
    if bundle is None or plan is None:
        return Forecast("NOT_READY", reasons=("MODEL_OR_REFERENCE_PLAN_NOT_READY",))
    model = bundle.record()
    if timestamp(model["trained_until"]) > timestamp(cutoff):
        return Forecast("NOT_READY", reasons=("MODEL_NOT_AVAILABLE_AT_CUTOFF",))
    policy = model["reference_policy"]
    if plan.policy_id != policy["policy_id"] or plan.policy_version != policy["version"]:
        return Forecast("INVALID", reasons=("FORECAST_SEED_POLICY_MISMATCH",))
    supplied = dict(features, entry=plan.entry, stop=plan.stop, reference_shares=float(plan.reference_shares),
                    entry_stop_atr=plan.stop_atr, **{f"context_{c}": float(c == plan.context) for c in CONTEXTS})
    values = [supplied.get(name) for name in model["feature_names"]]
    mask = [value is not None for value in values]
    if mask not in model["supported_masks"]:
        return Forecast("OOD", reasons=("UNSUPPORTED_FEATURE_VALIDITY_MASK",))
    vector = []
    for i, value in enumerate(values):
        value = model["imputation"][i] if value is None else value
        if value is None:
            return Forecast("NOT_READY", reasons=("FIXED_IMPUTATION_NOT_PROVIDED",))
        value = finite(value, "head_feature")
        lower, upper = model["feature_bounds"][i]
        if not finite(lower) <= value <= finite(upper):
            return Forecast("OOD", reasons=("FEATURE_OUTSIDE_TRAINED_SUPPORT",))
        vector.append(value)
    if any(is_active_fractal_feature(name) for name in model["feature_names"]):
        if not model.get("fractal_null_validation_report_id"):
            return Forecast("NOT_READY", reasons=("FRACTAL_NULL_VALIDATION_NOT_PROVIDED",))
    return project_outputs(head_outputs(model["heads"], vector), features, model["calibration"])


def head_outputs(heads: dict, values: list) -> dict:
    """Shared raw coefficient inference for runtime and past-only OOF fitting.

    This produces logits, not calibrated probabilities or entry permission.
    The caller validates feature support and the fitted coefficient contract.
    """
    v = np.array([1., *[finite(value, "head_feature") for value in values]])
    protected = heads.get("fill_only_base_count")
    if protected is not None:
        if type(protected) is not int or not 1 <= protected < len(values):
            raise ValueError("FILL_ONLY_HEAD_FEATURE_COUNT_INVALID")
        base = v[:protected + 1]
        return dict(log_quantiles=(np.asarray(heads["price"])[..., :protected + 1] @ base).tolist(),
                    timing_logits=(np.asarray(heads["timing"])[..., :protected + 1] @ base).tolist(),
                    raw_logits=dict(fill=float(np.asarray(heads["fill"]) @ v),
                        **{key: float(np.asarray(heads[key])[:protected + 1] @ base)
                           for key in ("rebound", "joint")}))
    return dict(log_quantiles=(np.asarray(heads["price"]) @ v).tolist(),
                timing_logits=(np.asarray(heads["timing"]) @ v).tolist(),
                raw_logits={key: float(np.asarray(heads[key]) @ v) for key in ("fill", "rebound", "joint")})


def model_input_profile(payload: dict, bundle_sha256: str) -> dict:
    """Bound predictor coordinates; legacy declarations retain their exact shape."""
    result = dict(schema="les-v7-model-input-profile/v1", bundle_sha256=bundle_sha256,
        feature_names=list(payload["feature_names"]), fractal_mode=payload.get("fractal_mode"))
    optional = ("execution_feature_profile", "fill_only_feature_extension", "fill_calibration_profile")
    if any(key in payload for key in optional):
        from copy import deepcopy
        result.update(schema="les-v7-model-input-profile/v2", feature_bounds=deepcopy(payload["feature_bounds"]))
        result.update({key: deepcopy(payload[key]) for key in optional if key in payload})
    return result


def project_outputs(outputs: dict, features: dict, calibration: dict) -> Forecast:
    """Exact runtime projections; offline calibration never invents a bundle."""
    raw = np.asarray(outputs["log_quantiles"])
    if not np.isfinite(raw).all():
        return Forecast("INVALID", reasons=("NONFINITE_FORECAST",))
    upper = math.log(features["low"] / features["close"])
    repaired = np.minimum(np.sort(raw), upper)
    prices = features["close"] * np.exp(repaired)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        return Forecast("INVALID", reasons=("INVALID_BOTTOM_PRICE_COORDINATES",))
    logits = np.asarray(outputs["timing_logits"]) / calibration["timing_temperature"]
    if not np.isfinite(logits).all():
        return Forecast("INVALID", reasons=("NONFINITE_TIMING_FORECAST",))
    pmf = np.exp(logits - np.max(logits))
    pmf /= pmf.sum()
    raw_p, probabilities = [], []
    for key in ("fill", "rebound", "joint"):
        logit = outputs["raw_logits"][key]
        if not math.isfinite(logit):
            return Forecast("INVALID", reasons=("NONFINITE_BINARY_FORECAST",))
        raw_p.append(sigmoid(logit))
        slope, intercept = calibration[key]
        calibrated = slope * logit + intercept
        if not math.isfinite(calibrated):
            return Forecast("INVALID", reasons=("NONFINITE_CALIBRATED_FORECAST",))
        probabilities.append(sigmoid(calibrated))
    repairs = []
    if not np.array_equal(repaired, raw): repairs.append("SORTED_KNOWN_LOW_PROJECTION")
    if probabilities[2] > probabilities[1]:
        # Equal-weight bounded isotonic projection for J <= R, frozen R1 policy.
        probabilities[1] = probabilities[2] = (probabilities[1] + probabilities[2]) / 2
        repairs.append("BOUNDED_ISOTONIC_JOINT_REBOUND")
    return Forecast("VALID", tuple(float(x) for x in prices), tuple(float(x) for x in pmf),
                    *probabilities, tuple(float(x) for x in raw), tuple(raw_p), tuple(repairs), (), calibration["version"])


def localization_gate(forecast: Forecast, plan: ReferencePlan, *, confirmed: bool) -> bool:
    if forecast.status != "VALID": return False
    b10, b50, b90 = forecast.quantiles
    return (forecast.p_fill >= .5 and forecast.p_soon >= .5 and forecast.p_entry >= (.6 if confirmed else .5)
            and (b90 - b10) / plan.atr <= 2 and (plan.entry - b50) / plan.atr <= 1)

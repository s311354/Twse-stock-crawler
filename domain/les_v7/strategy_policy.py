"""Immutable causal strategy configuration, separate from fitted model artifacts.

The record's availability is a declared policy publication bound, not financial
source availability or permission to trade. A compatibility extraction from an
old bundle cannot claim that its policy was published before that bundle.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json

from domain.les_v7.contracts import CONTEXTS, FEATURE_VERSION, MAPPING_VERSION, canonical, finite, identity, timestamp
from domain.les_v7.decision import make_plan, rule


STRATEGY_POLICY_SCHEMA = "les-v7-strategy-policy/v1"
POLICY_BLOCKS = ("context_rules", "trigger_rules", "reference_policy", "quality_policy")


def policy_blocks_id(payload: Mapping) -> str:
    """Include all four blocks: fitted quality filters are not interchangeable."""
    return identity({key: payload[key] for key in POLICY_BLOCKS})


def _source_clocks(value, published):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "available_at" and timestamp(item) > published:
                raise ValueError("STRATEGY_POLICY_REFERENCES_FUTURE_SOURCE")
            _source_clocks(item, published)
    elif isinstance(value, (list, tuple)):
        for item in value: _source_clocks(item, published)


@dataclass(frozen=True, slots=True)
class StrategyPolicy:
    payload_json: str
    sha256: str

    @classmethod
    def load(cls, record: Mapping, *, cutoff: str | None = None) -> StrategyPolicy:
        if (not isinstance(record, Mapping) or set(record) != {"payload", "sha256"}
                or not isinstance(record["payload"], Mapping)):
            raise ValueError("EXPLICIT_STRATEGY_POLICY_RECORD_REQUIRED")
        payload = record["payload"]
        required = {"schema", "policy_version", "available_at", "feature_profile", "mapping_version", *POLICY_BLOCKS}
        if set(payload) not in (required, required | {"source_bundle_sha256"}):
            raise ValueError("STRATEGY_POLICY_DEPENDENCIES_INCOMPLETE")
        if record["sha256"] != identity(payload):
            raise ValueError("STRATEGY_POLICY_HASH_MISMATCH")
        if (payload["schema"] != STRATEGY_POLICY_SCHEMA
                or not isinstance(payload["policy_version"], str) or not payload["policy_version"].strip()
                or payload["feature_profile"] != FEATURE_VERSION or payload["mapping_version"] != MAPPING_VERSION):
            raise ValueError("INCOMPATIBLE_STRATEGY_POLICY_VERSION")
        published = timestamp(payload["available_at"])
        if cutoff is not None and published > timestamp(cutoff):
            raise ValueError("STRATEGY_POLICY_NOT_AVAILABLE_AT_CUTOFF")
        if not isinstance(payload["quality_policy"], Mapping) or not payload["quality_policy"]:
            raise ValueError("STRATEGY_QUALITY_POLICY_REQUIRED")
        if set(payload["context_rules"]) != set(CONTEXTS):
            raise ValueError("CONTEXT_RULES_INCOMPLETE")
        if set(payload["trigger_rules"]) != {"near_pullback_support", "tight_base_or_retest"}:
            raise ValueError("TRIGGER_RULES_INCOMPLETE")
        for group in ("context_rules", "trigger_rules"):
            for node in payload[group].values(): rule(node, {})
        make_plan("REVERSAL", dict(close=100., support=90., atr=1., hh20=110.),
                  payload["reference_policy"], "STRATEGY_POLICY_SHAPE_VALIDATION_ONLY")
        finite(payload["reference_policy"].get("leadership_rr_min"), "leadership_rr_min", 0)
        for key in POLICY_BLOCKS: _source_clocks(payload[key], published)
        origin = payload.get("source_bundle_sha256")
        if origin is not None and (not isinstance(origin, str) or len(origin) != 64
                                   or any(c not in "0123456789abcdef" for c in origin)):
            raise ValueError("STRATEGY_POLICY_SOURCE_BUNDLE_ID_INVALID")
        encoded = canonical(payload)
        if len(encoded.encode("utf-8")) > 256 * 1024:
            raise ValueError("STRATEGY_POLICY_EXCEEDS_SIZE_LIMIT")
        return cls(encoded, record["sha256"])

    @classmethod
    def from_bundle(cls, bundle) -> StrategyPolicy:
        """Compatibility policy; does not change an existing bundle or report."""
        payload = bundle.record()
        return cls.load(make_strategy_policy(
            policy_version="BUNDLE_COMPATIBILITY_R1", available_at=payload["trained_until"],
            source_bundle_sha256=bundle.sha256, **{key: payload[key] for key in POLICY_BLOCKS}))

    def record(self) -> dict:
        return json.loads(self.payload_json)

    def export(self) -> dict:
        return dict(payload=self.record(), sha256=self.sha256)

    def bind_bundle(self, bundle) -> None:
        model, payload = bundle.record(), self.record()
        if policy_blocks_id(payload) != policy_blocks_id(model):
            raise ValueError("STRATEGY_POLICY_MODEL_BINDING_MISMATCH")
        if payload.get("source_bundle_sha256") is not None:
            if payload["source_bundle_sha256"] != bundle.sha256:
                raise ValueError("STRATEGY_POLICY_SOURCE_BUNDLE_MISMATCH")
            if timestamp(payload["available_at"]) < timestamp(model["trained_until"]):
                raise ValueError("STRATEGY_POLICY_COMPATIBILITY_CANNOT_BACKDATE")


def make_strategy_policy(*, policy_version: str, available_at: str, context_rules: Mapping,
                         trigger_rules: Mapping, reference_policy: Mapping, quality_policy: Mapping,
                         source_bundle_sha256: str | None = None) -> dict:
    """Explicit construction only; no runtime weights, trained heads or priors."""
    payload = dict(schema=STRATEGY_POLICY_SCHEMA, policy_version=policy_version,
                   available_at=timestamp(available_at).isoformat(), feature_profile=FEATURE_VERSION,
                   mapping_version=MAPPING_VERSION, context_rules=context_rules, trigger_rules=trigger_rules,
                   reference_policy=reference_policy, quality_policy=quality_policy)
    if source_bundle_sha256 is not None: payload["source_bundle_sha256"] = source_bundle_sha256
    record = dict(payload=payload, sha256=identity(payload))
    return StrategyPolicy.load(record).export()

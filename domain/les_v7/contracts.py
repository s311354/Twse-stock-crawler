"""Frozen v7 contracts and source-derived constants (IMP-R1)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from collections.abc import Mapping

SPEC_ID = "LES_V7_FABL_1.3_IMPLEMENTATION_SPEC"
SPEC_SHA256 = "155b814b76487f72ff6ff6d489568e1e86fe9cbe1a02d152367e453954115d6e"
FEATURE_VERSION = "V7_BASE_FRACTAL_R1"
MAPPING_VERSION = "BL_MAPPING_B1"
CONTEXTS = ("REVERSAL", "TREND_PULLBACK", "LEADERSHIP_CONTINUATION")
FACTORS = ("structure", "trend", "relative_strength", "demand", "momentum", "execution")
PRIOR = (.25, .15, .20, .20, .10, .10)  # Research prior, NEVER a runtime fallback.
QUALITY_R2_PROFILE = "QG_V7_EVIDENCE_R2"
QUALITY_R3_PROFILE = "QG_V7_REPORTED_PERIOD_R3"
QUALITY_R4_PROFILE = "QG_V7_REPORTED_PERIOD_R4"
QUALITY_R5_PROFILE = "QG_V7_REPORTED_PERIOD_R5"
QUALITY_R6_PROFILE = "QG_V7_REPORTED_PERIOD_R6"
QUALITY_R7_PROFILE = "QG_V7_REPORTED_PERIOD_R7"
QUALITY_REPORTED_PROFILES = (QUALITY_R3_PROFILE, QUALITY_R4_PROFILE, QUALITY_R5_PROFILE, QUALITY_R6_PROFILE, QUALITY_R7_PROFILE)


def quality_extension_schema(quality_profile):
    return {QUALITY_R2_PROFILE: "v7-quality-evidence/v2",
            QUALITY_R3_PROFILE: "v7-quality-evidence/v3",
            QUALITY_R4_PROFILE: "v7-quality-evidence/v4",
            QUALITY_R5_PROFILE: "v7-quality-evidence/v5",
            QUALITY_R6_PROFILE: "v7-quality-evidence/v6",
            QUALITY_R7_PROFILE: "v7-quality-evidence/v7"}.get(quality_profile)


def envelope_schema_version(quality_profile):
    return {QUALITY_R3_PROFILE: "LES_V7_ENVELOPE_3",
            QUALITY_R4_PROFILE: "LES_V7_ENVELOPE_4",
            QUALITY_R5_PROFILE: "LES_V7_ENVELOPE_5",
            QUALITY_R6_PROFILE: "LES_V7_ENVELOPE_6",
            QUALITY_R7_PROFILE: "LES_V7_ENVELOPE_7"}.get(quality_profile, "LES_V7_ENVELOPE_1")


def parameter_version(quality_profile=None):
    base = SPEC_ID + ":" + MAPPING_VERSION
    return base + ":" + quality_profile if quality_extension_schema(quality_profile) else base


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def identity(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def finite(value: object, name: str = "value", lo=None, hi=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name}: FINITE_NUMBER_REQUIRED")
    result = float(value)
    if lo is not None and result < lo or hi is not None and result > hi:
        raise ValueError(f"{name}: OUT_OF_RANGE")
    return result


def timestamp(value: str | datetime) -> datetime:
    date = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None or date.utcoffset() is None:
        raise ValueError("TIMEZONE_REQUIRED")
    return date.astimezone(timezone.utc)


def ramp(value: float | None, lower: float, upper: float) -> float | None:
    if value is None:
        return None
    if upper <= lower:
        raise ValueError("INVALID_RAMP_BOUNDS")
    return min(1., max(0., (finite(value) - lower) / (upper - lower)))


@dataclass(frozen=True)
class GateReceipt:
    gate_id: str
    gate_version: str
    state: str
    input_hash: str
    source_available_at: str | None
    reasons: tuple[str, ...]
    proof_record_ids: tuple[str, ...]
    input_ids: tuple[str, ...] = ()
    checked_at: str | None = None

    @classmethod
    def bind(cls, record: Mapping | None, *, gate_id: str, inputs: object, cutoff: str) -> GateReceipt:
        """Verify a upstream port receipt binding, not an official-source signature."""
        if record is None:
            return cls(gate_id, "UNAVAILABLE", "UNKNOWN", identity(inputs), None,
                       (f"{gate_id}_EVIDENCE_MISSING",), ())
        if (record.get("gate_id") != gate_id or not record.get("gate_version")
                or record.get("input_hash") != identity(inputs)
                or not record.get("proof_record_ids")
                or not record.get("input_ids") or not record.get("checked_at")
                or timestamp(record["source_available_at"]) > timestamp(cutoff)):
            raise ValueError(f"{gate_id}_RECEIPT_BINDING_INVALID")
        if record.get("state") not in {"PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE"}:
            raise ValueError("INVALID_GATE_STATE")
        if timestamp(record["checked_at"]) < timestamp(record["source_available_at"]):
            raise ValueError("INVALID_GATE_CHECK_TIME")
        return cls(gate_id, record["gate_version"], record["state"], record["input_hash"],
                   record["source_available_at"], tuple(record.get("reason_codes", ())),
                   tuple(record["proof_record_ids"]), tuple(record["input_ids"]), record["checked_at"])


@dataclass(frozen=True)
class ReferencePlan:
    policy_id: str
    policy_version: str
    evidence_id: str
    context: str
    entry: float
    stop: float
    target: float | None
    atr: float
    band: tuple[float, float]
    reference_shares: int
    cost_up: float
    cost_stop: float
    horizon: int = 20
    expiry: str = "NEXT_SESSION_CLOSE"

    def __post_init__(self):
        expected_expiry = ("THIRD_OFFICIAL_SESSION_CLOSE"
            if self.policy_version == "FR3_SOURCE_BOUND_LIMIT_H20_L2"
            and self.context == "LEADERSHIP_CONTINUATION" else "NEXT_SESSION_CLOSE")
        if (self.policy_id != "REFERENCE_H20" or not self.policy_version or self.context not in CONTEXTS
                or self.horizon != 20 or self.expiry != expected_expiry):
            raise ValueError("INVALID_REFERENCE_POLICY")
        entry, stop, atr = (finite(value, "reference_price", 0) for value in (self.entry, self.stop, self.atr))
        if not 0 < stop < entry or atr <= 0: raise ValueError("POSITIVE_REFERENCE_RISK_REQUIRED")
        if type(self.reference_shares) is not int or self.reference_shares <= 0:
            raise ValueError("POSITIVE_REFERENCE_SHARES_REQUIRED")
        if len(self.band) != 2 or not 0 < finite(self.band[0]) <= finite(self.band[1]):
            raise ValueError("INVALID_REFERENCE_ENTRY_BAND")
        if self.target is not None and finite(self.target) <= entry:
            raise ValueError("TARGET_MUST_EXCEED_ENTRY")
        finite(self.cost_up, "cost_up", 0)
        finite(self.cost_stop, "cost_stop", 0)

    @property
    def plan_id(self) -> str:
        from dataclasses import asdict
        return identity(asdict(self))

    @property
    def stop_atr(self) -> float:
        return (self.entry - self.stop) / self.atr

    @property
    def net_rr(self) -> float | None:
        if self.target is None:
            return None
        risk = self.entry - self.stop + self.cost_stop
        return (self.target - self.entry - self.cost_up) / risk if risk > 0 else None

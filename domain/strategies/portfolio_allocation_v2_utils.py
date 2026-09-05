"""Small deterministic helpers shared by PAE v2 policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict


def normalized_confidence(value: float | None) -> float:
    if value is None:
        return 0.0
    parsed = float(value)
    return parsed if parsed <= 1 else parsed / 100.0


def stable_dataclass_hash(value: object) -> str:
    payload = json.dumps(
        asdict(value),
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["normalized_confidence", "stable_dataclass_hash"]

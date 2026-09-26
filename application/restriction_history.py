"""Import a source-backed historical ledger; never download during replay."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Iterable

from domain.restriction_announcements import (
    BUNDLE_SCHEMA, HISTORICAL_SCOPE, check_history, timestamp, validate_ledger,
)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class HistoricalRestrictionLedger:
    """Raw source validation once per batch, compact proof per symbol/report.

    Hashes prove local integrity, not authenticity or completeness of a source.
    VERIFIED coverage additionally requires a documented human/provider review.
    """

    def __init__(self, bundle: dict):
        validate_ledger(bundle)
        for doc in bundle["documents"].values():
            body = doc.get("body_utf8")
            if not isinstance(body, str) or hashlib.sha256(body.encode("utf-8")).hexdigest() != doc["sha256"]:
                raise ValueError("HISTORICAL_DOCUMENT_HASH_MISMATCH")
        for item in bundle["coverage"] + bundle["announcements"]:
            for proof in item.get("proofs", []):
                doc = bundle["documents"].get(proof.get("document_id"), {})
                if not proof.get("quote") or proof["quote"] not in doc.get("body_utf8", ""):
                    raise ValueError("HISTORICAL_PROOF_NOT_IN_SOURCE")
        self.bundle_hash = canonical_hash(bundle)
        self._bundle = deepcopy(bundle)

    @classmethod
    def from_file(cls, path: str | Path) -> "HistoricalRestrictionLedger":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def reconstruct(self, symbols: Iterable[str], *, market: str, as_of: object,
                    market_data_date: str) -> dict[str, dict]:
        if market != self._bundle["market"]:
            raise ValueError("HISTORICAL_RESTRICTION_MARKET_MISMATCH")
        cutoff = timestamp(str(as_of)).isoformat()
        output = {}
        for symbol in sorted(set(map(str, symbols))):
            records = [deepcopy(row) for row in self._bundle["announcements"] if row["symbol"] == symbol]
            coverage = deepcopy(self._bundle["coverage"])
            ids = {ref["document_id"] for row in coverage + records for ref in row.get("proofs", [])}
            receipts = {key: {field: self._bundle["documents"][key][field]
                              for field in ("url", "sha256", "retrieved_at")} for key in sorted(ids)}
            evidence = {"schema_version": BUNDLE_SCHEMA, "scope": HISTORICAL_SCOPE,
                        "market": market, "symbol": symbol, "reconstruction_as_of": cutoff,
                        "bundle_sha256": self.bundle_hash, "documents": receipts,
                        "coverage": coverage, "announcements": records}
            # Report diagnostics are derived, never input to the restriction gate.
            evidence["reconstruction_result"] = check_history(
                evidence, market=market, market_data_date=market_data_date, decision_time=cutoff)
            output[symbol] = evidence
        return output

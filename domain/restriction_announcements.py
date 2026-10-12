"""Point-in-time restriction reconstruction, separate from live observations.

Coverage is an evidence-provider attestation, not something inferred from an
empty query. Raw documents and quoted proof are checked at the import boundary.
This compact ledger is re-evaluated when a serialized research report is read.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import re
from urllib.parse import urlparse

import pandas as pd


BUNDLE_SCHEMA = "restriction-announcements/v1"
HISTORICAL_SCOPE = "HISTORICAL_ANNOUNCEMENTS"
CATEGORIES = ("TRADING_HALT", "ALTERED_TRADING", "DISPOSITION")
OFFICIAL_HOSTS = {"TWSE": {"www.twse.com.tw", "twse.com.tw"},
                  "TPEX": {"www.tpex.org.tw", "tpex.org.tw"}}


def timestamp(value: object) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ANNOUNCEMENT_TIMESTAMP_REQUIRED")
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("ANNOUNCEMENT_TIMESTAMP_MUST_BE_AWARE")
    return result


def publication_bounds(record: Mapping) -> tuple[pd.Timestamp, pd.Timestamp]:
    value = record.get("published_at")
    if record.get("publication_precision") == "DATE":
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("INVALID_PUBLICATION_DATE")
        first = pd.Timestamp(value).tz_localize("Asia/Taipei")
        # No guessed midnight release: same-date publication is ambiguous until
        # the next midnight. A verified exact timestamp can resolve it earlier.
        return first, first + pd.Timedelta(days=1)
    if record.get("publication_precision") != "TIMESTAMP":
        raise ValueError("PUBLICATION_PRECISION_REQUIRED")
    point = timestamp(value)
    return point, point


def validate_receipt(receipt: Mapping, market: str) -> None:
    if not isinstance(receipt.get("url"), str) or not isinstance(receipt.get("sha256"), str):
        raise ValueError("INVALID_ANNOUNCEMENT_RECEIPT")
    url = urlparse(receipt.get("url", ""))
    if (url.scheme != "https" or url.hostname not in OFFICIAL_HOSTS.get(market, set())
            or url.username or url.password or url.port not in (None, 443)):
        raise ValueError("NON_OFFICIAL_ANNOUNCEMENT_SOURCE")
    if not re.fullmatch(r"[0-9a-f]{64}", receipt.get("sha256", "")):
        raise ValueError("INVALID_ANNOUNCEMENT_HASH")
    timestamp(receipt.get("retrieved_at"))


def validate_ledger(ledger: Mapping) -> None:
    """Validate the normalized contract, without assigning eligibility."""
    if not isinstance(ledger, Mapping):
        raise ValueError("INVALID_ANNOUNCEMENT_BUNDLE")
    if ledger.get("schema_version") != BUNDLE_SCHEMA or ledger.get("market") not in OFFICIAL_HOSTS:
        raise ValueError("INVALID_ANNOUNCEMENT_SCHEMA_OR_MARKET")
    receipts = ledger.get("documents")
    if not isinstance(receipts, dict):
        raise ValueError("ANNOUNCEMENT_DOCUMENTS_REQUIRED")
    for receipt in receipts.values():
        if not isinstance(receipt, Mapping):
            raise ValueError("INVALID_ANNOUNCEMENT_RECEIPT")
        validate_receipt(receipt, ledger["market"])

    def proofs(row):
        refs = row.get("proofs")
        if not isinstance(refs, list) or not refs:
            raise ValueError("ANNOUNCEMENT_SOURCE_PROOF_REQUIRED")
        for ref in refs:
            if (not isinstance(ref, Mapping) or ref.get("document_id") not in receipts
                    or not isinstance(ref.get("quote"), str) or not ref["quote"].strip()):
                raise ValueError("INVALID_ANNOUNCEMENT_SOURCE_PROOF")

    coverage, announcements = ledger.get("coverage"), ledger.get("announcements")
    if not isinstance(coverage, list) or not isinstance(announcements, list):
        raise ValueError("ANNOUNCEMENT_LEDGER_LISTS_REQUIRED")
    seen_categories = set()
    for item in coverage:
        if not isinstance(item, Mapping) or item.get("category") not in CATEGORIES:
            raise ValueError("INVALID_COVERAGE_CATEGORY")
        if item["category"] in seen_categories:
            raise ValueError("CONFLICTING_CATEGORY_COVERAGE")
        seen_categories.add(item["category"])
        if timestamp(item.get("start_at")) > timestamp(item.get("end_at")):
            raise ValueError("INVALID_COVERAGE_RANGE")
        if item.get("status") not in {"VERIFIED", "UNVERIFIED"}:
            raise ValueError("COVERAGE_STATUS_REQUIRED")
        if item["status"] == "VERIFIED":
            if (any(item.get(key) is not True for key in
                    ("complete", "includes_carry_in", "includes_all_revisions"))
                    or item.get("universe_scope") != "MARKET_WIDE"
                    or not isinstance(item.get("verified_by"), str) or not item["verified_by"].strip()):
                raise ValueError("COMPLETE_COVERAGE_ATTESTATION_REQUIRED")
            timestamp(item.get("verified_at"))
            proofs(item)
        elif item.get("proofs"):
            proofs(item)
    seen, identities = set(), {}
    for row in announcements:
        if (not isinstance(row, Mapping) or row.get("category") not in CATEGORIES
                or not isinstance(row.get("id"), str) or not row["id"].strip()
                or not isinstance(row.get("symbol"), str)
                or not re.fullmatch(r"[A-Za-z0-9]{4,12}", row["symbol"])
                or type(row.get("version")) is not int or row["version"] < 1
                or type(row.get("withdrawn")) is not bool):
            raise ValueError("INVALID_ANNOUNCEMENT_RECORD")
        key = row["category"], row["id"], row["version"]
        if key in seen:
            raise ValueError("DUPLICATE_ANNOUNCEMENT_VERSION")
        seen.add(key)
        identity = row["category"], row["id"]
        if identity in identities and identities[identity] != row["symbol"]:
            raise ValueError("ANNOUNCEMENT_IDENTITY_CHANGED")
        identities[identity] = row["symbol"]
        start = timestamp(row.get("effective_from"))
        if row.get("effective_until") is not None and timestamp(row["effective_until"]) <= start:
            raise ValueError("INVALID_EFFECTIVE_INTERVAL")
        first, _ = publication_bounds(row)
        proofs(row)
        # Download time is not publication time, but a document cannot be
        # retrieved before the earliest possible publication of its version.
        if any(timestamp(receipts[ref["document_id"]]["retrieved_at"]) < first for ref in row["proofs"]):
            raise ValueError("DOCUMENT_PREDATES_PUBLICATION")


def check_history(evidence: Mapping, *, market: str, market_data_date: str,
                  decision_time: object) -> dict:
    result = {"status": "UNKNOWN", "restricted": None, "reason": "INVALID_HISTORICAL_RESTRICTION_EVIDENCE",
              "market_data_date": market_data_date, "observation_date": None,
              "decision_date": None, "scope": HISTORICAL_SCOPE, "categories": {}}

    def fail(reason):
        result["reason"] = reason
        return result

    try:
        cutoff = timestamp(str(decision_time))
        bar = pd.Timestamp(market_data_date)
        if pd.isna(bar) or bar.tzinfo is not None or bar != bar.normalize():
            return fail("INVALID_RESTRICTION_TIMESTAMP")
        market_close = bar.tz_localize("Asia/Taipei") + pd.Timedelta(hours=13, minutes=30)
        if market_close > cutoff:
            return fail("INCOMPLETE_OR_FUTURE_MARKET_DATA")
        result["market_effective_at"] = market_close.isoformat()
        result["decision_effective_at"] = cutoff.isoformat()
        result["decision_date"] = cutoff.tz_convert("Asia/Taipei").date().isoformat()
        if evidence.get("market") != market:
            return fail("RESTRICTION_MARKET_MISMATCH")
        if timestamp(evidence.get("reconstruction_as_of")) != cutoff:
            return fail("HISTORICAL_RESTRICTION_CUTOFF_MISMATCH")
        if not re.fullmatch(r"[0-9a-f]{64}", evidence.get("bundle_sha256", "")):
            return fail("HISTORICAL_BUNDLE_HASH_REQUIRED")
        validate_ledger(evidence)
        symbol = evidence.get("symbol")
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z0-9]{4,12}", symbol):
            return fail("INVALID_HISTORICAL_RESTRICTION_SYMBOL")
        coverage = {row["category"]: row for row in evidence["coverage"]}
        for category in CATEGORIES:
            state = {"status": "UNKNOWN", "reason": "INCOMPLETE_HISTORICAL_COVERAGE", "active_ids": [],
                     "market_active_ids": [], "decision_active_ids": []}
            result["categories"][category] = state
            covered = coverage.get(category, {})
            complete = (covered.get("status") == "VERIFIED"
                        and timestamp(covered["start_at"]) <= market_close <= cutoff <= timestamp(covered["end_at"]))
            groups = defaultdict(list)
            for row in evidence["announcements"]:
                if row["category"] == category:
                    groups[row["id"]].append(row)
            uncertain = False
            for records in groups.values():
                records.sort(key=lambda row: row["version"])
                if len({row["symbol"] for row in records}) != 1:
                    return fail("ANNOUNCEMENT_IDENTITY_CHANGED")
                if records[0]["symbol"] != symbol:
                    continue
                versions = [row["version"] for row in records]
                if versions != list(range(1, max(versions) + 1)):
                    uncertain = True
                    state["reason"] = "MISSING_ORIGINAL_ANNOUNCEMENT_VERSION"
                    continue
                bounds = [publication_bounds(row) for row in records]
                if any(a[0] > b[0] for a, b in zip(bounds, bounds[1:])):
                    return fail("NON_MONOTONIC_ANNOUNCEMENT_VERSIONS")
                eligible = []
                for row, (first, last) in zip(records, bounds):
                    if first <= cutoff < last:
                        uncertain = True
                        state["reason"] = "AMBIGUOUS_PUBLICATION_TIME"
                    elif last <= cutoff:
                        eligible.append(row)
                if eligible:
                    current = eligible[-1]
                    end = current.get("effective_until")
                    # Publication eligibility uses the research clock. Coverage
                    # and restrictions must also apply to the actual signal bar;
                    # a later weekend release cannot erase that day's restriction.
                    if not current["withdrawn"]:
                        for at, key in ((market_close, "market_active_ids"), (cutoff, "decision_active_ids")):
                            if (timestamp(current["effective_from"]) <= at
                                    and (end is None or at < timestamp(end))):
                                state[key].append(current["id"])
                        if current["id"] in state["market_active_ids"] or current["id"] in state["decision_active_ids"]:
                            state["active_ids"].append(current["id"])
            # Known active restrictions still block when another category is
            # incomplete. Ambiguous revisions cannot certify a restriction ended.
            if state["active_ids"]:
                state.update(status="RESTRICTED", reason="HISTORICAL_ACTIVE_ANNOUNCEMENT")
            elif complete and not uncertain:
                state.update(status="CLEAR", reason="COMPLETE_HISTORICAL_COVERAGE")
        statuses = [row["status"] for row in result["categories"].values()]
        if "RESTRICTED" in statuses:
            result.update(status="RESTRICTED", restricted=True, reason="HISTORICAL_RESTRICTION_CONFIRMED")
        elif all(status == "CLEAR" for status in statuses):
            result.update(status="CLEAR", restricted=False, reason="HISTORICAL_COVERAGE_CONFIRMED")
        else:
            result["reason"] = "HISTORICAL_RESTRICTION_COVERAGE_OR_TIME_UNKNOWN"
    except (ValueError, TypeError, KeyError, OverflowError):
        return fail("INVALID_HISTORICAL_RESTRICTION_EVIDENCE")
    return result

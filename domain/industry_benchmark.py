"""Cutoff-selected official industry index and membership evidence.

Normalized receipts are supplied by the application import boundary. An index
hash is integrity, not a claim of historical publication or source authenticity.
"""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json

import pandas as pd

from domain.models import Stocktype
from domain.restriction_announcements import publication_bounds, timestamp, validate_receipt
from domain.strategies.low_entry_v6_quality import number

SCHEMA = "industry-benchmark/v1"
SOURCE_SCHEMA = "industry-benchmark-source/v1"
INDUSTRY_BENCHMARK_ATTR = "les_v6_industry_benchmark"


def identity(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sector_code(value: object) -> str:
    name = str(value).upper().removeprefix("STOCKTYPE.")
    member = Stocktype.__members__.get(name)
    return str(member.value[0]) if member is not None else name


def source_binding(packet: Mapping, kind: str, row: Mapping) -> dict:
    fields = (("version", "effective_from", "effective_until", "members") if kind == "membership" else
              ("trade_date", "close", "membership_version", "revision", "series_version"))
    return dict(kind=kind, market=packet["market"], sector=sector_code(packet["sector"]),
                benchmark_id=packet["benchmark_id"], source_sha256=row["document_sha256"],
                **{key: row.get(key) for key in fields})


def validate_packet(packet: Mapping) -> None:
    if (packet.get("schema") != SCHEMA or packet.get("market") not in {"TWSE", "TPEX"}
            or packet.get("kind") != "OFFICIAL_INDUSTRY_INDEX"
            or not isinstance(packet.get("benchmark_id"), str) or not packet["benchmark_id"]
            or not isinstance(packet.get("series_version"), str) or not packet["series_version"]
            or not sector_code(packet.get("sector")).isdecimal()):
        raise ValueError("INVALID_INDUSTRY_BENCHMARK_CONTRACT")
    for kind, rows in (("membership", packet["memberships"]), ("bar", packet["bars"])):
        if not isinstance(rows, list):
            raise ValueError("INDUSTRY_EVIDENCE_ROWS_REQUIRED")
        for row in rows:
            validate_receipt(row["receipt"], packet["market"])
            available = timestamp(row["available_at"])
            if row.get("binding_sha256") != identity(source_binding(packet, kind, row)):
                raise ValueError("INDUSTRY_SOURCE_BINDING_MISMATCH")
            if row.get("availability_source") == "FIRST_OBSERVED_DOWNLOAD":
                if available != timestamp(row["receipt"]["retrieved_at"]):
                    raise ValueError("INDUSTRY_OBSERVATION_CANNOT_BE_BACKDATED")
            elif row.get("availability_source") == "OFFICIAL_PUBLICATION_PROOF":
                evidence = row["publication_evidence"]
                if available != publication_bounds(evidence)[1]:
                    raise ValueError("INDUSTRY_PROOF_TIME_MISMATCH")
                timestamp(evidence["verified_at"])
                validate_receipt(evidence["receipt"], packet["market"])
                if not evidence.get("verified_by") or not evidence.get("verification_method"):
                    raise ValueError("INDUSTRY_PROOF_REVIEW_REQUIRED")
            else:
                raise ValueError("INDUSTRY_AVAILABILITY_NOT_VERIFIED")
            if row["document_sha256"] != row["receipt"]["sha256"]:
                raise ValueError("INDUSTRY_DOCUMENT_HASH_MISMATCH")
            if kind == "membership":
                start = pd.Timestamp(row["effective_from"])
                end = pd.Timestamp(row["effective_until"]) if row.get("effective_until") else None
                if (not isinstance(row["version"], str) or not row["version"] or start.tzinfo is not None
                        or pd.isna(start) or start != start.normalize()
                        or (end is not None and (pd.isna(end) or end.tzinfo is not None
                                                or end != end.normalize() or end <= start))
                        or not isinstance(row["members"], list) or not row["members"]
                        or any(not isinstance(s, str) or not s for s in row["members"])
                        or len(row["members"]) != len(set(row["members"]))):
                    raise ValueError("INVALID_INDUSTRY_MEMBERSHIP")
            else:
                day = pd.Timestamp(row["trade_date"])
                if (pd.isna(day) or day.tzinfo is not None or day != day.normalize()
                        or day.date().isoformat() != row["trade_date"]
                        or number(row["close"]) is None or number(row["close"]) <= 0
                        or type(row["revision"]) is not int or row["revision"] < 0
                        or row["series_version"] != packet["series_version"]
                        or available < day.tz_localize("Asia/Taipei") + pd.Timedelta(hours=13, minutes=30)):
                    raise ValueError("INVALID_INDUSTRY_INDEX_BAR")


def aligned_index(packet: Mapping | None, dates, *, symbol: str, sector: str,
                  market: str, as_of: object) -> tuple[pd.Series, dict]:
    """No ffill, current-membership substitution, or implicit peer basket.

    Membership must have been known by that session's close. Revised index
    observations may be selected only after their own availability time.
    """
    points = pd.Series(float("nan"), index=dates.index, dtype=float)
    result = dict(status="UNKNOWN", reason="INDUSTRY_BENCHMARK_MISSING", observations=[],
                  benchmark_id=None, series_version=None, selected_sha256=None)
    if not packet:
        return points, result
    try:
        validate_packet(packet)
        cutoff = timestamp(str(as_of))
        if packet["market"] != market or sector_code(packet["sector"]) != sector_code(sector):
            raise ValueError("INDUSTRY_MARKET_OR_SECTOR_MISMATCH")
        selected, gaps = [], []
        bars_by_date: dict[str, list] = {}
        for row in packet["bars"]:
            bars_by_date.setdefault(row["trade_date"], []).append(row)
        for index, day in dates.items():
            day = pd.Timestamp(day)
            close_at = day.tz_localize("Asia/Taipei") + pd.Timedelta(hours=13, minutes=30)
            memberships = [row for row in packet["memberships"] if
                timestamp(row["available_at"]) <= min(cutoff, close_at)
                and pd.Timestamp(row["effective_from"]) <= day
                and (not row.get("effective_until") or day < pd.Timestamp(row["effective_until"]))]
            if not memberships:
                gaps.append(dict(date=str(day.date()), reason="MEMBERSHIP_NOT_AVAILABLE")); continue
            # A late re-download of an old membership must not replace a newer
            # effective version. Equivalent receipts are not competing versions.
            newest = max((row["effective_from"], timestamp(row["available_at"])) for row in memberships)
            winners = {row["binding_sha256"]: row for row in sorted(memberships, key=identity, reverse=True)
                       if (row["effective_from"], timestamp(row["available_at"])) == newest}
            if len(winners) != 1:
                gaps.append(dict(date=str(day.date()), reason="MEMBERSHIP_CONFLICT")); continue
            membership = next(iter(winners.values()))
            if symbol not in membership["members"]:
                gaps.append(dict(date=str(day.date()), reason="SYMBOL_NOT_IN_HISTORICAL_MEMBERSHIP")); continue
            bars = [row for row in bars_by_date.get(str(day.date()), [])
                    if row["membership_version"] == membership["version"] and timestamp(row["available_at"]) <= cutoff]
            if not bars:
                gaps.append(dict(date=str(day.date()), reason="INDEX_BAR_NOT_AVAILABLE")); continue
            order = max((row["revision"], timestamp(row["available_at"])) for row in bars)
            winners = {row["binding_sha256"]: row for row in sorted(bars, key=identity, reverse=True)
                       if (row["revision"], timestamp(row["available_at"])) == order}
            if len(winners) != 1:
                gaps.append(dict(date=str(day.date()), reason="INDEX_VERSION_CONFLICT")); continue
            row = next(iter(winners.values()))
            points.loc[index] = number(row["close"])
            selected.append(dict(date=str(day.date()), close=row["close"], revision=row["revision"],
                available_at=row["available_at"], availability_source=row["availability_source"],
                index_binding_sha256=row["binding_sha256"], membership_version=membership["version"],
                membership_available_at=membership["available_at"], membership_binding_sha256=membership["binding_sha256"],
                index_receipt=row["receipt"], membership_receipt=membership["receipt"],
                index_publication_evidence=row.get("publication_evidence"),
                membership_publication_evidence=membership.get("publication_evidence")))
        result.update(status="PASS" if not gaps and selected else "UNKNOWN",
                      reason="DATE_ALIGNED_OFFICIAL_INDEX" if not gaps and selected else "INDUSTRY_HISTORY_INCOMPLETE",
                      benchmark_id=packet["benchmark_id"], series_version=packet["series_version"],
                      observations=selected, gaps=gaps, selected_sha256=identity(selected))
    except (KeyError, TypeError, ValueError) as error:
        points[:] = float("nan")
        result["reason"] = str(error)
    return points, result

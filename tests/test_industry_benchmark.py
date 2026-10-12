"""TEST_ONLY synthetic reviewed documents; never official historical evidence."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

from application.industry_benchmark import prepare_industry_benchmark, prepare_industry_context
from domain.industry_benchmark import SOURCE_SCHEMA, aligned_index, source_binding


def packet(dates=None, closes=None, *, proof=True):
    dates = list(dates) if dates is not None else pd.bdate_range("2026-06-19", periods=61)
    closes = list(closes) if closes is not None else np.linspace(100, 101, len(dates))
    result = dict(schema=SOURCE_SCHEMA, market="TWSE", sector="24", kind="OFFICIAL_INDUSTRY_INDEX",
                  benchmark_id="TEST_ONLY_SEMI", series_version="TEST_ONLY_v1", documents={},
                  memberships=[], bars=[])
    member = dict(version="TEST_ONLY_members_v1", effective_from="2026-01-01", effective_until=None,
                  members=["2330"], published_at="2025-12-31T18:00:00+08:00")
    add_row(result, "membership", member, proof=proof)
    for day, close in zip(dates, closes, strict=True):
        day = pd.Timestamp(day).date().isoformat()
        add_row(result, "bar", dict(trade_date=day, close=float(close), membership_version=member["version"],
            revision=0, series_version=result["series_version"], published_at=day+"T13:30:00+08:00"), proof=proof)
    return result


def add_row(packet, kind, row, *, proof=True):
    row = deepcopy(row)
    published = row.pop("published_at")
    body = "TEST_ONLY " + json.dumps(row, sort_keys=True) + " published=" + published
    digest = sha256(body.encode()).hexdigest()
    doc = dict(url="https://www.twse.com.tw/TEST_ONLY", body_utf8=body, sha256=digest,
               retrieved_at="2026-10-02T15:00:00+08:00")
    packet["documents"][digest] = doc
    row["document_sha256"] = digest
    binding = source_binding(packet, kind, row)
    row.update(source_binding=binding, extraction_verified_by="TEST_ONLY_reviewer", quotes=[body])
    if proof:
        row["publication_proof"] = dict(status="VERIFIED", verified_by="TEST_ONLY_reviewer",
            verification_method="TEST_ONLY_EXACT_SOURCE_VERSION", verified_at="2026-10-02T16:00:00+08:00",
            binding=binding, document=doc, quotes=[body], published_at=published, publication_precision="TIMESTAMP")
    packet["memberships" if kind == "membership" else "bars"].append(row)


def select(raw, *, dates=None, cutoff="2026-09-11T13:30:00+08:00", symbol="2330", sector="SEMI"):
    dates = pd.Series(pd.to_datetime(dates if dates is not None else sorted({r["trade_date"] for r in raw["bars"]})))
    return aligned_index(prepare_industry_benchmark(raw), dates, symbol=symbol, sector=sector,
                         market="TWSE", as_of=cutoff)


def test_late_download_without_proof_is_not_backdated_but_valid_version_proof_is_usable():
    raw = packet()
    values, evidence = select(raw)
    assert evidence["status"] == "PASS" and values.notna().all()
    observed = packet(proof=False)
    observed["memberships"][0]["available_at"] = "2025-01-01T00:00+08:00"
    points, blocked = select(observed)
    assert points.isna().all() and blocked["status"] == "UNKNOWN"
    assert all(g["reason"] == "MEMBERSHIP_NOT_AVAILABLE" for g in blocked["gaps"])


@pytest.mark.parametrize("part,field,value", [
    ("memberships", "members", ["OTHER"]), ("memberships", "version", "OTHER"),
    ("bars", "close", 999.), ("bars", "revision", 2), ("bars", "trade_date", "2026-01-01"),
])
def test_source_binding_detects_wrong_version_or_extraction(part, field, value):
    raw = packet()
    raw[part][0][field] = value
    with pytest.raises(ValueError, match="EXTRACTION_NOT_VERIFIED"):
        prepare_industry_benchmark(raw)


def test_proof_binding_and_source_hash_are_checked_independently():
    raw = packet()
    raw["bars"][0]["publication_proof"]["binding"] = dict(raw["bars"][0]["source_binding"], revision=2)
    with pytest.raises(ValueError, match="ORIGINAL_VERSION_BINDING"):
        prepare_industry_benchmark(raw)
    raw = packet()
    raw["documents"][raw["bars"][0]["document_sha256"]]["body_utf8"] += "changed"
    with pytest.raises(ValueError, match="SOURCE_DOCUMENT_HASH_MISMATCH"):
        prepare_industry_benchmark(raw)


def test_exact_cutoff_date_precision_and_revision_do_not_leak_future_versions():
    raw = packet()
    baseline, before = select(raw)
    row = raw["bars"][-1]
    add_row(raw, "bar", dict(trade_date=row["trade_date"], close=200., membership_version=row["membership_version"],
        revision=1, series_version=row["series_version"], published_at="2026-09-12T09:00:00+08:00"))
    same, preserved = select(raw)
    pd.testing.assert_series_equal(same, baseline)
    assert preserved == before
    changed, after = select(raw, cutoff="2026-09-12T09:00:00+08:00")
    assert changed.iloc[-1] == 200. and after["observations"][-1]["revision"] == 1
    raw = packet()
    proof = raw["bars"][-1]["publication_proof"]
    proof.update(published_at=raw["bars"][-1]["trade_date"], publication_precision="DATE")
    assert pd.isna(select(raw)[0].iloc[-1])
    assert select(raw, cutoff="2026-09-12T00:00:00+08:00")[0].notna().all()


def test_conflicting_versions_wrong_sector_and_membership_missing_remain_unknown():
    raw = packet()
    row = raw["bars"][-1]
    add_row(raw, "bar", dict(trade_date=row["trade_date"], close=200., membership_version=row["membership_version"],
        revision=0, series_version=row["series_version"], published_at=row["trade_date"]+"T13:30:00+08:00"))
    values, evidence = select(raw)
    assert pd.isna(values.iloc[-1]) and evidence["gaps"][-1]["reason"] == "INDEX_VERSION_CONFLICT"
    assert select(packet(), sector="28")[0].isna().all()
    assert select(packet(), symbol="OTHER")[0].isna().all()


def test_future_constituents_cannot_fill_earlier_sessions_or_peer_baskets_become_index():
    raw = packet()
    raw["memberships"][0]["publication_proof"].update(published_at="2026-09-01T13:00:00+08:00")
    _, evidence = select(raw)
    assert evidence["gaps"][0]["reason"] == "MEMBERSHIP_NOT_AVAILABLE"
    raw["kind"] = "SELECTED_UNIVERSE_PEER_MEDIAN"
    with pytest.raises(ValueError, match="INVALID_INDUSTRY_BENCHMARK_CONTRACT"):
        prepare_industry_benchmark(raw)
    with pytest.raises(ValueError, match="RAW_INDUSTRY_SOURCE"):
        prepare_industry_context({"industry_benchmark": prepare_industry_benchmark(packet())})


def test_public_contract_preserves_input_and_requires_timezone():
    raw = packet()
    original = deepcopy(raw)
    result = prepare_industry_context({"industry_benchmark": raw})
    assert raw == original and result["industry_benchmark"]["schema"] == "industry-benchmark/v1"
    _, evidence = select(raw, cutoff="2026-09-11T13:30:00")
    assert evidence["status"] == "UNKNOWN"


def test_equivalent_receipts_are_not_conflicts_and_late_old_revision_cannot_override_new():
    raw = packet()
    copy = deepcopy(raw["bars"][-1])
    raw["bars"].append(copy)
    copy["publication_proof"]["verified_at"] = "2026-10-02T17:00:00+08:00"
    assert select(raw)[1]["status"] == "PASS"
    row = raw["bars"][-1]
    for revision, close, when in ((1, 110., "2026-09-12T09:00:00+08:00"),
                                  (0, 105., "2026-09-13T09:00:00+08:00")):
        add_row(raw, "bar", dict(trade_date=row["trade_date"], close=close, membership_version=row["membership_version"],
            revision=revision, series_version=row["series_version"], published_at=when))
    points, evidence = select(raw, cutoff="2026-09-13T10:00:00+08:00")
    assert points.iloc[-1] == 110. and evidence["observations"][-1]["revision"] == 1

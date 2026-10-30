"""SYNTHETIC ONLY: independent coverage never manufactures index/member facts."""
from copy import deepcopy

import pytest

from application.industry_benchmark import prepare_industry_benchmark
from domain.les_v7.industry_score import HORIZON_VERSION, apply_industry_features
from domain.strategies.low_entry_v6_hybrid import known_features
from tests.strategies.test_v7_industry_score import CUTOFF, inputs, apply as legacy_apply


def apply(f, bars, raw, **kwargs):
    return apply_industry_features(f, bars, prepare_industry_benchmark(raw) if raw else None,
        symbol="2330", market="TWSE", sector="24", cutoff=CUTOFF, version=HORIZON_VERSION, **kwargs)


def test_old_early_gap_cannot_remove_complete_current_twenty_session_evidence():
    f, bars, raw = inputs()
    raw["bars"].pop(0)
    before = deepcopy(raw)
    old, _ = legacy_apply(f, bars, raw)
    result, audit = apply(f, bars, raw)
    assert old["industry_rs20"] is old["industry_rs60"] is None
    assert result["industry_status"] == audit["status"] == "PARTIAL"
    assert result["industry_rs20"] is not None and result["industry_rs60"] is None
    assert result["industry_period_status"] == {"20": "PASS", "60": "UNKNOWN"}
    assert known_features(result)["industry_rs20"] == result["industry_rs20"]
    assert known_features(result)["industry_rs60"] is None
    assert audit["periods"]["20"]["window"]["observations"] == 21
    assert audit["periods"]["60"]["evidence"]["gaps"]
    assert raw == before


@pytest.mark.parametrize("damage", ["current_bar", "membership", "future_publication"])
def test_missing_actual_horizon_sources_stay_unknown(damage):
    f, bars, raw = inputs()
    if damage == "current_bar":
        raw["bars"].pop()
    elif damage == "membership":
        raw["memberships"] = []
    else:
        # No earlier availability is inferred from the historical trade date.
        for row in raw["bars"]:
            row.pop("publication_proof", None)
        for document in raw["documents"].values():
            document["retrieved_at"] = "2026-09-12T09:00:00+08:00"
    result, audit = apply(f, bars, raw)
    assert result["industry_status"] == "UNKNOWN"
    assert result["industry_rs20"] is result["industry_rs60"] is None
    assert not any(result["score_availability"][f"industry_rs{n}"] for n in (20, 60))


def test_partial_history_only_supplies_exact_horizon_without_neutral_points():
    f, bars, raw = inputs(21)
    result, audit = apply(f, bars, raw)
    assert result["industry_status"] == "PARTIAL"
    assert result["industry_rs20"] is not None
    assert result["industry_rs60"] is None
    assert audit["periods"]["60"]["reason"] == "INDUSTRY_HORIZON_HISTORY_SHORT"


def test_missing_packet_and_bad_close_clear_stale_period_flags():
    f, bars, raw = inputs()
    f.update(industry_rs20=99., industry_rs60=99., industry_period_status={"20": "PASS", "60": "PASS"})
    result, _ = apply(f, bars, None)
    assert known_features(result)["industry_rs20"] is None
    result, audit = apply(dict(f, close=119.), bars, raw)
    assert audit["reason"] == "INDUSTRY_STOCK_CLOSE_BINDING_MISMATCH"
    assert result["industry_period_status"] == {"20": "UNKNOWN", "60": "UNKNOWN"}

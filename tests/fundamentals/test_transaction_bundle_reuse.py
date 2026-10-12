"""Synthetic/offline reuse inside one refresh; native cutoff selection remains."""
import pandas as pd
import pytest

from application.daily_performance import DailyPerformance
from infrastructure.crawler.twse_http_client import AccessBlockedError
from tests.fundamentals.test_incremental_snapshot_refresh import seeded, CUTOFF
from tests.fundamentals.test_filing_refresh_planner import _candidate, _fact


def test_monthly_no_change_decodes_once_with_full_reference_parity(tmp_path, monkeypatch):
    outputs, probes = [], []
    for mode in ("reference", "optimized"):
        repo, _, service = seeded(tmp_path / mode)
        original = service._load_cached_bundle
        def uncached(*args, **kwargs):
            kwargs.pop("local_reuse", None)
            return original(*args, **kwargs)
        probe = DailyPerformance()
        service.begin_run()
        with monkeypatch.context() as patch, probe.activate():
            if mode == "reference":
                patch.setattr(service, "_load_cached_bundle", uncached)
            result = service.load("2330", signal_date="2026-08-01T05:30:00+00:00", sector="24")
        outputs.append(result)
        probes.append(probe)
    assert outputs[0].snapshot == outputs[1].snapshot
    assert outputs[0].as_of == outputs[1].as_of
    pd.testing.assert_frame_equal(outputs[0].annual_fundamentals, outputs[1].annual_fundamentals)
    assert probes[0].counters["fundamental_snapshot_decodes"] == 2
    assert probes[1].counters["fundamental_snapshot_decodes"] == 1
    assert probes[1].counters["fundamental_transaction_bundle_reuses"] == 1


def test_blocked_refresh_reselects_before_reusing_fallback(tmp_path, monkeypatch):
    repo, _, service = seeded(tmp_path)
    def blocked(*args, **kwargs):
        raise AccessBlockedError("SYNTHETIC blocked refresh")
    monkeypatch.setattr(service.catalog_refresh, "prepare", blocked)
    service.begin_run()
    probe = DailyPerformance()
    with probe.activate():
        result = service.load("2330", signal_date="2026-08-01T05:30:00+00:00", sector="24")
    assert result.snapshot["remote_refresh_skipped"] is True
    assert result.snapshot["remote_refresh_status"] == "ACCESS_BLOCKED"
    assert probe.counters["fundamental_snapshot_decodes"] == 1
    assert probe.counters["fundamental_transaction_bundle_reuses"] == 1


def test_partial_refresh_before_block_cannot_reuse_old_selection(tmp_path, monkeypatch):
    repo, _, service = seeded(tmp_path)
    def partial_then_blocked(*args, **kwargs):
        repo.save_revision(_candidate(2025, 4), b"SYNTHETIC-PARTIAL-REVISION", [_fact(999.)],
            available_at="2026-07-30T05:00:00+00:00", fetched_at="2026-07-30T05:01:00+00:00",
            uow=kwargs["uow"])
        raise AccessBlockedError("SYNTHETIC blocked after partial ingestion")
    monkeypatch.setattr(service.catalog_refresh, "prepare", partial_then_blocked)
    service.begin_run()
    probe = DailyPerformance()
    with probe.activate():
        result = service.load("2330", signal_date="2026-08-01T05:30:00+00:00", sector="24")
    assert probe.counters["fundamental_transaction_bundle_reuses"] == 0
    assert result.snapshot["available_at"] == "2026-07-30T05:00:00+00:00"
    assert len(result.snapshot["les_v6_mops_filings"]) == 6
    assert result.snapshot["remote_refresh_skipped"] is True


@pytest.mark.parametrize("change", ["future", "revision", "provenance", "sector"])
def test_native_selection_invalidates_reuse_without_promoting_future(tmp_path, change):
    repo, _, service = seeded(tmp_path)
    probe = DailyPerformance()
    with repo.unit_of_work() as uow:
        selected = repo.load_filings_many(["2330"], as_of=CUTOFF, uow=uow)["2330"]
        prior = service._load_cached_bundle("2330", parsed_signal=pd.Timestamp(CUTOFF),
            sector="24", uow=uow, filings=selected)
        reuse = service._bundle_selection_key("2330", selected, "24"), prior
        if change in {"future", "revision"}:
            available = "2026-07-23T05:00:00+00:00" if change == "future" else "2026-07-20T05:00:00+00:00"
            repo.save_revision(_candidate(2025, 4), b"SYNTHETIC-NEW-REVISION", [_fact(999.)],
                available_at=available, fetched_at="2026-07-23T05:01:00+00:00", uow=uow)
        elif change == "provenance":
            repo.update_availability_provenance(selected[0], available_at="2026-07-20T00:00:00+00:00",
                availability_source="FIRST_OBSERVED_DOWNLOAD", uow=uow)
        else:
            service._company_sectors["2330"] = "25"
        with probe.activate():
            result = service._load_cached_bundle("2330", parsed_signal=pd.Timestamp(CUTOFF),
                sector="25" if change == "sector" else "24", uow=uow, local_reuse=reuse)
        assert probe.counters["fundamental_transaction_bundle_reuses"] == (1 if change == "future" else 0)
        if change == "future":
            assert result[0].snapshot == prior[0].snapshot
        else:
            assert result[0].snapshot != prior[0].snapshot

"""SYNTHETIC ONLY: cache-only history plumbing, not repaired financial evidence."""
from copy import deepcopy
from datetime import date, datetime, timedelta
import json
import sqlite3

import pytest

from domain.les_v7.contracts import identity, timestamp
from infrastructure.crawler import valuation_history as module


OBSERVED = "2026-07-15T05:00:00Z"
REFERENCE = "2026-07-14"


def projection(*, observed=OBSERVED, months=2):
    start, end = module._window(REFERENCE, months)
    days = {(start + timedelta(days=i)).isoformat(): "CLOSED" for i in range((end - start).days + 1)}
    if months == 2:
        days.update({"2026-05-29": "OPEN", "2026-06-30": "OPEN"})
    result = dict(schema=module.CALENDAR_SCHEMA, source="TWSE_OFFICIAL", days=days,
        days_sha256=identity(days), source_ids=["c" * 64], available_at=observed,
        window=dict(start=start.isoformat(), end=end.isoformat(), months=months), SYNTHETIC_ONLY=True)
    result["calendar_id"] = identity(result)
    return result


def rehash_projection(data):
    data["days_sha256"] = identity(data["days"])
    data["calendar_id"] = identity({key: value for key, value in data.items() if key != "calendar_id"})
    return data


def stock(symbol="2330", close="1,234.50"):
    return [symbol, "SYNTHETIC", "10000", "10", "12345000", "1200", "1250", "1190", close,
            "+", "1", "1234", "1", "1235", "1", "15"]


def database(path, *, rows=None, market_type="13", epoch=None, no_taiex=True):
    epoch = int(timestamp("2026-07-01T05:00:00Z").timestamp()) if epoch is None else epoch
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE market_history_cache (trade_date TEXT,dataset TEXT,market_type TEXT,schema_version INTEGER,payload_json TEXT,cached_at_epoch INTEGER)")
        for day, values in (rows or {"20260529": [stock(), stock("2454", "9.30")], "20260630": [stock(), stock("2454", "9.40")]}).items():
            connection.execute("INSERT INTO market_history_cache VALUES (?,?,?,?,?,?)", (day, "STOCK_TABLE", market_type, 1, json.dumps(values, ensure_ascii=False), epoch))
    return path


def collect(path, *, calendar=None, symbols=("2330", "2454"), observed=OBSERVED):
    return module.collect_valuation_history(path, symbols=symbols, reference_session=REFERENCE,
        observed_at=observed, calendar_projection=calendar or projection(), months=2)


def test_readonly_batch_query_normalizes_twd_not_legacy_thousand_scale_and_does_not_need_taiex(tmp_path, monkeypatch):
    path = database(tmp_path / "history.sqlite3")
    before = path.read_bytes(); original_connect = sqlite3.connect; statements = []
    def readonly_connect(*args, **kwargs):
        assert args[0].endswith("?mode=ro") and kwargs.get("uri") is True
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        denied = {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                  sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_DROP_TABLE}
        connection.set_authorizer(lambda action, *rest: sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK)
        return connection
    monkeypatch.setattr(module.sqlite3, "connect", readonly_connect)
    result = collect(path)
    assert path.read_bytes() == before
    assert len([statement for statement in statements if statement.lstrip().upper().startswith("SELECT")]) == 1
    assert all("CREATE" not in statement.upper() for statement in statements)
    assert result["by_symbol"]["2330"]["rows"] == [dict(Date="2026-05-29", Close=1234.5), dict(Date="2026-06-30", Close=1234.5)]
    assert result["by_symbol"]["2454"]["rows"][0]["Close"] == 9.3
    assert result["by_symbol"]["2330"]["status"] == "COMPLETE"
    assert len(result["documents"]) == 2
    assert module.verify_valuation_history(result) is result


def test_nonexistent_database_is_not_created_and_gaps_are_preserved(tmp_path):
    path = tmp_path / "absent" / "history.sqlite3"
    result = collect(path)
    assert not path.exists() and not path.parent.exists()
    item = result["by_symbol"]["2330"]
    assert item["status"] == "PARTIAL" and not item["rows"]
    assert {gap["reason"] for gap in item["coverage"]["gaps"]} == {"OFFICIAL_MONTH_END_CACHE_DAY_MISSING"}


def test_unknown_later_calendar_day_prevents_using_an_earlier_open_bar(tmp_path):
    path = database(tmp_path / "history.sqlite3")
    calendar = projection(); calendar["days"]["2026-05-31"] = "UNKNOWN"; rehash_projection(calendar)
    result = collect(path, calendar=calendar)
    item = result["by_symbol"]["2330"]
    assert item["coverage"]["unknown_calendar_months"] == ["2026-05"]
    assert item["rows"] == [dict(Date="2026-06-30", Close=1234.5)]
    assert result["summary"]["cached_month_end_days"] == 1


def test_missing_exact_monthend_does_not_fall_back_to_earlier_cache_bar(tmp_path):
    path = database(tmp_path / "history.sqlite3", rows={"20260528": [stock()], "20260630": [stock()]})
    result = collect(path)
    assert result["by_symbol"]["2330"]["rows"] == [dict(Date="2026-06-30", Close=1234.5)]
    assert result["by_symbol"]["2330"]["coverage"]["gaps"][0]["expected_session"] == "2026-05-29"


def test_other_market_dataset_cannot_fill_twse_elec_gap(tmp_path):
    path = database(tmp_path / "history.sqlite3", market_type="TPEX:24:v1")
    assert collect(path)["by_symbol"]["2330"]["rows"] == []


def test_symbol_missing_from_table_and_duplicate_rows_are_not_silently_resolved(tmp_path):
    path = database(tmp_path / "history.sqlite3", rows={"20260529": [stock("2454")], "20260630": [stock(), stock(close="100")]})
    item = collect(path)["by_symbol"]["2330"]
    assert item["status"] == "UNKNOWN" and item["rows"] == []
    assert [gap["reason"] for gap in item["coverage"]["gaps"]] == ["SYMBOL_NOT_IN_MONTH_END_TABLE", "CONFLICTING_SYMBOL_ROWS"]


@pytest.mark.parametrize("close", ["--", "", "nan", "inf", "-3", "0", "not-a-price"])
def test_nonfinite_or_absent_close_is_not_a_monthly_observation(tmp_path, close):
    path = database(tmp_path / "history.sqlite3", rows={"20260529": [stock(close=close)]})
    item = collect(path)["by_symbol"]["2330"]
    assert not item["rows"]
    assert item["coverage"]["gaps"][0]["reason"] == "MONTH_END_CLOSE_UNVERIFIED"


def test_cache_timestamp_is_last_write_not_historical_publication_and_future_write_is_unknown(tmp_path):
    path = database(tmp_path / "history.sqlite3", epoch=int(timestamp("2026-07-16T05:00:00Z").timestamp()))
    item = collect(path)["by_symbol"]["2330"]
    assert item["status"] == "UNKNOWN" and not item["rows"]
    assert all(gap["reason"] == "CACHE_LAST_WRITE_TIME_UNVERIFIED_OR_FUTURE" for gap in item["coverage"]["gaps"])


@pytest.mark.parametrize("damage", ["days", "time", "id"])
def test_calendar_tamper_or_future_observation_is_rejected(tmp_path, damage):
    path = database(tmp_path / "history.sqlite3"); calendar = projection()
    if damage == "days": calendar["days"]["2026-05-31"] = "OPEN"
    elif damage == "time": calendar["available_at"] = "2027-01-01T00:00:00Z"; rehash_projection(calendar)
    else: calendar["calendar_id"] = "0" * 64
    with pytest.raises(ValueError, match="CALENDAR_BINDING_INVALID"):
        collect(path, calendar=calendar)


@pytest.mark.parametrize("damage", ["quote", "unit", "raw_row", "cache_payload", "cache_epoch"])
def test_rehashed_outer_artifact_cannot_override_original_payload_membership_or_units(tmp_path, damage):
    result = collect(database(tmp_path / "history.sqlite3"))
    item = result["by_symbol"]["2330"]; record = item["records"][0]
    if damage == "quote": item["rows"][0]["Close"] *= 1000
    elif damage == "unit": record["price_unit"] = "TWD_THOUSAND"
    elif damage == "raw_row": record["raw_row"][8] = "2"; record["raw_row_sha256"] = identity(record["raw_row"])
    elif damage == "cache_payload": result["documents"][record["cache_source_id"]]["payload_json"] = "[]"
    else: record["cached_at_epoch"] -= 1
    item["source_id"] = identity({key: value for key, value in item.items() if key != "source_id"})
    result["evidence_id"] = identity({key: value for key, value in result.items() if key != "evidence_id"})
    with pytest.raises(ValueError, match="BINDING_INVALID|SOURCE_MISMATCH"):
        module.verify_valuation_history(result)


def test_missing_cache_schema_is_unknown_without_creating_a_table(tmp_path):
    path = tmp_path / "history.sqlite3"
    with sqlite3.connect(path) as connection: connection.execute("PRAGMA user_version=1")
    before = path.read_bytes(); result = collect(path)
    assert result["by_symbol"]["2330"]["status"] == "UNKNOWN"
    assert result["summary"]["cache_error"].startswith("VALUATION_HISTORY_CACHE_UNVERIFIED")
    assert path.read_bytes() == before


def test_capture_stamps_actual_completion_not_read_start_and_preserves_last_write(tmp_path, monkeypatch):
    start, end = "2026-07-15T04:59:00Z", OBSERVED
    path = database(tmp_path / "history.sqlite3", epoch=int(timestamp("2026-07-15T04:59:30Z").timestamp()))
    # Wrapper is fixed at 36 months; explicit test projection keeps all other
    # months UNKNOWN and the two official-shaped month ends verified.
    calendar = projection(observed=start, months=36)
    calendar["days"].update({"2026-05-29": "OPEN", "2026-06-30": "OPEN"}); rehash_projection(calendar)
    monkeypatch.setattr(module, "offline_calendar_projection", lambda **kwargs: deepcopy(calendar))
    clocks = iter((start, end))
    result = module.capture_valuation_history(path, symbols=["2330"], reference_session=REFERENCE, clock=lambda: next(clocks))
    assert result["capture_started_at"] == timestamp(start).isoformat()
    assert result["observed_at"] == timestamp(end).isoformat()
    assert result["by_symbol"]["2330"]["coverage"]["months_valid"] == 2
    assert "NOT_HISTORICAL_OBSERVATION" in result["availability_basis"]
    assert module.verify_valuation_history(result) is result


def test_offline_real_calendar_projection_does_not_call_network_and_keeps_missing_year_unknown(monkeypatch):
    from infrastructure.crawler.twse_http_client import TwseHttpClient
    def forbidden(*args, **kwargs): raise AssertionError("NETWORK_NOT_ALLOWED")
    monkeypatch.setattr(TwseHttpClient, "get_json", forbidden)
    result = module.offline_calendar_projection(reference_session="2026-10-14", observed_at="2026-10-15T00:00:00+08:00")
    assert result["missing_annual_years"] == [2023]
    assert result["days"]["2023-10-31"] == "UNKNOWN"
    assert result["days"]["2024-10-31"] == "CLOSED"  # Reviewed official exception, not missing-bar inference.
    assert result["annual_documents"]["2024"]["sha256"] in result["source_ids"]
    assert result["calendar_id"] == identity({key: value for key, value in result.items() if key != "calendar_id"})


def test_verification_decodes_each_full_market_payload_only_once(tmp_path, monkeypatch):
    result = collect(database(tmp_path / "history.sqlite3"))
    source_bodies = {source["payload_json"] for source in result["documents"].values()}
    calls = []; original = module.json.loads
    def counted(value, *args, **kwargs):
        if value in source_bodies: calls.append(value)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(module.json, "loads", counted)
    module.verify_valuation_history(result)
    assert len(calls) == len(result["documents"]) == 2  # Not symbol_count × month_count.


@pytest.mark.parametrize("damage", ["duplicate_date", "reversed_dates", "missing_symbol", "negative_price", "negative_row_index", "missing_gap"])
def test_rehashed_quote_order_scope_and_coverage_cannot_be_forged(tmp_path, damage):
    result = collect(database(tmp_path / "history.sqlite3"))
    item = result["by_symbol"]["2330"]
    if damage == "duplicate_date": item["rows"][1]["Date"] = item["rows"][0]["Date"]
    elif damage == "reversed_dates": item["rows"].reverse(); item["records"].reverse()
    elif damage == "missing_symbol": result["by_symbol"].pop("2454")
    elif damage == "negative_price": item["rows"][0]["Close"] = -1.
    elif damage == "negative_row_index": item["records"][0]["raw_row_index"] = -1
    else:
        item["rows"].pop(); item["records"].pop(); item["coverage"]["months_valid"] = 1
    item["source_id"] = identity({key: value for key, value in item.items() if key != "source_id"})
    result["evidence_id"] = identity({key: value for key, value in result.items() if key != "evidence_id"})
    with pytest.raises(ValueError): module.verify_valuation_history(result)


def test_retention_reuses_payloads_across_captures_and_restores_complete_integrity(tmp_path, monkeypatch):
    from infrastructure import research_archive
    result = collect(database(tmp_path / "history.sqlite3")); root = tmp_path / "retained"
    first = module.retain_valuation_history(result, root)
    assert module.load_valuation_history(first, root) == result
    source_files = sorted((root / "objects").glob("*.json")); before = [file.stat().st_mtime_ns for file in source_files]
    manifest = json.loads(first.read_text())
    assert all("payload_json" not in source for source in manifest["payload"]["documents"].values())
    original_write = research_archive.write_bytes_once
    def forbidden(path, *args, **kwargs):
        if path.parent.name == "objects": raise AssertionError("UNCHANGED_PAYLOAD_MUST_NOT_BE_WRITTEN")
        return original_write(path, *args, **kwargs)
    monkeypatch.setattr(research_archive, "write_bytes_once", forbidden)
    assert module.retain_valuation_history(result, root) == first
    second = deepcopy(result); second["observed_at"] = "2026-07-15T05:01:00+00:00"
    second["evidence_id"] = identity({key: value for key, value in second.items() if key != "evidence_id"})
    second_path = module.retain_valuation_history(second, root)
    assert second_path != first and module.load_valuation_history(second_path, root) == second
    assert sorted((root / "objects").glob("*.json")) == source_files
    assert [file.stat().st_mtime_ns for file in source_files] == before


def test_physical_archive_resolution_preserves_logical_selection_and_verifies_relocated_objects(tmp_path):
    import shutil
    result = collect(database(tmp_path / "history.sqlite3"))
    original = tmp_path / "original"
    manifest = module.retain_valuation_history(result, original)
    selection = module.project_valuation_selections(result, manifest=manifest)["2330"]
    relocated = tmp_path / "portable"
    shutil.copytree(original, relocated)
    physical = relocated / "manifests" / manifest.name
    # The original logical location no longer exists. It remains hash-bound
    # source metadata, not a path the archive verifier needs to access.
    shutil.rmtree(original)
    with module.ValuationSelectionVerifier(manifest_path_resolver=lambda logical: physical) as verifier:
        assert verifier.verify(selection) == selection
        assert verifier.verify(selection) == selection
        next((relocated / "objects").glob("*.json")).write_text("corrupt")
        with pytest.raises(ValueError, match="MISSING_OR_CHANGED"):
            verifier.verify(selection)


@pytest.mark.parametrize("damage", ["body", "manifest", "reference"])
def test_retained_body_manifest_and_reference_corruption_never_uses_runtime_cache(tmp_path, damage):
    result = collect(database(tmp_path / "history.sqlite3")); root = tmp_path / "retained"
    path = module.retain_valuation_history(result, root)
    if damage == "body":
        body = next((root / "objects").glob("*.json")); body.write_text("[]")
    else:
        manifest = json.loads(path.read_text())
        if damage == "manifest": manifest["evidence_id"] = "0" * 64
        else:
            source = next(iter(manifest["payload"]["documents"].values()))
            source["body_ref"]["path"] = "../outside.json"
            manifest["manifest_id"] = identity({key: value for key, value in manifest.items() if key != "manifest_id"})
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): module.load_valuation_history(path, root)


@pytest.mark.parametrize("months", [0,37,True])
def test_history_window_is_bounded(months):
    with pytest.raises(ValueError, match="MONTH_LIMIT_INVALID"):
        module._window(REFERENCE, months)


def retained_selections(tmp_path):
    capture = collect(database(tmp_path / "history.sqlite3"))
    root = tmp_path / "retained"
    manifest = module.retain_valuation_history(capture, root)
    return capture, manifest, module.project_valuation_selections(capture, manifest=manifest)


def rehash_selection(selection):
    selection["selection_id"] = identity({key: value for key, value in selection.items() if key != "selection_id"})
    return selection


def test_compact_projection_and_verification_bind_retained_capture_with_detached_results(tmp_path):
    capture, manifest, selections = retained_selections(tmp_path)
    original = deepcopy(capture)
    with module.ValuationSelectionVerifier() as verifier:
        result = verifier.verify(selections["2330"], symbol="2330", cutoff=OBSERVED)
        assert result == selections["2330"]
        result["rows"][0]["Close"] = 1.
        result["coverage"]["gaps"].append(dict(SYNTHETIC_MUTATION=True))
        assert verifier.verify(selections["2330"]) == selections["2330"]
        assert verifier.verify(selections["2454"]) == selections["2454"]
        assert verifier.stats["loads"] == 1 and verifier.stats["hits"] == 2
        assert verifier.stats["entries"] == 1 and verifier.stats["bytes"] <= verifier.stats["max_bytes"]
    assert capture == original
    with pytest.raises(ValueError, match="VERIFIER_CLOSED"): verifier.verify(selections["2330"])
    assert module.verify_valuation_selection(selections["2330"]) == selections["2330"]


@pytest.mark.parametrize("damage", ["price", "status", "source_id", "coverage", "time", "source_records",
                                  "calendar", "units", "authenticity", "availability", "capture_id"])
def test_rehashed_compact_claims_cannot_substitute_retained_monthly_values_or_provenance(tmp_path, damage):
    _, _, selections = retained_selections(tmp_path); selected = selections["2330"]
    if damage == "price": selected["rows"][0]["Close"] += 100.
    elif damage == "status": selected["status"] = "UNKNOWN"
    elif damage == "source_id": selected["source_id"] = "0" * 64
    elif damage == "coverage": selected["coverage"]["months_valid"] += 1
    elif damage == "time": selected["observed_at"] = "2026-07-15T04:59:59Z"
    elif damage == "source_records": selected["source_record_ids"] = ["0" * 64]
    elif damage == "calendar": selected["calendar_id"] = "0" * 64
    elif damage == "units": selected["price_unit"] = "TWD_THOUSAND"
    elif damage == "authenticity": selected["source_authenticity"] = "VERIFIED_HISTORICAL_PUBLICATION"
    elif damage == "availability": selected["availability_basis"] = "OBSERVED_IN_THE_PAST"
    else: selected["capture_evidence_id"] = "0" * 64
    rehash_selection(selected)
    with pytest.raises(ValueError, match="PROJECTION_MISMATCH|CAPTURE_ID_MISMATCH"):
        module.verify_valuation_selection(selected)


def test_original_changed_during_complete_restore_is_not_cached_as_verified(tmp_path, monkeypatch):
    from pathlib import Path
    _, _, selections = retained_selections(tmp_path)
    selected = selections["2330"]
    original_read = Path.read_bytes
    changed = []
    def changing_read(path):
        content = original_read(path)
        if path.parent.name == "objects" and not changed:
            changed.append(str(path))
            path.write_bytes(content + b" ")
        return content
    monkeypatch.setattr(Path, "read_bytes", changing_read)
    with pytest.raises(ValueError, match="MISSING_OR_CHANGED"):
        module.verify_valuation_selection(selected)
    assert len(changed) == 1


@pytest.mark.parametrize("which", ["manifest", "object"])
def test_missing_retained_artifact_is_not_replaced_with_runtime_cache(tmp_path, which):
    _, manifest, selections = retained_selections(tmp_path)
    target = manifest if which == "manifest" else next((manifest.parent.parent / "objects").glob("*.json"))
    target.unlink()
    with pytest.raises(ValueError): module.verify_valuation_selection(selections["2330"])


@pytest.mark.parametrize("which", ["manifest", "object", "deleted_object"])
def test_warm_verifier_restats_retained_references_before_cache_hits(tmp_path, which):
    _, manifest, selections = retained_selections(tmp_path)
    verifier = module.ValuationSelectionVerifier()
    verifier.verify(selections["2330"])
    target = manifest if which == "manifest" else next((manifest.parent.parent / "objects").glob("*.json"))
    if which == "deleted_object": target.unlink()
    else: target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="MISSING_OR_CHANGED"):
        verifier.verify(selections["2454"])
    verifier.close()


def test_work_cache_is_bounded_and_clears_without_sharing_another_job(tmp_path):
    capture, first, selections = retained_selections(tmp_path)
    later = deepcopy(capture); later["observed_at"] = "2026-07-15T05:01:00+00:00"
    later["evidence_id"] = identity({key: value for key, value in later.items() if key != "evidence_id"})
    second = module.retain_valuation_history(later, first.parent.parent)
    second_selections = module.project_valuation_selections(later, manifest=second)
    with module.ValuationSelectionVerifier(max_captures=1) as verifier:
        verifier.verify(selections["2330"]); verifier.verify(second_selections["2330"])
        assert verifier.stats["entries"] == 1 and verifier.stats["evictions"] == 1
        verifier.clear()
        assert verifier.stats["entries"] == verifier.stats["bytes"] == 0
        verifier.verify(selections["2330"])
        assert verifier.stats["loads"] == 3
    with module.ValuationSelectionVerifier(max_bytes=0) as verifier:
        verifier.verify(selections["2330"]); verifier.verify(selections["2454"])
        assert verifier.stats["entries"] == 0 and verifier.stats["loads"] == 2


def test_injected_loader_cannot_bypass_original_capture_verification(tmp_path):
    capture, manifest, selections = retained_selections(tmp_path)
    invalid = deepcopy(capture); invalid["by_symbol"]["2330"]["rows"][0]["Close"] += 1.
    with pytest.raises(ValueError, match="CONTENT_ID_MISMATCH"):
        module.verify_valuation_selection(selections["2330"], manifest_loader=lambda _: invalid)
    calls = []
    def loader(path):
        calls.append(path)
        return module.load_valuation_history(path, path.parent.parent)
    with module.ValuationSelectionVerifier(manifest_loader=loader) as verifier:
        verifier.verify(selections["2330"]); verifier.verify(selections["2454"])
        assert calls == [manifest.resolve()]


def failure_selection():
    return rehash_selection(dict(schema=module.SELECTION_SCHEMA, market="TWSE", symbol="2330",
        observed_at=OBSERVED, rows=[], status="UNKNOWN", diagnostics=["SYNTHETIC_ONLY_CAPTURE_UNAVAILABLE"],
        historical_information_set_certified=False))


def test_unbound_failure_projection_is_explicit_unknown_and_never_loads_prices():
    failure = failure_selection()
    def forbidden(_): raise AssertionError("FAILURE_HAS_NO_ARTIFACT")
    assert module.verify_valuation_selection(failure, manifest_loader=forbidden) == failure


@pytest.mark.parametrize("damage", ["rows", "status", "diagnostic", "source_id", "reference", "months"])
def test_unbound_projection_cannot_claim_valid_pricing_by_rehashing(tmp_path, damage):
    failed = failure_selection()
    if damage == "rows": failed["rows"] = [dict(Date="2026-05-29", Close=1.)]
    elif damage == "status": failed["status"] = "COMPLETE"
    elif damage == "diagnostic": failed["diagnostics"] = []
    elif damage == "source_id": failed["source_id"] = "0" * 64
    elif damage == "reference": failed["manifest"] = "SYNTHETIC"
    else: failed["coverage"] = dict(months_valid=24)
    rehash_selection(failed)
    with pytest.raises(ValueError, match="UNBOUND_FAILURE_INVALID"):
        module.verify_valuation_selection(failed)


def test_selection_external_symbol_and_cutoff_are_checked_even_when_cached(tmp_path):
    _, _, selections = retained_selections(tmp_path)
    with module.ValuationSelectionVerifier() as verifier:
        verifier.verify(selections["2330"])
        with pytest.raises(ValueError, match="BINDING_INVALID"):
            verifier.verify(selections["2330"], symbol="2454")
        with pytest.raises(ValueError, match="AFTER_CUTOFF"):
            verifier.verify(selections["2330"], cutoff="2026-07-15T04:59:59Z")

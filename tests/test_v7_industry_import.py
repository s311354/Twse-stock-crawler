"""SYNTHETIC ONLY: explicit local industry original import, no source discovery."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from application.industry_benchmark import load_industry_benchmarks, prepare_industry_benchmark
from config.settings import AppSettings, SettingsError
from domain.industry_benchmark import aligned_index
from infrastructure.legacy_crawler_facade import LegacyCrawlerFacade
from tests.test_industry_benchmark import packet


def source_file(tmp_path, value=None):
    target = tmp_path / "industry.json"
    target.write_text(json.dumps({"SEMI": packet()} if value is None else value), encoding="utf-8")
    return target


def configured(path):
    return AppSettings.from_env({"TWSE_INDUSTRY_BENCHMARK_FILE": str(path)}, validate_paths=False)


def test_explicit_file_import_verifies_once_and_settings_retains_detached_content(tmp_path, monkeypatch):
    import application.industry_benchmark as importer
    original = importer.prepare_industry_benchmark
    calls = []
    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(importer, "prepare_industry_benchmark", observed)
    path = source_file(tmp_path)
    settings = configured(path)
    assert len(calls) == 1
    expected = json.loads(settings.industry_benchmarks_json)
    assert set(expected) == {"24"}
    assert expected["24"]["schema"] == "industry-benchmark/v1"
    assert expected["24"]["bars"][-1]["receipt"]["sha256"]
    assert "TEST_ONLY" not in repr(settings)
    # Runtime construction decodes the validated snapshot, never rereads a file
    # that could change between configuration and the model's daily cutoff.
    path.write_text("corrupted after import", encoding="utf-8")
    facade = LegacyCrawlerFacade()
    facade._initialize_legacy_facade(app_settings=settings, analysis_dataset_builder=object())
    assert facade._industry_benchmarks == expected
    assert len(calls) == 1


def test_missing_option_preserves_existing_constructor_and_does_not_read_file(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled import must not read or fetch")
    monkeypatch.setattr("application.industry_benchmark.load_industry_benchmarks", forbidden)
    assert AppSettings().industry_benchmarks_json is None
    settings = AppSettings.from_env({}, validate_paths=False)
    assert settings.industry_benchmarks_json is None
    facade = LegacyCrawlerFacade()
    facade._initialize_legacy_facade(app_settings=settings, analysis_dataset_builder=object())
    assert facade._industry_benchmarks == {}


def test_imported_evidence_preserves_exact_historical_stock_membership(tmp_path):
    path = source_file(tmp_path)
    normalized = json.loads(configured(path).industry_benchmarks_json)["24"]
    dates = pd.Series(pd.bdate_range("2026-06-19", periods=61))
    known, evidence = aligned_index(normalized, dates, symbol="2330", sector="SEMI",
        market="TWSE", as_of="2026-09-11T13:30:00+08:00")
    assert evidence["status"] == "PASS" and known.notna().all()
    missing, rejected = aligned_index(normalized, dates, symbol="OTHER", sector="SEMI",
        market="TWSE", as_of="2026-09-11T13:30:00+08:00")
    assert missing.isna().all() and rejected["status"] == "UNKNOWN"


@pytest.mark.parametrize("value,reason", [
    ([], "INDUSTRY_SECTOR_MAPPING_REQUIRED"),
    ({"bad": {}}, "INVALID_INDUSTRY_SECTOR_KEY"),
    ({"28": packet()}, "INDUSTRY_SECTOR_KEY_MISMATCH"),
    ({"SEMI": packet(), "24": packet()}, "DUPLICATE_NORMALIZED_INDUSTRY_SECTOR"),
    ({"24": prepare_industry_benchmark(packet())}, "RAW_INDUSTRY_SOURCE_PACKET_REQUIRED"),
    ({"24": None}, "RAW_INDUSTRY_SOURCE_PACKET_REQUIRED"),
])
def test_explicit_bad_contract_fails_even_without_path_validation(tmp_path, value, reason):
    with pytest.raises(SettingsError, match=reason):
        configured(source_file(tmp_path, value))


def test_duplicate_json_keys_are_not_silently_selected(tmp_path):
    target = tmp_path / "duplicate.json"
    raw = json.dumps(packet())
    target.write_text('{"24":' + raw + ',"24":' + raw + '}', encoding="utf-8")
    with pytest.raises(SettingsError, match="DUPLICATE_INDUSTRY_JSON_KEY"):
        configured(target)


def test_content_hash_failure_is_not_downgraded_to_disabled(tmp_path):
    raw = packet()
    changed = deepcopy(raw)
    changed["documents"][changed["bars"][0]["document_sha256"]]["body_utf8"] += "tampered"
    with pytest.raises(SettingsError, match="SOURCE_DOCUMENT_HASH_MISMATCH"):
        configured(source_file(tmp_path, {"24": changed}))


def test_missing_file_and_bounded_input_raise_clear_configuration_errors(tmp_path, monkeypatch):
    with pytest.raises(SettingsError, match="TWSE_INDUSTRY_BENCHMARK_FILE"):
        configured(tmp_path / "missing.json")
    monkeypatch.setattr("application.industry_benchmark.MAX_SOURCE_FILE_BYTES", 128)
    with pytest.raises(SettingsError, match="BOUNDED_INDUSTRY_SOURCE_FILE_REQUIRED"):
        configured(source_file(tmp_path))


def test_facade_supplies_validated_sector_mapping_to_market_history_input(tmp_path, monkeypatch):
    settings = configured(source_file(tmp_path))
    facade = LegacyCrawlerFacade()
    facade._initialize_legacy_facade(app_settings=settings, analysis_dataset_builder=object())
    for name in ("stocknumbers", "stocknames", "daily_dates", "daily_stocks", "daily_highs", "daily_lows",
                 "daily_closes", "daily_volumes", "daily_trade_counts", "daily_trade_values", "daily_bid_prices",
                 "daily_ask_prices", "daily_pe_ratios", "daily_taiex_closes", "iso_scheduled_times"):
        setattr(facade, name, [])
    for name in ("low_entry_v4_signal_dates_by_stock", "annual_fundamentals_by_stock",
                 "fundamental_snapshots_by_stock", "fundamental_status_by_stock", "fundamental_status_detail_by_stock"):
        setattr(facade, name, {})
    facade.stocklistsize, facade.low_entry_v4_account_equity = 0, 1_000_000.
    market = facade._market_history_input()
    assert market.industry_benchmarks == json.loads(settings.industry_benchmarks_json)
    assert market.industry_benchmarks["24"]["bars"][-1]["binding_sha256"]

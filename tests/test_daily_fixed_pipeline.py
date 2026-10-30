"""Synthetic adapter checks; never financial coverage or performance claims."""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from application.market_history import MarketHistoryAssembler
from application.daily_stock import isolate_stock_market
from examples.benchmark_daily_computation import synthetic_market
from examples.daily_fixed_pipeline import ReadOnlyFundamentals, _account, _maxprofits, fixture_valuation_scope
from infrastructure.crawler import valuation_history as history
from infrastructure.storage.csv_repository import CsvRepository
from infrastructure.storage.report_evidence import iter_report_bundle
from application.replay_contracts import canonical_json, digest
from examples.daily_fixed_pipeline import _dataframe_canonical_chunks, _dataframe_sha256, _normal


def test_retained_market_observation_reaches_native_assembler_without_new_clock():
    market = synthetic_market(1, 30)
    packet = dict(symbol="2300", source_available_at="2026-09-30T07:00:00Z",
                  source_final=False, original_observation="SYNTHETIC_ONLY")
    market = replace(market, entry_model="v7", les_v7_native_evidence=True,
        market_observed_at="2027-01-01T00:00:00Z", captured_market_evidence={"2300": packet})
    frame = MarketHistoryAssembler().build(market, 0)
    assert frame.attrs["les_v7_market_evidence"] == packet
    assert frame.attrs["les_v7_market_evidence"] is not packet
    frame.attrs["les_v7_market_evidence"]["source_final"] = True
    assert packet["source_final"] is False


def test_explicit_empty_capture_is_not_replaced_with_current_observation():
    market = replace(synthetic_market(1, 30), entry_model="v7", les_v7_native_evidence=True,
        market_observed_at="2027-01-01T00:00:00Z", captured_market_evidence={"2300": {}})
    assert MarketHistoryAssembler().build(market, 0).attrs["les_v7_market_evidence"] == {}


def test_worker_transport_keeps_only_the_requested_captured_observation():
    market = replace(synthetic_market(2, 30), captured_market_evidence={
        "2300": {"symbol": "2300"}, "2301": {"symbol": "2301"}})
    assert isolate_stock_market(market, 1).captured_market_evidence == {"2301": {"symbol": "2301"}}


def test_fixture_fundamental_connection_is_read_only(tmp_path):
    with sqlite3.connect(tmp_path / "mops.sqlite3") as connection:
        connection.execute("CREATE TABLE marker(value TEXT)")
    repository = ReadOnlyFundamentals(tmp_path)
    with repository.unit_of_work() as unit:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            unit.connection.execute("INSERT INTO marker VALUES ('forbidden')")


def test_missing_or_nonempty_account_cannot_claim_cold_start():
    account = dict(value=1_000_000., available_cash=1_000_000., positions=[],
                   intended_weight=.25, capacity_weight=.25, as_of="2026-10-16T01:00:00Z")
    manifest = dict(evaluation_cutoff="2026-10-16T15:16:04Z", stocks=[dict(account=account)])
    assert _account(manifest)["as_of"] == "2026-10-16T15:16:04+00:00"
    for key, value in (("positions", None), ("positions", [dict(symbol="2330", shares=1)]),
                       ("available_cash", 900000.)):
        changed = deepcopy(manifest); changed["stocks"][0]["account"][key] = value
        with pytest.raises(ValueError, match="EXPLICIT_ALL_CASH"):
            _account(changed)


def test_profit_diagnostic_uses_production_opening_prices_and_rounding():
    market = synthetic_market(1, 30)
    result = _maxprofits(market)
    assert len(result) == 1 and len(result[0]) == 5
    assert result[0][4] == round(result[0][3] / market.opens[0][-1], 2)


def test_unfrozen_monthly_alias_is_rejected_without_reading_live_path(tmp_path):
    with fixture_valuation_scope(tmp_path, dict(valuation_aliases={})) as verify:
        # A bound-looking selection must be rejected before reading its path.
        selection = dict(schema=history.SELECTION_SCHEMA, symbol="2330", market="TWSE",
            observed_at="2026-07-15T05:00:00Z", capture_evidence_id="a" * 64,
            manifest="/must-not-read-live/manifest.json")
        selection["selection_id"] = history.identity(selection)
        with pytest.raises(ValueError, match="NOT_FROZEN"):
            verify(selection)


def test_native_publication_uses_explicit_source_verifier_scope(tmp_path):
    from tests.strategies.test_daily_v7_pipeline import build, prepared_synthetic_market
    builder, decisions = build(prepared_synthetic_market())
    entered = []
    @contextmanager
    def scope():
        entered.append(True)
        yield lambda selection: selection
    CsvRepository(tmp_path, valuation_history_verifier_scope=scope).write_analysis_dataset(
        "SYNTHETIC", builder.project_decisions(decisions))
    assert entered == [True]


def test_report_stream_uses_injected_verifier_and_propagates_source_failure(tmp_path, monkeypatch):
    from tests.strategies.test_daily_v7_pipeline import build, prepared_synthetic_market
    from infrastructure.storage import valuation_source_verification as source_module
    builder, decisions = build(prepared_synthetic_market())
    CsvRepository(tmp_path).write_analysis_dataset("SYNTHETIC", builder.project_decisions(decisions))
    bundle = tmp_path / "SYNTHETIC_les_v7_reports.json"
    marker, entered, verified = object(), [], []
    @contextmanager
    def scope():
        entered.append(True)
        yield marker
    def forbidden():
        raise AssertionError("DEFAULT_LIVE_SOURCE_SCOPE_FORBIDDEN")
    def verify(report, *, verifier):
        assert verifier is marker
        verified.append(report["symbol"])
    monkeypatch.setattr(source_module, "valuation_history_verifier_scope", forbidden)
    monkeypatch.setattr(source_module, "verify_report_valuation_sources", verify)
    reports = list(iter_report_bundle(bundle, valuation_history_verifier_scope=scope))
    assert entered == [True] and verified == [reports[0]["symbol"]]
    def fail(report, *, verifier):
        assert verifier is marker
        raise ValueError("SYNTHETIC_SOURCE_NOT_VERIFIED")
    monkeypatch.setattr(source_module, "verify_report_valuation_sources", fail)
    with pytest.raises(ValueError, match="SOURCE_NOT_VERIFIED"):
        list(iter_report_bundle(bundle, valuation_history_verifier_scope=scope))


def _old_dataframe_payload(frame):
    return dict(columns=list(frame.columns),
        rows=_normal(frame.astype(object).where(pd.notna(frame), None).to_dict("records")))


def _dataset_cases():
    class SyntheticEnum(Enum):
        SOURCE = "SYNTHETIC_ONLY"
    return [
        pd.DataFrame(),
        pd.DataFrame(columns=["empty", "欄位"]),
        pd.DataFrame({"z": [1., float("nan"), float("inf"), -float("inf"), -0.0],
                      "a": [True, False, None, True, False]}),
        pd.DataFrame({"number": pd.Series([1, pd.NA, 3], dtype="Int64"),
                      "boolean": pd.Series([True, pd.NA, False], dtype="boolean"),
                      "text": pd.Series(["字串", pd.NA, ""], dtype="string"),
                      "float": pd.Series([1.5, pd.NA, 3.], dtype="Float64")}),
        pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-16", None], utc=True),
                      "date": [date(2026, 10, 16), None],
                      "local": [datetime(2026, 10, 16, 1, 2, 3, tzinfo=timezone.utc), pd.NaT]}),
        pd.DataFrame({"nested": [{"b": (1, np.float64(2), float("nan")), "a": ["中", None]},
                                  {1: "first", "1": "last", "enum": SyntheticEnum.SOURCE}],
                      "text": ['"\\\n\t\b\f\r雪😀' * 7000, ""]}),
        pd.DataFrame({"u64": np.array([2 ** 63 + 1, 2 ** 64 - 1], dtype=np.uint64),
                      "f32": np.array([1.2345678, np.nan], dtype=np.float32),
                      "category": pd.Categorical(["A", None])}),
        pd.DataFrame([[1, 2, "three"], [4, 5, "six"]], columns=["dup", "dup", 7]),
        pd.DataFrame([[1, "last"], [2, "winner"]], columns=[1, "1"]),
        pd.DataFrame({"datetime_object": pd.Series([
            np.datetime64("2026-10-16T01:02:03.123456789"), np.datetime64("NaT", "ns"),
            np.datetime64("2026-10-16", "D"), np.datetime64("2026", "Y")], dtype=object)}),
        pd.DataFrame({"numpy_object": pd.Series([
            np.longdouble("1.234567890123456789"), np.float32(1.2345),
            np.uint64(2 ** 64 - 1), np.bool_(True)], dtype=object)}),
    ]


@pytest.mark.parametrize("frame", _dataset_cases())
def test_streaming_full_dataframe_hash_is_exact_old_canonical_bytes(frame):
    # Small synthetic oracle only. The measured implementation must never
    # materialize the universe-sized records list used by this reference.
    expected = _old_dataframe_payload(frame)
    assert "".join(_dataframe_canonical_chunks(frame)) == canonical_json(expected)
    assert _dataframe_sha256(frame) == digest(expected)


def test_streaming_dataframe_hash_does_not_copy_or_encode_whole_dataframe(monkeypatch):
    from examples import daily_fixed_pipeline as module
    frame = pd.DataFrame({"LES report": ["雪😀\"\\" * 40000] * 3,
                          "score": pd.Series([pd.NA, 1, 2], dtype="Int64")})
    expected = digest(_old_dataframe_payload(frame))
    def forbidden(*args, **kwargs):
        raise AssertionError("WHOLE_DATAFRAME_MATERIALIZATION_FORBIDDEN")
    monkeypatch.setattr(frame, "astype", forbidden)
    monkeypatch.setattr(frame, "where", forbidden)
    monkeypatch.setattr(frame, "to_dict", forbidden)
    original = module.canonical_json
    def scalar_only(value):
        assert not isinstance(value, (dict, list, tuple))
        if isinstance(value, str):
            assert len(value) <= 64 * 1024
        return original(value)
    monkeypatch.setattr(module, "canonical_json", scalar_only)
    assert module._dataframe_sha256(frame) == expected


def test_streaming_dataframe_nested_unsupported_na_preserves_old_failure():
    frame = pd.DataFrame({"nested": [{"unknown": pd.NA}]})
    with pytest.raises(TypeError):
        digest(_old_dataframe_payload(frame))
    with pytest.raises(TypeError):
        _dataframe_sha256(frame)

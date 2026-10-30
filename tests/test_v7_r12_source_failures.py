"""SYNTHETIC ONLY: failed monthly evidence cannot become absent or poison peers."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from application.market_history import MarketHistoryInput
from application.valuation_peers_r10 import _independent_prices, capture_comparison_contexts
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.rule_baseline import CLOSE_AVAILABLE_AT, INDUSTRY_AVAILABLE_AT
from infrastructure.runtime.native_source_capture import NativeSourceCapture


def after_release():
    return (timestamp(CLOSE_AVAILABLE_AT) + timedelta(seconds=1)).isoformat()


def crawler():
    return SimpleNamespace(stocknumbers=["2330", "2317"], daily_dates=[["2026-10-14"], ["2026-10-14"]])


def check_failure(result, *, observed):
    assert set(result) == {"2330", "2317"}
    for symbol, entry in result.items():
        selected = entry["valuation_price_history"]
        assert selected["symbol"] == symbol and selected["status"] == "UNKNOWN"
        assert selected["observed_at"] == timestamp(observed).isoformat()
        assert selected["selection_id"] == identity({k: v for k, v in selected.items() if k != "selection_id"})
        assert not selected["rows"] and selected["capture_error"] and selected["diagnostics"]
        assert not {"capture_evidence_id", "source_id", "manifest"} & set(selected)


@pytest.mark.parametrize("failure", [OSError("SYNTHETIC_READ_FAILURE"), ValueError("SYNTHETIC_INTEGRITY_FAILURE")])
def test_capture_failure_remains_explicit_unknown_instead_of_missing_fallback(tmp_path, failure):
    def fail(*args, **kwargs): raise failure
    at = after_release()
    capture = NativeSourceCapture(tmp_path, valuation_history_database=tmp_path / "unused.sqlite3",
        clock=lambda: timestamp(at), valuation_history_collector=fail)
    capture.market = "TWSE"
    capture._collect_valuation_history(crawler(), reference_session="2026-10-14")
    selected = capture.selections(crawler(), cutoff=at)
    check_failure(selected, observed=at)
    rows = [dict(Date="2026-10-14", Close=100.)]
    same, source = _independent_prices(selected["2330"]["valuation_price_history"], rows=rows,
        symbol="2330", cutoff=at, source=dict(source_final=True))
    assert same == rows and source["source_final"] is False


def test_retained_integrity_failure_preserves_future_observation_for_cutoff_rejection(tmp_path, monkeypatch):
    at = after_release()
    future = (timestamp(at) + timedelta(days=1)).isoformat()
    artifact = dict(reference_session="2026-10-14", by_symbol={"2330": {}, "2317": {}},
        window=dict(months=36), observed_at=future)
    def fail(*args, **kwargs): raise ValueError("SYNTHETIC_RETAINED_OBJECT_CORRUPT")
    monkeypatch.setattr("infrastructure.crawler.valuation_history.retain_valuation_history", fail)
    capture = NativeSourceCapture(tmp_path, valuation_history_database=tmp_path / "unused.sqlite3",
        clock=lambda: timestamp(at), valuation_history_collector=lambda *args, **kwargs: artifact)
    capture.market = "TWSE"
    capture._collect_valuation_history(crawler(), reference_session="2026-10-14")
    result = capture.selections(crawler(), cutoff=at)
    check_failure(result, observed=future)
    with pytest.raises(ValueError, match="SELECTION_INVALID"):
        _independent_prices(result["2330"]["valuation_price_history"], rows=[dict(Date="2026-10-14", Close=100.)],
            symbol="2330", cutoff=at, source=dict(source_final=True))


def two_stock_market(*, cutoff=None):
    from infrastructure.crawler.security_master import capture_security_master, select_security_master
    from tests.test_security_master import synthetic_instruments, synthetic_issuers, OBSERVED
    at = cutoff or after_release()
    instruments = [("股票", [symbol + "　合成普通股", isin, "1994/09/05", "上市", "半導體業", "ESVUFR", ""])
        for symbol, isin in (("2330", "TW0002330008"), ("2317", "TW0002317005"))]
    issuers = [dict(公司代號=s, 公司簡稱="合成", 公司名稱="合成股份有限公司", 營利事業統一編號="12345678", 上市日期="19940905")
        for s in ("2330", "2317")]
    master = capture_security_master(market="TWSE", instrument_content=synthetic_instruments(rows=instruments),
        issuer_content=synthetic_issuers(rows=issuers), fetched_at=OBSERVED)
    sources = {symbol: dict(security=select_security_master(master, symbol, at)) for symbol in ("2330", "2317")}
    dates = pd.bdate_range(end="2026-10-14", periods=61).strftime("%Y-%m-%d").tolist()
    repeated = lambda value: [[value] * 61, [value] * 61]
    return MarketHistoryInput(stock_count=2, stock_numbers=["2330", "2317"], stock_names=["Synthetic A", "Synthetic B"],
        dates=[list(dates), list(dates)], opens=repeated(100000.), highs=repeated(101000.), lows=repeated(99000.),
        closes=repeated(100000.), volumes=repeated(10000.), trade_counts=repeated(100.),
        trade_values=repeated(1_000_000.), bids=repeated(99900.), asks=repeated(100000.),
        pe_ratios=repeated(20.), taiex_closes=repeated(10000.), annual_fundamentals={}, fundamental_snapshots={},
        fundamental_statuses={}, fundamental_status_details={}, signal_dates={"2330": at, "2317": at},
        account_equity=1_000_000., default_sector="SEMI", entry_model="v7", les_v7_native_evidence=True,
        les_v7_rule_baseline=True, les_v7_source_repair=True, market_observed_at=at,
        market_breadth_snapshots={"2026-10-14": dict(as_of="2026-10-14", universe="SYNTHETIC_ONLY",
            eligible_count=2, valid_return_count=2, advancers=0, decliners=0, unchanged=2,
            advancer_ratio=0., median_return=0., coverage=1., source_final=True, taiex_return=0.)},
        native_source_selections=sources)


def monthly_failure(symbol, cutoff):
    result = dict(schema="les-v7-monthly-valuation-selection/v1", symbol=symbol, market="TWSE", status="UNKNOWN",
        rows=[], price_unit="TWD", observed_at=cutoff, capture_error="SYNTHETIC_MONTHLY_FAILURE",
        diagnostics=["SYNTHETIC_MONTHLY_FAILURE"], historical_information_set_certified=False)
    return dict(result, selection_id=identity(result))


def test_monthly_unknown_stays_on_issuer_and_never_poisons_peer_current_quote():
    market = two_stock_market()
    sources = deepcopy(market.native_source_selections)
    sources["2317"]["valuation_price_history"] = monthly_failure("2317", market.signal_dates["2317"])
    before = deepcopy(sources)
    result = capture_comparison_contexts(replace(market, native_source_selections=sources))
    issuer = result.native_source_selections["2317"]["r10_valuation_context"]["issuer"]
    assert issuer["market_evidence"]["source_final"] is False
    assert issuer["market_evidence"]["valuation_history_selection"]["status"] == "UNKNOWN"
    peer = result.native_source_selections["2330"]["r10_valuation_context"]["peers"][0]
    assert peer["symbol"] == "2317" and len(peer["market_rows"]) == 1
    assert peer["market_evidence"]["source_final"] is True
    assert "valuation_history_selection" not in peer["market_evidence"]
    assert peer["market_evidence"]["bars_sha256"] == identity(peer["market_rows"])
    assert sources == before


def test_r11_never_consumes_r12_monthly_failure_projection():
    at = (timestamp(INDUSTRY_AVAILABLE_AT) + timedelta(seconds=1)).isoformat()
    market = two_stock_market(cutoff=at)
    sources = deepcopy(market.native_source_selections)
    sources["2317"]["valuation_price_history"] = monthly_failure("2317", at)
    result = capture_comparison_contexts(replace(market, native_source_selections=sources))
    for symbol in ("2330", "2317"):
        issuer = result.native_source_selections[symbol]["r10_valuation_context"]["issuer"]
        assert issuer["market_evidence"]["source_final"] is True
        assert "valuation_history_selection" not in issuer["market_evidence"]

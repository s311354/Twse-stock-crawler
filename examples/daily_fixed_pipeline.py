"""Native daily computation over an isolated, immutable local fixture.

Only the observation adapter is supplied by the fixture. Financial derivation,
LES, valuation, EREM, typed decisions, COLD_START PAE and report publication use
the production public pipeline. No refresh, SMTP, orders or formal state exist
on this path. Counterfactual evaluation is not historical deployment evidence.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from application.analysis_dataset import AnalysisDatasetBuilder
from application.daily_performance import DailyPerformance, measure_stage
from application.fundamental_service import MopsFundamentalService
from application.market_history import MarketHistoryInput
from application.portfolio_allocation_v2 import PortfolioAllocationV2Pipeline
from application.portfolio_contracts import PortfolioRuntimeContext
from application.replay_contracts import canonical_json, digest
from domain.les_v7.contracts import timestamp
from domain.les_v7.report import validate_report
from examples.compare_les_r14 import reselect_packet
from infrastructure.crawler.valuation_history import ValuationSelectionVerifier
from infrastructure.daily_work import open_daily_work
from infrastructure.fundamentals.mops_queries import MopsQueryRepository
from infrastructure.fundamentals.mops_unit_of_work import SqliteFundamentalUnitOfWork
from infrastructure.legacy_crawler_facade import LegacyCrawlerFacade
from infrastructure.research_archive import write_once
from infrastructure.storage.csv_repository import CsvRepository
from infrastructure.storage.report_evidence import STORE_NAME, iter_report_bundle


PARITY_FIELDS = ("financial_inputs_sha256", "full_decisions_sha256", "full_dataset_sha256",
                 "full_reports_sha256", "references_sha256", "node_bytes_sha256",
                 "account_sha256", "pae_prestate_sha256", "pae_poststate_sha256", "csv_sha256")


class ReadOnlyFundamentals:
    """Query-only facade; even an unexpected command fails at SQLite."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.database_path = self.root / "mops.sqlite3"
        self.queries = MopsQueryRepository(self._connect, lambda: None, raw_root=self.root / "raw")

    def _connect(self):
        connection = sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=1")
        return connection

    def unit_of_work(self):
        return SqliteFundamentalUnitOfWork(self._connect())

    def __getattr__(self, name):
        if name.startswith(("load_", "last_")):
            return getattr(self.queries, name)
        raise AttributeError(name)


class NoRemotePorts:
    """Constructor-compatible ports with no network implementation."""
    def configure_refresh_control(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        raise RuntimeError("FIXED_BENCHMARK_REMOTE_ACCESS_FORBIDDEN:" + name)


def financial_service(fixture, manifest):
    database = Path(fixture) / manifest["mops"]["path"]
    remote = NoRemotePorts()
    return MopsFundamentalService(remote, repository=ReadOnlyFundamentals(database.parent),
        filing_documents=remote, filing_parser=remote)


def _normal(value):
    if isinstance(value, dict):
        return {str(key): _normal(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normal(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _normal(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _hash_records(records):
    sha = hashlib.sha256()
    for record in records:
        encoded = canonical_json(_normal(record)).encode()
        sha.update(len(encoded).to_bytes(8, "big")); sha.update(encoded)
    return sha.hexdigest()


def _canonical_normal_chunks(value, *, string_chunk_chars=64 * 1024):
    """Exactly encode ``canonical_json(_normal(value))`` without a big string.

    Dict key stringification, last-key collision semantics, JSON sorting,
    datetime/numpy adapters and nonfinite-to-null follow the existing helper.
    Strings are escaped in independent character chunks, including JSON report
    strings; no complete escaped report or universe-sized byte string is built.
    """
    if isinstance(value, dict):
        # _normal also stringifies keys before JSON sort_keys. A lightweight
        # key-to-original-key index preserves its last collision winner.
        keys = {str(key): key for key in value}
        yield "{"
        for index, key in enumerate(sorted(keys)):
            if index:
                yield ","
            yield from _canonical_normal_chunks(key, string_chunk_chars=string_chunk_chars)
            yield ":"
            yield from _canonical_normal_chunks(value[keys[key]], string_chunk_chars=string_chunk_chars)
        yield "}"
    elif isinstance(value, (tuple, list)):
        yield "["
        for index, item in enumerate(value):
            if index:
                yield ","
            yield from _canonical_normal_chunks(item, string_chunk_chars=string_chunk_chars)
        yield "]"
    elif isinstance(value, Enum):
        yield from _canonical_normal_chunks(value.value, string_chunk_chars=string_chunk_chars)
    elif isinstance(value, (date, datetime)):
        yield from _canonical_normal_chunks(value.isoformat(), string_chunk_chars=string_chunk_chars)
    elif hasattr(value, "item"):
        yield from _canonical_normal_chunks(value.item(), string_chunk_chars=string_chunk_chars)
    elif isinstance(value, str):
        yield '"'
        for offset in range(0, len(value), string_chunk_chars):
            yield canonical_json(value[offset:offset + string_chunk_chars])[1:-1]
        yield '"'
    elif isinstance(value, float) and not math.isfinite(value):
        yield "null"
    else:
        yield canonical_json(value)


def _dataframe_canonical_chunks(dataset):
    """Byte-equivalent to the old all-at-once full dataset hash input.

    Only one row's shallow mapping exists at once. Scalar missing cells follow
    ``astype(object).where(notna(...), None)``; nested containers are not treated
    as whole-cell NA and retain the old recursive normalization. Column order,
    row order, duplicate-key last winners, scalar types and every field remain
    part of the canonical hash; the dataframe is never mutated or copied.
    """
    columns = list(dataset.columns)
    yield '{"columns":'
    # The old payload normalizes rows, not column labels. Preserve that exact
    # behavior (including a TypeError for unsupported column label objects).
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    yield from encoder.iterencode(columns)
    yield ',"rows":['
    for index, values in enumerate(dataset.itertuples(index=False, name=None)):
        if index:
            yield ","
        row = {}
        for column, value in zip(columns, values, strict=True):
            if not isinstance(value, (dict, tuple, list)):
                missing = pd.isna(value)
                if isinstance(missing, bool) or type(missing).__module__ == "numpy" and getattr(missing, "ndim", None) == 0:
                    if bool(missing):
                        value = None
            # DataFrame.to_dict("records") boxes object-dtype numpy scalars,
            # unlike itertuples. Keep the old cell-level contract without
            # changing normalization of numpy values nested inside containers.
            if isinstance(value, np.datetime64):
                value = pd.Timestamp(value)
            elif isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.bool_):
                value = bool(value)
            row[column] = value
        yield from _canonical_normal_chunks(row)
    yield "]}"


def _dataframe_sha256(dataset):
    sha = hashlib.sha256()
    for chunk in _dataframe_canonical_chunks(dataset):
        sha.update(chunk.encode("utf-8"))
    return sha.hexdigest()


def _file_sha(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def _tree_bytes(root):
    return sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())


def _account(manifest):
    accounts = [row["account"] for row in manifest["stocks"]]
    if not accounts or any(account != accounts[0] for account in accounts):
        raise ValueError("FIXED_BENCHMARK_IDENTICAL_EXPLICIT_ACCOUNT_REQUIRED")
    account = accounts[0]
    if (any(account.get(key) is None for key in
            ("value", "available_cash", "positions", "intended_weight", "capacity_weight", "as_of"))
            or account["positions"] != [] or account["value"] <= 0
            or account["available_cash"] != account["value"]):
        raise ValueError("FIXED_BENCHMARK_EXPLICIT_ALL_CASH_ACCOUNT_REQUIRED")
    return dict(account, as_of=timestamp(manifest["evaluation_cutoff"]).isoformat())


def build_market(manifest, fixture, service, retention):
    """Re-select at original financial cutoff, retain one issuer at a time."""
    from examples.daily_fixed_fixture import load_symbol
    fixture = Path(fixture)
    cutoff = timestamp(manifest["evaluation_cutoff"]).isoformat()
    account = _account(manifest)
    arrays = {key: [] for key in ("Date", "Open", "High", "Low", "Close", "Volume",
                                  "Trades", "Amount", "Bid", "Ask", "PE", "TAIEX Close")}
    snapshots = retention.snapshots()
    annual, native, restrictions, captured, capture_cutoffs, names = {}, {}, {}, {}, {}, []
    industry, financial_hashes, sectors = {}, [], {}
    sessions = set()
    for stock in manifest["stocks"]:
        frozen = load_symbol(fixture, manifest, stock)
        symbol, market = stock["symbol"], stock["market"]
        if market != "TWSE":
            raise ValueError("FIXED_BENCHMARK_EXPECTS_FROZEN_TWSE_UNIVERSE")
        packet = frozen["source_packet"]
        if timestamp(packet["cutoff"]) != timestamp(stock["source_cutoff"]):
            raise ValueError("FIXED_BENCHMARK_SOURCE_CUTOFF_CHANGED")
        selected = reselect_packet(packet, evaluation_cutoff=cutoff)
        source_cutoff = min((stock["source_cutoff"], packet.get("financial_capture_cutoff", stock["source_cutoff"])), key=timestamp)
        if timestamp(source_cutoff) > timestamp(cutoff):
            raise ValueError("FIXED_BENCHMARK_FUTURE_FINANCIAL_SELECTION")
        sectors[symbol] = str((packet.get("industry_input") or {}).get("sector") or
            ((packet.get("security_evidence") or {}).get("instrument") or {}).get("sector") or "ELEC")
        with measure_stage("fundamental_historical_derivation"):
            bundle = service.load_historical(stock.get("source_issuer") or symbol,
                as_of=timestamp(source_cutoff), sector=sectors[symbol], market=market)
        snapshot = dict(bundle.snapshot)
        records = snapshot.get("les_v6_mops_filings", snapshot.get("les_v6_filings", ()))
        with measure_stage("v7_financial_sources"):
            additions = service.load_v7_financial_sources(stock.get("source_issuer") or symbol,
                as_of=source_cutoff, records=records, market=market, source_repair=True)
        snapshot["les_v7_context"] = dict(snapshot.get("les_v7_context", {}), **additions)
        snapshots.retain(symbol, snapshot, cutoff=source_cutoff, market=market)
        annual[symbol] = bundle.annual_fundamentals
        financial_hashes.append(dict(symbol=symbol, source_cutoff=source_cutoff,
            annual=_normal(bundle.annual_fundamentals.reset_index().to_dict("records")),
            snapshot_sha256=digest(_normal(snapshot))))
        rows = packet.get("quote_market_rows", packet["market_rows"])
        if not rows:
            raise ValueError("FIXED_BENCHMARK_EMPTY_SAVED_QUOTES:" + symbol)
        for key in arrays:
            values = [row.get(key) for row in rows]
            if key in {"Open", "High", "Low", "Close", "Bid", "Ask"}:
                values = [None if value is None else value * 1000 for value in values]
            arrays[key].append(values)
        names.append(stock["name"])
        sessions.add(packet["expected_session"])
        captured[symbol] = packet.get("market_evidence", {})
        capture_cutoffs[symbol] = source_cutoff
        restrictions[symbol] = selected.get("restriction_evidence", {})
        sources = dict(security=selected.get("security_evidence"), official_pe=selected.get("official_pe", []))
        if selected.get("calendar_evidence") is not None:
            sources["calendar_evidence"] = selected["calendar_evidence"]
        comparison = selected.get("r10_valuation_context") or {}
        selection = ((comparison.get("issuer") or {}).get("market_evidence") or {}).get("valuation_history_selection")
        if selection is not None:
            sources["valuation_price_history"] = selection
        native[symbol] = sources
        benchmark = (packet.get("industry_input") or {}).get("benchmark")
        if benchmark is not None:
            sector = sectors[symbol]
            if sector in industry and digest(industry[sector]) != digest(benchmark):
                raise ValueError("FIXED_BENCHMARK_CONFLICTING_INDUSTRY_SOURCE")
            industry[sector] = benchmark
        del frozen, packet, selected, bundle, snapshot, records, additions
    if len(sessions) != 1:
        raise ValueError("FIXED_BENCHMARK_ONE_ANALYSIS_SESSION_REQUIRED")
    symbols = list(manifest["symbols"])
    if symbols != [row["symbol"] for row in manifest["stocks"]]:
        raise ValueError("FIXED_BENCHMARK_UNIVERSE_ORDER_CHANGED")
    market = MarketHistoryInput(stock_count=len(symbols), stock_numbers=symbols, stock_names=names,
        dates=arrays["Date"], opens=arrays["Open"], highs=arrays["High"], lows=arrays["Low"], closes=arrays["Close"],
        volumes=arrays["Volume"], trade_counts=arrays["Trades"], trade_values=arrays["Amount"], bids=arrays["Bid"],
        asks=arrays["Ask"], pe_ratios=arrays["PE"], taiex_closes=arrays["TAIEX Close"],
        annual_fundamentals=annual, fundamental_snapshots=snapshots, fundamental_statuses={},
        fundamental_status_details={}, signal_dates={symbol: cutoff for symbol in symbols},
        account_equity=account["value"], default_sector="ELEC", les_v6_portfolio=account,
        trading_restrictions=restrictions, fallback_date=next(iter(sessions)), market_source="TWSE",
        industry_benchmarks=industry, entry_model="v7", les_v7_native_evidence=True,
        les_v7_rule_baseline=True, les_v7_source_repair=True, native_source_selections=native,
        fundamental_capture_cutoffs=capture_cutoffs, captured_market_evidence=captured)
    return market, dict(financial_inputs_sha256=digest(financial_hashes), sectors=sectors, account=account)


@contextmanager
def fixture_valuation_scope(fixture, manifest):
    root = Path(fixture).resolve()
    def resolve(logical):
        relative = manifest.get("valuation_aliases", {}).get(str(logical))
        if relative is None:
            raise ValueError("FIXED_BENCHMARK_MONTHLY_SOURCE_NOT_FROZEN")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("FIXED_BENCHMARK_MONTHLY_SOURCE_OUTSIDE_FIXTURE")
        return path
    verifier = ValuationSelectionVerifier(manifest_path_resolver=resolve)
    try:
        yield verifier.verify
    finally:
        verifier.close()


def _maxprofits(market):
    # Reuse the actual legacy profit diagnostic, including opening-price input,
    # rounding, scaled fee and missing-data policy; these are not model targets.
    facade = LegacyCrawlerFacade()
    facade.stocklistsize = market.stock_count
    facade.daily_stocks = market.opens
    facade.stocknumbers = market.stock_numbers
    facade.stocksprice = []
    return facade.cal_max_profit()


def run_round(*, fixture, manifest, output, workspace, cache_path, iteration, stage_callback=None):
    """New work context per round; persistent parse cache and CAS are reused."""
    fixture, output = Path(fixture).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / f"round-{iteration}-receipt.json"
    if receipt_path.exists():
        raise ValueError("FIXED_BENCHMARK_ROUND_ALREADY_EXISTS")
    before_bytes = _tree_bytes(output)
    probe = DailyPerformance(stage_callback=stage_callback)
    with probe.activate(), measure_stage("total"):
        with open_daily_work(parse_cache_path=cache_path) as work:
            service = financial_service(fixture, manifest)
            with measure_stage("fundamentals"):
                market, inputs = build_market(manifest, fixture, service, work.snapshot_retention)
            scope = lambda: fixture_valuation_scope(fixture, manifest)
            builder = AnalysisDatasetBuilder(valuation_history_verifier_scope=scope)
            with measure_stage("strategies"):
                with measure_stage("profit_diagnostics"):
                    maxprofits = _maxprofits(market)
                decisions = builder.build_decisions(market, maxprofits)
                with measure_stage("pae"):
                    decisions = PortfolioAllocationV2Pipeline(portfolio_id="fixed-elec-performance").enrich(decisions,
                        PortfolioRuntimeContext(account_equity=inputs["account"]["value"],
                            current_cash=inputs["account"]["available_cash"], positions=(),
                            sector_by_symbol=inputs["sectors"], default_sector="ELEC"))
                dataset = builder.project_decisions(decisions)
            with measure_stage("full_result_hashing"):
                decision_hash = _hash_records(asdict(row) for row in decisions)
                dataset_hash = _dataframe_sha256(dataset)
            prefix = f"round-{iteration}"
            with measure_stage("report"):
                CsvRepository(output, workspace_root=workspace, valuation_history_verifier_scope=scope).write_analysis_dataset(prefix, dataset)
            with measure_stage("complete_result_verification"):
                bundle = output / f"{prefix}_les_v7_reports.json"
                reports = hashlib.sha256()
                expected_reports = {str(row.value("證券代號")): digest(json.loads(row.value("LES v7 Report"))) for row in decisions}
                seen = []
                for report in iter_report_bundle(bundle, valuation_history_verifier_scope=scope):
                    validate_report(report)
                    if digest(report) != expected_reports.get(report["symbol"]):
                        raise ValueError("FIXED_BENCHMARK_RESTORED_FULL_REPORT_MISMATCH")
                    encoded = canonical_json(report).encode()
                    reports.update(len(encoded).to_bytes(8, "big")); reports.update(encoded)
                    seen.append(report["symbol"])
                if seen != list(manifest["symbols"]):
                    raise ValueError("FIXED_BENCHMARK_REPORT_UNIVERSE_CHANGED")
                refs_hash = digest(json.loads(bundle.read_bytes()))
                nodes = hashlib.sha256()
                with closing(sqlite3.connect((output / STORE_NAME).as_uri() + "?mode=ro", uri=True)) as connection:
                    connection.execute("PRAGMA query_only=1")
                    if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                        raise ValueError("FIXED_BENCHMARK_CLOSED_STORE_CORRUPT")
                    node_count = 0
                    for sha, body in connection.execute("SELECT sha256,body FROM report_objects ORDER BY sha256"):
                        nodes.update(sha.encode()); nodes.update(body); node_count += 1
                empty_state = dict(schema="benchmark-explicit-pae-state/v1", observations=[], sessions=[])
                values = dict(financial_inputs_sha256=inputs["financial_inputs_sha256"],
                    full_decisions_sha256=decision_hash, full_dataset_sha256=dataset_hash,
                    full_reports_sha256=reports.hexdigest(), references_sha256=refs_hash,
                    node_bytes_sha256=nodes.hexdigest(), account_sha256=digest(inputs["account"]),
                    pae_prestate_sha256=digest(empty_state), pae_poststate_sha256=digest(empty_state),
                    result_count=len(decisions), restored_report_count=len(seen), node_count=node_count,
                    csv_sha256=_file_sha(output / f"{prefix}_analysis_dataset.csv"),
                    evidence_store_bytes=(output / STORE_NAME).stat().st_size,
                    new_retention_bytes=_tree_bytes(output) - before_bytes)
            work_stats = work.stats
    measurement = dict(schema="daily-fixed-native-round/v1", iteration=iteration,
        temperature="process_cold_new_caches" if iteration == 0 else "warm_disk_new_work_context",
        input_id=manifest["input_id"], evaluation_cutoff=manifest["evaluation_cutoff"],
        account_policy="EXPLICIT_RETAINED_ALL_CASH_COUNTERFACTUAL",
        pae_state_policy="EXPLICIT_EMPTY_NULL_PERSISTENCE_NO_FORMAL_STATE",
        no_network=True, no_mail=True, no_orders=True, work_stats=work_stats, **values, **probe.snapshot())
    write_once(receipt_path, measurement)
    return measurement

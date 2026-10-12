"""Cutoff selection over a pinned archive, using the existing PIT derivation."""
from __future__ import annotations

import datetime as dt
import json
import math
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass

import pandas as pd

from application.fundamental_service import MopsFundamentalService
from application.market_history import MarketHistoryInput
from application.replay_contracts import ReplayRequest, aware_timestamp, canonical_json, digest
from application.research_evidence import RAW_EVIDENCE_SCHEMA, FilingHistoryIndex, calendar_at, materialize_filings
from application.pdf_upload_policy import PDF_UPLOAD_SOURCE
from application.restriction_history import HistoricalRestrictionLedger
from domain.data_provenance import TRUSTED_AVAILABILITY_SOURCES
from domain.fundamental_derivation import FiledFact, FiledFinancials
from domain.fundamental_models import FundamentalSnapshot
from domain.models import TwseColumns as C
from domain.strategies.low_entry_v6_quality import verification_diagnostics
from domain.trading_restrictions import restriction_check


class ArchiveFundamentals:
    """Minimal read-only repository for the existing load_historical use case."""

    def __init__(self, filings, batches):
        self.filings, self.batches = filings, batches

    def unit_of_work(self):
        return nullcontext()

    def load_filed_financials(self, symbol, as_of, *, uow=None):
        return [f for f in self.filings.get(symbol, ()) if aware_timestamp(f.available_at) <= aware_timestamp(as_of)]

    def load_financial_batch_snapshots(self, symbol, *, as_of, uow=None):
        by_period = {}
        for batch in sorted(self.batches.get(symbol, ()), key=lambda b: aware_timestamp(b.observed_at)):
            if aware_timestamp(batch.observed_at) <= aware_timestamp(as_of):
                by_period[batch.period] = batch
        return list(by_period.values())


@dataclass(frozen=True)
class ReplayInputs:
    market: MarketHistoryInput
    sessions: tuple[str, ...]
    signal_date: str | None
    diagnostics: dict
    evidence: dict
    account_validation: dict
    calibrations: dict


class PinnedReplayEvidence:
    def __init__(self, evidence_id: str, payload: dict, *, parse_filing=None, annual_calendar=None,
                 document_reader=None, document_check=None, filing_index: FilingHistoryIndex | None = None):
        if digest(payload) != evidence_id:
            raise ValueError("REPLAY_EVIDENCE_SIGNATURE_MISMATCH")
        self.evidence_id = evidence_id
        self._json = canonical_json(payload)
        self._parse_filing = parse_filing
        self._annual_calendar = annual_calendar
        self._payload = None
        self._cache = None
        self._document_reader, self._document_check = document_reader, document_check
        self._filing_index = filing_index

    @classmethod
    def from_file(cls, path, *, loader, parse_filing, annual_calendar, cache, version,
                  document_reader=None, document_check=None, filing_index_factory=None):
        """Loader validates once; sole ownership stays private in this context.

        No public caller-owned payload or mutable selected reference is retained.
        The ordinary constructor remains the independent uncached oracle.
        """
        evidence_id, payload = loader(path)
        obj = cls.__new__(cls)
        obj.evidence_id, obj._payload, obj._json = evidence_id, payload, None
        obj._parse_filing, obj._annual_calendar = parse_filing, annual_calendar
        obj._cache, obj._version = cache, version
        obj._document_reader, obj._document_check = document_reader, document_check
        obj._filing_index = filing_index_factory(evidence_id) if filing_index_factory is not None else None
        obj._bars_by_day = {}
        for item in payload['bars']:
            day = dt.datetime.strptime(item['date'], '%Y%m%d').date().isoformat()
            obj._bars_by_day.setdefault(day, []).append(item)
        return obj

    def close(self):
        if self._cache is not None:
            self._cache.clear()
        self._payload = self._json = None
        self._bars_by_day = {}

    def _prepared(self, request):
        if self._payload is None:
            return json.loads(self._json)
        payload = dict(self._payload)
        payload['bars'] = [row for day, rows in self._bars_by_day.items()
                           if str(request.history_start) <= day <= str(request.history_end) for row in rows]
        return payload

    def select(self, request: ReplayRequest) -> ReplayInputs:
        if self._document_check is not None:
            self._document_check()
        payload = self._prepared(request)
        if request.evidence_id != self.evidence_id or request.market != payload["market"] or request.sector != payload["sector"]:
            raise ValueError("REPLAY_EVIDENCE_ID_OR_MARKET_MISMATCH")
        if not set(request.symbols).issubset(payload["symbols"]):
            raise ValueError("REPLAY_UNIVERSE_NOT_IN_EVIDENCE")
        if request.history_start.isoformat() < payload["start"] or request.history_end.isoformat() > payload["end"]:
            raise ValueError("REPLAY_RANGE_OUTSIDE_EVIDENCE")
        strict_raw = payload.get("schema") == RAW_EVIDENCE_SCHEMA
        def verify(proof, *, binding):
            from application.research_evidence import verify_publication
            if self._cache is None:
                return verify_publication(proof, binding=binding, document_reader=self._document_reader)
            return self._cache.memo('publication', (self._version, digest(proof), digest(binding)),
                                    lambda: verify_publication(proof, binding=binding,
                                                               document_reader=self._document_reader))
        days, calendar_excluded = calendar_at(payload["calendar"], request.as_of, strict=strict_raw,
                                             annual_calendar=self._annual_calendar, verifier=verify)
        raw_errors = {}
        if strict_raw:
            if self._parse_filing is None:
                raise ValueError("RAW_FILING_ADAPTER_REQUIRED")
            payload['filings'] = {s: payload.get('filings', {}).get(s, []) for s in request.symbols}
            if self._filing_index is not None:
                payload["filings"], raw_errors = self._filing_index.load(payload, request.symbols, request.as_of)
            elif self._cache is None:
                payload["filings"], raw_errors = materialize_filings(payload, parse_filing=self._parse_filing,
                                                                  document_reader=self._document_reader)
            else:
                records, raw_errors = {}, {}
                for symbol in request.symbols:
                    records[symbol], raw_errors[symbol] = [], []
                    for item in payload.get('filings', {}).get(symbol, []):
                        key = (self.evidence_id, self._version, symbol, digest(item))
                        def materialize(item=item, symbol=symbol):
                            return materialize_filings(dict(raw_documents=payload['raw_documents'],
                                filings={symbol: [item]}), parse_filing=self._parse_filing,
                                cache=self._cache, version=self._version, document_reader=self._document_reader)
                        rows, errors = self._cache.memo('filing_proof', key, materialize)
                        records[symbol].extend(rows[symbol]); raw_errors[symbol].extend(errors[symbol])
                payload['filings'] = records
            from domain.market_breadth import derive_twse_market_breadth
            payload['breadth'] = {}
            for day, item in payload.get('breadth_sources', {}).items():
                if not str(request.history_start) <= day <= str(request.history_end):
                    continue
                def compute(day=day, item=item):
                    return dict(snapshot=derive_twse_market_breadth(item['rows'], as_of=day,
                        universe=item['universe'], source_final=True).to_mapping(),
                        source_rows_sha256=digest(item['rows']), market_type=item['market_type'])
                payload['breadth'][day] = (compute() if self._cache is None else
                    self._cache.memo('breadth', (self.evidence_id, self._version, day), compute))
        selected_days = [(request.history_start + dt.timedelta(days=i)).isoformat()
                         for i in range((request.history_end - request.history_start).days + 1)]
        unknown_calendar = [day for day in selected_days if days.get(day) not in {"OPEN", "CLOSED"}]
        if unknown_calendar and not strict_raw:
            raise ValueError("REPLAY_CALENDAR_UNKNOWN")
        sessions = tuple(day for day in selected_days if days.get(day) == "OPEN")
        signal = sessions[-1] if sessions else None
        # UNKNOWN calendar is still a blocker, but existing official bars can
        # support diagnostic technical scores. This does NOT label the day OPEN
        # or make it a successful PAE persistence observation.
        usable_dates = set(sessions)
        if strict_raw:
            observed = {dt.datetime.strptime(item["date"], "%Y%m%d").date().isoformat()
                        for item in payload["bars"] if item["dataset"] == "STOCK_TABLE" and item["data"]}
            usable_dates.update(observed & set(unknown_calendar))
            signal = max(usable_dates) if usable_dates else None
        diagnostic = dict(calendar_status=days.get(request.history_end.isoformat(), "UNKNOWN"),
                          evidence_schema=payload.get("schema"),
                          calendar_unknown_dates=unknown_calendar, calendar_excluded=calendar_excluded,
                          raw_filing_errors=raw_errors,
                          non_trading_policy="KEEP_CUTOFF_USE_LAST_OPEN_WITHIN_EXPLICIT_RANGE",
                          missing_bars={}, invalid_bars=[], preparation_gaps=payload.get("preparation_gaps", []))
        bars, benchmarks = {}, {}
        benchmark_conflicts = set()
        conflicts = set()
        for item in payload["bars"]:
            day = dt.datetime.strptime(item["date"], "%Y%m%d").date().isoformat()
            if day not in usable_dates:
                continue
            if item["dataset"] == "TAIEX":
                if day in benchmarks and benchmarks[day] != item["data"]:
                    benchmark_conflicts.add(day)
                benchmarks[day] = item["data"]
                continue
            for row in item["data"]:
                if not isinstance(row, list) or len(row) < C.REQUIRED_WIDTH:
                    diagnostic["invalid_bars"].append(dict(date=day, reason="SCHEMA_CHANGED"))
                    continue
                key = (str(row[C.STOCK_NO]), day)
                if key[0] not in request.symbols:
                    continue
                if key in bars and bars[key] != row:
                    conflicts.add(key)
                bars[key] = row
        for key in conflicts:
            bars.pop(key, None)
            diagnostic["invalid_bars"].append(dict(symbol=key[0], date=key[1], reason="CONFLICTING_MARKET_ROWS"))
        for day in benchmark_conflicts:
            benchmarks.pop(day, None)
            diagnostic["invalid_bars"].append(dict(date=day, reason="CONFLICTING_BENCHMARK_ROWS"))

        filings, batches, receipts, excluded = {}, {}, {}, {}
        for symbol in request.symbols:
            filings[symbol], batches[symbol], receipts[symbol], excluded[symbol] = [], [], [], []
            for record in payload.get("filings", {}).get(symbol, []):
                values = dict(record)
                receipt = values.pop("receipt", {})
                reason = None
                try:
                    at = aware_timestamp(values["available_at"])
                    if at > request.as_of:
                        reason = "AFTER_CUTOFF"
                    elif (values.get("availability_source") not in TRUSTED_AVAILABILITY_SOURCES
                          and not (strict_raw and values.get("availability_source") == PDF_UPLOAD_SOURCE)):
                        reason = "UNVERIFIED_AVAILABILITY"
                    elif values.get("period_end") and dt.date.fromisoformat(values["period_end"]) > request.as_of.date():
                        reason = "FUTURE_FISCAL_PERIOD"
                except (KeyError, ValueError, TypeError):
                    reason = "INVALID_AVAILABILITY"
                audit = {k: values.get(k) for k in ("filing_id", "content_sha256", "available_at", "availability_source", "fiscal_year", "quarter", "revision_number", "statement_scope", "source_url")}
                audit.update(receipt)
                if reason:
                    excluded[symbol].append(dict(audit, reason=reason))
                    continue
                values["facts"] = tuple(FiledFact(**dict(fact, dimensions=tuple(tuple(d) for d in fact.get("dimensions", ())))) for fact in values["facts"])
                values["taxonomy_schema_refs"] = tuple(values.get("taxonomy_schema_refs", ()))
                filings[symbol].append(FiledFinancials(**values))
                receipts[symbol].append(audit)
            # Scope precedence is period-local; do not join consolidated and
            # individual facts into an invented statement or mix revision times.
            from application.research_evidence import select_filing_versions
            indices, version_conflicts = select_filing_versions([
                {key: getattr(f, key) for key in ("fiscal_year", "quarter", "statement_scope", "available_at",
                                                "revision_number", "content_sha256")} for f in filings[symbol]])
            selected_filings = [filings[symbol][i] for i in indices]
            excluded[symbol].extend({k: v for k, v in conflict.items() if k != "indices"} for conflict in version_conflicts)
            filings[symbol] = selected_filings
            selected_ids = {(f.filing_id, f.content_sha256) for f in selected_filings}
            receipts[symbol] = [dict(r, selected=(r["filing_id"], r["content_sha256"]) in selected_ids) for r in receipts[symbol]]
            if any(r.get("publication_policy") and r["selected"] for r in receipts[symbol]):
                diagnostic.setdefault("publication_time_assumptions", {})[symbol] = [
                    r for r in receipts[symbol] if r.get("publication_policy") and r["selected"]]
            for raw in payload.get("batches", {}).get(symbol, []) if request.market == "TWSE" else []:
                try:
                    reason = "AFTER_CUTOFF" if aware_timestamp(raw["observed_at"]) > request.as_of else None
                except (KeyError, ValueError, TypeError):
                    reason = "INVALID_AVAILABILITY"
                if reason is None:
                    batches[symbol].append(FundamentalSnapshot(**raw))
                else:
                    excluded[symbol].append(dict(source="TWSE_OPENAPI", available_at=raw.get("observed_at"),
                                                  content_sha256=raw.get("content_sha256"), reason=reason))
        repo = ArchiveFundamentals(filings, batches)
        # No live ports can be invoked by this read-only native derivation path.
        service = MopsFundamentalService(object(), repo, filing_documents=object(), filing_parser=object())
        annual, snapshots, financial_diagnostics = {}, {}, {}
        for symbol in request.symbols:
            bundle = service.load_historical(symbol, as_of=request.as_of, sector=request.sector, market=request.market)
            annual[symbol], snapshots[symbol] = bundle.annual_fundamentals, bundle.snapshot
            records = bundle.snapshot.get("les_v6_filings", [])
            financial_diagnostics[symbol] = verification_diagnostics(records, symbol=symbol, as_of=request.as_of, market=request.market)
            # Native diagnostics own the reported-period requirement. A date's
            # last completed quarter is not a publication deadline, and eight
            # diagnostic quarters are not mandatory Quality scoring inputs.

        restrictions, restriction_diagnostics = {}, {}
        ledger = HistoricalRestrictionLedger(payload["restriction_ledger"]) if payload.get("restriction_ledger") else None
        reconstructed = ledger.reconstruct(request.symbols, market=request.market, market_data_date=signal,
                                           as_of=request.as_of) if ledger and signal else {}
        for symbol in request.symbols:
            candidates = []
            for row in payload.get("restrictions", []):
                if row["symbol"] != symbol:
                    continue
                try:
                    reason = "AFTER_CUTOFF" if aware_timestamp(row["evidence"]["available_at"]) > request.as_of else None
                except (KeyError, ValueError, TypeError):
                    reason = "INVALID_AVAILABILITY"
                if reason is None:
                    candidates.append(row["evidence"])
                else:
                    excluded[symbol].append(dict(source="TRADING_RESTRICTIONS", evidence=row["evidence"], reason=reason))
            evidence = {}
            if candidates:
                latest = max(aware_timestamp(row["available_at"]) for row in candidates)
                newest = {digest(row): row for row in candidates if aware_timestamp(row["available_at"]) == latest}
                if len(newest) == 1:
                    evidence = next(iter(newest.values()))
                else:
                    restriction_diagnostics[symbol] = {"status": "UNKNOWN", "reason": "CONFLICTING_RESTRICTION_OBSERVATIONS"}
            if symbol not in restriction_diagnostics:
                check = restriction_check(evidence, market=request.market, symbol=symbol, market_data_date=signal or request.history_end.isoformat(),
                                          decision_time=request.as_of, allow_historical=True)
                if symbol in reconstructed:
                    historical = restriction_check(reconstructed[symbol], market=request.market, symbol=symbol, market_data_date=signal,
                                                   decision_time=request.as_of, allow_historical=True)
                    if check["status"] != "UNKNOWN" and historical["status"] != "UNKNOWN" and check["status"] != historical["status"]:
                        check = {"status": "UNKNOWN", "reason": "CONFLICTING_RESTRICTION_SOURCES"}
                        evidence = {}
                    elif check["status"] == "UNKNOWN":
                        evidence, check = reconstructed[symbol], historical
                restriction_diagnostics[symbol] = check
                if check["status"] != "UNKNOWN":
                    restrictions[symbol] = evidence

        def numeric(value):
            try:
                number = float(str(value).replace(",", ""))
                return number if math.isfinite(number) else None
            except (TypeError, ValueError):
                return None

        calibrations, calibration_diagnostics = {}, {}
        for symbol in request.symbols:
            eligible, rejected = [], []
            for row in payload.get("kelly_calibrations", []):
                if row.get("symbol") != symbol:
                    continue
                reason = None
                try:
                    available = aware_timestamp(row["available_at"])
                    trained = aware_timestamp(row["training_end"])
                    weight = numeric(row.get("half_kelly_weight"))
                    training_start = aware_timestamp(row["training_start"]) if strict_raw else None
                    if not trained <= available <= request.as_of or (training_start is not None and training_start > trained):
                        reason = "CALIBRATION_AFTER_CUTOFF_OR_INVALID_TRAINING_RANGE"
                    elif (row.get("market") != request.market or row.get("model_version") != request.versions.model
                          or row.get("parameter_sha256") != request.versions.parameters):
                        reason = "CALIBRATION_VERSION_OR_MARKET_MISMATCH"
                    elif (weight is None or not 0 < weight <= 1 or not row.get("source_id") or not row.get("method")
                          or type(row.get("sample_count")) is not int or row["sample_count"] <= 0
                          or (strict_raw and (not row.get("calibration_version") or not row.get("source_sha256")))):
                        reason = "INVALID_EMPIRICAL_CALIBRATION"
                except (KeyError, ValueError, TypeError):
                    reason = "INVALID_CALIBRATION_EVIDENCE"
                if reason:
                    rejected.append(dict(row, exclusion_reason=reason))
                else:
                    eligible.append(row)
            if eligible:
                latest = max(aware_timestamp(row["available_at"]) for row in eligible)
                newest = {digest(row): row for row in eligible if aware_timestamp(row["available_at"]) == latest}
                if len(newest) == 1:
                    calibrations[symbol] = next(iter(newest.values()))
            calibration_diagnostics[symbol] = dict(status="PASS" if symbol in calibrations else "UNKNOWN",
                reason=None if symbol in calibrations else "MISSING_OR_CONFLICTING_KELLY_CALIBRATION", excluded=rejected)
        diagnostic["kelly"] = calibration_diagnostics

        histories = {symbol: [] for symbol in request.symbols}
        for (symbol, day), row in sorted(bars.items()):
            o, h, l, c, v = [numeric(row[i]) for i in (C.OPEN, C.HIGH, C.LOW, C.CLOSE, C.VOLUME)]
            if any(value is None for value in (o, h, l, c, v)) or not (0 < l <= min(o, c) <= max(o, c) <= h) or v < 0:
                diagnostic["invalid_bars"].append(dict(symbol=symbol, date=day, reason="OHLCV_UNAVAILABLE"))
                continue
            histories[symbol].append(dict(date=day, name=row[C.NAME], open=o*1000, high=h*1000, low=l*1000, close=c*1000,
                volume=v, trades=numeric(row[C.TRADE_COUNT]), amount=numeric(row[C.TRADE_VALUE]),
                bid=(numeric(row[C.BID_PRICE]) or 0)*1000 or None, ask=(numeric(row[C.ASK_PRICE]) or 0)*1000 or None,
                pe=numeric(row[C.PE]), taiex=numeric(benchmarks.get(day))))
        diagnostic["bar_exemptions"] = {}
        diagnostic["lifecycle_excluded"] = []
        for symbol, rows in histories.items():
            exempt = {}
            # Listing/halt evidence uses the same reviewed publication contract.
            # No inference from the first observed bar or an empty quote query.
            for event in payload.get("security_lifecycle", {}).get(symbol, []):
                try:
                    binding = {k: event[k] for k in ("symbol", "market", "kind", "start", "end")}
                    at = aware_timestamp(verify(event["proof"], binding=binding))
                    if binding["symbol"] != symbol or binding["market"] != request.market or at > request.as_of:
                        continue
                    if event["kind"] not in {"PRE_LISTING", "SUSPENDED", "DELISTED"}:
                        continue
                    for day in sessions:
                        if event["start"] <= day <= event["end"]:
                            if any(row["date"] == day for row in rows):
                                diagnostic["invalid_bars"].append(dict(symbol=symbol, date=day,
                                    reason="BAR_CONFLICTS_WITH_SECURITY_LIFECYCLE"))
                            else:
                                exempt[day] = event["kind"]
                except (KeyError, ValueError, TypeError) as error:
                    diagnostic["lifecycle_excluded"].append(dict(symbol=symbol, reason=str(error), event=event))
            diagnostic["bar_exemptions"][symbol] = exempt
            diagnostic["missing_bars"][symbol] = sorted(set(sessions) - {row["date"] for row in rows} - exempt.keys())
        breadth = {}
        previous_taiex = None
        for day in sorted(usable_dates):
            current_taiex = numeric(benchmarks.get(day))
            if day in payload.get("breadth", {}):
                breadth[day] = dict(payload["breadth"][day]["snapshot"], taiex_return=(
                    current_taiex / previous_taiex - 1 if current_taiex is not None and previous_taiex not in (None, 0) else None))
            previous_taiex = current_taiex
        diagnostic["missing_breadth_dates"] = sorted(set(sessions) - breadth.keys())
        diagnostic["missing_taiex_dates"] = [day for day in sessions if numeric(benchmarks.get(day)) is None]
        diagnostic.update(financials=financial_diagnostics, excluded_evidence=excluded, restrictions=restriction_diagnostics)
        active = tuple(s for s in sorted(request.symbols) if histories[s])
        def values(key):
            return tuple(tuple(row[key] for row in histories[s]) for s in active)
        positions = {h.symbol: h for h in request.account.holdings}
        from application.industry_benchmark import prepare_industry_benchmark
        from domain.industry_benchmark import sector_code
        industry_benchmarks = {}
        industry_errors = {}
        for sector, packet in payload.get("industry_benchmarks", {}).items():
            try:
                def prepare():
                    return prepare_industry_benchmark(packet, document_reader=self._document_reader)
                prepared = (prepare() if self._cache is None else self._cache.memo(
                    "industry_benchmark", (self.evidence_id, self._version, digest(packet)), prepare))
                if prepared is None or sector_code(prepared["sector"]) != sector_code(sector):
                    raise ValueError("INDUSTRY_SECTOR_KEY_MISMATCH")
                industry_benchmarks[sector_code(sector)] = prepared
            except (ValueError, KeyError, TypeError) as error:
                industry_errors[sector_code(sector)] = str(error)
        diagnostic["industry_benchmark_import_errors"] = industry_errors
        market = MarketHistoryInput(
            stock_count=len(active), stock_numbers=active, stock_names=tuple(histories[s][-1]["name"] for s in active),
            dates=values("date"), opens=values("open"), highs=values("high"), lows=values("low"), closes=values("close"),
            volumes=values("volume"), trade_counts=values("trades"), trade_values=values("amount"), bids=values("bid"), asks=values("ask"),
            pe_ratios=values("pe"), taiex_closes=values("taiex"), annual_fundamentals=annual, fundamental_snapshots=snapshots,
            fundamental_statuses={}, fundamental_status_details={}, signal_dates={s: request.as_of.isoformat() for s in active},
            account_equity=request.account.account_equity, default_sector=request.sector,
            les_v6_portfolio=dict(value=request.account.account_equity, available_cash=request.account.cash, intended_weight=.25, capacity_weight=.25),
            trading_restrictions=restrictions, fallback_date=signal or request.history_end.isoformat(), market_source=request.market,
            market_breadth_snapshots=breadth,
            industry_benchmarks=industry_benchmarks,
            replay_as_of=request.as_of.isoformat(), entry_les_by_stock={s:h.entry_les for s,h in positions.items() if h.entry_les is not None},
            entry_dates_by_stock={s:h.position().entry_date for s,h in positions.items()},
            entry_decisions_by_stock={s:h.entry_decision for s,h in positions.items() if h.entry_decision is not None})
        marks = {s: dict(price=rows[-1]["close"]/1000, date=rows[-1]["date"], source=request.market)
                 for s, rows in histories.items() if rows and rows[-1]["date"] == signal}
        missing_marks = sorted(set(positions) - marks.keys())
        account = dict(status="UNKNOWN" if missing_marks else "PASS", missing_marks=missing_marks,
                       marks={s:marks[s] for s in positions if s in marks})
        if not missing_marks:
            computed = request.account.cash + sum(h.shares * marks[s]["price"] for s,h in positions.items())
            account["computed_equity"] = computed
            if abs(computed - request.account.account_equity) > max(.01, request.account.account_equity * 1e-6):
                raise ValueError("ACCOUNT_EQUITY_RECONCILIATION_FAILED")
        selected = dict(filings=receipts, batches={s:[dict(source="TWSE_OPENAPI", observed_at=b.observed_at, content_sha256=b.content_sha256) for b in values] for s,values in batches.items()},
                        restrictions=restrictions, bars=histories, breadth={day: payload["breadth"][day] for day in breadth},
                        calibrations=calibrations, market_policy=payload.get("market_policy"))
        state_sessions = tuple(sorted(day for day, status in days.items()
                                      if status == "OPEN" and day <= request.history_end.isoformat()))
        result = ReplayInputs(market, state_sessions, signal, diagnostic, selected, account, calibrations)
        # Selection owns every nested value; neither model nor audit can mutate
        # the context's raw evidence or a subsequent day's cached inputs.
        return deepcopy(result) if self._payload is not None else result

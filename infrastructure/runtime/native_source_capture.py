"""One bounded current-data capture, detached per-stock source projections.

No historical replay, model inference, accounts, notifications or PAE writes.
Raw originals are retained once; stocks carry compact inclusion proofs only.
"""
from datetime import datetime, timezone
from pathlib import Path
import logging

from domain.errors import DataSourceError
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.official_pe import OfficialPeIndex
from infrastructure.crawler.official_pe import OfficialPeClient, OfficialPeRepository
from infrastructure.crawler.security_master import (
    collect_security_master, load_security_master, prepare_security_master, select_security_master,
)
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig


class NativeSourceCapture:
    """At most two master requests and one current-session P/E request."""
    def __init__(self, root, *, http_client=None, clock=None, collector=collect_security_master,
                 master_loader=load_security_master, pe_client=None, pe_repository=None,
                 valuation_history_database=None, valuation_history_collector=None, calendar_collector=None):
        self.root = Path(root)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.http_client = http_client or TwseHttpClient(TwseHttpConfig(
            timeout_seconds=30, max_attempts=1, min_request_interval_seconds=1,
            allow_insecure_ssl_fallback=False))
        self.collector, self.master_loader = collector, master_loader
        self.pe_client = pe_client or OfficialPeClient(self.http_client, clock=self.clock)
        self.pe_repository = pe_repository or OfficialPeRepository(self.root / "official-pe.sqlite3")
        self.master = None
        self.pe_index = OfficialPeIndex([])
        self.diagnostics = []
        self.master_manifest = None
        self.market = None
        self.valuation_history_database = valuation_history_database
        self.valuation_history_collector = valuation_history_collector
        self.valuation_history = {}
        self.calendar_collector = calendar_collector
        self.calendar_evidence = None

    def collect(self, crawler, *, market):
        self.master = None
        self.master_manifest = None
        self.diagnostics = []
        self.pe_index = OfficialPeIndex([])
        self.market = market
        self.valuation_history = {}
        self.calendar_evidence = None
        dates = sorted(set(str(day)[:10] for day in crawler.iso_scheduled_times))[-60:]
        if not dates:
            self.diagnostics.append("NO_COMPLETED_SESSION_FOR_SOURCE_CAPTURE")
            return self
        # Fetch only the newest completed table. Previously captured exact-date
        # observations are reusable; no hidden 60-request backfill is performed.
        try:
            path = self.collector(self.root / "security-master", market,
                client=self.http_client, now=lambda: self.clock().isoformat())
            packet, documents = self.master_loader(path)
            self.master = prepare_security_master(packet, document_reader=documents.read,
                                                   check_sources=documents.check)
            self.master_manifest = str(path)
        except (DataSourceError, OSError, ValueError) as error:
            self.diagnostics.append("SECURITY_MASTER_CAPTURE_UNAVAILABLE:" + str(error))
            logging.warning("Native security master unavailable; source gates remain UNKNOWN: %s", error)
        try:
            observed = self.pe_client.fetch(market=market, trade_date=dates[-1])
            self.pe_repository.record(observed)
        except (DataSourceError, OSError, ValueError) as error:
            self.diagnostics.append("OFFICIAL_PE_CAPTURE_UNAVAILABLE:" + str(error))
            logging.warning("Dated official P/E unavailable; no source substitution: %s", error)
        # Integrity failures in an existing cache are not permission to trust a
        # different live batch or pretend that the missing history is complete.
        self.pe_index = OfficialPeIndex(self.pe_repository.load_as_of(
            market=market, trade_dates=dates, cutoff=self.clock().isoformat()))
        from domain.les_v7.rule_baseline import CLOSE_POLICIES, current_source_baseline_version
        if (self.valuation_history_database is not None and market == "TWSE"
                and str(getattr(crawler, "analysis_universe", "")) in {"ELEC", "Stocktype.ELEC"}
                and current_source_baseline_version(self.clock()) in CLOSE_POLICIES):
            self._collect_valuation_history(crawler, reference_session=dates[-1])
        from domain.les_v7.rule_baseline import LIMIT_POLICIES, UNIFIED_BASELINE_POLICY
        # Every descendant of the R18 LIMIT route still consumes the same
        # official session evidence. A literal list ending at R19 dropped it
        # for R20--R23 and manufactured structural UNKNOWNs after upgrades.
        # Preserve R16/older capture behavior and the historical adapter's
        # separate no-current-observation contract.
        if current_source_baseline_version(self.clock()) in {UNIFIED_BASELINE_POLICY, *LIMIT_POLICIES}:
            self._collect_calendar(crawler, market=market)
        return self

    def _collect_calendar(self, crawler, *, market):
        """One source-bound calendar for all required per-stock 120-day windows."""
        from application.evidence_context import ReadOnlyEvidence
        from infrastructure.calendar.current_observation import capture_current_calendar, unavailable
        try:
            windows = [sorted(set(str(day)[:10] for day in values))[-121:]
                       for values in crawler.daily_dates if values]
            starts, ends = [values[0] for values in windows if values], [values[-1] for values in windows if values]
            if not starts or not ends:
                raise ValueError("R17_CALENDAR_REQUIRED_SESSION_SCOPE_MISSING")
            self.calendar_evidence = (self.calendar_collector or capture_current_calendar)(
                self.root / "current-calendar", market=market, start=min(starts), end=max(ends),
                client=self.http_client, clock=self.clock)
        except (DataSourceError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            reason = "R17_CURRENT_CALENDAR_CAPTURE_UNAVAILABLE:" + str(error)
            self.calendar_evidence = unavailable(market=market, observed_at=self.clock(), reason=reason)
            self.diagnostics.append(reason)
            logging.warning("Official current calendar unavailable; session coverage remains UNKNOWN: %s", error)
        self.calendar_evidence = ReadOnlyEvidence(self.calendar_evidence)

    def _collect_valuation_history(self, crawler, *, reference_session):
        """One independent cache-only price capture; no technical-window change."""
        from infrastructure.crawler.valuation_history import (
            capture_valuation_history, retain_valuation_history,
        )
        symbols = [str(s) for s in crawler.stocknumbers if s]
        capture = None
        try:
            capture = (self.valuation_history_collector or capture_valuation_history)(
                self.valuation_history_database, symbols=symbols,
                reference_session=reference_session, clock=self.clock)
            if (capture.get("reference_session") != reference_session or set(capture.get("by_symbol", {})) != set(symbols)
                    or capture.get("window", {}).get("months") != 36):
                raise ValueError("VALUATION_HISTORY_REQUEST_SCOPE_MISMATCH")
            # Retention verifies originals/projections once before publishing.
            manifest = retain_valuation_history(capture, self.root / "valuation-history")
            checked = capture
            for symbol, item in checked["by_symbol"].items():
                # Large monthly source bodies are retained once, not copied
                # into 436 DataFrames/contexts or decoded in each stock worker.
                selection = dict(schema="les-v7-monthly-valuation-selection/v1", symbol=symbol, market="TWSE",
                    observed_at=checked["observed_at"], capture_evidence_id=checked["evidence_id"],
                    source_id=item["source_id"], rows=item["rows"], status=item["status"],
                    coverage=item["coverage"], price_unit="TWD", availability_basis=checked["availability_basis"],
                    source_authenticity=checked["source_authenticity"], historical_information_set_certified=False,
                    source_record_ids=sorted({r["cache_source_id"] for r in item["records"]}),
                    calendar_id=checked["calendar_evidence"]["calendar_id"], manifest=str(manifest))
                self.valuation_history[symbol] = dict(selection, selection_id=identity(selection))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            reason = "VALUATION_HISTORY_CAPTURE_UNAVAILABLE:" + str(error)
            observed = timestamp(self.clock())
            if isinstance(capture, dict) and capture.get("observed_at") is not None:
                try:
                    # An invalid future capture is not made cutoff-visible by
                    # replacing its observation with today's failure clock.
                    observed = max(observed, timestamp(capture["observed_at"]))
                except (TypeError, ValueError):
                    pass  # The explicit failure remains UNKNOWN below.
            self.valuation_history = {}
            for symbol in symbols:
                failure = dict(schema="les-v7-monthly-valuation-selection/v1", symbol=symbol, market="TWSE",
                    observed_at=observed.isoformat(), status="UNKNOWN", rows=[], price_unit="TWD",
                    reference_session=reference_session, capture_error=reason, diagnostics=[reason],
                    availability_basis="CAPTURE_FAILURE_NOT_VERIFIED_OBSERVATION",
                    source_authenticity="UNVERIFIED_CAPTURE_FAILED", historical_information_set_certified=False,
                    source_record_ids=[], coverage=dict(months_requested=36, months_valid=0,
                        gaps=[dict(reason=reason)], unknown_calendar_months=[]))
                # No capture/source/manifest ID is invented for a failed
                # capture or claimed for an integrity-invalid original.
                self.valuation_history[symbol] = dict(failure, selection_id=identity(failure))
            self.diagnostics.append(reason)
            logging.warning("Independent valuation history unavailable; no historical EPS substitution: %s", error)

    def selections(self, crawler, *, cutoff):
        timestamp(cutoff)
        result = {}
        for symbol, dates in zip(crawler.stocknumbers, crawler.daily_dates):
            if not symbol:
                continue
            proof = select_security_master(self.master, str(symbol), cutoff) if self.master else None
            pe = [self.pe_index.select(market=self.market, symbol=str(symbol),
                trade_date=str(day)[:10], cutoff=cutoff) for day in dates[-60:]]
            result[str(symbol)] = dict(security=proof, official_pe=pe,
                capture_diagnostics=list(self.diagnostics),
                availability_policy="CURRENT_OBSERVATIONS_NOT_HISTORICAL_BACKFILL")
            history = self.valuation_history.get(str(symbol))
            if history is not None:
                # Retain a supplied future observation so cutoff validation
                # rejects it, rather than treating it as an absent optional
                # history and silently choosing a different valuation path.
                result[str(symbol)]["valuation_price_history"] = history
            if self.calendar_evidence is not None:
                result[str(symbol)]["calendar_evidence"] = self.calendar_evidence
        return result

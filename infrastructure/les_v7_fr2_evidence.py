"""Read-only, cutoff-local source audit for the isolated FR2 comparison.

This is not a second Quality engine or a publication-time repair. It reuses
the native proof, filing-version, calendar and restriction validators. A final
daily archive is useful retrospective price evidence, not proof that those
bytes were observed at the historical cutoff. The audit never emits gates.
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
from pathlib import Path

from application.filing_publication import verify_filing_publication
from application.replay_contracts import aware_timestamp, digest
from application.research_evidence import RAW_EVIDENCE_SCHEMA, calendar_at, select_filing_versions, verify_publication
from application.restriction_history import HistoricalRestrictionLedger
from domain.evidence_publication import verified_document
from domain.trading_restrictions import restriction_check
from infrastructure.calendar.twse_calendar import TwseCalendar
from infrastructure.replay_evidence import load_evidence
from infrastructure.research_documents import EvidenceDocuments

SCHEMA = "les-v7-fr2-source-audit/v1"
MAX_EVIDENCE_BYTES = 512 * 1024 * 1024


class Fr2HistoricalSources:
    """One validated immutable raw dataset; no HTTP, cache or account writes.

    ``select`` returns detached summaries, never shared mutable raw inputs.
    Raw content integrity is verified once per document in this work scope;
    proof eligibility and version selection are evaluated for every cutoff.
    Actual Quality/valuation must still run via the frozen native v7 engine.
    """

    def __init__(self, evidence_file):
        self.path = Path(evidence_file).resolve()
        self.evidence_id, self._payload = load_evidence(self.path, max_bytes=MAX_EVIDENCE_BYTES)
        if self._payload.get("schema") != RAW_EVIDENCE_SCHEMA:
            raise ValueError("FR2_SOURCE_AUDIT_REQUIRES_RAW_V2")
        self._documents = EvidenceDocuments.for_evidence(self.path, self.evidence_id)
        self._stamp = self._file_stamp()
        self._verified = {}
        self._restrictions = {}
        for row in self._payload.get("restrictions", ()):
            self._restrictions.setdefault(row.get("symbol"), []).append(row)
        ledger = self._payload.get("restriction_ledger")
        self._ledger = HistoricalRestrictionLedger(ledger) if ledger else None

    def _file_stamp(self):
        stat = self.path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _original(self, sha):
        source = self._payload.get("raw_documents", {}).get(sha)
        if source is None:
            raise ValueError("ORIGINAL_NOT_ACQUIRED")
        if source.get("sha256") != sha:
            raise ValueError("FILING_ORIGINAL_HASH_MISMATCH")
        # The reader checks content hashes; a changed on-disk object must not
        # be hidden by a warm document cache.
        if "body_ref" in source:
            def stamp():
                stat = self._documents.objects.path(sha).stat()
                return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
            before = stamp()
            if sha in self._verified:
                if before != self._verified[sha]:
                    raise ValueError("FR2_SOURCE_ORIGINAL_CHANGED")
                return
            verified_document(source, document_reader=self._documents.read)
            if stamp() != before:
                raise ValueError("FR2_SOURCE_ORIGINAL_CHANGED")
            self._verified[sha] = before
        elif sha not in self._verified:
            verified_document(source, document_reader=self._documents.read)
            self._verified[sha] = None

    def _filings(self, symbol, cutoff):
        rows = []
        for item in self._payload.get("filings", {}).get(symbol, ()):
            row = {key: item.get(key) for key in ("fiscal_year", "quarter", "period_end",
                "statement_scope", "revision_number", "content_sha256", "source_version", "source_url")}
            row.update(available_at=None, status="UNKNOWN", reason=None,
                eligible_at_cutoff=False, selected=False,
                stored_available_at=item.get("available_at"), fetched_at=item.get("receipt", {}).get("fetched_at"))
            try:
                self._original(item["content_sha256"])
                row["original_integrity"] = "VERIFIED"
                if item.get("publication_proof") is not None:
                    available = verify_filing_publication(symbol, item, document_reader=self._documents.read)
                    row.update(available_at=available, availability_basis="VERIFIED_HISTORICAL_PUBLICATION",
                        publication_proof_id=digest(item["publication_proof"]))
                    row["status"] = "VERIFIED_AVAILABLE" if aware_timestamp(available) <= cutoff else "AFTER_CUTOFF"
                else:
                    # A PDF upload assumption is not upgraded to verified exact
                    # iXBRL version evidence, including if a legacy replay accepts it.
                    available = max(aware_timestamp(item["available_at"]),
                        aware_timestamp(item["receipt"]["fetched_at"])).isoformat()
                    row.update(available_at=available, availability_basis="FIRST_OBSERVATION")
                    row["status"] = "OBSERVED_AVAILABLE" if aware_timestamp(available) <= cutoff else "PUBLICATION_PROOF_REQUIRED"
                    if item.get("publication_time_policy"):
                        row["unverified_time_assumption_id"] = digest(item["publication_time_policy"])
                row["eligible_at_cutoff"] = row["status"] in {"VERIFIED_AVAILABLE", "OBSERVED_AVAILABLE"}
                if item.get("period_end") and item["period_end"] > cutoff.date().isoformat():
                    row.update(eligible_at_cutoff=False, status="FUTURE_FISCAL_PERIOD")
            except (KeyError, ValueError, TypeError, OSError) as error:
                row.update(status="SOURCE_OR_VERSION_UNVERIFIED", reason=str(error))
            rows.append(row)
        eligible = [row for row in rows if row["eligible_at_cutoff"]]
        indices, conflicts = select_filing_versions(eligible)
        for conflict in conflicts:
            for index in conflict["indices"]:
                eligible[index].update(status="CONFLICTING_FILING_VERSION", eligible_at_cutoff=False,
                    reason=conflict["reason"])
        for index in indices:
            eligible[index]["selected"] = True
        return rows

    def _restriction(self, symbol, cutoff, session):
        candidates, excluded = [], []
        for item in self._restrictions.get(symbol, ()):
            evidence = item.get("evidence", {})
            try:
                available = aware_timestamp(evidence["available_at"])
                if available > cutoff:
                    excluded.append(dict(evidence_id=digest(evidence), reason="AFTER_CUTOFF"))
                else:
                    candidates.append(evidence)
            except (KeyError, ValueError, TypeError):
                excluded.append(dict(evidence_id=digest(evidence), reason="INVALID_AVAILABILITY"))
        source, check = {}, None
        if candidates:
            latest = max(aware_timestamp(row["available_at"]) for row in candidates)
            newest = {digest(row): row for row in candidates if aware_timestamp(row["available_at"]) == latest}
            if len(newest) == 1:
                source = next(iter(newest.values()))
            else:
                check = dict(status="UNKNOWN", reason="CONFLICTING_RESTRICTION_OBSERVATIONS")
        if check is None:
            check = restriction_check(source, market=self._payload["market"], symbol=symbol,
                market_data_date=session, decision_time=cutoff, allow_historical=True)
        if self._ledger:
            historical = self._ledger.reconstruct([symbol], market=self._payload["market"],
                market_data_date=session, as_of=cutoff.isoformat())[symbol]
            checked = restriction_check(historical, market=self._payload["market"], symbol=symbol,
                market_data_date=session, decision_time=cutoff, allow_historical=True)
            if check["status"] != "UNKNOWN" and checked["status"] != "UNKNOWN" and check["status"] != checked["status"]:
                source, check = {}, dict(status="UNKNOWN", reason="CONFLICTING_RESTRICTION_SOURCES")
            elif check["status"] == "UNKNOWN":
                source, check = historical, checked
        return dict(check=check, source_id=digest(source) if source else None, excluded=excluded)

    def select(self, *, symbols, cutoff):
        if self._file_stamp() != self._stamp:
            raise ValueError("FR2_SOURCE_DATASET_CHANGED")
        at = aware_timestamp(cutoff)
        supplied = tuple(symbols)
        if not supplied or any(not isinstance(s, str) or not s for s in supplied) or len(supplied) != len(set(supplied)):
            raise ValueError("FR2_EXPLICIT_FIXED_UNIVERSE_REQUIRED")
        requested = tuple(sorted(supplied))
        end, start = at.date(), at.date() - timedelta(days=699)
        days, excluded = calendar_at(self._payload["calendar"], at, strict=True,
            annual_calendar=lambda data: TwseCalendar(fetcher=lambda: data),
            verifier=lambda proof, **options: verify_publication(
                proof, document_reader=self._documents.read, **options))
        dates = [(start + timedelta(days=i)).isoformat() for i in range(700)]
        status_counts = Counter(days.get(day, "UNKNOWN") for day in dates)
        bar_dates = {symbol: set() for symbol in requested}
        for item in self._payload["bars"]:
            if item.get("dataset") != "STOCK_TABLE":
                continue
            day = f"{item['date'][:4]}-{item['date'][4:6]}-{item['date'][6:8]}"
            if start.isoformat() <= day <= end.isoformat():
                for row in item.get("data", ()):
                    if row and str(row[0]) in bar_dates:
                        bar_dates[str(row[0])].add(day)
        session = max((day for dateset in bar_dates.values() for day in dateset), default=end.isoformat())
        stocks = []
        for symbol in requested:
            filings = self._filings(symbol, at)
            restriction = self._restriction(symbol, at, session)
            stocks.append(dict(symbol=symbol, in_dataset_universe=symbol in self._payload["symbols"],
                observed_bar_dates=sorted(bar_dates[symbol]), observed_bar_day_count=len(bar_dates[symbol]),
                bar_count_semantics="SOURCE_ROWS_NOT_OHLC_OR_LIFECYCLE_VALIDATED",
                historical_market_publication=dict(status="UNKNOWN", reason="FINAL_DAILY_ARCHIVE_HAS_NO_ORIGINAL_OBSERVATION_PROOF"),
                filings=filings, filing_status_counts=dict(Counter(row["status"] for row in filings)),
                selected_filing_ids=[row["content_sha256"] for row in filings if row["selected"]],
                trading_restrictions=restriction,
                native_quality_valuation=dict(status="NOT_EVALUATED", reason="REQUIRES_FROZEN_NATIVE_R3_QUALITY_AND_ANNUAL_VALUATION")))
        result = dict(schema=SCHEMA, dataset_id=self.evidence_id, cutoff=at.isoformat(),
            market=self._payload["market"], sector=self._payload["sector"], symbols=list(requested),
            fixed_universe_id=digest(list(requested)), source_market_policy=self._payload.get("market_policy"),
            window=dict(start=start.isoformat(), end=end.isoformat(), calendar_days=700),
            calendar=dict(status="UNKNOWN" if status_counts["UNKNOWN"] else "VERIFIED_SOURCE_COVERAGE",
                day_status_counts=dict(status_counts), excluded=excluded), stocks=stocks,
            scope="SOURCE_AVAILABILITY_AND_VERSION_SELECTION_NOT_NATIVE_GATE_CLEARANCE",
            source_authenticity="LOCAL_CONTENT_INTEGRITY_NOT_OFFICIAL_AUTHENTICITY_CERTIFICATION",
            verified_original_documents=len({row["content_sha256"] for stock in stocks for row in stock["filings"]
                if row.get("original_integrity") == "VERIFIED"}), research_only=True,
            execution_enabled=False, approved_for_live=False)
        return dict(result, audit_id=digest(result))


def retain_source_audit(evidence_file, *, symbols, cutoffs, output_root):
    """Publish exact per-cutoff audits and a last completion, without overwrites.

    Same business inputs have stable identifiers; local paths and wall clocks
    are not part of source identity. Missing fixed-scope members are retained.
    This receipt is an audit completion, never research/eligibility completion.
    """
    from infrastructure.research_archive import write_once
    requested, times = tuple(symbols), tuple(cutoffs)
    if not times or len(set(times)) != len(times):
        raise ValueError("FR2_SOURCE_AUDIT_EXPLICIT_UNIQUE_CUTOFFS_REQUIRED")
    source = Fr2HistoricalSources(evidence_file)
    reports = [source.select(symbols=requested, cutoff=value) for value in times]
    summaries = []
    for report in reports:
        summaries.append(dict(cutoff=report["cutoff"], audit_id=report["audit_id"],
            calendar=report["calendar"],
            universe_missing_symbols=[stock["symbol"] for stock in report["stocks"] if not stock["in_dataset_universe"]],
            filing_status_counts=dict(sum((Counter(stock["filing_status_counts"]) for stock in report["stocks"]), Counter())),
            symbols_with_selected_filings=sum(bool(stock["selected_filing_ids"]) for stock in report["stocks"]),
            restriction_status_counts=dict(Counter(stock["trading_restrictions"]["check"]["status"] for stock in report["stocks"])),
            market_publication_status_counts=dict(Counter(stock["historical_market_publication"]["status"] for stock in report["stocks"])),
            verified_original_documents=report["verified_original_documents"]))
    body = dict(schema="les-v7-fr2-source-audit-series/v1", dataset_id=source.evidence_id,
        symbols=sorted(requested), fixed_universe_id=reports[0]["fixed_universe_id"], cutoffs=summaries,
        scope="SOURCE_AUDIT_ONLY_NOT_FROZEN_NATIVE_QUALITY_OR_ACCOUNT_PERFORMANCE",
        status="PARTIAL", research_only=True, execution_enabled=False, approved_for_live=False)
    audit_id = digest(body)
    folder = Path(output_root) / audit_id
    for report in reports:
        write_once(folder / "cutoffs" / (report["audit_id"] + ".json"), report)
    write_once(folder / "summary.json", dict(body, audit_id=audit_id))
    write_once(folder / "completion.json", dict(schema="les-v7-fr2-source-audit-completion/v1",
        audit_id=audit_id, dataset_id=source.evidence_id, result_ids=[report["audit_id"] for report in reports],
        status="AUDIT_COMPLETED_EVIDENCE_STILL_PARTIAL", research_only=True))
    return folder, dict(body, audit_id=audit_id)

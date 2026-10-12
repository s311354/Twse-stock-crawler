"""Bounded, resumable collection of official sources; never auto-VERIFY proofs."""
from __future__ import annotations

from dataclasses import asdict
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from application.replay_contracts import digest
from application.research_evidence import OFFICIAL_HOSTS, verified_document
from domain.errors import DataSourceError
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.fundamentals.mops_client import (
    MopsClient, MOPS_CATALOG_URL, MOPS_ANNOUNCEMENT_URL, MOPS_CORRECTION_URL,
    _parse_table, _is_access_blocked_content,
)
from infrastructure.fundamentals.ixbrl_parser import decode_mops_content
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments


def pdf_download_links(content: bytes, url: str) -> list[dict]:
    """Capture ordinary official download links, never execute site scripts."""
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links, self.href, self.label = [], None, []
        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                self.href, self.label = dict(attrs).get('href'), []
        def handle_data(self, text):
            if self.href is not None: self.label.append(text)
        def handle_endtag(self, tag):
            if tag == 'a' and self.href:
                target, label = urljoin(url, self.href), ''.join(self.label).strip()
                parsed = urlsplit(target)
                if (parsed.scheme == 'https' and parsed.hostname == 'doc.twse.com.tw'
                        and not parsed.username and not parsed.password
                        and parsed.path.endswith('.pdf') and label.endswith('.pdf')):
                    self.links.append(dict(url=target, filename=label))
                self.href = None
    parser = Links()
    parser.feed(decode_mops_content(content))
    return parser.links


def collection_plan(payload: dict, audit: dict) -> list[dict]:
    """Known official queries only, deduplicated across symbols and periods."""
    tasks = {}
    def add(kind, symbols, url, data=None, **metadata):
        task = dict(kind=kind, symbols=sorted(symbols), method="POST" if data else "GET",
                    url=url, data=data, **metadata)
        tasks[digest(task)] = task
    pending = [r for r in audit["rows"] if r["status"] not in
               {"VERIFIED_AVAILABLE", "OBSERVED_AVAILABLE", "PUBLISHED_AFTER_CUTOFF"}]
    pending += [dict(symbol=symbol, period=period, original_present=False, content_sha256=None)
                for symbol, financial in audit["financials"].items()
                for period in financial.get("missing_original_periods", [])]
    for symbol in sorted({r["symbol"] for r in pending}):
        base = dict(step="1", firstin="1", off="1", TYPEK="all", co_id=symbol)
        add("CATALOG", [symbol], MOPS_CATALOG_URL, base)
        # Include the following year for Q4 / subsequent correction disclosures.
        years = {int(r["period"][:4]) for r in pending if r["symbol"] == symbol}
        for year in sorted(years | {min(y + 1, int(audit["as_of"][:4])) for y in years}):
            add("ANNOUNCEMENTS", [symbol], MOPS_ANNOUNCEMENT_URL,
                dict(base, year=str(year-1911), month="all", b_date="", e_date=""), calendar_year=year)
    for period in sorted({r["period"] for r in pending}):
        symbols = {r["symbol"] for r in pending if r["period"] == period}
        add("CORRECTIONS", symbols, MOPS_CORRECTION_URL,
            dict(step="1", firstin="1", off="1", isQuery="Y", TYPEK="sii" if payload["market"] == "TWSE" else "otc",
                 year=str(int(period[:4])-1911), season=f"{int(period[-1]):02d}"), period=period)
    for row in pending:
        if row["original_present"] or not row["content_sha256"]:
            continue
        item = next(r for r in payload["filings"][row["symbol"]] if r["content_sha256"] == row["content_sha256"])
        add("ORIGINAL", [row["symbol"]], item["source_url"], expected_sha256=row["content_sha256"], period=row["period"])
    return [dict(request_id=key, **tasks[key]) for key in sorted(tasks)]


def classify_source(task, content, url):
    if task["kind"] == "PDF_ORIGINAL" and content.startswith(b"%PDF-"):
        return dict(status="SOURCE_CAPTURED_REVIEW_REQUIRED", reason="PDF_CONTENT_NOT_HISTORICAL_VERSION_PROOF", candidates=[])
    if _is_access_blocked_content(content):
        return dict(status="ACCESS_BLOCKED", reason="OFFICIAL_SECURITY_GUARD_PAGE", candidates=[])
    if task["kind"] == "PDF_ORIGINAL":
        return dict(status="SOURCE_UNSUPPORTED", reason="PDF_ORIGINAL_REQUIRED_NOT_HTML", candidates=[])
    if task["kind"] == "PDF_DOWNLOAD_PAGE":
        links = pdf_download_links(content, url)
        return dict(status="SOURCE_CAPTURED_REVIEW_REQUIRED" if links else "SOURCE_UNSUPPORTED",
                    reason="PDF_DOWNLOAD_LINK_REVIEW_REQUIRED", candidates=links)
    if task["kind"] == "ORIGINAL":
        same = hashlib.sha256(content).hexdigest() == task["expected_sha256"]
        return dict(status="ORIGINAL_RECOVERED" if same else "ORIGINAL_VERSION_UNRECOVERABLE",
                    reason="EXACT_HASH_MATCH_NOT_PUBLICATION_PROOF" if same else "CURRENT_CONTENT_NOT_PINNED_HISTORICAL_VERSION",
                    candidates=[])
    rows = _parse_table(content).rows
    if task["kind"] == "CATALOG":
        candidates = [asdict(f) for f in MopsClient._parse_catalog(task["symbols"][0], content, url).filings]
    elif task["kind"] == "ANNOUNCEMENTS":
        candidates = [dict(cells=r.cells, actions=r.actions) for r in rows
                      if any(word in " ".join(r.cells) for word in ("財務報", "重編", "更正", "修訂"))]
    else:
        candidates = [dict(cells=r.cells, actions=r.actions) for r in rows
                      if r.cells and r.cells[0] in task["symbols"]]
    text = decode_mops_content(content)
    recognized_table = any("公司代號" in r.cells and "公告日期" in r.cells for r in rows)
    explicit_empty = recognized_table or any(marker in text for marker in ("查無資料", "查無符合", "無符合條件"))
    return dict(status="SOURCE_CAPTURED_REVIEW_REQUIRED" if candidates else
                "NO_MATCH_UNPROVEN" if explicit_empty else "SOURCE_UNSUPPORTED",
                reason="VERSION_LINK_AND_PUBLICATION_REVIEW_REQUIRED" if candidates else
                "EMPTY_QUERY_IS_NOT_ABSENCE_OF_FILINGS_OR_REVISIONS" if explicit_empty else
                "NO_RECOGNIZED_HISTORICAL_ROWS", candidates=candidates, table_rows=len(rows))


def collect_sources(plan, root: Path, *, client=None, max_requests=12, retry_unresolved=False, now=utc_now):
    """Each completed request is immutable; a retry creates a new attempt.

    No formal cache/database is opened. An exhausted budget leaves explicit
    PENDING work. Two consecutive source failures stop this collection only.
    """
    if type(max_requests) is not int or not 1 <= max_requests <= 100:
        raise ValueError("COLLECTION_REQUEST_BUDGET_MUST_BE_1_TO_100")
    docs = EvidenceDocuments(root / "objects")
    write_once(root / "plan.json", dict(schema="filing-collection-plan/v1", tasks=plan))
    owned = client is None
    response_url = None
    if owned:
        client = MopsClient(http_client=TwseHttpClient(TwseHttpConfig(timeout_seconds=15,
            max_attempts=2, min_request_interval_seconds=1, allow_insecure_ssl_fallback=False)),
            network_rounds=1, network_backoff_seconds=0, catalog_empty_attempts=1,
            catalog_empty_backoff_seconds=0, min_request_interval_seconds=1)
        client.http_client.session.max_redirects = 3
        def preserve_official_origin(response, *args, **kwargs):
            nonlocal response_url
            urls = [response.url]
            if response.is_redirect:
                urls.append(urljoin(response.url, response.headers["Location"]))
            for url in urls:
                parsed = urlsplit(url)
                if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_HOSTS or parsed.username or parsed.password:
                    raise ValueError("OFFICIAL_EVIDENCE_REDIRECT_REQUIRED")
            response_url = response.url
            return response
        def prepare_source_attempt():
            # The shared transport can replace a poisoned Session on retry.
            # Reinstall provenance checks before that retry or redirect too.
            session = client.http_client.session
            session.max_redirects = 3
            if preserve_official_origin not in session.hooks['response']:
                session.hooks['response'].append(preserve_official_origin)
    results, used, failures = [], 0, 0
    try:
        for task in plan:
            key = task["request_id"]
            if key != digest({k: v for k, v in task.items() if k != "request_id"}):
                raise ValueError("COLLECTION_REQUEST_HASH_MISMATCH")
            previous = sorted((root / "requests" / key).glob("*.json"))
            old = json.loads(previous[-1].read_text()) if previous else None
            if old:
                if old["request"] != task or old["receipt_sha256"] != digest({k: v for k,v in old.items() if k != "receipt_sha256"}):
                    raise ValueError("COLLECTION_RECEIPT_MISMATCH")
                if old.get("document"):
                    verified_document(old["document"], document_reader=docs.read)
                if not retry_unresolved or old["status"] in {"SOURCE_CAPTURED_REVIEW_REQUIRED", "ORIGINAL_RECOVERED"}:
                    results.append(old)
                    continue
            if used >= max_requests or failures >= 2:
                results.append(dict(request=task, status="PENDING", reason="REQUEST_BUDGET_OR_SOURCE_CIRCUIT_OPEN"))
                continue
            used += 1
            result = dict(request=task, attempt=len(previous)+1, retrieved_at=now(), document=None)
            try:
                response_url = None
                response, url = client.fetch_official_source(task["method"], task["url"],
                    data=task["data"], context=f"historical filing evidence {task['kind']} {','.join(task['symbols'])}",
                    **(dict(before_attempt=prepare_source_attempt) if owned else {}))
                if owned and response_url:
                    url = response_url
                result["retrieved_at"] = now()
                result["document"] = dict(url=url, retrieved_at=result["retrieved_at"],
                    sha256=hashlib.sha256(response.content).hexdigest(), body_ref=docs.store(response.content))
                result.update(classify_source(task, response.content, url), http_status=response.status_code,
                              transport_attempts=response.attempts)
            except (DataSourceError, OSError, ValueError) as error:
                category = getattr(error, "category", None)
                result.update(status=getattr(category, "value", "SOURCE_UNAVAILABLE").upper(), reason=str(error), candidates=[])
            failures = failures+1 if result["status"] in {"NETWORK_ERROR", "ACCESS_BLOCKED", "SOURCE_UNAVAILABLE"} else 0
            result["receipt_sha256"] = digest(result)
            write_once(root / "requests" / key / f"{result['attempt']:04d}.json", result)
            results.append(result)
    finally:
        if owned:
            client.http_client.session.close()
    report = dict(schema="filing-source-collection/v1", plan_sha256=digest(plan), results=results,
                  status="UNVERIFIED_REQUIRES_PUBLICATION_VERSION_REVIEW")
    write_once(root / "reports" / f"{digest(report)}.json", report)
    return report

"""Explicit research assumption, NOT proof of an iXBRL historical version.

The official PDF listing supplies a real upload timestamp. Applying that time
to separately captured iXBRL bytes is a user-accepted hypothesis. Keep it out of
the strict publication proof validator and of live trusted availability sources.
"""
from __future__ import annotations

from copy import deepcopy
from html.parser import HTMLParser
import re
import unicodedata
from urllib.parse import urlsplit

from application.replay_contracts import aware_timestamp, digest
from application.research_evidence import filing_binding, verified_document
from domain.restriction_announcements import publication_bounds

PDF_UPLOAD_POLICY = "filing-pdf-upload-policy/v1"
PDF_UPLOAD_SOURCE = "OFFICIAL_PDF_UPLOAD_USER_ASSUMPTION"


class _Rows(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = self.cell = None


def pdf_listing_time(document, binding, filename, *, document_reader=None):
    """Match the actual official row, not caller-supplied extracted text/dates."""
    url = urlsplit(document["url"])
    if url.hostname != "doc.twse.com.tw" or url.path != "/server-java/t57sb01":
        raise ValueError("OFFICIAL_PDF_LISTING_REQUIRED")
    content = verified_document(document, document_reader=document_reader)
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("big5")
    parser = _Rows()
    parser.feed(text)
    headers = ["證券代號", "資料年度", "資料細節說明", "電子檔案", "上傳日期", "財務報告更(補)正"]
    header = next((r for r in parser.rows if all(k in r for k in headers)), None)
    if header is None:
        raise ValueError("PDF_LISTING_SCHEMA_CHANGED")
    indexes = {k: header.index(k) for k in headers}
    rows = [dict(zip(headers, (r[indexes[k]] for k in headers))) for r in parser.rows
            if len(r) == len(header) and r[indexes["電子檔案"]] == filename]
    if len(rows) != 1:
        raise ValueError("PDF_LISTING_ROW_MISSING_OR_CONFLICTING")
    row = rows[0]
    scope = {"consolidated": ("C", "AI1", "IFRSs合併財報"),
             "individual": ("A", "AI3", "IFRSs個體財報")}.get(binding["statement_scope"])
    if scope is None:
        raise ValueError("PDF_STATEMENT_SCOPE_UNSUPPORTED")
    report_id, suffix, label = scope
    year, quarter, symbol = binding["fiscal_year"], binding["quarter"], binding["symbol"]
    if type(year) is not int or type(quarter) is not int or not 1 <= quarter <= 4:
        raise ValueError("PDF_FISCAL_PERIOD_INVALID")
    period = f"{year-1911}年第{'一二三四'[quarter-1]}季"
    normalized = lambda value: re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))
    if (filename != f"{year}{quarter:02d}_{symbol}_{suffix}.pdf"
            or row["證券代號"] != symbol or normalized(row["資料年度"]) != period
            or row["資料細節說明"] != label or binding["report_id"] != report_id):
        raise ValueError("PDF_COMPANY_PERIOD_SCOPE_MISMATCH")
    if row["財務報告更(補)正"] != "無":
        raise ValueError("PDF_CORRECTION_REQUIRES_VERSION_REVIEW")
    # 'No correction' is only a row check, never certification of revision history.
    match = re.fullmatch(r"(\d{3,4})/(\d{2})/(\d{2})(?: (\d{2}:\d{2}:\d{2}))?", row["上傳日期"])
    if not match:
        raise ValueError("PDF_UPLOAD_TIME_INVALID")
    y, month, day, clock = match.groups()
    year = int(y) + (1911 if len(y) == 3 else 0)
    published = f"{year:04d}-{month}-{day}" + (f"T{clock}+08:00" if clock else "")
    precision = "TIMESTAMP" if clock else "DATE"
    first, available = publication_bounds(dict(published_at=published, publication_precision=precision))
    if aware_timestamp(document["retrieved_at"]) < first:
        raise ValueError("PDF_LISTING_PREDATES_UPLOAD")
    return dict(pdf_uploaded_at=published, publication_precision=precision,
                available_at=available.isoformat(), source_row=row)


def verify_pdf_upload_policy(symbol, item, *, document_reader=None):
    policy = item["publication_time_policy"]
    if (policy.get("schema") != PDF_UPLOAD_POLICY
            or policy.get("status") != "USER_ACCEPTED_ASSUMPTION"
            or not policy.get("accepted_by") or policy.get("binding") != filing_binding(symbol, item)):
        raise ValueError("EXPLICIT_PDF_UPLOAD_ASSUMPTION_REQUIRED")
    if item.get("publication_proof") or item.get("revision_number", 1) != 1:
        raise ValueError("PDF_POLICY_CANNOT_OVERRIDE_PROOF_OR_KNOWN_REVISION")
    if aware_timestamp(policy["accepted_at"]) < aware_timestamp(policy["document"]["retrieved_at"]):
        raise ValueError("PDF_POLICY_ACCEPTANCE_PREDATES_SOURCE")
    result = pdf_listing_time(policy["document"], policy["binding"], policy["pdf_filename"],
                              document_reader=document_reader)
    if item.get("period_end") and result["available_at"][:10] < item["period_end"]:
        raise ValueError("PDF_UPLOAD_PREDATES_REPORTING_PERIOD")
    return result


def apply_pdf_upload_policy(payload, review, *, document_reader=None):
    result, changes = deepcopy(payload), []
    for entry in review["reviews"]:
        binding = entry["binding"]
        candidates = [r for r in result["filings"].get(binding["symbol"], [])
                      if filing_binding(binding["symbol"], r) == binding]
        if len(candidates) != 1:
            raise ValueError("PDF_POLICY_REQUIRES_EXACT_EXISTING_FILING")
        row = candidates[0]
        peers = [r for r in result["filings"][binding["symbol"]]
                 if (r["fiscal_year"], r["quarter"], r["statement_scope"]) ==
                    (row["fiscal_year"], row["quarter"], row["statement_scope"])]
        if len({r["content_sha256"] for r in peers}) != 1:
            raise ValueError("PDF_POLICY_CANNOT_RESOLVE_MULTIPLE_VERSIONS")
        policy = dict(schema=PDF_UPLOAD_POLICY, status="USER_ACCEPTED_ASSUMPTION",
                      accepted_by=review["accepted_by"], accepted_at=review["accepted_at"],
                      binding=binding, pdf_filename=entry["pdf_filename"], document=entry["document"])
        if row.get("publication_time_policy") not in (None, policy):
            raise ValueError("PDF_POLICY_ALREADY_PRESENT")
        verify_pdf_upload_policy(binding["symbol"], dict(row, publication_time_policy=policy),
                                 document_reader=document_reader)
        row["publication_time_policy"] = policy
        changes.append(dict(binding=binding, policy_sha256=digest(policy),
                            reason="USER_ACCEPTED_PDF_UPLOAD_TIME_NOT_VERIFIED_VERSION_PROOF"))
    return result, changes

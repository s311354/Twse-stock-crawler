"""Offline PDF extraction/comparison using installed Poppler, no Python deps."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import parse_qs, unquote, urlsplit

from application.filing_pdf_comparison import COMPARISON_VERSION, compare_pdf_facts
from application.pdf_upload_policy import verify_pdf_upload_policy
from application.replay_contracts import digest
from application.research_evidence import filing_binding, verified_document
from infrastructure.fundamentals.ixbrl_parser import parse_ixbrl
from infrastructure.filing_evidence_collection import collect_sources, pdf_download_links
from infrastructure.publication_repair import open_dataset
from infrastructure.research_archive import write_once
from infrastructure.research_documents import EvidenceDocuments


def collect_filing_pdf(evidence: Path, *, symbol: str, filing_sha256: str, output: Path) -> dict:
    """Two bounded/resumable source queries, separate from offline replay.

    Only an already accepted official listing selects the filename. The server
    may return a timestamp-suffixed temporary URL; retain that link page too.
    A captured PDF is current content, NOT proof of its historical vintage.
    """
    eid, payload, documents = open_dataset(evidence)
    rows = [r for r in payload.get('filings', {}).get(symbol, []) if r['content_sha256'] == filing_sha256]
    if len(rows) != 1:
        raise ValueError('PDF_COMPARISON_REQUIRES_EXACT_FILING')
    item = rows[0]
    verify_pdf_upload_policy(symbol, item, document_reader=documents.read)
    filename = item['publication_time_policy']['pdf_filename']
    task = dict(kind='PDF_DOWNLOAD_PAGE', symbols=[symbol], method='POST',
                url='https://doc.twse.com.tw/server-java/t57sb01',
                data=dict(step='9', kind='A', co_id=symbol, filename=filename, colorchg='1'))
    page_root = output/'download-page'
    report = collect_sources([dict(request_id=digest(task), **task)], page_root, max_requests=1)
    receipt = report['results'][0]
    if receipt['status'] != 'SOURCE_CAPTURED_REVIEW_REQUIRED':
        return dict(status=receipt['status'], reason=receipt.get('reason'), dataset_id=eid)
    page = receipt['document']
    page_store = EvidenceDocuments(page_root/'objects')
    links = [link for link in pdf_download_links(page_store.read(page['body_ref']), page['url'])
             if link['filename'] == filename]
    if len(links) != 1:
        raise ValueError('PDF_DOWNLOAD_LINK_MISSING_OR_AMBIGUOUS')
    task = dict(kind='PDF_ORIGINAL', symbols=[symbol], method='GET', url=links[0]['url'], data=None)
    pdf_root = output/'download-original'
    report = collect_sources([dict(request_id=digest(task), **task)], pdf_root, max_requests=1)
    receipt = report['results'][0]
    if receipt['status'] != 'SOURCE_CAPTURED_REVIEW_REQUIRED':
        return dict(status=receipt['status'], reason=receipt.get('reason'), dataset_id=eid)
    store = EvidenceDocuments(output/'objects')
    document = store.externalize(receipt['document'], document_reader=EvidenceDocuments(pdf_root/'objects').read)
    document['download_page'] = store.externalize(page, document_reader=page_store.read)
    write_once(output/'pdf.json', document)
    return dict(status='PDF_CAPTURED_NOT_VERSION_VERIFIED', dataset_id=eid,
                pdf_document=str(output/'pdf.json'), pdf_sha256=document['sha256'])


def extract_pdf_pages(content: bytes) -> tuple[tuple[str, ...], str]:
    if not content.startswith(b"%PDF-"):
        raise ValueError("PDF_ORIGINAL_REQUIRED_NOT_HTML")
    executable = shutil.which("pdftotext")
    if executable is None:
        raise ValueError("PDFTOTEXT_REQUIRED_INSTALL_POPPLER")
    with tempfile.TemporaryDirectory(prefix="twse-pdf-") as directory:
        source, target = Path(directory) / "source.pdf", Path(directory) / "text.txt"
        source.write_bytes(content)
        try:
            version = subprocess.run([executable, "-v"], capture_output=True, timeout=10, check=True)
            subprocess.run([executable, "-layout", "-enc", "UTF-8", str(source), str(target)],
                           capture_output=True, timeout=60, check=True)
        except (subprocess.SubprocessError, OSError) as error:
            raise ValueError("PDF_TEXT_EXTRACTION_FAILED") from error
        text = target.read_text(encoding="utf-8")
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if not any(page.strip() for page in pages):
        raise ValueError("PDF_SCANNED_OR_NO_EXTRACTABLE_TEXT_REVIEW_REQUIRED")
    return tuple(pages), (version.stdout + version.stderr).decode("utf-8", errors="replace").splitlines()[0]


def compare_filing_pdf(evidence: Path, *, symbol: str, filing_sha256: str,
                       pdf_document: Path, mapping_file: Path | None = None) -> dict:
    """Use original bytes only. No source fetch, dataset mutation or auto-proof."""
    eid, payload, documents = open_dataset(evidence)
    candidates = [r for r in payload.get("filings", {}).get(symbol, [])
                  if r["content_sha256"] == filing_sha256]
    if len(candidates) != 1:
        raise ValueError("PDF_COMPARISON_REQUIRES_EXACT_FILING")
    item = candidates[0]
    assessed = verify_pdf_upload_policy(symbol, item, document_reader=documents.read)
    policy = item["publication_time_policy"]
    document = json.loads(Path(pdf_document).read_text())
    pdf_store = EvidenceDocuments(Path(pdf_document).parent / "objects")
    url = urlsplit(document["url"])
    filename = policy["pdf_filename"]
    linked = False
    if document.get("download_page"):
        page_doc = document["download_page"]
        page_url = urlsplit(page_doc["url"])
        if page_url.hostname != "doc.twse.com.tw" or page_url.path != "/server-java/t57sb01":
            raise ValueError("PDF_OFFICIAL_DOWNLOAD_PAGE_REQUIRED")
        links = pdf_download_links(verified_document(page_doc, document_reader=pdf_store.read), page_doc["url"])
        linked = sum(link == dict(url=document["url"], filename=filename) for link in links) == 1
        if not linked:
            raise ValueError("PDF_DOWNLOAD_LINK_MISMATCH")
    if (not linked and Path(unquote(url.path)).name != filename and
            parse_qs(url.query).get("filename") != [filename]):
        raise ValueError("PDF_FILENAME_DOES_NOT_MATCH_OFFICIAL_LISTING")
    content = verified_document(document, document_reader=pdf_store.read)
    pages, extractor = extract_pdf_pages(content)
    original_document = payload["raw_documents"][filing_sha256]
    if original_document["sha256"] != filing_sha256:
        raise ValueError("FILING_ORIGINAL_HASH_MISMATCH")
    original = verified_document(original_document, document_reader=documents.read)
    facts = [asdict(fact) for fact in parse_ixbrl(original)]
    binding = filing_binding(symbol, item)
    mapping = (json.loads(Path(mapping_file).read_text()) if mapping_file else
               dict(schema=COMPARISON_VERSION, binding=binding, pdf_sha256=document["sha256"], fields=[]))
    if mapping.get("binding") != binding or mapping.get("pdf_sha256") != document["sha256"]:
        raise ValueError("PDF_COMPARISON_BINDING_MISMATCH")
    report = compare_pdf_facts(facts, pages, mapping, statement_scope=item["statement_scope"])
    # Retain references, not Base64, and expose source text/page locations for
    # review. Neither extracted_text supplied by a caller nor stored facts win.
    document = pdf_store.externalize(document)
    report.update(dataset_id=eid, binding=binding, pdf_document=document, pdf_filename=filename,
                  publication_time=assessed, publication_time_status="ASSUMED_NOT_VERIFIED",
                  extractor=extractor, extracted_text_sha256=hashlib.sha256("\f".join(pages).encode()).hexdigest(),
                  pages=[dict(page=i+1, text=text) for i, text in enumerate(pages)],
                  mapping_template=dict(schema=COMPARISON_VERSION, binding=binding,
                      pdf_sha256=document["sha256"], statement_scope=item["statement_scope"],
                      reviewed_by=None, identity_evidence={}, fields=[]))
    report["report_sha256"] = digest(report)
    return json.loads(json.dumps(report, ensure_ascii=False, allow_nan=False))

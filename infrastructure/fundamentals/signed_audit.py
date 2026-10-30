"""Manual, append-only issuer PDF corroboration. No download or schedule here."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from application.daily_work import current_work
from domain.les_v7.contracts import identity, timestamp
from domain.les_v7.signed_audit import SCHEMA, POLICY, validate_signed_source
from infrastructure.research_archive import write_once
from infrastructure.research_documents import EvidenceDocuments

EXTRACTION = "POPPLER_LAYOUT_UTF8_FORMFEED_PAGES_R1"


def extract_pages(content):
    if not content.startswith(b"%PDF") or not 0 < len(content) <= 12 * 1024 * 1024:
        raise ValueError("SIGNED_REPORT_PDF_FORMAT_OR_SIZE_INVALID")
    result = subprocess.run(["pdftotext", "-layout", "-", "-"], input=content, capture_output=True,
                            check=True, timeout=30)
    if len(result.stdout) > 12 * 1024 * 1024:
        raise ValueError("SIGNED_REPORT_EXTRACTION_CAPACITY_EXCEEDED")
    return result.stdout.decode("utf-8").split("\f")


class SignedAuditStore:
    """References are relative CAS refs, never replace an existing filing.

    Publication requires an explicit visual-review attestation bound to the
    actual retained PDF hash. A hash or attestation is not a digital signature
    or independent certification of source authenticity.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.documents = EvidenceDocuments(self.root / "objects")

    def publish(self, content, binding, row, *, cutoff):
        packet = deepcopy(binding)
        packet.pop("source_id", None)
        pages = extract_pages(content)
        wanted = {1, *packet["audit_pages"], *(a["page"] for a in packet["financial_anchors"].values())}
        packet.update(schema=SCHEMA, policy=POLICY, pdf_sha256=hashlib.sha256(content).hexdigest(),
            body_ref=self.documents.store(content), extraction_policy=EXTRACTION,
            pages={str(p): pages[p - 1] for p in sorted(wanted)},
            availability_basis="ACTUAL_FETCH_AND_SOURCE_TIEOUT_NO_HISTORICAL_BACKDATE",
            available_at=packet["verified_at"], historical_publication_at=None,
            source_authenticity="ISSUER_HOST_AND_VISUAL_REVIEW_NOT_CRYPTOGRAPHIC_CERTIFICATION")
        packet["extracted_pages_sha256"] = identity(packet["pages"])
        packet["source_id"] = identity(packet)
        validate_signed_source(packet, row, cutoff=cutoff)
        path = self.root / "bindings" / packet["filing_sha256"] / (packet["source_id"] + ".json")
        write_once(path, packet)
        return packet

    def select(self, row, *, cutoff):
        sha = row.get("content_sha256", "")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("SIGNED_REPORT_INVALID_FILING_ID")
        candidates = []
        for path in sorted((self.root / "bindings" / sha).glob("*.json")):
            packet = json.loads(path.read_bytes())
            if packet.get("source_id") != path.stem or identity({k: v for k, v in packet.items() if k != "source_id"}) != path.stem:
                raise ValueError("SIGNED_REPORT_MANIFEST_CORRUPT")
            if timestamp(packet["available_at"]) > timestamp(cutoff):
                continue  # Never substitute today's original for earlier research.
            candidates.append(packet)
        if not candidates:
            return None
        latest = max(timestamp(p["available_at"]) for p in candidates)
        chosen = [p for p in candidates if timestamp(p["available_at"]) == latest]
        if len(chosen) != 1:
            raise ValueError("SIGNED_REPORT_VERSION_CONFLICT")
        packet = chosen[0]
        validate_signed_source(packet, row, cutoff=cutoff)
        content = self.documents.read(packet["body_ref"])  # Verify bytes, even on memo hits.
        def parse():
            pages = extract_pages(content)
            return {p: pages[int(p) - 1] for p in packet["pages"]}
        work = current_work()
        pages = work.memo_json("signed_report_extraction", identity(dict(
            pdf=packet["pdf_sha256"], pages=sorted(packet["pages"]), parser=EXTRACTION)), parse) if work else parse()
        if pages != packet["pages"] or packet.get("extraction_policy") != EXTRACTION:
            raise ValueError("SIGNED_REPORT_EXTRACTION_DOES_NOT_MATCH_ORIGINAL")
        return deepcopy(packet)

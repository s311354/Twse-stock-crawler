"""Import official industry evidence using the existing document/proof verifier."""
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path

from application.research_evidence import verified_document, verify_publication
from domain.industry_benchmark import SCHEMA, SOURCE_SCHEMA, identity, source_binding, validate_packet

MAX_SOURCE_FILE_BYTES = 16 * 1024 * 1024


def load_industry_benchmarks(path, *, document_reader=None) -> dict:
    """Import an explicitly supplied bounded raw-source mapping, with no I/O fetch.

    The file maps official sector codes (or their existing enum aliases) to
    ``industry-benchmark-source/v1`` packets. Public input never accepts a
    self-certified normalized PASS packet. No path is inferred when disabled.
    """
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DUPLICATE_INDUSTRY_JSON_KEY")
            result[key] = value
        return result

    with Path(path).expanduser().open("rb") as source:
        body = source.read(MAX_SOURCE_FILE_BYTES + 1)
    if len(body) > MAX_SOURCE_FILE_BYTES:
        raise ValueError("BOUNDED_INDUSTRY_SOURCE_FILE_REQUIRED")
    payload = json.loads(body.decode("utf-8"), object_pairs_hook=unique_object)
    if not isinstance(payload, dict):
        raise ValueError("INDUSTRY_SECTOR_MAPPING_REQUIRED")
    from domain.industry_benchmark import sector_code
    prepared = {}
    for key, packet in payload.items():
        code = sector_code(key)
        if not code.isdecimal():
            raise ValueError("INVALID_INDUSTRY_SECTOR_KEY")
        if code in prepared:
            raise ValueError("DUPLICATE_NORMALIZED_INDUSTRY_SECTOR")
        if not isinstance(packet, Mapping):
            raise ValueError("RAW_INDUSTRY_SOURCE_PACKET_REQUIRED")
        normalized = prepare_industry_benchmark(packet, document_reader=document_reader)
        if normalized is None or sector_code(normalized["sector"]) != code:
            raise ValueError("INDUSTRY_SECTOR_KEY_MISMATCH")
        prepared[code] = normalized
    return prepared


def prepare_industry_context(context: Mapping, *, document_reader=None) -> dict:
    """Public JSON boundary accepts source documents, never self-certified receipts."""
    result = deepcopy(dict(context))
    if "industry_benchmark" in result:
        result["industry_benchmark"] = prepare_industry_benchmark(
            result["industry_benchmark"], document_reader=document_reader)
    return result


def prepare_industry_benchmark(packet: Mapping | None, *, document_reader=None) -> dict | None:
    """Keep first download time unless exact version publication is proven.

    Extraction is an explicit provider/reviewer attestation, not inferred from
    a checksum. No source discovery, guessed deadline or live fetch occurs.
    """
    if not packet:
        return None
    if packet.get("schema") != SOURCE_SCHEMA:
        raise ValueError("RAW_INDUSTRY_SOURCE_PACKET_REQUIRED")
    result = {key: deepcopy(packet[key]) for key in
              ("market", "sector", "kind", "benchmark_id", "series_version")}
    result["schema"] = SCHEMA
    for kind, name in (("membership", "memberships"), ("bar", "bars")):
        result[name] = []
        for raw in packet[name]:
            row = deepcopy(dict(raw))
            document = packet["documents"][row["document_sha256"]]
            body = verified_document(document, document_reader=document_reader)
            text = body.decode("utf-8", errors="replace")
            binding = source_binding(result, kind, row)
            if (row.get("source_binding") != binding or not row.get("extraction_verified_by")
                    or not isinstance(row.get("quotes"), list) or not row["quotes"]
                    or any(not isinstance(q, str) or not q.strip() or q not in text for q in row["quotes"])):
                raise ValueError("INDUSTRY_EXTRACTION_NOT_VERIFIED")
            proof = row.pop("publication_proof", None)
            if proof is not None:
                available = verify_publication(proof, binding=binding, document_reader=document_reader)
                source = "OFFICIAL_PUBLICATION_PROOF"
                row["publication_evidence"] = {
                    key: deepcopy(proof[key]) for key in ("published_at", "publication_precision", "verified_at",
                                                         "verified_by", "verification_method")}
                row["publication_evidence"]["receipt"] = {
                    key: proof["document"][key] for key in ("url", "sha256", "retrieved_at")}
            else:
                available, source = document["retrieved_at"], "FIRST_OBSERVED_DOWNLOAD"
            row.update(available_at=available, availability_source=source, binding_sha256=identity(binding),
                       receipt={key: document[key] for key in ("url", "sha256", "retrieved_at")})
            for key in ("source_binding", "quotes"):
                row.pop(key, None)
            result[name].append(row)
    validate_packet(result)
    return result

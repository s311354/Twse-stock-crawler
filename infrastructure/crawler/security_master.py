"""Immutable official TWSE/TPEx instrument and issuer master capture.

One bounded batch per source/market; current tables are usable no earlier than
their actual acquisition. Raw documents are shared by content reference. Sparse
selections are re-derived from those originals, not trusted caller PASS flags.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from urllib.parse import urlparse

from domain.security_master import (SCHEMA, SELECTION_SCHEMA, POLICY, TIME_POLICY,
    PARSER_VERSION, ISIN_URLS, ISSUER_URLS, LIMITATIONS, Instrument, Issuer, content_id, timestamp, decide_identity,
    instrument_from_row, issuer_from_row, master_identity, merkle_levels, merkle_proof,
    verify_compact_security_selection)
from domain.errors import AccessBlockedError
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.research_archive import utc_now, write_once
from infrastructure.research_documents import EvidenceDocuments

MAX_BYTES = 16 * 1024 * 1024
MAX_ROWS = 100_000
# A small cache stores immutable rows, never mutable public dictionaries.
_PARSED: OrderedDict[tuple, tuple] = OrderedDict()
_TREES: OrderedDict[tuple, tuple] = OrderedDict()


def clear_security_master_cache() -> None:
    _PARSED.clear()
    _TREES.clear()


def _tree(kind, market, content, rows):
    key = kind, market, hashlib.sha256(content).hexdigest(), PARSER_VERSION
    if key in _TREES:
        _TREES.move_to_end(key)
        return _TREES[key]
    levels = merkle_levels(rows)
    _TREES[key] = levels
    while len(_TREES) > 4:
        _TREES.popitem(last=False)
    return levels


def _memo(kind, market, content, parse):
    key = kind, market, hashlib.sha256(content).hexdigest(), PARSER_VERSION
    if key in _PARSED:
        _PARSED.move_to_end(key)
        return _PARSED[key]
    rows = tuple(parse())
    _PARSED[key] = rows
    while len(_PARSED) > 4:
        _PARSED.popitem(last=False)
    return rows


def _bounded(content):
    if not isinstance(content, bytes) or not 0 < len(content) <= MAX_BYTES:
        raise ValueError("SECURITY_MASTER_BODY_SIZE_LIMIT")
    if any(word in content[:8192].lower() for word in (b"access denied", b"request rejected", b"access blocked")):
        raise AccessBlockedError("Official security master access blocked", retryable=False)


class _Rows(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "tr":
            self.row = []
        elif tag.lower() in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag.lower() in {"td", "th"} and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag.lower() == "tr" and self.row is not None:
            self.rows.append(self.row)
            if len(self.rows) > MAX_ROWS:
                raise ValueError("SECURITY_MASTER_ROW_LIMIT")
            self.row = None


def parse_isin_master(content: bytes, market: str) -> tuple[Instrument, ...]:
    """MS950 official seven-column table, with authoritative section and CFI."""
    _bounded(content)
    if market not in ISIN_URLS:
        raise ValueError("SECURITY_MASTER_MARKET_INVALID")
    def parse():
        text = content.decode("cp950")
        if "因為安全性考量" in text[:8192] or "所執行的頁面無法呈現" in text[:8192]:
            raise AccessBlockedError("Official security master access blocked", retryable=False)
        expected_title = "本國上市證券國際證券辨識號碼一覽表" if market == "TWSE" else "本國上櫃證券國際證券辨識號碼一覽表"
        if expected_title not in text or not re.search(r"最近更新日期:\s*\d{4}/\d{2}/\d{2}", text):
            raise ValueError("SECURITY_MASTER_OFFICIAL_TABLE_TITLE_UNVERIFIED")
        parser = _Rows(); parser.feed(text)
        fields = ["有價證券代號及名稱", "國際證券辨識號碼(ISIN Code)", "上市日", "市場別", "產業別", "CFICode", "備註"]
        headers = [row for row in parser.rows if row == fields]
        if len(headers) != 1:
            raise ValueError("SECURITY_MASTER_COLUMN_SCHEMA_MISMATCH")
        category, seen, rows = None, set(), []
        for row in parser.rows[parser.rows.index(fields) + 1:]:
            if len(row) == 1:
                category = row[0]
                continue
            if len(row) != 7 or not category:
                raise ValueError("SECURITY_MASTER_ROW_WIDTH_OR_CATEGORY_MISSING")
            instrument = instrument_from_row(row, category, market)
            if instrument.symbol in seen:
                raise ValueError("SECURITY_MASTER_DUPLICATE_INSTRUMENT_CONFLICT")
            seen.add(instrument.symbol)
            rows.append(instrument)
        if not rows:
            raise ValueError("SECURITY_MASTER_EMPTY_TABLE_NOT_COVERAGE")
        return rows
    return _memo("instruments", market, content, parse)


def parse_issuer_master(content: bytes, market: str) -> tuple[Issuer, ...]:
    """Only identity/lifecycle fields; issuer shares/financials are not reused."""
    _bounded(content)
    if market not in ISSUER_URLS:
        raise ValueError("SECURITY_MASTER_MARKET_INVALID")
    def parse():
        def pairs(values):
            result = {}
            for key, value in values:
                if key in result:
                    raise ValueError("SECURITY_MASTER_DUPLICATE_JSON_KEY")
                result[key] = value
            return result
        payload = json.loads(content.decode("utf-8-sig"), object_pairs_hook=pairs)
        if not isinstance(payload, list) or not 1 <= len(payload) <= 8192:
            raise ValueError("SECURITY_MASTER_ISSUER_TABLE_UNVERIFIED")
        result, seen = [], set()
        for row in payload:
            issuer = issuer_from_row(row, market)
            if issuer.symbol in seen:
                raise ValueError("SECURITY_MASTER_DUPLICATE_ISSUER_CONFLICT")
            seen.add(issuer.symbol)
            result.append(issuer)
        return result
    return _memo("issuers", market, content, parse)


def _document_content(document, *, document_reader=None):
    url = urlparse(document.get("url", ""))
    if (document.get("url") not in set(ISIN_URLS.values()) | set(ISSUER_URLS.values())
            or url.scheme != "https" or url.username or url.password):
        raise ValueError("SECURITY_MASTER_OFFICIAL_SOURCE_REQUIRED")
    if sum(key in document for key in ("body_ref", "body_base64")) != 1:
        raise ValueError("SECURITY_MASTER_ORIGINAL_BYTES_REQUIRED")
    if "body_ref" in document:
        from application.research_evidence import validate_document_ref
        ref = document["body_ref"]; validate_document_ref(ref)
        if document_reader is None or ref["sha256"] != document.get("sha256"):
            raise ValueError("SECURITY_MASTER_DOCUMENT_READER_OR_HASH_REQUIRED")
        content = document_reader(ref)
        if len(content) != ref["size_bytes"]:
            raise ValueError("SECURITY_MASTER_SOURCE_SIZE_MISMATCH")
    else:
        content = base64.b64decode(document["body_base64"], validate=True)
    _bounded(content)
    if hashlib.sha256(content).hexdigest() != document.get("sha256"):
        raise ValueError("SECURITY_MASTER_SOURCE_HASH_MISMATCH")
    timestamp(document["retrieved_at"])
    return content


def _identity(packet):
    return master_identity(packet)


def capture_security_master(*, market: str, instrument_content: bytes, issuer_content: bytes,
                            fetched_at: str, document_store=None, instrument_fetched_at=None,
                            issuer_fetched_at=None) -> dict:
    """Build a detached packet from exact originals, with conservative times."""
    timestamp(fetched_at)
    times = (instrument_fetched_at or fetched_at, issuer_fetched_at or fetched_at)
    if max(map(timestamp, times)) != timestamp(fetched_at):
        raise ValueError("SECURITY_MASTER_CAPTURE_COMPLETION_TIME_MISMATCH")
    instruments = parse_isin_master(instrument_content, market)
    issuers = parse_issuer_master(issuer_content, market)
    def doc(url, content, at):
        result = dict(url=url, sha256=hashlib.sha256(content).hexdigest(), retrieved_at=at,
                      representation="ORIGINAL_TRANSPORT_BYTES")
        if document_store:
            result["body_ref"] = document_store(content)
        else:
            result["body_base64"] = base64.b64encode(content).decode("ascii")
        return result
    packet = dict(schema=SCHEMA, policy=POLICY, market=market, parser_version=PARSER_VERSION,
        time_policy=TIME_POLICY, fetched_at=fetched_at, available_at=fetched_at,
        availability_basis="ACTUAL_FIRST_OBSERVATION", original_publication_at=None,
        historical_information_set_certified=False, source_authenticity="OFFICIAL_HTTPS_SOURCE_HASH_BOUND_NOT_CERTIFIED",
        limitations=list(LIMITATIONS), instruments_document=doc(ISIN_URLS[market], instrument_content, times[0]),
        issuers_document=doc(ISSUER_URLS[market], issuer_content, times[1]),
        instrument_row_count=len(instruments), issuer_row_count=len(issuers),
        instruments_root=_tree("instruments", market, instrument_content, instruments)[-1][0],
        issuers_root=_tree("issuers", market, issuer_content, issuers)[-1][0])
    packet["evidence_id"] = _identity(packet)
    return packet


def validate_security_master(packet: dict, *, document_reader=None) -> dict:
    if (not isinstance(packet, dict) or packet.get("schema") != SCHEMA or packet.get("policy") != POLICY
            or packet.get("market") not in ISIN_URLS or packet.get("parser_version") != PARSER_VERSION
            or packet.get("time_policy") != TIME_POLICY or packet.get("available_at") != packet.get("fetched_at")
            or packet.get("original_publication_at") is not None or packet.get("historical_information_set_certified") is not False
            or packet.get("availability_basis") != "ACTUAL_FIRST_OBSERVATION" or packet.get("limitations") != LIMITATIONS
            or packet.get("source_authenticity") != "OFFICIAL_HTTPS_SOURCE_HASH_BOUND_NOT_CERTIFIED"):
        raise ValueError("SECURITY_MASTER_PACKET_POLICY_OR_TIME_INVALID")
    market = packet["market"]; timestamp(packet["available_at"])
    for key, url in (("instruments_document", ISIN_URLS[market]), ("issuers_document", ISSUER_URLS[market])):
        if (not isinstance(packet.get(key), dict) or packet[key].get("url") != url
                or timestamp(packet[key].get("retrieved_at")) > timestamp(packet["fetched_at"])):
            raise ValueError("SECURITY_MASTER_SOURCE_TIME_OR_MARKET_BINDING_MISMATCH")
    if max(timestamp(packet[key]["retrieved_at"]) for key in ("instruments_document", "issuers_document")) != timestamp(packet["available_at"]):
        raise ValueError("SECURITY_MASTER_CAPTURE_COMPLETION_TIME_MISMATCH")
    instrument_content = _document_content(packet["instruments_document"], document_reader=document_reader)
    issuer_content = _document_content(packet["issuers_document"], document_reader=document_reader)
    instruments = parse_isin_master(instrument_content, market)
    issuers = parse_issuer_master(issuer_content, market)
    if (packet.get("instrument_row_count") != len(instruments) or packet.get("issuer_row_count") != len(issuers)
            or packet.get("evidence_id") != _identity(packet)
            or packet.get("instruments_root") != _tree("instruments", market, instrument_content, instruments)[-1][0]
            or packet.get("issuers_root") != _tree("issuers", market, issuer_content, issuers)[-1][0]):
        raise ValueError("SECURITY_MASTER_PACKET_CONTENT_ID_MISMATCH")
    return deepcopy(packet)


@dataclass(frozen=True, slots=True)
class SecurityMasterContext:
    """Work-scoped validated immutable rows/index; no public mutable evidence."""
    header_json: str
    instruments: tuple[Instrument, ...]
    issuers: tuple[Issuer, ...]
    instrument_levels: tuple[tuple[str, ...], ...]
    issuer_levels: tuple[tuple[str, ...], ...]
    instrument_index: object
    issuer_index: object
    check_sources: object = None


def prepare_security_master(packet: dict, *, document_reader=None, check_sources=None) -> SecurityMasterContext:
    """Full original verification once per work, bounded immutable projection."""
    validated = validate_security_master(packet, document_reader=document_reader)
    instrument_content = _document_content(validated["instruments_document"], document_reader=document_reader)
    issuer_content = _document_content(validated["issuers_document"], document_reader=document_reader)
    instruments = parse_isin_master(instrument_content, validated["market"])
    issuers = parse_issuer_master(issuer_content, validated["market"])
    header = {key: value for key, value in validated.items() if key not in {"instruments_document", "issuers_document"}}
    for key in ("instruments_document", "issuers_document"):
        header[key] = {field: value for field, value in validated[key].items() if field != "body_base64"}
    return SecurityMasterContext(json.dumps(header, ensure_ascii=False, sort_keys=True), instruments, issuers,
        _tree("instruments", validated["market"], instrument_content, instruments),
        _tree("issuers", validated["market"], issuer_content, issuers),
        MappingProxyType({row.symbol: index for index, row in enumerate(instruments)}),
        MappingProxyType({row.symbol: index for index, row in enumerate(issuers)}), check_sources)


def select_security_master(packet: dict | SecurityMasterContext, symbol: str, cutoff: str, *, document_reader=None) -> dict:
    context = packet if isinstance(packet, SecurityMasterContext) else prepare_security_master(packet, document_reader=document_reader)
    if context.check_sources is not None:
        context.check_sources()
    validated = json.loads(context.header_json)
    symbol = str(symbol).strip().upper()
    # Compact self-contained inclusion proof references originals but does not
    # embed their multi-megabyte bodies in every stock's native report.
    selected = dict(schema=SELECTION_SCHEMA, policy=POLICY, market=validated["market"], market_symbol=symbol,
        cutoff=cutoff, master_evidence_id=validated["evidence_id"],
        master=validated, status="UNKNOWN", reason="SECURITY_MASTER_NOT_AVAILABLE_AT_CUTOFF",
        security_type="UNKNOWN", issuer_symbol=None, low_entry_model_supported=False,
        instrument=None, issuer=None)
    if timestamp(validated["available_at"]) <= timestamp(cutoff):
        instrument_index = context.instrument_index.get(symbol)
        issuer_index = context.issuer_index.get(symbol)
        instrument = context.instruments[instrument_index] if instrument_index is not None else None
        issuer = context.issuers[issuer_index] if issuer_index is not None else None
        selected.update(decide_identity(instrument, issuer, market=validated["market"], symbol=symbol, cutoff=cutoff))
        if selected["instrument"] is not None:
            selected["instrument_inclusion"] = merkle_proof(context.instrument_levels, instrument_index)
        if selected["issuer"] is not None:
            selected["issuer_inclusion"] = merkle_proof(context.issuer_levels, issuer_index)
    selected["selection_id"] = content_id({key: value for key, value in selected.items() if key != "master"})
    # Canonical detach makes tuple/list representation stable after JSON save.
    selected = json.loads(json.dumps(selected, ensure_ascii=False, allow_nan=False))
    return verify_compact_security_selection(selected, cutoff=cutoff)


def verify_security_selection(selection: dict, *, cutoff: str, document_reader=None) -> dict:
    if (not isinstance(selection, dict) or selection.get("schema") != SELECTION_SCHEMA
            or selection.get("policy") != POLICY or timestamp(selection.get("cutoff")) != timestamp(cutoff)):
        raise ValueError("SECURITY_MASTER_SELECTION_CUTOFF_OR_SCHEMA_MISMATCH")
    validated = verify_compact_security_selection(selection, cutoff=cutoff)
    # Full audit validates roots against originals. The body references are
    # retained when originals live in the immutable shared byte store.
    if document_reader is not None:
        validate_security_master(selection["master"], document_reader=document_reader)
    return validated


def collect_security_master(root, market: str, *, client=None, now=utc_now, max_requests=2) -> Path:
    """Public manual/daily hook. Exactly two requests, secure shared transport."""
    if market not in ISIN_URLS or type(max_requests) is not int or max_requests != 2:
        raise ValueError("SECURITY_MASTER_TWO_BATCH_REQUEST_BUDGET_REQUIRED")
    transport = client or TwseHttpClient(TwseHttpConfig(timeout_seconds=30, max_attempts=1,
        min_request_interval_seconds=1, allow_insecure_ssl_fallback=False))
    if transport.config.allow_insecure_ssl_fallback or transport.config.min_request_interval_seconds < 1:
        raise ValueError("SECURITY_MASTER_SECURE_RATE_LIMITED_TRANSPORT_REQUIRED")
    calls = 0
    def reserve():
        nonlocal calls
        if calls >= max_requests:
            raise ValueError("SECURITY_MASTER_REQUEST_BUDGET_EXHAUSTED")
        calls += 1
    instruments = transport.request_bytes("GET", ISIN_URLS[market], before_attempt=reserve,
        context="Official observed instrument identity master", content_validator=lambda raw: parse_isin_master(raw, market))
    instrument_at = now()
    issuers = transport.request_bytes("GET", ISSUER_URLS[market], before_attempt=reserve,
        context="Official observed financial issuer master", content_validator=lambda raw: parse_issuer_master(raw, market))
    issuer_at = now()
    root = Path(root); documents = EvidenceDocuments(root / "objects")
    packet = capture_security_master(market=market, instrument_content=instruments.content,
        issuer_content=issuers.content, fetched_at=max((instrument_at, issuer_at), key=timestamp), document_store=documents.store,
        instrument_fetched_at=instrument_at, issuer_fetched_at=issuer_at)
    validate_security_master(packet, document_reader=documents.read)
    path = root / packet["evidence_id"] / "manifest.json"
    write_once(path, packet)
    return path


def load_security_master(path) -> tuple[dict, EvidenceDocuments]:
    path = Path(path)
    with path.open("rb") as stream:
        content = stream.read(4 * MAX_BYTES + 1)
    if len(content) > 4 * MAX_BYTES:
        raise ValueError("SECURITY_MASTER_MANIFEST_SIZE_LIMIT")
    packet = json.loads(content)
    documents = EvidenceDocuments(path.parent.parent / "objects")
    documents.verify(packet)
    packet = validate_security_master(packet, document_reader=documents.read)
    documents.check()
    return packet, documents


__all__ = ["collect_security_master", "load_security_master", "capture_security_master",
    "validate_security_master", "select_security_master", "verify_security_selection",
    "parse_isin_master", "parse_issuer_master", "clear_security_master_cache", "prepare_security_master", "SecurityMasterContext"]

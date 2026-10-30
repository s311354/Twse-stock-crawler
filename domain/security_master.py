"""Cutoff-bound security identity rules, distinct from financial issuer data.

An observed current master is not evidence of past universe membership. Names
and suffixes are never used to manufacture a financial issuer mapping.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import re
from zoneinfo import ZoneInfo

SCHEMA = "security-master/v1"
SELECTION_SCHEMA = "security-master-selection/v1"
POLICY = "OFFICIAL_OBSERVED_SECURITY_MASTER_R1"
TIME_POLICY = "FIRST_VERIFIED_OBSERVATION_NO_HISTORICAL_BACKDATE"
PARSER_VERSION = "OFFICIAL_SECURITY_MASTER_PARSER_R1"
ISIN_URLS = {"TWSE": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
             "TPEX": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"}
ISSUER_URLS = {"TWSE": "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
               "TPEX": "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"}
LIMITATIONS = ["CURRENT_OBSERVATION_NOT_HISTORICAL_MEMBERSHIP_PROOF",
    "LISTING_DATE_REPORTED_FORMAL_ANNOUNCEMENT_TAKES_PRECEDENCE",
    "MISSING_ROW_DOES_NOT_PROVE_DELISTED_OR_NO_FINANCIAL_ISSUER",
    "NON_COMMON_SECURITY_NOT_MAPPED_BY_SUFFIX", "HASH_INTEGRITY_NOT_SOURCE_AUTHENTICITY_CERTIFICATION"]


def content_id(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("SECURITY_MASTER_AWARE_TIME_REQUIRED")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("SECURITY_MASTER_AWARE_TIME_REQUIRED")
    return parsed


@dataclass(frozen=True, slots=True)
class Instrument:
    market: str
    symbol: str
    name: str
    isin: str
    cfi: str
    source_category: str
    listing_date: str | None
    sector: str
    remark: str
    source_row_sha256: str
    raw_cells: tuple[str, ...]

    @property
    def security_type(self) -> str:
        # Both official table section and published CFI group must agree.
        # Unknown new categories/CFI groups remain OTHER, never inferred common.
        if self.source_category in {"股票", "創新板"} and self.cfi.startswith("ES"):
            return "COMMON"
        if self.source_category == "特別股" and self.cfi.startswith("EP"):
            return "PREFERRED"
        if self.source_category == "臺灣存託憑證(TDR)" and self.cfi.startswith("ED"):
            return "DR"
        return "OTHER"


@dataclass(frozen=True, slots=True)
class Issuer:
    market: str
    symbol: str
    name: str
    company_name: str
    legal_id: str
    listing_date: str | None
    source_row_sha256: str
    raw_row_json: str


def instrument_from_row(row, category, market) -> Instrument:
    if not isinstance(row, (list, tuple)) or len(row) != 7 or any(not isinstance(x, str) for x in row):
        raise ValueError("SECURITY_MASTER_INSTRUMENT_ROW_INVALID")
    matched = re.fullmatch(r"([0-9A-Z]{4,12})\s+(.+)", row[0])
    expected_market = ("上市臺灣創新板" if market == "TWSE" and category == "創新板" else
                       "上市" if market == "TWSE" else "上櫃")
    if (not matched or not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d", row[1])
            or market not in ISIN_URLS or row[3] != expected_market
            or not re.fullmatch(r"[A-Z]{6}", row[5]) or not isinstance(category, str) or not category):
        raise ValueError("SECURITY_MASTER_ROW_IDENTITY_OR_CLASSIFICATION_INVALID")
    symbol, name = matched.groups()
    listed = datetime.strptime(row[2], "%Y/%m/%d").date().isoformat() if row[2] else None
    return Instrument(market, symbol, name.strip(), row[1], row[5], category, listed,
                      row[4], row[6], content_id(dict(category=category, row=list(row))), tuple(row))


def issuer_from_row(row, market) -> Issuer:
    fields = (("公司代號", "公司簡稱", "公司名稱", "營利事業統一編號", "上市日期") if market == "TWSE" else
              ("SecuritiesCompanyCode", "CompanyAbbreviation", "CompanyName", "UnifiedBusinessNo.", "DateOfListing"))
    if (market not in ISSUER_URLS or not isinstance(row, dict)
            or any(not isinstance(row.get(field), str) for field in fields)
            or not re.fullmatch(r"[0-9A-Z]{4,12}", row[fields[0]])
            or not row[fields[1]].strip() or not row[fields[2]].strip()
            or not re.fullmatch(r"\d{8}", row[fields[3]])):
        raise ValueError("SECURITY_MASTER_ISSUER_IDENTITY_INVALID")
    listed = datetime.strptime(row[fields[4]], "%Y%m%d").date().isoformat() if row[fields[4]] else None
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return Issuer(market, row[fields[0]], row[fields[1]], row[fields[2]], row[fields[3]], listed, content_id(row), raw)


def _node(left: str, right: str) -> str:
    return hashlib.sha256(b"security-master-node-v1\0" + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def merkle_levels(rows) -> tuple[tuple[str, ...], ...]:
    """Domain-separated binary inclusion tree; odd terminal nodes duplicate."""
    leaves = tuple(content_id(asdict(row)) for row in rows)
    if not leaves:
        raise ValueError("SECURITY_MASTER_EMPTY_MERKLE_TREE")
    levels = [leaves]
    while len(levels[-1]) > 1:
        level = levels[-1]
        levels.append(tuple(_node(level[i], level[i + 1] if i + 1 < len(level) else level[i])
                            for i in range(0, len(level), 2)))
    return tuple(levels)


def merkle_proof(levels, index: int) -> dict:
    count = len(levels[0])
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("SECURITY_MASTER_MERKLE_INDEX_INVALID")
    offset, siblings = index, []
    for level in levels[:-1]:
        sibling = offset ^ 1
        siblings.append(level[sibling] if sibling < len(level) else level[offset])
        offset //= 2
    return dict(index=index, count=count, siblings=siblings)


def verify_inclusion(row, proof: dict, *, root: str, count: int) -> None:
    if (not isinstance(proof, dict) or type(proof.get("index")) is not int
            or type(proof.get("count")) is not int or proof["count"] != count
            or not 0 <= proof["index"] < count or not isinstance(proof.get("siblings"), list)):
        raise ValueError("SECURITY_MASTER_INCLUSION_PROOF_INVALID")
    level_count, offset = count, proof["index"]
    value = content_id(asdict(row))
    expected_length = (count - 1).bit_length()
    if len(proof["siblings"]) != expected_length:
        raise ValueError("SECURITY_MASTER_INCLUSION_PROOF_LENGTH_INVALID")
    for sibling in proof["siblings"]:
        if not isinstance(sibling, str) or not re.fullmatch(r"[0-9a-f]{64}", sibling):
            raise ValueError("SECURITY_MASTER_INCLUSION_HASH_INVALID")
        if offset % 2 == 0 and offset + 1 >= level_count and sibling != value:
            raise ValueError("SECURITY_MASTER_ODD_LEAF_BINDING_MISMATCH")
        value = _node(sibling, value) if offset % 2 else _node(value, sibling)
        offset //= 2; level_count = (level_count + 1) // 2
    if value != root:
        raise ValueError("SECURITY_MASTER_INCLUSION_ROOT_MISMATCH")


def master_identity(header: dict) -> str:
    return content_id(dict(schema=SCHEMA, policy=POLICY, market=header["market"],
        parser_version=PARSER_VERSION, time_policy=TIME_POLICY,
        documents={key: {field: header[key][field] for field in ("url", "sha256", "retrieved_at")}
                   for key in ("instruments_document", "issuers_document")},
        instrument_row_count=header["instrument_row_count"], issuer_row_count=header["issuer_row_count"],
        instruments_root=header["instruments_root"], issuers_root=header["issuers_root"]))


def verify_compact_security_selection(selection: dict, *, cutoff: str) -> dict:
    """Validate compact projection/inclusion, not certify official signatures.

    The capture boundary additionally reparses every full original to establish
    these roots. The source-policy binding must retain the same master identity.
    """
    from copy import deepcopy
    if (not isinstance(selection, dict) or selection.get("schema") != SELECTION_SCHEMA
            or selection.get("policy") != POLICY or timestamp(selection.get("cutoff")) != timestamp(cutoff)):
        raise ValueError("SECURITY_MASTER_SELECTION_CUTOFF_OR_SCHEMA_MISMATCH")
    header = selection.get("master", {})
    if (not isinstance(header, dict) or header.get("schema") != SCHEMA or header.get("policy") != POLICY
            or header.get("market") not in ISIN_URLS or header.get("parser_version") != PARSER_VERSION
            or header.get("time_policy") != TIME_POLICY or header.get("available_at") != header.get("fetched_at")
            or header.get("historical_information_set_certified") is not False
            or header.get("original_publication_at") is not None
            or header.get("availability_basis") != "ACTUAL_FIRST_OBSERVATION" or header.get("limitations") != LIMITATIONS
            or header.get("source_authenticity") != "OFFICIAL_HTTPS_SOURCE_HASH_BOUND_NOT_CERTIFIED"):
        raise ValueError("SECURITY_MASTER_COMPACT_HEADER_INVALID")
    market = header["market"]
    for key, url in (("instruments_document", ISIN_URLS[market]), ("issuers_document", ISSUER_URLS[market])):
        doc = header.get(key)
        if (not isinstance(doc, dict) or doc.get("url") != url
                or timestamp(doc.get("retrieved_at")) > timestamp(header["fetched_at"])
                or not re.fullmatch(r"[0-9a-f]{64}", str(doc.get("sha256", "")))):
            raise ValueError("SECURITY_MASTER_SOURCE_TIME_OR_MARKET_BINDING_MISMATCH")
    if max(timestamp(header[key]["retrieved_at"]) for key in ("instruments_document", "issuers_document")) != timestamp(header["available_at"]):
        raise ValueError("SECURITY_MASTER_CAPTURE_COMPLETION_TIME_MISMATCH")
    for key in ("instrument_row_count", "issuer_row_count"):
        if type(header.get(key)) is not int or not 1 <= header[key] <= 100_000:
            raise ValueError("SECURITY_MASTER_COMPACT_COUNT_INVALID")
    if (header.get("evidence_id") != master_identity(header)
            or selection.get("master_evidence_id") != header["evidence_id"]):
        raise ValueError("SECURITY_MASTER_PACKET_CONTENT_ID_MISMATCH")
    available = timestamp(header["available_at"])
    instrument_record, issuer_record = selection.get("instrument"), selection.get("issuer")
    if available > timestamp(cutoff):
        expected = dict(market=market, market_symbol=selection["market_symbol"], issuer_symbol=None,
            security_type="UNKNOWN", status="UNKNOWN", reason="SECURITY_MASTER_NOT_AVAILABLE_AT_CUTOFF",
            low_entry_model_supported=False, instrument=None, issuer=None)
    else:
        # Missing-row proofs deliberately do NOT prove delisting or no issuer.
        instrument = None
        if instrument_record is not None:
            instrument = instrument_from_row(instrument_record["raw_cells"], instrument_record["source_category"], market)
            if content_id(asdict(instrument)) != content_id(instrument_record):
                raise ValueError("SECURITY_MASTER_INSTRUMENT_PROJECTION_MISMATCH")
            verify_inclusion(instrument, selection.get("instrument_inclusion"),
                root=header["instruments_root"], count=header["instrument_row_count"])
        issuer = None
        if issuer_record is not None:
            issuer = issuer_from_row(json.loads(issuer_record["raw_row_json"]), market)
            if content_id(asdict(issuer)) != content_id(issuer_record):
                raise ValueError("SECURITY_MASTER_ISSUER_PROJECTION_MISMATCH")
            verify_inclusion(issuer, selection.get("issuer_inclusion"),
                root=header["issuers_root"], count=header["issuer_row_count"])
        expected = decide_identity(instrument, issuer, market=market, symbol=selection["market_symbol"], cutoff=cutoff)
    for key, value in expected.items():
        if content_id(selection.get(key)) != content_id(value):
            raise ValueError("SECURITY_MASTER_SELECTION_SOURCE_BINDING_MISMATCH")
    if selection.get("selection_id") != content_id({key: value for key, value in selection.items()
                                                    if key not in {"master", "selection_id"}}):
        raise ValueError("SECURITY_MASTER_SELECTION_ID_MISMATCH")
    return deepcopy(selection)


def decide_identity(instrument: Instrument | None, issuer: Issuer | None, *,
                    market: str, symbol: str, cutoff: str) -> dict:
    """Pure selection; original-byte validation is the infrastructure boundary."""
    if market not in {"TWSE", "TPEX"} or not re.fullmatch(r"[0-9A-Z]{4,12}", symbol):
        raise ValueError("SECURITY_MASTER_MARKET_OR_SYMBOL_INVALID")
    when = timestamp(cutoff)
    result = dict(market=market, market_symbol=symbol, issuer_symbol=None,
        security_type="UNKNOWN", status="UNKNOWN", reason="SECURITY_MASTER_ROW_MISSING",
        low_entry_model_supported=False, instrument=None, issuer=None)
    if instrument is None:
        return result
    if instrument.market != market or instrument.symbol != symbol:
        raise ValueError("SECURITY_MASTER_INSTRUMENT_BINDING_MISMATCH")
    result.update(instrument=asdict(instrument), security_type=instrument.security_type)
    if instrument.listing_date and date.fromisoformat(instrument.listing_date) > when.astimezone(ZoneInfo("Asia/Taipei")).date():
        result["reason"] = "SECURITY_NOT_YET_LISTED"
        return result
    if instrument.security_type != "COMMON":
        result.update(status="OUT_OF_SCOPE", reason="NON_COMMON_EQUITY_MODEL_UNSUPPORTED")
        # The official ISIN row identifies a security, not a proof of the
        # preferred/DR-to-issuer relation. Do not strip A or reuse ordinary EPS.
        return result
    if issuer is None:
        result["reason"] = "OFFICIAL_ISSUER_ROW_MISSING"
        return result
    if issuer.market != market or issuer.symbol != symbol:
        raise ValueError("SECURITY_MASTER_ISSUER_BINDING_MISMATCH")
    result["issuer"] = asdict(issuer)
    if (instrument.listing_date is None or issuer.listing_date is None
            or instrument.listing_date != issuer.listing_date):
        result.update(status="CONFLICT" if instrument.listing_date and issuer.listing_date else "UNKNOWN",
                      reason="SECURITY_ISSUER_LISTING_DATE_UNVERIFIED")
        return result
    result.update(issuer_symbol=issuer.symbol, status="VERIFIED", reason="OFFICIAL_COMMON_SECURITY_AND_ISSUER_BOUND",
        low_entry_model_supported=True)
    return result


__all__ = ["SCHEMA", "SELECTION_SCHEMA", "POLICY", "TIME_POLICY", "PARSER_VERSION", "Instrument", "Issuer",
    "content_id", "timestamp", "decide_identity", "instrument_from_row", "issuer_from_row",
    "master_identity", "merkle_levels", "merkle_proof", "verify_compact_security_selection"]

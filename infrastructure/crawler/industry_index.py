"""Bounded official-final price-index capture, NOT historical membership proof.

The monthly endpoint returns a whole month even when asked for an earlier day.
Keep its exact bytes, filter rows explicitly, and never manufacture publication
times, revisions, constituents, or an industry-benchmark-source/v1 PASS packet.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from urllib.parse import urlencode

from application.replay_contracts import digest
from domain.industry_benchmark import sector_code
from domain.les_v7.contracts import timestamp
from infrastructure.crawler.twse_http_client import TwseHttpClient, TwseHttpConfig
from infrastructure.research_archive import write_bytes_once, write_once

ENDPOINT = "https://www.twse.com.tw/indicesReport/EFTRI_HIST"
SCHEMA = "official-industry-index-capture/v1"
PARSER_VERSION = "TWSE_EFTRI_HIST_PRICE_R1"
SECTOR_SCHEMA = "official-industry-index-capture/v2"
SECTOR_PARSER_VERSION = "TWSE_EFTRI_HIST_NAMED_SECTOR_PRICE_R2"
COLUMN_MAPPING_VERSION = "TWSE_EFTRI_HIST_OFFICIAL_SECTOR_COLUMNS_R1"
FIELDS = ["日　期", "電子類指數", "半導體類指數", "電腦及週邊設備類指數", "光電類指數",
          "通信網路類指數", "電子零組件類指數", "電子通路類指數", "資訊服務類指數", "其他電子類指數", "金融保險類指數"]
MAX_BODY_BYTES = 2 * 1024 * 1024
SECTOR = sector_code("ELEC")
# These are named columns in the exact verified exchange table, not inferred
# stock membership or an alias for the broad electronics index.
INDEX_COLUMNS = MappingProxyType({
    "13": "電子類指數", "24": "半導體類指數", "25": "電腦及週邊設備類指數",
    "26": "光電類指數", "27": "通信網路類指數", "28": "電子零組件類指數",
    "29": "電子通路類指數", "30": "資訊服務類指數", "31": "其他電子類指數",
})


def _index_spec(sector) -> tuple[str, str]:
    # Existing enum names/codes are supported. Exact official field names are
    # also explicit selectors; arbitrary company/industry text is not guessed.
    by_name = {name: code for code, name in INDEX_COLUMNS.items()}
    code = by_name.get(sector) if isinstance(sector, str) else None
    code = code or sector_code(sector)
    if code not in INDEX_COLUMNS:
        raise ValueError("UNSUPPORTED_OFFICIAL_INDUSTRY_SECTOR")
    return code, INDEX_COLUMNS[code]


def _day(value: str) -> date:
    day = date.fromisoformat(value)
    if day.isoformat() != value:
        raise ValueError("INDUSTRY_ISO_DATE_REQUIRED")
    return day


def months_in_range(start: str, end: str, *, max_months: int = 4) -> list[str]:
    """Plan before network I/O. Never silently truncate an over-budget request."""
    first, last = _day(start), _day(end)
    if first > last or type(max_months) is not int or not 1 <= max_months <= 24:
        raise ValueError("INVALID_INDUSTRY_CAPTURE_RANGE_OR_BUDGET")
    months, day = [], first.replace(day=1)
    while day <= last:
        months.append(day.isoformat())
        if len(months) > max_months:
            raise ValueError("INDUSTRY_CAPTURE_REQUEST_BUDGET_EXCEEDED")
        day = date(day.year + (day.month == 12), day.month % 12 + 1, 1)
    return months


def parse_month(body: bytes, *, month: str) -> list[dict]:
    """Strict observed exchange schema; missing rows are not evidence of closure."""
    return _parse_month(body, month=month, field="電子類指數")


def parse_sector_month(body: bytes, *, month: str, sector) -> list[dict]:
    """Select one explicitly named official price-index column, never members."""
    _, field = _index_spec(sector)
    return _parse_month(body, month=month, field=field)


def _parse_month(body: bytes, *, month: str, field: str) -> list[dict]:
    requested = _day(month)
    if requested.day != 1 or len(body) > MAX_BODY_BYTES:
        raise ValueError("INVALID_INDUSTRY_MONTH_OR_BODY_SIZE")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DUPLICATE_INDUSTRY_RESPONSE_KEY")
            result[key] = value
        return result

    packet = json.loads(body.decode("utf-8"), object_pairs_hook=unique)
    title = f"{requested.year - 1911}年{requested.month:02d}月 電子類指數及金融保險類指數"
    if (not isinstance(packet, dict) or packet.get("stat") != "OK"
            or packet.get("title") != title or packet.get("date") != requested.strftime("%Y%m%d")
            or packet.get("fields") != FIELDS or not isinstance(packet.get("data"), list)
            or type(packet.get("total")) is not int or packet["total"] != len(packet["data"])):
        raise ValueError("OFFICIAL_INDUSTRY_PRICE_SCHEMA_MISMATCH")
    rows, seen = [], set()
    for row in packet["data"]:
        if (not isinstance(row, list) or len(row) != len(FIELDS)
                or not isinstance(row[0], str) or not re.fullmatch(r"\d{3}/\d{2}/\d{2}", row[0])):
            raise ValueError("INVALID_INDUSTRY_PRICE_ROW")
        year, number, day = map(int, row[0].split("/"))
        session = date(year + 1911, number, day)
        if (session.year, session.month) != (requested.year, requested.month) or session in seen:
            raise ValueError("INDUSTRY_PRICE_DATE_CONFLICT")
        seen.add(session)
        value = row[FIELDS.index(field)]
        if not isinstance(value, str) or not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", value):
            raise ValueError("INVALID_INDUSTRY_PRICE_VALUE")
        try:
            close = Decimal(value.replace(",", ""))
        except InvalidOperation as error:
            raise ValueError("INVALID_INDUSTRY_PRICE_VALUE") from error
        if not close.is_finite() or close <= 0 or not float(close) < float("inf"):
            raise ValueError("INVALID_INDUSTRY_PRICE_VALUE")
        rows.append(dict(trade_date=session.isoformat(), close=float(close)))
    return sorted(rows, key=lambda row: row["trade_date"])


def _selection(documents, read, *, start, end, cutoff, sector=None):
    first, last, at = _day(start), _day(end), timestamp(cutoff)
    selected, excluded = [], []
    for document in documents:
        body = read(document)
        if len(body) != document["size_bytes"] or sha256(body).hexdigest() != document["sha256"]:
            raise ValueError("INDUSTRY_CAPTURE_ORIGINAL_HASH_MISMATCH")
        if document["url"] != ENDPOINT + "?" + urlencode(dict(date=_day(document["month"]).strftime("%Y%m%d"), response="json")):
            raise ValueError("INDUSTRY_CAPTURE_SOURCE_BINDING_MISMATCH")
        observed = timestamp(document["retrieved_at"])
        rows = (parse_month(body, month=document["month"]) if sector is None else
                parse_sector_month(body, month=document["month"], sector=sector))
        for row in rows:
            day = _day(row["trade_date"])
            close_at = timestamp(day.isoformat() + "T13:30:00+08:00")
            reason = ("OUTSIDE_REQUESTED_DATE_RANGE" if not first <= day <= last else
                      "AFTER_CUTOFF_SESSION_CLOSE" if close_at > at else
                      "DOCUMENT_OBSERVED_BEFORE_SESSION_CLOSE" if close_at > observed else
                      "DOCUMENT_NOT_AVAILABLE_AT_CUTOFF" if observed > at else None)
            if reason:
                excluded.append(dict(trade_date=day.isoformat(), reason=reason, document_sha256=document["sha256"]))
            else:
                selected.append(dict(row, document_sha256=document["sha256"], available_at=document["retrieved_at"],
                    availability_source="FIRST_OBSERVED_DOWNLOAD", original_publication_time=None,
                    original_revision=None, data_kind="POSTHOC_OFFICIAL_FINAL_PRICE_INDEX"))
    if len({row["trade_date"] for row in selected}) != len(selected):
        raise ValueError("INDUSTRY_CAPTURE_DATE_CONFLICT")
    return sorted(selected, key=lambda row: row["trade_date"]), excluded


def capture_index(root, *, start, end, max_months=4, client=None, clock=None, sector=None) -> Path:
    """Manual explicit download; stops on the first HTTP/schema/integrity error.

    Reuses the project's rate-limited TLS-verifying transport and immutable
    publication. No financial refresh, model, account, mail or order side effect.
    """
    if sector is not None:
        code, _ = _index_spec(sector)
        return capture_indices(root, sectors=[code], start=start, end=end,
            max_months=max_months, client=client, clock=clock)[code]
    months = months_in_range(start, end, max_months=max_months)
    root = Path(root)
    clock = clock or (lambda: datetime.now(timezone.utc))
    client = client or TwseHttpClient(TwseHttpConfig(timeout_seconds=30, max_attempts=1,
        min_request_interval_seconds=1, allow_insecure_ssl_fallback=False))
    documents, empty_months = [], []
    for month in months:
        params = dict(date=_day(month).strftime("%Y%m%d"), response="json")
        response = client.request_bytes("GET", ENDPOINT, params=params, context="Official-final industry index " + month)
        observed = clock().isoformat()
        timestamp(observed)
        if not parse_month(response.content, month=month):
            empty_months.append(month)
        content_id = sha256(response.content).hexdigest()
        write_bytes_once(root / "objects" / (content_id + ".json"), response.content)
        documents.append(dict(month=month, url=ENDPOINT + "?" + urlencode(params), sha256=content_id,
                              size_bytes=len(response.content), retrieved_at=observed))
    cutoff = clock().isoformat()
    rows, excluded = _selection(documents, lambda d: (root / "objects" / (d["sha256"] + ".json")).read_bytes(),
                                 start=start, end=end, cutoff=cutoff)
    payload = dict(schema=SCHEMA, parser_version=PARSER_VERSION, market="TWSE", sector=SECTOR,
        index_name="電子類指數", index_type="PRICE_INDEX", start=start, end=end, cutoff=cutoff,
        documents=documents, rows=rows, excluded_rows=excluded, eligible_for_industry_scoring=False,
        status="PARTIAL", empty_months=empty_months,
        gaps=["HISTORICAL_MEMBERSHIP_EVIDENCE_NOT_PROVIDED"] +
            (["NO_INDEX_ROWS_IN_REQUESTED_RANGE"] if not rows else []) + ["EMPTY_INDEX_MONTH:" + m for m in empty_months],
        historical_information_set_certified=False, membership_evidence=[],
        missing_rows_policy="NO_MARKET_CLOSURE_INFERENCE", no_accounts_or_mail=True)
    capture_id = digest(payload)
    path = root / "captures" / (capture_id + ".json")
    write_once(path, dict(payload, capture_id=capture_id))
    return path


def capture_indices(root, *, sectors, start, end, max_months=4, client=None, clock=None) -> dict[str, Path]:
    """Capture requested subindices with one bounded request per month.

    All requested sectors share the same verified response bytes, URL and
    observation time. The new v2 artifacts remain PARTIAL without membership
    proof and cannot be fed to the scored-source importer as PASS packets.
    Default capture_index() remains the byte-compatible v1 electronics path.
    """
    if not isinstance(sectors, (list, tuple)) or not sectors:
        raise ValueError("EXPLICIT_OFFICIAL_INDUSTRY_SECTORS_REQUIRED")
    selected = [_index_spec(sector) for sector in sectors]
    codes = [code for code, _ in selected]
    if len(codes) != len(set(codes)):
        raise ValueError("DUPLICATE_OFFICIAL_INDUSTRY_SECTOR")
    months = months_in_range(start, end, max_months=max_months)
    root = Path(root)
    clock = clock or (lambda: datetime.now(timezone.utc))
    client = client or TwseHttpClient(TwseHttpConfig(timeout_seconds=30, max_attempts=1,
        min_request_interval_seconds=1, allow_insecure_ssl_fallback=False))
    documents, empty = [], {code: [] for code in codes}
    for month in months:
        params = dict(date=_day(month).strftime("%Y%m%d"), response="json")
        response = client.request_bytes("GET", ENDPOINT, params=params,
            context="Official-final industry index " + month)
        observed = clock().isoformat()
        timestamp(observed)
        # Validate every requested series before a manifest is published. A bad
        # requested column stops the work, not silently dropping that sector.
        for code in codes:
            if not parse_sector_month(response.content, month=month, sector=code):
                empty[code].append(month)
        content_id = sha256(response.content).hexdigest()
        write_bytes_once(root / "objects" / (content_id + ".json"), response.content)
        documents.append(dict(month=month, url=ENDPOINT + "?" + urlencode(params), sha256=content_id,
            size_bytes=len(response.content), retrieved_at=observed))
    cutoff = clock().isoformat()
    paths = {}
    for code, name in selected:
        rows, excluded = _selection(documents,
            lambda d: (root / "objects" / (d["sha256"] + ".json")).read_bytes(),
            start=start, end=end, cutoff=cutoff, sector=code)
        payload = dict(schema=SECTOR_SCHEMA, parser_version=SECTOR_PARSER_VERSION,
            market="TWSE", sector=code, index_name=name, index_type="PRICE_INDEX",
            index_selection=dict(mapping_version=COLUMN_MAPPING_VERSION, sector=code, field=name),
            start=start, end=end, cutoff=cutoff, documents=documents, rows=rows, excluded_rows=excluded,
            eligible_for_industry_scoring=False, status="PARTIAL", empty_months=empty[code],
            gaps=["HISTORICAL_MEMBERSHIP_EVIDENCE_NOT_PROVIDED"] +
                (["NO_INDEX_ROWS_IN_REQUESTED_RANGE"] if not rows else []) +
                ["EMPTY_INDEX_MONTH:" + month for month in empty[code]],
            historical_information_set_certified=False, membership_evidence=[],
            missing_rows_policy="NO_MARKET_CLOSURE_INFERENCE", no_accounts_or_mail=True)
        capture_id = digest(payload)
        path = root / "captures" / (capture_id + ".json")
        write_once(path, dict(payload, capture_id=capture_id))
        paths[code] = path
    return paths


def verify_capture(path, root) -> dict:
    """Reparse retained originals, not a self-certified list of extracted prices."""
    packet = json.loads(Path(path).read_text(encoding="utf-8"))
    capture_id = packet.pop("capture_id")
    legacy = packet.get("schema") == SCHEMA and packet.get("parser_version") == PARSER_VERSION
    named = packet.get("schema") == SECTOR_SCHEMA and packet.get("parser_version") == SECTOR_PARSER_VERSION
    if not (legacy or named) or digest(packet) != capture_id:
        raise ValueError("INDUSTRY_CAPTURE_ID_MISMATCH")
    if (packet.get("eligible_for_industry_scoring") is not False or packet.get("status") != "PARTIAL"
            or packet.get("membership_evidence") != [] or packet.get("historical_information_set_certified") is not False):
        raise ValueError("INDUSTRY_CAPTURE_CANNOT_CERTIFY_MEMBERSHIP")
    code, name = (SECTOR, "電子類指數") if legacy else _index_spec(packet.get("sector"))
    if (packet.get("market") != "TWSE" or packet.get("sector") != code
            or packet.get("index_name") != name or packet.get("index_type") != "PRICE_INDEX"
            or packet.get("missing_rows_policy") != "NO_MARKET_CLOSURE_INFERENCE"
            or packet.get("no_accounts_or_mail") is not True):
        raise ValueError("INDUSTRY_CAPTURE_SOURCE_BINDING_MISMATCH")
    if named and packet.get("index_selection") != dict(
            mapping_version=COLUMN_MAPPING_VERSION, sector=code, field=name):
        raise ValueError("INDUSTRY_CAPTURE_COLUMN_BINDING_MISMATCH")
    planned = months_in_range(packet["start"], packet["end"], max_months=24)
    if [d["month"] for d in packet["documents"]] != planned:
        raise ValueError("INDUSTRY_CAPTURE_MONTH_COVERAGE_MISMATCH")

    def read(document):
        content_id = document["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", content_id):
            raise ValueError("INVALID_INDUSTRY_CAPTURE_CONTENT_ID")
        return (Path(root) / "objects" / (content_id + ".json")).read_bytes()

    rows, excluded = _selection(packet["documents"], read, start=packet["start"], end=packet["end"], cutoff=packet["cutoff"],
        sector=code if named else None)
    empty_months = [d["month"] for d in packet["documents"] if not (
        parse_sector_month(read(d), month=d["month"], sector=code) if named else parse_month(read(d), month=d["month"]))]
    gaps = (["HISTORICAL_MEMBERSHIP_EVIDENCE_NOT_PROVIDED"] + (["NO_INDEX_ROWS_IN_REQUESTED_RANGE"] if not rows else [])
            + ["EMPTY_INDEX_MONTH:" + m for m in empty_months])
    if (rows != packet["rows"] or excluded != packet["excluded_rows"] or packet["gaps"] != gaps
            or packet.get("empty_months") != empty_months):
        raise ValueError("INDUSTRY_CAPTURE_EXTRACTION_MISMATCH")
    return dict(packet, capture_id=capture_id)

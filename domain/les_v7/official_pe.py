"""Dated official P/E observations, never inferred earnings or historical clocks.

The exchanges explicitly omit P/E for some securities. That is distinct from a
missing response/row and is not, by itself, proof of a loss or an unsafe company.
Reconstruction from today's final table is available today, not on its date.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from bisect import bisect_left
from copy import deepcopy
from datetime import date, datetime, time
import hashlib
import html
import json
import math
import re
import unicodedata
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from domain.errors import NoDataError, SchemaChangedError
from domain.les_v7.contracts import canonical, identity, timestamp


SCHEMA = "official-dated-pe/v1"
PARSER_VERSION = "OFFICIAL_DATED_PE_R1"
ENDPOINTS = {
    "TWSE": "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d",
    "TPEX": "https://www.tpex.org.tw/www/zh-tw/afterTrading/peQryDate",
}
AVAILABILITY_BASIS = "FINAL_DATA_OBSERVED_AT_FETCH"
TAIPEI = ZoneInfo("Asia/Taipei")
MANIFEST_SCHEMA = "official-dated-pe-manifest/v1"
MAX_RAW_BYTES = 8 * 1024 * 1024


def session_date(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("OFFICIAL_PE_DATE_REQUIRED")
    result = (datetime.strptime(value, "%Y%m%d").date() if re.fullmatch(r"\d{8}", value)
              else date.fromisoformat(value))
    return result.isoformat()


def request_parameters(market: str, trade_date: str) -> dict:
    day = session_date(trade_date)
    if market == "TWSE":
        return dict(date=day.replace("-", ""), selectType="ALL", response="json")
    if market == "TPEX":
        return dict(date=day.replace("-", "/"), cate="", response="json")
    raise ValueError("OFFICIAL_PE_MARKET_UNSUPPORTED")


def _text(value: object) -> str:
    return unicodedata.normalize("NFKC", html.unescape(str(value))).strip()


def _fields(values: list) -> list[str]:
    return [re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", _text(value))) for value in values]


def _json_object(pairs):
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("OFFICIAL_PE_DUPLICATE_JSON_KEY")
    return dict(pairs)


def _leaf(record):
    return hashlib.sha256(b"L" + canonical(record).encode()).hexdigest()


def _branch(left, right):
    return hashlib.sha256(b"N" + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def _merkle_levels(records):
    levels = [[_leaf(row) for row in records]]
    while len(levels[-1]) > 1:
        nodes = levels[-1]
        levels.append([_branch(nodes[index], nodes[min(index + 1, len(nodes) - 1)])
                       for index in range(0, len(nodes), 2)])
    return tuple(tuple(level) for level in levels)


def _parse(payload: object, *, market: str, trade_date: str) -> list[dict]:
    day = date.fromisoformat(session_date(trade_date))
    compact = day.strftime("%Y%m%d")
    if not isinstance(payload, dict):
        raise SchemaChangedError("Official P/E response must be an object")
    stat = str(payload.get("stat", "")).strip()
    if stat.lower() != "ok":
        if any(marker in stat for marker in ("查無資料", "沒有符合", "無資料")):
            raise NoDataError(f"Official P/E no data for {market} {compact}")
        raise SchemaChangedError("Official P/E status is not OK")
    if payload.get("date") != compact:
        raise SchemaChangedError("Official P/E response date mismatch")
    if market == "TWSE":
        table = payload
        if table.get("selectType") != "ALL":
            raise SchemaChangedError("Official TWSE P/E must cover ALL securities")
        code, name, count = "證券代號", "證券名稱", "total"
    elif market == "TPEX":
        tables = payload.get("tables")
        if not isinstance(tables, list) or len(tables) != 1 or not isinstance(tables[0], dict):
            raise SchemaChangedError("Official TPEx P/E requires one full-market table")
        table = tables[0]
        if (table.get("date") != f"{day.year - 1911}/{day.month:02d}/{day.day:02d}"
                or table.get("stkCategory") != "全部產業"):
            raise SchemaChangedError("Official TPEx P/E table date/scope mismatch")
        code, name, count = "股票代號", "公司名稱", "totalCount"
    else:
        raise ValueError("OFFICIAL_PE_MARKET_UNSUPPORTED")
    fields, rows = table.get("fields"), table.get("data")
    if (not isinstance(fields, list) or not all(isinstance(field, str) for field in fields)
            or not isinstance(rows, list)):
        raise SchemaChangedError("Official P/E fields/data missing")
    names = _fields(fields)
    if len(names) != len(set(names)) or not {code, name, "本益比", "財報年/季"}.issubset(names):
        raise SchemaChangedError("Official P/E fields missing or duplicate")
    if type(table.get(count)) is not int or table[count] != len(rows):
        raise SchemaChangedError("Official P/E row count mismatch")
    if not rows:
        raise NoDataError(f"Official P/E empty table for {market} {compact}")
    by_symbol: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != len(names):
            raise SchemaChangedError("Official P/E row width mismatch")
        record = dict(zip(names, row))
        if not isinstance(record[code], str) or not isinstance(record[name], str):
            raise SchemaChangedError("Official P/E security identity must be text")
        symbol, label = _text(record[code]), _text(record[name])
        if not re.fullmatch(r"[0-9A-Z]{4,10}", symbol) or not label:
            raise SchemaChangedError("Official P/E security identity missing")
        raw_pe = _text(record["本益比"])
        pe = None
        if raw_pe in {"-", "--", "---", "----", "N/A"}:
            status = "NOT_REPORTED"
        elif raw_pe == "":
            status = "MISSING_VALUE"
        else:
            if not re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?", raw_pe):
                raise SchemaChangedError("Official P/E numeric value malformed")
            try:
                pe = float(raw_pe.replace(",", ""))
            except ValueError as error:
                raise SchemaChangedError("Official P/E numeric value malformed") from error
            if not math.isfinite(pe):
                raise SchemaChangedError("Official P/E numeric value not finite")
            status = "REPORTED" if pe > 0 else "NON_POSITIVE"
        financial_period = _text(record["財報年/季"])
        reported_close, raw_close, close_status = None, None, "NOT_IN_SOURCE"
        if "收盤價" in record:
            raw_close = _text(record["收盤價"])
            close_status = "NOT_REPORTED"
            if raw_close not in {"", "-", "--", "---", "----", "N/A"}:
                if not re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?", raw_close):
                    raise SchemaChangedError("Official P/E closing price malformed")
                reported_close = float(raw_close.replace(",", ""))
                if not math.isfinite(reported_close):
                    raise SchemaChangedError("Official P/E closing price invalid")
                # Official BWIBBU_d actually returns 0.00 for some securities
                # while retaining a reference P/E. Preserve this source value
                # and flag it; it is not a quote, and must not poison other rows
                # or be replaced with a favourable old/derived price.
                close_status = "REPORTED" if reported_close > 0 else "NON_POSITIVE"
        item = dict(market=market, symbol=symbol, security_name=label,
                    session_date=day.isoformat(), pe=pe, status=status,
                    raw_pe=raw_pe, financial_period=financial_period,
                    reported_close=reported_close,
                    raw_close=raw_close, close_status=close_status,
                    raw_cells=deepcopy(row))
        by_symbol.setdefault(symbol, []).append(item)
    result = []
    for symbol, items in sorted(by_symbol.items()):
        if len(items) == 1:
            result.append(items[0])
        else:
            # Even identical duplicate rows violate the full-table contract;
            # retain all versions, do not silently pick a tradable value.
            result.append(dict(market=market, symbol=symbol, session_date=day.isoformat(),
                               pe=None, status="CONFLICT", reason="DUPLICATE_SYMBOL",
                               conflicting_records=items))
    return result


def build_snapshot(raw_content: bytes, *, market: str, trade_date: str,
                   fetched_at: str) -> dict:
    """Bind the exact response bytes to its observed availability and mapping.

    No official publication time is claimed. The response date is an applicable
    session, not a historical observation time. Synthetic callers must label
    their enclosing fixture; a local content hash is not an official signature.
    """
    if not isinstance(raw_content, bytes):
        raise ValueError("OFFICIAL_PE_RAW_BYTES_REQUIRED")
    if len(raw_content) > MAX_RAW_BYTES:
        raise ValueError("OFFICIAL_PE_RAW_CAPACITY_EXCEEDED")
    day, observed = session_date(trade_date), timestamp(fetched_at)
    if observed.astimezone(TAIPEI) < datetime.combine(date.fromisoformat(day), time(13, 30), TAIPEI):
        raise ValueError("OFFICIAL_PE_SESSION_NOT_COMPLETED_AT_FETCH")
    try:
        raw_text = raw_content.decode("utf-8-sig")
        payload = json.loads(raw_text, object_pairs_hook=_json_object)
    except (UnicodeDecodeError, ValueError) as error:
        raise SchemaChangedError("Official P/E raw response is not UTF-8 JSON") from error
    params = request_parameters(market, day)
    records = _parse(payload, market=market, trade_date=day)
    root = _merkle_levels(records)[-1][0]
    table = payload if market == "TWSE" else payload["tables"][0]
    manifest = dict(schema=MANIFEST_SCHEMA, parser_version=PARSER_VERSION, market=market,
        session_date=day, scope="ALL_SECURITIES", source=market + "_DATED_PE",
        source_url=ENDPOINTS[market], request_parameters=params,
        content_sha256=hashlib.sha256(raw_content).hexdigest(),
        field_names=_fields(table["fields"]), record_count=len(records), record_root=root,
        fetched_at=observed.isoformat(), available_at=observed.isoformat(),
        original_publication_at=None, availability_basis=AVAILABILITY_BASIS,
        verification="LOCAL_RAW_BINDING_AND_MEMBERSHIP_NOT_EXCHANGE_SIGNATURE")
    body = dict(schema=SCHEMA, parser_version=PARSER_VERSION, market=market,
                session_date=day, scope="ALL_SECURITIES",
                source=market + "_DATED_PE", source_url=ENDPOINTS[market],
                request_parameters=params,
                request_url=ENDPOINTS[market] + "?" + urlencode(params),
                content_sha256=hashlib.sha256(raw_content).hexdigest(),
                raw_response_utf8=raw_content.decode("utf-8"),
                fetched_at=observed.isoformat(), available_at=observed.isoformat(),
                original_publication_at=None, availability_basis=AVAILABILITY_BASIS,
                source_authenticity="OFFICIAL_ENDPOINT_CAPTURE_NOT_CRYPTOGRAPHIC_SIGNATURE",
                records=records, record_manifest=manifest, manifest_id=identity(manifest))
    return dict(body, evidence_id=identity(body))


def validate_snapshot(snapshot: Mapping, *, cutoff: str | None = None) -> dict:
    """Reparse exact retained bytes and compare every derived/provenance field."""
    try:
        if not isinstance(snapshot, Mapping):
            raise ValueError("OFFICIAL_PE_SNAPSHOT_INVALID")
        actual = deepcopy(dict(snapshot))
        expected = build_snapshot(actual["raw_response_utf8"].encode("utf-8"),
                                  market=actual["market"], trade_date=actual["session_date"],
                                  fetched_at=actual["fetched_at"])
        if canonical(actual) != canonical(expected):
            raise ValueError("OFFICIAL_PE_SNAPSHOT_BINDING_INVALID")
        if cutoff is not None and timestamp(actual["available_at"]) > timestamp(cutoff):
            raise ValueError("OFFICIAL_PE_NOT_AVAILABLE_AT_CUTOFF")
        return expected
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError("OFFICIAL_PE_SNAPSHOT_INVALID") from error


class OfficialPeIndex:
    """Work-scoped bounded index: raw validation once, detached JSON on select.

    Index values contain immutable JSON strings, not caller-owned dict/list
    references. Each cutoff still independently selects effective observation
    versions. Nothing depends on an account, model, or future financial data.
    """
    def __init__(self, snapshots: Sequence[Mapping], *, max_observations: int = 128,
                 max_bytes: int = 128 * 1024 * 1024):
        if len(snapshots) > max_observations or max_observations < 1 or max_bytes < 1:
            raise ValueError("OFFICIAL_PE_INDEX_CAPACITY_EXCEEDED")
        self._index = {}
        used = 0
        for item in snapshots:
            checked = validate_snapshot(item)
            levels = _merkle_levels(checked["records"])
            rows = {row["symbol"]: canonical(row) for row in checked["records"]}
            manifest = canonical(checked["record_manifest"])
            used += (sum(len(value.encode()) for value in rows.values()) + len(manifest.encode())
                     + sum(len(node) for level in levels for node in level))
            if used > max_bytes:
                raise ValueError("OFFICIAL_PE_INDEX_CAPACITY_EXCEEDED")
            entry = (timestamp(checked["available_at"]), checked["evidence_id"], rows,
                     manifest, checked["manifest_id"], tuple(sorted(rows)), levels)
            self._index.setdefault((checked["market"], checked["session_date"]), []).append(entry)
        self.byte_count = used

    def select(self, *, market: str, symbol: str, trade_date: str, cutoff: str) -> dict:
        """Latest exact-market/date version, including disappearance/conflicts."""
        day = session_date(trade_date)
        request_parameters(market, day)
        at = timestamp(cutoff)
        selected = [entry for entry in self._index.get((market, day), ()) if entry[0] <= at]
        missing = dict(market=market, symbol=symbol, session_date=day, pe=None,
                       status="MISSING", reason="OFFICIAL_PE_OBSERVATION_MISSING",
                       source_record_ids=[], source_available_at=None, selection_proofs=[])
        if not selected:
            return missing
        latest = max(entry[0] for entry in selected)
        selected = [entry for entry in selected if entry[0] == latest]
        proofs = []
        for entry in selected:
            proof = dict(manifest=json.loads(entry[3]), manifest_id=entry[4],
                         snapshot_evidence_id=entry[1])
            position = bisect_left(entry[5], symbol)
            if symbol in entry[2]:
                proof["membership"] = self._membership(entry, position)
            else:
                keys = entry[5]
                proof["absence"] = dict(before=self._membership(entry, position - 1) if position else None,
                    after=self._membership(entry, position) if position < len(keys) else None)
            proofs.append(proof)
        return _selection_result(proofs, market=market, symbol=symbol, day=day)

    @staticmethod
    def _membership(entry, original):
        index, siblings = original, []
        for level in entry[6][:-1]:
            siblings.append(level[min(index ^ 1, len(level) - 1)])
            index //= 2
        return dict(record=json.loads(entry[2][entry[5][original]]), leaf_index=original, siblings=siblings)


def _selection_result(proofs, *, market, symbol, day):
    # Stable order and dedup make repeated identical snapshots idempotent.
    unique = {canonical(proof): proof for proof in proofs}
    proofs = [unique[key] for key in sorted(unique)]
    items = [proof.get("membership", {}).get("record") for proof in proofs]
    latest = max(timestamp(proof["manifest"]["available_at"]) for proof in proofs)
    missing = dict(market=market, symbol=symbol, session_date=day, pe=None,
                   status="MISSING", reason="SYMBOL_NOT_IN_OFFICIAL_PE_TABLE")
    if len({identity(item) for item in items}) != 1:
        result = dict(missing, status="CONFLICT", reason="CONFLICTING_OFFICIAL_PE_OBSERVATIONS")
    elif items[0] is None:
        result = missing
    else:
        result = deepcopy(items[0])
    return dict(result, source_record_ids=sorted({proof["snapshot_evidence_id"] for proof in proofs}),
                source_available_at=latest.isoformat(), parser_version=PARSER_VERSION,
                availability_basis=AVAILABILITY_BASIS, selection_proofs=deepcopy(proofs))


def _hex_id(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _check_manifest(proof, *, market, day, cutoff):
    manifest = proof.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("OFFICIAL_PE_MANIFEST_INVALID")
    keys = {"schema", "parser_version", "market", "session_date", "scope", "source", "source_url",
            "request_parameters", "content_sha256", "field_names", "record_count", "record_root",
            "fetched_at", "available_at", "original_publication_at", "availability_basis", "verification"}
    params = request_parameters(market, day)
    required = {"本益比", "財報年/季", *({"證券代號", "證券名稱"} if market == "TWSE"
                                          else {"股票代號", "公司名稱"})}
    fields = manifest.get("field_names")
    if (set(manifest) != keys or manifest.get("schema") != MANIFEST_SCHEMA
            or manifest.get("parser_version") != PARSER_VERSION
            or manifest.get("market") != market or manifest.get("session_date") != day
            or manifest.get("scope") != "ALL_SECURITIES" or manifest.get("source") != market + "_DATED_PE"
            or manifest.get("source_url") != ENDPOINTS[market] or manifest.get("request_parameters") != params
            or manifest.get("manifest_id") is not None or proof.get("manifest_id") != identity(manifest)
            or not _hex_id(proof.get("snapshot_evidence_id"))
            or not _hex_id(manifest.get("content_sha256")) or not _hex_id(manifest.get("record_root"))
            or type(manifest.get("record_count")) is not int or manifest["record_count"] < 1
            or not isinstance(fields, list) or not all(isinstance(name, str) for name in fields)
            or len(fields) != len(set(fields)) or not required.issubset(fields)
            or manifest.get("original_publication_at") is not None
            or manifest.get("availability_basis") != AVAILABILITY_BASIS
            or manifest.get("verification") != "LOCAL_RAW_BINDING_AND_MEMBERSHIP_NOT_EXCHANGE_SIGNATURE"
            or manifest.get("available_at") != manifest.get("fetched_at")):
        raise ValueError("OFFICIAL_PE_MANIFEST_BINDING_INVALID")
    observed = timestamp(manifest["available_at"])
    if (observed > timestamp(cutoff) or observed.astimezone(TAIPEI)
            < datetime.combine(date.fromisoformat(day), time(13, 30), TAIPEI)):
        raise ValueError("OFFICIAL_PE_NOT_AVAILABLE_AT_CUTOFF")
    return manifest


def _check_membership(member, manifest):
    if not isinstance(member, dict) or set(member) != {"record", "leaf_index", "siblings"}:
        raise ValueError("OFFICIAL_PE_MEMBERSHIP_INVALID")
    record, index, siblings = member["record"], member["leaf_index"], member["siblings"]
    count = manifest["record_count"]
    if (not isinstance(record, dict) or type(index) is not int or not 0 <= index < count
            or not isinstance(siblings, list) or not all(_hex_id(node) for node in siblings)):
        raise ValueError("OFFICIAL_PE_MEMBERSHIP_INVALID")
    # Recreate the row mapping from raw cells and the frozen field names. A
    # positive derived PE cannot replace an official N/A raw cell by rehashing.
    raw_rows = ([record["raw_cells"]] if "raw_cells" in record else
                [row["raw_cells"] for row in record.get("conflicting_records", ())])
    market, day = manifest["market"], manifest["session_date"]
    common = dict(stat="OK", date=day.replace("-", ""))
    table = dict(fields=manifest["field_names"], data=raw_rows)
    if market == "TWSE":
        payload = dict(common, **table, selectType="ALL", total=len(raw_rows))
    else:
        target = date.fromisoformat(day)
        payload = dict(common, tables=[dict(table, totalCount=len(raw_rows), stkCategory="全部產業",
                        date=f"{target.year - 1911}/{target.month:02d}/{target.day:02d}")])
    parsed = _parse(payload, market=market, trade_date=day)
    if canonical(parsed) != canonical([record]):
        raise ValueError("OFFICIAL_PE_ROW_MAPPING_INVALID")
    digest = _leaf(record)
    position = 0
    while count > 1:
        if position >= len(siblings):
            raise ValueError("OFFICIAL_PE_MEMBERSHIP_INVALID")
        sibling = siblings[position]
        if index % 2:
            digest = _branch(sibling, digest)
        else:
            if index == count - 1 and sibling != digest:
                raise ValueError("OFFICIAL_PE_MEMBERSHIP_INVALID")
            digest = _branch(digest, sibling)
        position += 1
        index //= 2
        count = (count + 1) // 2
    if position != len(siblings) or digest != manifest["record_root"]:
        raise ValueError("OFFICIAL_PE_MEMBERSHIP_INVALID")
    return record


def validate_pe_selection(selection: Mapping, *, market: str, symbol: str,
                          trade_date: str, cutoff: str) -> dict:
    """Validate compact row/absence proofs without embedding full raw tables.

    This verifies the content-addressed local manifest and membership; official
    authenticity/publication cannot be established by a hash alone. A full
    archive audit still validates its referenced raw snapshot with
    validate_snapshot(). Native reports must bind this record to their input ID.
    """
    try:
        day = session_date(trade_date)
        if not isinstance(selection, Mapping):
            raise ValueError("OFFICIAL_PE_SELECTION_INVALID")
        proofs = selection.get("selection_proofs")
        if not isinstance(proofs, list) or not proofs:
            raise ValueError("OFFICIAL_PE_SELECTION_PROOF_MISSING")
        times = set()
        for proof in proofs:
            if not isinstance(proof, dict):
                raise ValueError("OFFICIAL_PE_SELECTION_INVALID")
            if set(proof) not in ({"manifest", "manifest_id", "snapshot_evidence_id", "membership"},
                                  {"manifest", "manifest_id", "snapshot_evidence_id", "absence"}):
                raise ValueError("OFFICIAL_PE_SELECTION_INVALID")
            manifest = _check_manifest(proof, market=market, day=day, cutoff=cutoff)
            times.add(manifest["available_at"])
            if "membership" in proof and "absence" not in proof:
                if _check_membership(proof["membership"], manifest)["symbol"] != symbol:
                    raise ValueError("OFFICIAL_PE_SELECTION_IDENTITY_MISMATCH")
            elif "absence" in proof and "membership" not in proof:
                absence = proof["absence"]
                if not isinstance(absence, dict) or set(absence) != {"before", "after"}:
                    raise ValueError("OFFICIAL_PE_ABSENCE_INVALID")
                before, after = absence["before"], absence["after"]
                if before is None and after is None:
                    raise ValueError("OFFICIAL_PE_ABSENCE_INVALID")
                lower = _check_membership(before, manifest) if before is not None else None
                upper = _check_membership(after, manifest) if after is not None else None
                if (lower and lower["symbol"] >= symbol or upper and upper["symbol"] <= symbol
                        or lower is None and after["leaf_index"] != 0
                        or upper is None and before["leaf_index"] != manifest["record_count"] - 1
                        or lower and upper and after["leaf_index"] != before["leaf_index"] + 1):
                    raise ValueError("OFFICIAL_PE_ABSENCE_INVALID")
            else:
                raise ValueError("OFFICIAL_PE_SELECTION_INVALID")
        if (len(times) != 1 or canonical(dict(selection))
                != canonical(_selection_result(proofs, market=market, symbol=symbol, day=day))):
            raise ValueError("OFFICIAL_PE_SELECTION_BINDING_INVALID")
        return deepcopy(dict(selection))
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("OFFICIAL_PE_SELECTION_INVALID") from error


def select_pe(snapshots: Sequence[Mapping], *, market: str, symbol: str,
              trade_date: str, cutoff: str) -> dict:
    """Compatible one-shot entry; jobs should construct OfficialPeIndex once."""
    return OfficialPeIndex(snapshots).select(market=market, symbol=symbol,
                                            trade_date=trade_date, cutoff=cutoff)


def join_pe_rows(rows: Sequence[Mapping], snapshots: Sequence[Mapping], *, market: str,
                 symbol: str, cutoff: str) -> tuple[list[dict], list[dict]]:
    """Detached explicit date/symbol join; missing/unreported PE stays null.

    This is opt-in, not a mutation of legacy quote columns or old policies.
    Keep returned diagnostics with the joined inputs so source identity/time
    can be bound by the native valuation port. Do not turn a numeric join into
    an independently verified annual/Quality gate.
    """
    joined, evidence = [], []
    # Validate once per join rather than reparsing the same all-market raw table
    # for every date; detached local index never exposes shared mutable inputs.
    index = OfficialPeIndex(snapshots)
    for raw in rows:
        if not isinstance(raw, Mapping) or "Date" not in raw:
            raise ValueError("OFFICIAL_PE_JOIN_DATE_REQUIRED")
        row = deepcopy(dict(raw))
        day = session_date(str(row["Date"])[:10])
        if row.get("market", market) != market or str(row.get("symbol", symbol)) != symbol:
            raise ValueError("OFFICIAL_PE_JOIN_IDENTITY_MISMATCH")
        observed = index.select(market=market, symbol=symbol, trade_date=day, cutoff=cutoff)
        quote_pe = row.get("PE")
        quote_known = (not isinstance(quote_pe, bool) and isinstance(quote_pe, (int, float))
                       and math.isfinite(quote_pe))
        if (observed["status"] != "MISSING" and quote_known
                and (observed["status"] != "REPORTED"
                     or not math.isclose(float(quote_pe), observed["pe"], rel_tol=0., abs_tol=1e-9))):
            observed = dict(observed, status="CONFLICT", pe=None,
                            reason="OFFICIAL_PE_QUOTE_VALUE_CONFLICT", quote_pe=quote_pe)
        quote_close, reported_close = row.get("Close"), observed.get("reported_close")
        if (not isinstance(quote_close, bool) and isinstance(quote_close, (int, float))
                and math.isfinite(quote_close) and reported_close is not None
                and not math.isclose(float(quote_close), reported_close, rel_tol=0., abs_tol=1e-9)):
            observed = dict(observed, status="CONFLICT", pe=None,
                            reason="OFFICIAL_PE_QUOTE_CLOSE_CONFLICT", quote_close=quote_close)
        row["PE"] = observed["pe"] if observed["status"] == "REPORTED" else None
        joined.append(row)
        evidence.append(observed)
    return joined, evidence

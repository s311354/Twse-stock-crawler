"""Deterministic, reviewed PDF cell mappings against original iXBRL facts.

No fuzzy value search, inferred periods, OCR, or publication verification.
Unmapped facts (including notes/audit text) remain explicitly unverified.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import re
import unicodedata

from application.replay_contracts import digest
from domain.text_normalization import normalize_cjk_text

COMPARISON_VERSION = "filing-pdf-ixbrl-comparison/v1"
CONTEXT_KEYS = ("start_date", "end_date", "instant", "dimensions", "unit")


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", normalize_cjk_text(text))


def _number(token: str) -> Decimal:
    token = unicodedata.normalize("NFKC", token).replace(",", "").replace("−", "-")
    if re.fullmatch(r"\(\d+(?:\.\d+)?\)", token):
        token = "-" + token[1:-1]
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", token):
        raise ValueError("PDF_CELL_NOT_A_NUMBER")
    value = Decimal(token)
    if not value.is_finite():
        raise ValueError("PDF_CELL_NOT_FINITE")
    return value


def _page(pages: tuple[str, ...], number: int) -> str:
    if type(number) is not int or not 1 <= number <= len(pages):
        raise ValueError("PDF_PAGE_OUT_OF_RANGE")
    return pages[number - 1]


def _quote(pages: tuple[str, ...], evidence: dict) -> None:
    quote = evidence.get("quote")
    if not isinstance(quote, str) or not _compact(quote):
        raise ValueError("PDF_CONTEXT_QUOTE_REQUIRED")
    if _compact(quote) not in _compact(_page(pages, evidence.get("page"))):
        raise ValueError("PDF_CONTEXT_QUOTE_NOT_FOUND")


def compare_pdf_facts(facts: list[dict], pages: tuple[str, ...], mapping: dict,
                      *, statement_scope: str) -> dict:
    """Compare exact fact IDs/contexts to explicitly reviewed PDF row/columns.

    Numeric columns are zero-based after the row label, including percentages.
    Currency symbols are separators; dashes are unknown, NEVER assumed zero.
    The mapping records a human's column/period interpretation with source
    quotes. Matching values do not certify that interpretation or historical
    vintage. Every parsed fact is represented, not only successful matches.
    """
    if mapping.get("schema") != COMPARISON_VERSION:
        raise ValueError("PDF_COMPARISON_SCHEMA_MISMATCH")
    entries = mapping.get("fields", [])
    if not isinstance(entries, list):
        raise ValueError("PDF_FIELD_MAPPINGS_REQUIRED")
    unique = {digest(fact): fact for fact in facts}
    mapped = {}
    if entries:
        if not mapping.get("reviewed_by"):
            raise ValueError("PDF_COLUMN_MAPPING_REVIEW_REQUIRED")
        if mapping.get("statement_scope") != statement_scope:
            raise ValueError("PDF_STATEMENT_SCOPE_MISMATCH")
        for key in ("company", "period", "scope"):
            _quote(pages, mapping.get("identity_evidence", {}).get(key, {}))
    for entry in entries:
        key = entry["fact_id"]
        if key not in unique or key in mapped:
            raise ValueError("PDF_UNKNOWN_OR_DUPLICATE_FACT_MAPPING")
        mapped[key] = entry
    rows = []
    for key, fact in unique.items():
        row = dict(fact_id=key, fact=fact, status="NOT_MAPPED", reason="PDF_CELL_MAPPING_REQUIRED")
        rows.append(row)
        entry = mapped.get(key)
        if entry is None:
            continue
        row["mapping"] = entry
        try:
            context = entry.get("context", {})
            if any(name not in context or digest(context[name]) != digest(fact.get(name)) for name in CONTEXT_KEYS):
                raise ValueError("PDF_IXBRL_CONTEXT_MISMATCH")
            _quote(pages, entry.get("period_evidence", {}))
            if fact.get("numeric_value") is None:
                fragments = entry.get("text_evidence", [])
                if not fragments or not fact.get("text_value"):
                    row.update(status="TEXT_REVIEW_REQUIRED", reason="COMPLETE_TEXT_EVIDENCE_REQUIRED")
                    continue
                for fragment in fragments:
                    _quote(pages, fragment)
                text = "".join(fragment["quote"] for fragment in fragments)
                equal = _compact(text) == _compact(fact["text_value"])
                row.update(status="MATCH" if equal else "TEXT_REVIEW_REQUIRED",
                           reason="FULL_NORMALIZED_TEXT_EQUAL" if equal else "TEXT_DIFFERS_OR_INCOMPLETE",
                           pdf_text=text)
                continue
            _quote(pages, entry.get("unit_evidence", {}))
            page = _page(pages, entry.get("page"))
            label = entry.get("row_label", "")
            if not _compact(label):
                raise ValueError("PDF_ROW_LABEL_REQUIRED")
            # Allow CJK spacing but require an exact row prefix, not a number
            # found anywhere on the page or in an unrelated footnote.
            pattern = r"^\s*" + r"\s*".join(re.escape(c) for c in _compact(label))
            hits = [(line, re.match(pattern, unicodedata.normalize("NFKC", line))) for line in page.splitlines()]
            hits = [(line, match) for line, match in hits if match]
            if len(hits) != 1:
                raise ValueError("PDF_ROW_MISSING_OR_AMBIGUOUS")
            line, match = hits[0]
            row["pdf_row"] = line
            tail = unicodedata.normalize("NFKC", line)[match.end():].strip()
            cells = tail.replace("$", " ").split()
            index = entry.get("number_index")
            if type(index) is not int or index < 0 or index >= len(cells):
                raise ValueError("PDF_COLUMN_OUT_OF_RANGE")
            scale = Decimal(str(entry.get("scale")))
            if not scale.is_finite() or scale not in {Decimal(1), Decimal(1000), Decimal(1000000)}:
                raise ValueError("PDF_UNIT_SCALE_UNSUPPORTED")
            value = _number(cells[index]) * scale
            expected = Decimal(str(fact["numeric_value"]))
            if not expected.is_finite():
                raise ValueError("IXBRL_VALUE_NOT_FINITE")
            row.update(pdf_cell=cells[index], pdf_value=str(value), ixbrl_value=str(expected),
                       delta=str(value - expected), status="MATCH" if value == expected else "MISMATCH",
                       reason="EXACT_NORMALIZED_VALUE" if value == expected else "PDF_IXBRL_VALUE_DIFFERENCE")
        except (ValueError, InvalidOperation, TypeError, KeyError) as error:
            row.update(status="UNRESOLVED", reason=str(error))
    counts = dict(Counter(r["status"] for r in rows))
    return dict(schema=COMPARISON_VERSION, mapping_sha256=digest(mapping),
                mapping_reviewed_by=mapping.get("reviewed_by"), rows=rows, counts=counts,
                parsed_fact_count=len(facts), unique_fact_count=len(rows),
                matched_fact_count=counts.get("MATCH", 0),
                status="DIFFERENCES_FOUND" if counts.get("MISMATCH") else
                       "ALL_MAPPED_VALUES_MATCH" if rows and counts.get("MATCH") == len(rows) else "PARTIAL",
                publication_version_verified=False,
                interpretation="REVIEWED_CELL_MAPPING_NOT_AUTOMATIC_VERSION_PROOF")

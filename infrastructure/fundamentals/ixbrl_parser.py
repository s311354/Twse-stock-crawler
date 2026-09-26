"""Normalize MOPS Inline XBRL facts without taxonomy-specific dependencies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import pow

from domain.fundamental_models import FilingDocument
from domain.fundamental_models import NormalizedFact
from domain.fundamental_models import XbrlContext
from infrastructure.crawler.twse_http_client import TwseSchemaChangedError
from infrastructure.fundamentals.ixbrl_units import UnitParser


# Preserve visible text boundaries without inserting spaces between inline spans.
_TEXT_LAYOUT_TAGS = frozenset({"br", "p", "div", "li", "tr", "td", "th"})


@dataclass
class _RawFact:
    concept: str
    context_id: str
    unit: str | None
    scale: int
    sign: str
    is_numeric: bool
    is_nil: bool
    text: str = ""


class _InlineXbrlParser(UnitParser):
    def __init__(self) -> None:
        super().__init__()
        self.contexts: dict[str, XbrlContext] = {}
        self.raw_facts: list[_RawFact] = []
        self._context_id: str | None = None
        self._context_values: dict[str, str | list[tuple[str, str]] | None] = {}
        self._context_field: str | None = None
        self._context_text: list[str] = []
        self._dimension_name: str | None = None
        self._fact: _RawFact | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        attributes = {name.lower(): value or "" for name, value in attrs}
        normalized_tag = tag.lower()
        self._preserve_text_boundary(normalized_tag)
        if normalized_tag == "xbrli:context":
            self._context_id = attributes.get("id")
            self._context_values = {
                "start_date": None,
                "end_date": None,
                "instant": None,
                "dimensions": [],
            }
            return

        context_fields = {
            "xbrli:startdate": "start_date",
            "xbrli:enddate": "end_date",
            "xbrli:instant": "instant",
        }
        if self._context_id is not None and normalized_tag in context_fields:
            self._context_field = context_fields[normalized_tag]
            self._context_text = []
            return
        if self._context_id is not None and normalized_tag == "xbrldi:explicitmember":
            self._context_field = "dimension"
            self._dimension_name = attributes.get("dimension")
            self._context_text = []
            return

        if normalized_tag not in {"ix:nonfraction", "ix:nonnumeric"}:
            return
        scale_text = attributes.get("scale", "0")
        try:
            scale = int(scale_text)
        except ValueError:
            scale = 0
        self._fact = _RawFact(
            concept=attributes.get("name", ""),
            context_id=attributes.get("contextref", ""),
            unit=attributes.get("unitref") or None,
            scale=scale,
            sign=attributes.get("sign", ""),
            is_numeric=normalized_tag == "ix:nonfraction",
            is_nil=attributes.get("xsi:nil", "").lower() == "true",
        )

    def handle_data(self, data: str) -> None:
        super().handle_data(data)
        if self._fact is not None:
            self._fact.text += data
        if self._context_field is not None:
            self._context_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        super().handle_endtag(tag)
        normalized_tag = tag.lower()
        self._preserve_text_boundary(normalized_tag)
        context_end_tags = {
            "xbrli:startdate": "start_date",
            "xbrli:enddate": "end_date",
            "xbrli:instant": "instant",
            "xbrldi:explicitmember": "dimension",
        }
        if self._context_id is not None and normalized_tag in context_end_tags:
            field = context_end_tags[normalized_tag]
            value = _normalize_text("".join(self._context_text))
            if field == "dimension":
                dimensions = self._context_values["dimensions"]
                if isinstance(dimensions, list) and self._dimension_name and value:
                    dimensions.append((self._dimension_name, value))
                self._dimension_name = None
            else:
                self._context_values[field] = value or None
            self._context_field = None
            self._context_text = []
            return

        if normalized_tag == "xbrli:context" and self._context_id:
            dimensions = self._context_values.get("dimensions", [])
            self.contexts[self._context_id] = XbrlContext(
                context_id=self._context_id,
                start_date=_optional_string(self._context_values.get("start_date")),
                end_date=_optional_string(self._context_values.get("end_date")),
                instant=_optional_string(self._context_values.get("instant")),
                dimensions=tuple(dimensions) if isinstance(dimensions, list) else (),
            )
            self._context_id = None
            self._context_values = {}
            return

        if self._fact is None:
            return
        expected_tag = "ix:nonfraction" if self._fact.is_numeric else "ix:nonnumeric"
        if normalized_tag == expected_tag:
            self.raw_facts.append(self._fact)
            self._fact = None

    def _preserve_text_boundary(self, tag: str) -> None:
        if (
            tag in _TEXT_LAYOUT_TAGS
            and self._fact is not None
            and not self._fact.is_numeric
            and self._fact.text
            and not self._fact.text.endswith(("\n", "\r"))
        ):
            self._fact.text += "\n"


def decode_mops_content(content: bytes) -> str:
    """Decode MOPS bytes deterministically while tolerating legacy encodings."""

    for encoding in ("utf-8", "big5", "cp950", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def parse_ixbrl(content: bytes | str) -> list[NormalizedFact]:
    """Resolve periods/numbers while retaining decoded text for audit evidence.

    Text facts keep source whitespace and HTML layout boundaries. CJK matching
    normalizes a separate copy in the domain layer; raw HTML bytes remain in
    the filing repository for exact source reconstruction.
    """

    parser = _InlineXbrlParser()
    parser.feed(decode_mops_content(content) if isinstance(content, bytes) else content)
    facts: list[NormalizedFact] = []
    for raw in parser.raw_facts:
        if not raw.concept or not raw.context_id:
            continue
        context = parser.contexts.get(raw.context_id, XbrlContext(raw.context_id))
        text = _normalize_text(raw.text)
        numeric_value = _parse_number(text, raw.scale, raw.sign) if raw.is_numeric and not raw.is_nil else None
        facts.append(
            NormalizedFact(
                concept=raw.concept,
                context_id=raw.context_id,
                unit=parser.units.get(raw.unit),
                numeric_value=numeric_value,
                text_value=(
                    raw.text if not raw.is_numeric and text else None
                ),
                start_date=context.start_date,
                end_date=context.end_date,
                instant=context.instant,
                dimensions=context.dimensions,
            )
        )
    return facts


class IxbrlFilingParser:
    """Normalize one downloaded iXBRL document without persistence access."""

    def parse(self, document: FilingDocument) -> list[NormalizedFact]:
        facts = parse_ixbrl(document.content)
        if facts:
            return facts
        candidate = document.candidate
        raise TwseSchemaChangedError(
            "MOPS iXBRL {} {}Q{} contains no normalized facts".format(
                candidate.symbol,
                candidate.fiscal_year,
                candidate.quarter,
            )
        )


def _parse_number(text: str, scale: int, sign: str) -> float | None:
    normalized = text.replace(",", "").replace(" ", "")
    if not normalized or normalized in {"-", "--", "N/A"}:
        return None
    negative_parentheses = normalized.startswith("(") and normalized.endswith(")")
    normalized = normalized.strip("()")
    try:
        value = float(normalized) * pow(10, scale)
    except ValueError:
        return None
    if sign == "-" or negative_parentheses:
        value = -abs(value)
    return value


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = [
    "IxbrlFilingParser",
    "NormalizedFact",
    "XbrlContext",
    "decode_mops_content",
    "parse_ixbrl",
]

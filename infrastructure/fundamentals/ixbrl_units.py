"""Resolve iXBRL unit references from declarations, never from an ID's spelling."""

from __future__ import annotations

from html.parser import HTMLParser


XBRLI = "http://www.xbrl.org/2003/instance"
ISO4217 = "http://www.xbrl.org/2003/iso4217"
_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input",
                   "link", "meta", "param", "source", "track", "wbr"})


class UnitParser(HTMLParser):
    """Unit-only parser also used as the full fact parser's base class.

    Canonical strings retain the measure namespace (e.g. iso4217:TWD and
    iso4217:TWD/xbrli:shares). Conflicting or unsupported declarations resolve
    to None, rather than falling back to a potentially misleading unit ID.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.units: dict[str, str | None] = {}
        self._ns_stack: list[tuple[str, dict[str, str]]] = []
        self._unit_id: str | None = None
        self._unit_parts: dict[str, list[str | None]] = {}
        self._unit_side = "simple"
        self._divided = False
        self._measure_text: list[str] | None = None
        self._measure_namespaces: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        namespaces = dict(self._ns_stack[-1][1]) if self._ns_stack else {}
        attributes = dict(attrs)
        for key, value in attrs:
            if key.startswith("xmlns:") and value:
                namespaces[key[6:]] = value
        if tag not in _VOID:
            self._ns_stack.append((tag, namespaces))
        prefix, _, local = tag.partition(":")
        if namespaces.get(prefix) != XBRLI:
            return
        if local == "unit":
            self._unit_id = attributes.get("id")
            self._unit_parts = {"simple": [], "numerator": [], "denominator": []}
            self._unit_side, self._divided = "simple", False
        elif self._unit_id is not None:
            if local == "divide":
                self._divided = True
            elif local in {"unitnumerator", "unitdenominator"}:
                self._unit_side = local.removeprefix("unit")
            elif local == "measure":
                self._measure_text = []
                self._measure_namespaces = namespaces

    def handle_data(self, data: str) -> None:
        if self._measure_text is not None:
            self._measure_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        namespaces = self._ns_stack[-1][1] if self._ns_stack else {}
        prefix, _, local = tag.partition(":")
        if namespaces.get(prefix) == XBRLI and self._unit_id is not None:
            if local == "measure" and self._measure_text is not None:
                name = "".join(self._measure_text).strip()
                measure_prefix, _, measure = name.partition(":")
                uri = self._measure_namespaces.get(measure_prefix)
                canonical = None
                if uri == ISO4217 and len(measure) == 3 and measure.isupper() and measure.isalpha():
                    canonical = f"iso4217:{measure}"
                elif uri == XBRLI and measure in {"shares", "pure"}:
                    canonical = f"xbrli:{measure}"
                self._unit_parts[self._unit_side].append(canonical)
                self._measure_text = None
            elif local in {"unitnumerator", "unitdenominator"}:
                self._unit_side = "simple"
            elif local == "unit":
                simple, numerator, denominator = (self._unit_parts[key] for key in
                                                  ("simple", "numerator", "denominator"))
                canonical = None
                if not self._divided and len(simple) == 1 and not numerator and not denominator:
                    canonical = simple[0]
                elif (self._divided and not simple and len(numerator) == len(denominator) == 1
                      and numerator[0] is not None and denominator[0] is not None):
                    canonical = f"{numerator[0]}/{denominator[0]}"
                if self._unit_id in self.units and self.units[self._unit_id] != canonical:
                    canonical = None
                self.units[self._unit_id] = canonical
                self._unit_id = None
        for index in range(len(self._ns_stack) - 1, -1, -1):
            if self._ns_stack[index][0] == tag:
                del self._ns_stack[index:]
                break


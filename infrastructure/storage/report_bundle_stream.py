"""Bounded, incremental JSON bundle records; no whole-file/whole-array loads."""
from __future__ import annotations

import codecs
from collections.abc import Iterator
import json
from pathlib import Path
from typing import BinaryIO, Any

from infrastructure.storage.evidence_graph import MAX_REPORT_BYTES


class _JsonStream:
    def __init__(self, file: BinaryIO, chunk_bytes: int, max_record_bytes: int):
        self.file = file
        self.chunk_bytes = chunk_bytes
        self.max_record_bytes = max_record_bytes
        self.decoder = codecs.getincrementaldecoder("utf-8-sig")()
        self.buffer = ""
        self.position = 0
        self.eof = False

    def peek(self) -> str:
        while self.position == len(self.buffer) and not self.eof:
            content = self.file.read(self.chunk_bytes)
            self.eof = not content
            self.buffer = self.decoder.decode(content, final=self.eof)
            self.position = 0
        return self.buffer[self.position:self.position + 1]

    def whitespace(self) -> str:
        while (char := self.peek()) and char in " \t\r\n":
            self.position += 1
        return char

    def expect(self, expected: str) -> None:
        if self.whitespace() != expected:
            raise ValueError("INVALID_REPORT_BUNDLE_JSON")
        self.position += 1

    def value(self) -> Any:
        """Scan a single value linearly, then decode it exactly once."""
        first = self.whitespace()
        if not first:
            raise ValueError("TRUNCATED_REPORT_BUNDLE_JSON")
        container = first in "[{"
        string = first == '"'
        primitive = not container and not string
        stack: list[str] = []
        quoted = False
        escaped = False
        parts: list[str] = []
        size = 0
        started = False
        complete = False
        while not complete:
            if not self.peek():
                if not primitive or not started:
                    raise ValueError("TRUNCATED_REPORT_BUNDLE_JSON")
                break
            begin = self.position
            while self.position < len(self.buffer):
                char = self.buffer[self.position]
                if primitive and char in " \t\r\n,]}":
                    complete = True
                    break
                self.position += 1
                started = True
                if quoted:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                        if string:
                            complete = True
                elif char == '"':
                    quoted = True
                elif container and char in "[{":
                    stack.append(char)
                    if len(stack) > 128:
                        raise ValueError("REPORT_BUNDLE_TOO_DEEP")
                elif container and char in "]}":
                    if not stack or stack.pop() != {"]": "[", "}": "{"}[char]:
                        raise ValueError("INVALID_REPORT_BUNDLE_JSON")
                    complete = not stack
                if complete:
                    break
            part = self.buffer[begin:self.position]
            size += len(part.encode("utf-8"))
            if size > self.max_record_bytes:
                raise ValueError("REPORT_BUNDLE_RECORD_TOO_LARGE")
            parts.append(part)
        try:
            return json.loads("".join(parts))
        except (ValueError, RecursionError) as error:
            raise ValueError("INVALID_REPORT_BUNDLE_JSON") from error

    def skip(self, depth: int = 0) -> None:
        """Validate unselected metadata/arrays without retaining their contents."""
        if depth > 128:
            raise ValueError("REPORT_BUNDLE_TOO_DEEP")
        kind = self.whitespace()
        if kind not in ("[", "{"):
            self.value()
            return
        self.position += 1
        closing = "]" if kind == "[" else "}"
        if self.whitespace() == closing:
            self.position += 1
            return
        while True:
            if kind == "{":
                if not isinstance(self.value(), str):
                    raise ValueError("INVALID_REPORT_BUNDLE_JSON")
                self.expect(":")
            self.skip(depth + 1)
            if self.whitespace() == closing:
                self.position += 1
                return
            self.expect(",")

    def records(self, buckets: frozenset[str], *, context=None) -> Iterator[dict[str, Any]]:
        self.expect("{")
        seen = set()
        if self.whitespace() != "}":
            while True:
                key = self.value()
                if not isinstance(key, str) or len(key) > 512 or key in seen:
                    raise ValueError("INVALID_REPORT_BUNDLE_KEY")
                # A bundle has a small, bounded envelope, not an unbounded key set.
                if len(seen) >= 1024:
                    raise ValueError("REPORT_BUNDLE_TOO_MANY_KEYS")
                seen.add(key)
                self.expect(":")
                if key in buckets:
                    if context is not None:
                        context.bucket = key
                    self.expect("[")
                    if self.whitespace() != "]":
                        while True:
                            yield self.value()
                            if self.whitespace() == "]":
                                break
                            self.expect(",")
                    self.expect("]")
                elif context is not None and key in {"signal_date", "storage_schema"}:
                    value = self.value()
                    if value is not None and not isinstance(value, str):
                        raise ValueError("INVALID_REPORT_BUNDLE_METADATA")
                    context.metadata[key] = value
                else:
                    self.skip()
                if self.whitespace() == "}":
                    break
                self.expect(",")
        self.expect("}")
        if self.whitespace():
            raise ValueError("INVALID_REPORT_BUNDLE_TRAILING_DATA")


def iter_bundle_records(
    path: str | Path, *, buckets: tuple[str, ...], chunk_bytes: int = 64 * 1024,
    max_record_bytes: int = MAX_REPORT_BYTES, context=None,
) -> Iterator[dict[str, Any]]:
    for name, value, upper in (
        ("CHUNK_BYTES", chunk_bytes, 1024 * 1024),
        ("MAX_RECORD_BYTES", max_record_bytes, MAX_REPORT_BYTES),
    ):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"INVALID_REPORT_BUNDLE_{name}")
    if not buckets or any(bucket not in {"reports", "stale_reports"} for bucket in buckets):
        raise ValueError("INVALID_REPORT_BUNDLE_BUCKETS")
    with Path(path).open("rb") as file:
        yield from _JsonStream(file, chunk_bytes, max_record_bytes).records(frozenset(buckets), context=context)

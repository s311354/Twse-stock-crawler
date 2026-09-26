"""Deterministic text normalization for CJK filing evidence."""

from __future__ import annotations

from html import unescape
import re
import unicodedata


_CJK_CHARACTER_CLASS = (
    "\u3040-\u30ff"
    "\u3100-\u312f"
    "\u31a0-\u31bf"
    "\u3400-\u4dbf"
    "\u4e00-\u9fff"
    "\uac00-\ud7af"
    "\uf900-\ufaff"
    "\U00020000-\U0002fa1f"
)
_ZERO_WIDTH_CHARACTERS = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_UNICODE_WHITESPACE = re.compile(
    r"[\s\u0085\u00a0\u1680\u180e\u2000-\u200a"
    r"\u2028\u2029\u202f\u205f\u3000]+"
)
_CJK_LAYOUT_WHITESPACE = re.compile(
    rf"(?<=[{_CJK_CHARACTER_CLASS}]) +(?=[{_CJK_CHARACTER_CLASS}])"
)


def normalize_cjk_text(value: str | None) -> str:
    """Normalize presentation artifacts without destroying English spacing.

    HTML entities (including nested escapes) and full-width compatibility forms
    are decoded to a fixed point, so normalization is idempotent.
    Zero-width formatting characters are removed, Unicode whitespace is
    collapsed, and layout spaces between adjacent CJK characters are removed.
    """

    if value is None:
        return ""
    normalized = str(value)
    while True:
        decoded = unicodedata.normalize("NFKC", unescape(normalized))
        decoded = _ZERO_WIDTH_CHARACTERS.sub("", decoded)
        decoded = _UNICODE_WHITESPACE.sub(" ", decoded).strip()
        decoded = _CJK_LAYOUT_WHITESPACE.sub("", decoded)
        if decoded == normalized:
            return decoded
        normalized = decoded


__all__ = ["normalize_cjk_text"]

"""Lossless, source-range decoding for damaged legacy MOPS documents.

Latin-1 is only a reversible byte carrier, never a verified financial text
encoding. A broken unrelated note cannot invalidate a strictly decodable fact;
an undecodable/ambiguous fact cannot supply audit or numeric evidence.
"""
from hashlib import sha256
from html.parser import HTMLParser
from domain.text_normalization import UNVERIFIED_TEXT_ENCODING_PREFIX


IXBRL_PARSER_VERSION = "ixbrl-source-range-decoding/v1"
_CODECS = ("utf-8", "big5", "cp950")
_LAYOUT = frozenset({"br", "p", "div", "li", "tr", "td", "th"})


def decode_document(content):
    failures = []
    for codec in _CODECS:
        try:
            return content.decode(codec), codec, failures
        except UnicodeDecodeError as error:
            failures.append(dict(encoding=codec, byte_start=error.start, byte_end=error.end))
    return content.decode("latin-1"), None, failures


class _TextFragment(HTMLParser):
    def __init__(self, *, numeric):
        super().__init__(convert_charrefs=True)
        self.text, self.numeric = "", numeric

    def handle_data(self, data):
        self.text += data

    def _boundary(self, tag):
        if not self.numeric and tag in _LAYOUT and self.text and not self.text.endswith(("\n", "\r")):
            self.text += "\n"

    def handle_starttag(self, tag, attrs):
        self._boundary(tag)

    def handle_endtag(self, tag):
        self._boundary(tag)


def decode_fact_range(content, start, end, *, concept, context_id, numeric, complete=True):
    """Decode one complete HTML fact payload, not favorable substrings.

    Every successful codec must round-trip to the exact original bytes. If
    successful codecs disagree, no charset declaration or language heuristic
    chooses a favorable interpretation. HTML entities/layout are interpreted
    only after this verification, identically to the ordinary fact parser.
    """
    raw = content[start:end]
    raw_hash = sha256(raw).hexdigest()
    receipt = dict(concept=concept, context_id=context_id, byte_start=start,
                   byte_end=end, raw_sha256=raw_hash, encoding=None,
                   status="PENDING_VERIFICATION", reason="UNDECODABLE_FACT_BYTES")
    if not complete:
        receipt["reason"] = "INCOMPLETE_FACT_STRUCTURE"
        return UNVERIFIED_TEXT_ENCODING_PREFIX + raw_hash + "]", receipt
    candidates = []
    for codec in _CODECS:
        try:
            text = raw.decode(codec)
            if text.encode(codec) == raw:
                candidates.append((codec, text))
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    if candidates and len({text for _, text in candidates}) == 1:
        codec, text = candidates[0]
        fragment = _TextFragment(numeric=numeric)
        fragment.feed(text)
        fragment.close()
        value = fragment.text
        receipt.update(status="VERIFIED_DECODED", reason="STRICT_REVERSIBLE_FACT_RANGE",
                       encoding="ascii" if raw.isascii() else codec,
                       equivalent_encodings=[codec for codec, _ in candidates],
                       text_sha256=sha256(value.encode("utf-8")).hexdigest())
        return value, receipt
    if candidates:
        receipt.update(reason="AMBIGUOUS_FACT_ENCODING",
                       candidate_encodings=[codec for codec, _ in candidates])
    # Nonempty diagnostic text prevents a damaged audit body from being treated
    # as absent while its unqualified abstract alone certifies financial safety.
    return UNVERIFIED_TEXT_ENCODING_PREFIX + raw_hash + "]", receipt


__all__ = ["IXBRL_PARSER_VERSION", "decode_document", "decode_fact_range"]

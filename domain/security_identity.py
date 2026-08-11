"""Security identity mapping between TWSE instruments and MOPS issuers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


MARKET_SYMBOL_FIELD = "market_symbol"
ISSUER_SYMBOL_FIELD = "issuer_symbol"
SECURITY_TYPE_FIELD = "security_type"
SECURITY_IDENTITY_SOURCE_FIELD = "security_identity_source"


class SecurityType(str, Enum):
    """Instrument categories relevant to fundamental model eligibility."""

    COMMON = "COMMON"
    PREFERRED = "PREFERRED"
    OTHER = "OTHER"


@dataclass(frozen=True)
class SecurityIdentity:
    """Keep the traded security separate from its financial issuer."""

    market_symbol: str
    issuer_symbol: str
    stock_name: str
    security_type: SecurityType
    mapping_source: str

    @property
    def low_entry_model_supported(self) -> bool:
        return self.security_type is SecurityType.COMMON

    def snapshot_fields(self) -> dict[str, str]:
        return {
            MARKET_SYMBOL_FIELD: self.market_symbol,
            ISSUER_SYMBOL_FIELD: self.issuer_symbol,
            SECURITY_TYPE_FIELD: self.security_type.value,
            SECURITY_IDENTITY_SOURCE_FIELD: self.mapping_source,
        }


_COMMON_STOCK_PATTERN = re.compile(r"^\d{4}$")
_SUFFIXED_SECURITY_PATTERN = re.compile(r"^(?P<issuer>\d{4})[A-Z]$")


def resolve_security_identity(
    market_symbol: str,
    stock_name: str = "",
) -> SecurityIdentity:
    """Resolve an instrument without blindly stripping alphabetic suffixes.

    TWSE preferred-share names contain ``特`` and use a four-digit issuer code
    followed by a letter. Both signals are required before MOPS is queried with
    the issuer code; unknown suffixed instruments remain unmapped and fail safe.
    """

    symbol = str(market_symbol).strip().upper()
    name = str(stock_name).strip()
    if not symbol:
        raise ValueError("market_symbol cannot be empty")

    if _COMMON_STOCK_PATTERN.fullmatch(symbol):
        return SecurityIdentity(
            market_symbol=symbol,
            issuer_symbol=symbol,
            stock_name=name,
            security_type=SecurityType.COMMON,
            mapping_source="TWSE_MARKET_SYMBOL",
        )

    suffixed = _SUFFIXED_SECURITY_PATTERN.fullmatch(symbol)
    if suffixed and "特" in name:
        return SecurityIdentity(
            market_symbol=symbol,
            issuer_symbol=suffixed.group("issuer"),
            stock_name=name,
            security_type=SecurityType.PREFERRED,
            mapping_source="TWSE_SYMBOL_AND_NAME",
        )

    return SecurityIdentity(
        market_symbol=symbol,
        issuer_symbol=symbol,
        stock_name=name,
        security_type=SecurityType.OTHER,
        mapping_source="UNMAPPED_MARKET_SYMBOL",
    )


__all__ = [
    "ISSUER_SYMBOL_FIELD",
    "MARKET_SYMBOL_FIELD",
    "SECURITY_IDENTITY_SOURCE_FIELD",
    "SECURITY_TYPE_FIELD",
    "SecurityIdentity",
    "SecurityType",
    "resolve_security_identity",
]

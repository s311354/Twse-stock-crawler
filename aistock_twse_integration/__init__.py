"""Versioned AIStock integration boundary for Twse-stock-crawler."""

from .contract import CONTRACT_NAME
from .contract import CONTRACT_VERSION
from .contract import PACKAGE_VERSION
from .contract import package_contract

__all__ = (
    "CONTRACT_NAME",
    "CONTRACT_VERSION",
    "PACKAGE_VERSION",
    "package_contract",
)

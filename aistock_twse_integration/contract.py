"""Stable package and JSON contract metadata shared with AIStock consumers."""

from __future__ import annotations

from typing import Final


PACKAGE_VERSION: Final = "1.0.0"
CONTRACT_NAME: Final = "aistock.twse.owner-research"
CONTRACT_VERSION: Final = 1


def package_contract() -> dict[str, str | int]:
    return {
        "name": CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "packageVersion": PACKAGE_VERSION,
    }

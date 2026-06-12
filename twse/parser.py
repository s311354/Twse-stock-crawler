"""Parsing helpers for TWSE row values."""


def parse_price(price: object) -> float | None:
    normalized_price = str(price).replace(",", "").strip()
    if normalized_price in ("", "--"):
        return None

    try:
        return float(normalized_price)
    except ValueError:
        return None


def clean_cell(value: str) -> str:
    return str(value).replace(",", "").strip()

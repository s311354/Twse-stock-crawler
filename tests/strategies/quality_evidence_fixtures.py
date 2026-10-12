"""Explicitly SYNTHETIC originals for unit tests, never historical evidence."""
from hashlib import sha256

import pandas as pd

from domain.strategies.low_entry_v6_quality import BALANCE_FIELDS, CAPITAL_BALANCE_FIELDS, FLOW_FIELDS


def bind_synthetic_contexts(row):
    period = pd.Period(row["period_end"], freq="Q")
    row.setdefault("content_sha256", sha256(f"SYNTHETIC:{period}".encode()).hexdigest())
    row["field_evidence"] = {}
    for field in (*FLOW_FIELDS, *BALANCE_FIELDS, *CAPITAL_BALANCE_FIELDS):
        instant = field in (*BALANCE_FIELDS, *CAPITAL_BALANCE_FIELDS)
        eps = field == "basic_eps"
        start = (period.start_time.date().isoformat() if row["flow_basis"] == "QUARTER"
                 else f"{period.year}-01-01")
        proof = dict(status="RESOLVED", reason="REPORTED_EXACT_CONTEXT", basis="INSTANT" if instant else row["flow_basis"],
            start_date=None if instant else start, end_date=row["period_end"],
            **{key: row.get(key) for key in ("symbol", "scope", "currency", "revision_id", "source", "source_url",
                "content_sha256", "available_at")}, unit="TWD/share" if eps else "thousand")
        value = row.get(field)
        proof["contexts"] = [dict(concept="synthetic:" + field, context_id=f"SYNTHETIC:{period}:{field}",
            start_date=None if instant else start, end_date=None if instant else row["period_end"],
            instant=row["period_end"] if instant else None, dimensions=[],
            numeric_value=value * (1 if eps else 1000) if value is not None else None,
            unit="iso4217:TWD/xbrli:shares" if eps else "iso4217:TWD")]
        row["field_evidence"][field] = proof
    return row

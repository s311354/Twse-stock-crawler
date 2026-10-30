"""Original-source binding at I/O boundaries, outside the pure model engine.

The domain validator reproduces arithmetic and hashes. This additional boundary
checks that an R12 monthly-price projection really came from the named retained
capture. One bounded reader can be shared across an entire publication/render.
"""
from contextlib import contextmanager

from domain.les_v7.contracts import canonical, timestamp
from infrastructure.crawler.valuation_history import ValuationSelectionVerifier


@contextmanager
def valuation_history_verifier_scope():
    verifier = ValuationSelectionVerifier()
    try:
        yield verifier.verify
    finally:
        verifier.close()


def verify_report_valuation_sources(report, *, verifier=None):
    """Verify only the optional R12 source graph; old reports need no files."""
    native = report.get("diagnostics", {}).get("native_evidence") or {}
    context = (native.get("source_packet") or {}).get("r10_valuation_context")
    if not context:
        return
    if verifier is None:
        with valuation_history_verifier_scope() as verify:
            return verify_report_valuation_sources(report, verifier=verify)
    for block in (context.get("issuer"), *context.get("peers", ())):
        if not block:
            continue
        source = block.get("market_evidence", {})
        selection = source.get("valuation_history_selection")
        if selection is None:
            continue
        checked = verifier(selection)
        if checked["symbol"] != block.get("symbol"):
            raise ValueError("R12_MONTHLY_SOURCE_SYMBOL_BINDING_INVALID")
        cutoff = report.get("envelope", {}).get("event", {}).get("cutoff_at") or context.get("cutoff")
        if (cutoff is None or timestamp(checked["observed_at"]) > timestamp(cutoff)
                or context.get("cutoff") is not None and timestamp(context["cutoff"]) != timestamp(cutoff)):
            raise ValueError("R12_MONTHLY_SOURCE_CUTOFF_BINDING_INVALID")
        available = source.get("source_available_at")
        if (available is None and source.get("source_final") is not False
                or available is not None and timestamp(available) < timestamp(checked["observed_at"])):
            raise ValueError("R12_MONTHLY_SOURCE_OBSERVATION_CANNOT_BE_BACKDATED")
        # UNKNOWN/conflicting histories were deliberately not substituted.
        # A known invalid source cannot be relabelled final by a caller.
        if (checked["status"] == "UNKNOWN" or source.get("valuation_history_conflicts")
                or source.get("valuation_history_verification_error")):
            if source.get("source_final") is not False:
                raise ValueError("R12_UNKNOWN_MONTHLY_SOURCE_CANNOT_BE_FINAL")
            continue
        rows = block.get("market_rows", ())
        if not rows:
            raise ValueError("R12_MONTHLY_SOURCE_PRICE_BINDING_INVALID")
        session = rows[-1]["Date"]
        expected = [row for row in checked["rows"] if row["Date"] < session]
        if canonical(rows[:-1]) != canonical(expected):
            raise ValueError("R12_MONTHLY_SOURCE_PRICE_BINDING_INVALID")

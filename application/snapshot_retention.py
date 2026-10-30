"""Lightweight metadata/materialization contract for work-owned snapshots.

This is temporary native-work retention, not an evidence dataset or a new
financial selection policy. Legacy mappings keep their existing behavior.
"""


def snapshot_metadata(snapshots, symbol):
    reader = getattr(snapshots, "metadata", None)
    return reader(symbol) if callable(reader) else snapshots.get(symbol, {})


def snapshot_materialize(snapshots, symbol, *, cutoff=None, market=None):
    reader = getattr(snapshots, "materialize", None)
    return (reader(symbol, cutoff=cutoff, market=market) if callable(reader)
            else snapshots.get(symbol, {}))


def materialize_history_snapshot(market, symbol, *, cutoff):
    """Validate a collected packet at its retained clock, then evaluate later.

    Capture binding and final model cutoff are separate, explicit identities.
    Only native current daily v7 may declare this relationship; existing
    exact-cutoff retention checks and every domain source-time check remain.
    No original availability is modified or newly queried at the later clock.
    """
    capture = market.fundamental_capture_cutoffs.get(symbol)
    if capture is not None:
        from domain.les_v7.contracts import timestamp
        if (market.entry_model != "v7" or not market.les_v7_native_evidence
                or market.replay_as_of or timestamp(capture) > timestamp(cutoff)):
            raise ValueError("FUNDAMENTAL_CAPTURE_CUTOFF_BINDING_MISMATCH")
    return snapshot_materialize(market.fundamental_snapshots, symbol,
        cutoff=capture if capture is not None else cutoff, market=market.market_source)

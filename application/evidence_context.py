"""Work-scoped, read-only financial evidence, outside pandas metadata.

The caller owns collected inputs for the duration of the work. Only the current
stock is frozen; no second universe-sized financial cache is retained here.
No I/O, availability inference or model gate is performed. Native validators
and normalizers may share immutable intermediates within the current packet.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from uuid import uuid4

import pandas as pd

from application.daily_performance import count_work, measure_stage
from application.replay_contracts import digest
from domain.industry_benchmark import INDUSTRY_BENCHMARK_ATTR, sector_code
from domain.numeric_history import metadata_free_frame
from domain.strategies.low_entry_score_v4 import (
    ANNUAL_FUNDAMENTALS_ATTR, FUNDAMENTAL_SNAPSHOT_ATTR, SIGNAL_DATE_ATTR,
)


EVIDENCE_CONTEXT_ID_ATTR = "evidence_context_id"


class ReadOnlyEvidence(dict):
    """Detached immutable JSON mapping, including recursively frozen children.

    Native JSON encoding can traverse this dict without allocating a second
    dict/list graph. Ordinary mutation APIs (including reinitialization) are
    disabled. Like other Python frozen containers, deliberate calls to base
    class mutation methods are outside the read-only consumer contract.
    """

    __slots__ = ("_initialized", "__weakref__")

    def __init__(self, values, memo=None):
        if getattr(self, "_initialized", False):
            raise TypeError("READ_ONLY_EVIDENCE")
        memo = {} if memo is None else memo
        dict.__init__(self, ((key, _freeze(value, memo)) for key, value in values.items()))
        object.__setattr__(self, "_initialized", True)

    def _readonly(self, *args, **kwargs):
        raise TypeError("READ_ONLY_EVIDENCE")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly
    __setattr__ = __delattr__ = _readonly

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _freeze(value, memo):
    # Most nodes are JSON scalars. Avoid ABC Mapping tests and import work for
    # every leaf in the provenance graph; preserve subclasses via the fallback.
    kind = type(value)
    if kind in (str, int, float, bool, type(None)):
        return value
    if isinstance(value, ReadOnlyEvidence):
        return value
    if kind in (dict, list, tuple) or isinstance(value, (Mapping, list, tuple)):
        identity = id(value)
        if identity not in memo:
            memo[identity] = (ReadOnlyEvidence(value, memo) if kind is dict or isinstance(value, Mapping)
                              else tuple(_freeze(item, memo) for item in value))
        return memo[identity]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Financial packets contain JSON scalars, plus immutable date/numpy scalars.
    import datetime
    if isinstance(value, (datetime.date, datetime.time)):
        return value
    if hasattr(value, "item"):
        return _freeze(value.item(), memo)
    raise TypeError("UNSUPPORTED_MUTABLE_EVIDENCE_VALUE:" + type(value).__name__)


def _mutable_record(value):
    if isinstance(value, Mapping):
        return {key: _mutable_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_record(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class StockEvidence:
    symbol: str
    cutoff: str
    market: str
    evidence_id: str
    snapshot: ReadOnlyEvidence
    industry_benchmark: object
    research_context: ReadOnlyEvidence
    _annual: pd.DataFrame = field(repr=False, compare=False)
    _legacy_metadata: Callable[[], dict] | None = field(default=None, repr=False, compare=False)
    _filings_sha256: str | None = field(default=None, init=False, repr=False, compare=False)
    _quality_boundary: tuple | None = field(default=None, init=False, repr=False, compare=False)
    _quality_values: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _diagnostics_sha256: str | None = field(default=None, init=False, repr=False, compare=False)

    def quality_preprocessing(self, kind, compute, *, records, symbol, market, cutoff, policy):
        """Reuse three immutable intermediates, bounded to one stock/cutoff.

        Records must be the captured immutable source, not a caller's edited
        view. A new cutoff/policy/work version clears all intermediates and
        reselects evidence; neither QualityResult nor account decisions live here.
        """
        if (records is not self.snapshot.get("les_v6_filings", ())
                or symbol != self.symbol or market != self.market):
            raise ValueError("QUALITY_PREPROCESSING_BINDING_MISMATCH")
        if kind not in {"selection", "quarters", "reported"}:
            raise ValueError("UNKNOWN_QUALITY_PREPROCESSING_STAGE")
        if pd.isna(cutoff) or cutoff > pd.to_datetime(self.cutoff, utc=True, errors="raise"):
            raise ValueError("QUALITY_PREPROCESSING_CUTOFF_MISMATCH")
        from application.daily_work import current_work
        work = current_work()
        boundary = (cutoff.isoformat(), policy, work.version if work is not None else None)
        if boundary != self._quality_boundary:
            self._quality_values.clear()
            object.__setattr__(self, "_quality_boundary", boundary)
            object.__setattr__(self, "_diagnostics_sha256", None)
        if kind not in self._quality_values:
            with measure_stage("quality_prepare_" + kind):
                value = _freeze(compute(), {})
                self._quality_values[kind] = value
                count_work("quality_prepare_" + kind + "_builds")
        else:
            count_work("quality_prepare_" + kind + "_hits")
        return self._quality_values[kind]

    def diagnostics_content_key(self, *, symbol, market, as_of) -> str:
        """Hash selected versions/proofs, only after cutoff-aware validation.

        Future/unselected historical versions do not invalidate a successful
        diagnostic, but they are still rechecked by selection in every packet.
        On selection failure the caller uses the complete-record fallback key.
        """
        from domain.strategies.low_entry_v6_quality import select_quality_inputs
        cutoff = pd.to_datetime(as_of, utc=True, errors="raise")
        if pd.isna(cutoff):
            raise ValueError("INVALID_DECISION_TIME")
        periods, ledger, excluded = select_quality_inputs(
            self.snapshot.get("les_v6_filings", ()), symbol=symbol, market=market,
            cutoff=cutoff, preprocessing=self.quality_preprocessing)
        if self._diagnostics_sha256 is None:
            with measure_stage("quality_diagnostics_input_hash"):
                # Shallow projections only: source proof children stay frozen.
                # The internal Timestamp is deterministically represented, not
                # serialized with a permissive default=str fallback.
                selected = [dict(period=str(period), row={
                    key: value.isoformat() if key == "_available" else value
                    for key, value in row.items()}) for period, row in periods.items()]
                value = digest(dict(selected=selected, ledger=ledger, excluded=excluded))
                object.__setattr__(self, "_diagnostics_sha256", value)
                count_work("quality_diagnostics_input_hashes")
                count_work("quality_diagnostics_selected_filings", len(periods))
        return self._diagnostics_sha256

    def filings_content_key(self) -> str:
        """Hash actual captured values/proofs once, not their claimed file SHA.

        This preprocessing identity is internal, not a dataset or report ID.
        Cutoff/market/symbol/policy/work versions remain separate memo inputs.
        The source is detached and read-only, so account-only retries can reuse
        this identity without another expanded Python graph.
        """
        if self._filings_sha256 is None:
            with measure_stage("quality_diagnostics_input_hash"):
                value = digest(self.snapshot.get("les_v6_filings", ()))
                object.__setattr__(self, "_filings_sha256", value)
                count_work("quality_diagnostics_input_hashes")
        return self._filings_sha256

    def annual(self) -> pd.DataFrame:
        """Small annual table is copy-on-read; no mutable evidence escapes."""
        return metadata_free_frame(self._annual)

    def validate(self, history: pd.DataFrame) -> None:
        identity = history.attrs.get("les_v6_daily_identity", {})
        if (str(identity.get("symbol", "")) != self.symbol
                or str(history.attrs.get(SIGNAL_DATE_ATTR)) != self.cutoff
                or identity.get("market", "TWSE") != self.market
                or history.attrs.get(EVIDENCE_CONTEXT_ID_ATTR) != self.evidence_id):
            raise ValueError("EVIDENCE_CONTEXT_BINDING_MISMATCH")

    def fundamental_attrs(self) -> dict:
        return {FUNDAMENTAL_SNAPSHOT_ATTR: self.snapshot,
                ANNUAL_FUNDAMENTALS_ATTR: self.annual(),
                INDUSTRY_BENCHMARK_ATTR: self.industry_benchmark,
                "les_v6_research_context": self.research_context}

    def legacy_attrs(self) -> dict:
        """Opt-in old injected consumers get isolated native dict/list values."""
        attrs = self._legacy_metadata() if self._legacy_metadata is not None else {}
        attrs.update({key: value if isinstance(value, pd.DataFrame) else _mutable_record(value)
                      for key, value in self.fundamental_attrs().items()})
        return attrs


class EvidenceContext:
    """One work, at most one active per-stock frozen financial packet."""

    def __init__(self, market):
        self._market = market
        self._active = False
        self._work_id = uuid4().hex
        self._sequence = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._market = None

    @contextmanager
    def stock(self, item: int, history: pd.DataFrame):
        if self._market is None or self._active:
            raise ValueError("EVIDENCE_CONTEXT_CLOSED_OR_ACTIVE")
        market = self._market
        symbol = str(market.stock_numbers[item])
        self._active = True
        try:
            with measure_stage("evidence_context"):
                memo = {}
                snapshot = _freeze(market.fundamental_snapshots.get(symbol, {}), memo)
                benchmark = _freeze(market.industry_benchmarks.get(
                    sector_code(snapshot.get("sector") or market.default_sector)), memo)
                research = _freeze(market.les_v6_research_context.get(symbol, {}), memo)
                annual = metadata_free_frame(market.annual_fundamentals.get(symbol, pd.DataFrame()))
                count_work("evidence_context_container_nodes", len(memo))
                cutoff = str(history.attrs.get(SIGNAL_DATE_ATTR))
                # A local binding token, NOT a dataset/content identity. Source
                # hashes and availability checks stay in the native evidence
                # validators; avoid serializing/hashing that entire tree again.
                self._sequence += 1
                binding = f"{self._work_id}:{self._sequence}:{symbol}"
                def legacy_metadata():
                    # Only old injected consumers need breadth-history attrs.
                    # Assemble on demand, but exclude financial references
                    # before copying; finance comes from the frozen packet.
                    from copy import deepcopy
                    from application.market_history import MarketHistoryAssembler
                    attrs = MarketHistoryAssembler().build(market, item).attrs
                    financial = {FUNDAMENTAL_SNAPSHOT_ATTR, ANNUAL_FUNDAMENTALS_ATTR,
                                 INDUSTRY_BENCHMARK_ATTR, "les_v6_research_context"}
                    return deepcopy({key: value for key, value in attrs.items() if key not in financial})
                packet = StockEvidence(symbol=symbol, cutoff=cutoff, market=market.market_source,
                                       evidence_id=binding, snapshot=snapshot, industry_benchmark=benchmark,
                                       research_context=research, _annual=annual, _legacy_metadata=legacy_metadata)
                history.attrs[EVIDENCE_CONTEXT_ID_ATTR] = binding
                packet.validate(history)
                count_work("evidence_context_stocks")
            yield packet
        finally:
            # No registry of earlier StockEvidence/annual tables is retained.
            self._active = False

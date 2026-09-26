"""Bound remote refresh independently of financial-data eligibility."""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

from application.fundamental_ports import FundamentalRepositoryPort, FundamentalUnitOfWork
from domain.errors import AccessBlockedError, DataSourceError, NoDataError


class RefreshDeferred(NoDataError):
    """An operational deferral, never evidence that financial data is valid."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason, retryable=False)
        self.reason = reason


@dataclass(frozen=True)
class MopsRefreshPolicy:
    max_symbols: int = 20
    max_requests: int = 120
    max_filings_per_symbol: int = 4
    cooldown_seconds: float = 86400
    blocked_threshold: int = 2

    def __post_init__(self) -> None:
        if min(self.max_symbols, self.max_requests, self.max_filings_per_symbol) < 0:
            raise ValueError("MOPS refresh limits cannot be negative")
        if not isfinite(self.cooldown_seconds) or self.cooldown_seconds < 0 or self.blocked_threshold < 1:
            raise ValueError("Invalid MOPS refresh cooldown or blocked threshold")


class MopsRefreshControl:
    HOST_KEY = "__MOPS_HOST__"
    FAIR_TURN_KEY = "__MOPS_PRIORITY_TURN__"

    def __init__(
        self, repository: FundamentalRepositoryPort, policy: MopsRefreshPolicy,
        now_provider: Callable[[], dt.datetime],
    ) -> None:
        self.repository = repository
        self.policy = policy
        self.now_provider = now_provider
        self.begin_run()

    def begin_run(self) -> None:
        self.symbols: set[str] = set()
        self.requests = 0
        self.blocked_streak = 0
        self.circuit_open = False
        self.deferred: Counter[str] = Counter()
        self.outcomes: Counter[str] = Counter()
        self._host_checked = False
        self.scheduled_symbols: set[str] | None = None
        self.scheduled_order: list[str] = []
        self.priority_symbols: set[str] = set()
        self._fair_lanes = False

    def schedule(self, symbols: list[str], *, priority_symbols=()) -> None:
        """Alternate research/generic lanes; oldest first within each lane.

        Persist the next lane on actual attempts, not scheduling, so even a
        one-symbol/request batch cannot starve ordinary issuers across restarts.
        """
        ranked = []
        now = self.now()
        with self.repository.unit_of_work() as uow:
            for index, symbol in enumerate(dict.fromkeys(symbols)):
                state = self.repository.load_refresh_attempt(symbol, uow=uow)
                if state and now - state[0] < self.policy.cooldown_seconds:
                    continue
                ranked.append((state[0] if state else float("-inf"), index, symbol))
        self.priority_symbols = set(priority_symbols) & {row[2] for row in ranked}
        priority = [row[2] for row in sorted(ranked) if row[2] in self.priority_symbols]
        general = [row[2] for row in sorted(ranked) if row[2] not in self.priority_symbols]
        self._fair_lanes = bool(priority and general)
        state = self.repository.load_refresh_attempt(self.FAIR_TURN_KEY) if self._fair_lanes else None
        lanes = [general, priority] if state and state[1] == "GENERAL" else [priority, general]
        ordered = []
        for i in range(max(len(priority), len(general))):
            for lane in lanes:
                if i < len(lane):
                    ordered.append(lane[i])
        self.scheduled_order = ordered[:self.policy.max_symbols]
        self.scheduled_symbols = set(self.scheduled_order)

    def now(self) -> float:
        value = self.now_provider()
        if value.tzinfo is None:
            raise ValueError("MOPS refresh clock must be timezone-aware")
        return value.timestamp()

    def check_host(self, *, uow: FundamentalUnitOfWork | None = None) -> None:
        if not self._host_checked:
            state = self.repository.load_refresh_attempt(self.HOST_KEY, uow=uow)
            self.circuit_open = bool(
                state and self.now() - state[0] < self.policy.cooldown_seconds
            )
            self._host_checked = True
        if self.circuit_open:
            raise RefreshDeferred("ACCESS_BLOCKED_CIRCUIT_OPEN")

    def start_symbol(self, symbol: str, *, uow: FundamentalUnitOfWork) -> None:
        self.check_host(uow=uow)
        state = self.repository.load_refresh_attempt(symbol, uow=uow)
        if symbol in self.symbols or (
            state and self.now() - state[0] < self.policy.cooldown_seconds
        ):
            raise RefreshDeferred("REFRESH_COOLDOWN")
        if len(self.symbols) >= self.policy.max_symbols:
            raise RefreshDeferred("SYMBOL_BUDGET_EXHAUSTED")
        if self.scheduled_symbols is not None and symbol not in self.scheduled_symbols:
            raise RefreshDeferred("SYMBOL_BATCH_DEFERRED")
        if self.requests >= self.policy.max_requests:
            raise RefreshDeferred("REQUEST_BUDGET_EXHAUSTED")
        self.symbols.add(symbol)
        self.repository.save_refresh_attempt(symbol, self.now(), "STARTED", uow=uow)
        if self._fair_lanes:
            self.repository.save_refresh_attempt(
                self.FAIR_TURN_KEY, self.now(),
                "GENERAL" if symbol in self.priority_symbols else "PRIORITY", uow=uow,
            )

    def before_request(self) -> None:
        """Called before *each* HTTP attempt, including retries and SSL fallback."""
        self.check_host()
        if self.requests >= self.policy.max_requests:
            raise RefreshDeferred("REQUEST_BUDGET_EXHAUSTED")
        self.requests += 1

    def response(self, *, blocked: bool) -> None:
        self.blocked_streak = self.blocked_streak + 1 if blocked else 0
        if self.blocked_streak >= self.policy.blocked_threshold:
            self.circuit_open = True

    def finish(
        self, key: str, status: str, *, uow: FundamentalUnitOfWork | None = None,
    ) -> None:
        self.outcomes[status] += 1
        self.repository.save_refresh_attempt(key, self.now(), status, uow=uow)
        if self.circuit_open:
            self.repository.save_refresh_attempt(
                self.HOST_KEY, self.now(), "ACCESS_BLOCKED", uow=uow,
            )

    def failed(
        self, key: str, error: DataSourceError, *,
        uow: FundamentalUnitOfWork | None = None, transport_counted: bool = False,
    ) -> None:
        if isinstance(error, RefreshDeferred):
            self.deferred[error.reason] += 1
            # Do not postpone unattempted issuers: the next batch must progress.
            if key not in self.symbols:
                return
        elif not transport_counted:
            self.response(blocked=isinstance(error, AccessBlockedError))
        self.finish(key, getattr(error, "reason", error.category.value), uow=uow)

    def summary(self) -> dict:
        return {
            "remote_symbols": len(self.symbols), "http_attempts": self.requests,
            "circuit_open": self.circuit_open, "deferred": dict(self.deferred),
            "refresh_outcomes": dict(self.outcomes),
        }

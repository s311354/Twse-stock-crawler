"""Versioned research-only contracts shared by CLI and simulation clients."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from application.historical_replay import SESSION_CLOSE, TAIPEI
from domain.models import PortfolioPosition

REPLAY_SCHEMA = "strict-replay/v1"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def aware_timestamp(value: str | dt.datetime) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(parsed, dt.datetime) or parsed.utcoffset() is None:
        raise ValueError("REPLAY_REQUIRES_TIMEZONE")
    return parsed.astimezone(TAIPEI)


def calendar_window(end: dt.date, days: int = 700) -> tuple[dt.date, dt.date]:
    if type(days) is not int or days < 1:
        raise ValueError("INVALID_CALENDAR_DAYS")
    return end - dt.timedelta(days=days - 1), end


@dataclass(frozen=True)
class ReplayHolding:
    symbol: str
    shares: int
    entry_at: str
    entry_les: float | None = None
    entry_decision: str | None = None

    def validate(self, cutoff: dt.datetime) -> None:
        if not isinstance(self.symbol, str) or not re.fullmatch(r"[0-9A-Z]{4,10}", self.symbol):
            raise ValueError("INVALID_HOLDING_SYMBOL")
        if not isinstance(self.entry_at, str) or not self.entry_at:
            raise ValueError("HOLDING_ENTRY_DATE_REQUIRED")
        if isinstance(self.shares, bool) or not isinstance(self.shares, (int, float)) or not math.isfinite(self.shares) or self.shares <= 0 or self.shares != int(self.shares):
            raise ValueError("HOLDING_SHARES_REQUIRE_POSITIVE_INTEGER")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.entry_at):
            entered = dt.date.fromisoformat(self.entry_at)
            if entered > cutoff.date():
                raise ValueError("FUTURE_HOLDING_ENTRY")
            if entered == cutoff.date() and cutoff.time().replace(tzinfo=None) < SESSION_CLOSE:
                raise ValueError("SAME_DAY_HOLDING_REQUIRES_EXECUTION_TIME")
        elif aware_timestamp(self.entry_at) > cutoff:
            raise ValueError("FUTURE_HOLDING_ENTRY")
        self.position()  # Reuse native LES memory validation without losing timestamp above.

    def position(self) -> PortfolioPosition:
        day = (dt.date.fromisoformat(self.entry_at) if len(self.entry_at) == 10 else aware_timestamp(self.entry_at).date())
        return PortfolioPosition(self.symbol, shares=self.shares, entry_date=day,
                                 entry_les=self.entry_les, entry_decision=self.entry_decision)


@dataclass(frozen=True)
class ReplayAccount:
    cash: float
    account_equity: float
    holdings: tuple[ReplayHolding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "holdings", tuple(self.holdings))

    def validate(self, cutoff: dt.datetime) -> None:
        for value in (self.cash, self.account_equity):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("ACCOUNT_VALUES_MUST_BE_FINITE")
        if self.cash < 0 or self.account_equity <= 0 or self.cash > self.account_equity:
            raise ValueError("INVALID_ACCOUNT_BALANCE")
        if len(self.holdings) > 4 or len({h.symbol for h in self.holdings}) != len(self.holdings):
            raise ValueError("INVALID_OR_DUPLICATE_HOLDINGS")
        for holding in self.holdings:
            holding.validate(cutoff)
        if not self.holdings and abs(self.cash - self.account_equity) > 0.01:
            raise ValueError("EMPTY_ACCOUNT_EQUITY_MUST_EQUAL_CASH")


@dataclass(frozen=True)
class ReplayVersions:
    model: str
    parameters: str
    code: str


@dataclass(frozen=True)
class ReplayRequest:
    replay_id: str
    as_of: dt.datetime
    history_start: dt.date
    history_end: dt.date
    symbols: tuple[str, ...]
    market: str
    sector: str
    account: ReplayAccount
    versions: ReplayVersions
    evidence_id: str
    state_path: Path

    def __post_init__(self) -> None:
        cutoff = aware_timestamp(self.as_of)
        object.__setattr__(self, "as_of", cutoff)
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "state_path", Path(self.state_path).expanduser().resolve())
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", self.replay_id):
            raise ValueError("INVALID_REPLAY_ID")
        self.account.validate(cutoff)
        # This version intentionally has no premarket/intraday daily-bar mode.
        if cutoff.time().replace(tzinfo=None) != SESSION_CLOSE:
            raise ValueError("REPLAY_CLOSE_ONLY: use 13:30:00+08:00")
        if self.history_start > self.history_end or self.history_end != cutoff.date():
            raise ValueError("REPLAY_RANGE_REQUIRES_START_LE_END_EQUALS_CUTOFF_DATE")
        if self.market not in {"TWSE", "TPEX"}:
            raise ValueError("UNSUPPORTED_MARKET")
        if not self.symbols or len(set(self.symbols)) != len(self.symbols) or any(not re.fullmatch(r"[0-9A-Z]{4,10}", s) for s in self.symbols):
            raise ValueError("REPLAY_REQUIRES_EXPLICIT_UNIQUE_SYMBOLS")
        if not {h.symbol for h in self.account.holdings}.issubset(self.symbols):
            raise ValueError("HOLDINGS_MUST_BE_IN_RESEARCH_UNIVERSE")
        if not self.evidence_id or not all(asdict(self.versions).values()):
            raise ValueError("REPLAY_REQUIRES_PINNED_VERSIONS_AND_EVIDENCE")

    def identity(self) -> dict:
        """Stable across daily account/window changes, immutable within a job."""
        return dict(schema=REPLAY_SCHEMA, market=self.market, sector=self.sector,
                    symbols=sorted(self.symbols), versions=asdict(self.versions), evidence_id=self.evidence_id)

    def inputs(self) -> dict:
        return dict(self.identity(), replay_id=self.replay_id, as_of=self.as_of.isoformat(),
                    history_start=self.history_start.isoformat(), history_end=self.history_end.isoformat(),
                    account=asdict(self.account))

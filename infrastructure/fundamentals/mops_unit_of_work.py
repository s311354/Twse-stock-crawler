"""Explicit SQLite transaction boundary for MOPS persistence commands."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import closing
from contextlib import contextmanager
from types import TracebackType

from application.fundamental_ports import FundamentalUnitOfWork


class SqliteFundamentalUnitOfWork:
    """Own one explicit SQLite transaction and connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._finished = False

    def __enter__(self) -> "SqliteFundamentalUnitOfWork":
        self.connection.execute("BEGIN")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_value, traceback
        try:
            if not self._finished:
                if exc_type is None:
                    self.commit()
                else:
                    self.rollback()
        finally:
            self.connection.close()
        return False

    def commit(self) -> None:
        if not self._finished:
            self.connection.commit()
            self._finished = True

    def rollback(self) -> None:
        if not self._finished:
            self.connection.rollback()
            self._finished = True


@contextmanager
def borrow_connection(
    connect: Callable[[], sqlite3.Connection],
    connection: sqlite3.Connection | None,
    uow: FundamentalUnitOfWork | None,
) -> Iterator[sqlite3.Connection]:
    if connection is not None and uow is not None:
        raise ValueError("Pass either uow or connection, not both")
    if uow is not None:
        if not isinstance(uow, SqliteFundamentalUnitOfWork):
            raise TypeError("MopsRepository requires SqliteFundamentalUnitOfWork")
        yield uow.connection
        return
    if connection is not None:
        yield connection
        return
    with closing(connect()) as owned_connection:
        yield owned_connection


@contextmanager
def write_transaction(
    connection: sqlite3.Connection,
    *,
    externally_managed: bool,
) -> Iterator[None]:
    if externally_managed:
        yield
        return
    with connection:
        yield


__all__ = [
    "SqliteFundamentalUnitOfWork",
    "borrow_connection",
    "write_transaction",
]

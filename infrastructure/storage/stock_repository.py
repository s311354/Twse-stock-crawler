"""CSV, SQLite, and optional DuckDB repositories for stock datasets."""

from __future__ import annotations

import importlib
import re
import sqlite3
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR
from domain.repositories import StockRepository

_SYMBOL_COLUMN = "_symbol"
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_name(value: str) -> str:
    safe = _SAFE_NAME_PATTERN.sub("_", str(value).strip()).strip("_")
    if not safe:
        raise ValueError("Repository symbol/table identifiers cannot be empty.")
    return safe


class CsvStockRepository(StockRepository):
    """Persist each stock symbol as one CSV file."""

    def __init__(self, output_dir: str | Path = DATA_DIR) -> None:
        self.output_dir = Path(output_dir)

    def save(self, symbol: str, stocks: pd.DataFrame) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stocks.to_csv(self._path_for_symbol(symbol), index=False)

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._path_for_symbol(symbol)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _path_for_symbol(self, symbol: str) -> Path:
        return self.output_dir / "{}.csv".format(_safe_name(symbol))


class SQLiteStockRepository(StockRepository):
    """Persist stock rows into a single SQLite table keyed by symbol."""

    def __init__(self, database_path: str | Path, table_name: str = "stock_prices") -> None:
        self.database_path = Path(database_path)
        self.table_name = _safe_name(table_name)

    def save(self, symbol: str, stocks: pd.DataFrame) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        symbol_key = str(symbol)
        dataset = stocks.copy()
        dataset.insert(0, _SYMBOL_COLUMN, symbol_key)

        with sqlite3.connect(self.database_path) as connection:
            self._delete_symbol_rows(connection, symbol_key)
            dataset.to_sql(self.table_name, connection, if_exists="append", index=False)

    def load(self, symbol: str) -> pd.DataFrame:
        if not self.database_path.exists():
            return pd.DataFrame()

        with sqlite3.connect(self.database_path) as connection:
            if not self._table_exists(connection):
                return pd.DataFrame()
            dataset = pd.read_sql_query(
                'SELECT * FROM "{}" WHERE "{}" = ? ORDER BY rowid'.format(self.table_name, _SYMBOL_COLUMN),
                connection,
                params=(str(symbol),),
            )

        return dataset.drop(columns=[_SYMBOL_COLUMN], errors="ignore")

    def _delete_symbol_rows(self, connection: sqlite3.Connection, symbol: str) -> None:
        if not self._table_exists(connection):
            return
        connection.execute(
            'DELETE FROM "{}" WHERE "{}" = ?'.format(self.table_name, _SYMBOL_COLUMN),
            (symbol,),
        )

    def _table_exists(self, connection: sqlite3.Connection) -> bool:
        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (self.table_name,),
        )
        return cursor.fetchone() is not None


class DuckDBStockRepository(StockRepository):
    """Persist stock rows into DuckDB when the optional duckdb package exists."""

    def __init__(self, database_path: str | Path, table_name: str = "stock_prices") -> None:
        try:
            self._duckdb = importlib.import_module("duckdb")
        except ModuleNotFoundError as error:
            raise ImportError(
                "DuckDBStockRepository requires the optional 'duckdb' package. "
                "Install it only if the DuckDB backend is needed."
            ) from error

        self.database_path = Path(database_path)
        self.table_name = _safe_name(table_name)

    def save(self, symbol: str, stocks: pd.DataFrame) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = stocks.copy()
        dataset.insert(0, _SYMBOL_COLUMN, str(symbol))

        with self._duckdb.connect(str(self.database_path)) as connection:
            connection.register("stock_repository_input", dataset)
            connection.execute(
                'CREATE TABLE IF NOT EXISTS "{}" AS SELECT * FROM stock_repository_input WHERE 1 = 0'.format(
                    self.table_name
                )
            )
            connection.execute(
                'DELETE FROM "{}" WHERE "{}" = ?'.format(self.table_name, _SYMBOL_COLUMN),
                [str(symbol)],
            )
            connection.execute('INSERT INTO "{}" SELECT * FROM stock_repository_input'.format(self.table_name))
            connection.unregister("stock_repository_input")

    def load(self, symbol: str) -> pd.DataFrame:
        if not self.database_path.exists():
            return pd.DataFrame()

        with self._duckdb.connect(str(self.database_path)) as connection:
            exists = connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [self.table_name],
            ).fetchone()[0]
            if not exists:
                return pd.DataFrame()
            dataset = connection.execute(
                'SELECT * FROM "{}" WHERE "{}" = ?'.format(self.table_name, _SYMBOL_COLUMN),
                [str(symbol)],
            ).fetchdf()

        return dataset.drop(columns=[_SYMBOL_COLUMN], errors="ignore")


def create_stock_repository(kind: str, location: str | Path) -> StockRepository:
    """Create a repository backend by name."""

    normalized_kind = kind.strip().lower()
    if normalized_kind == "csv":
        return CsvStockRepository(location)
    if normalized_kind == "sqlite":
        return SQLiteStockRepository(location)
    if normalized_kind == "duckdb":
        return DuckDBStockRepository(location)
    raise ValueError("Unknown stock repository kind '{}'. Use csv, sqlite, or duckdb.".format(kind))

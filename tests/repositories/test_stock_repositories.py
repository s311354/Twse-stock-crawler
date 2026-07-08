import pandas as pd
import pandas.testing as pdt

from infrastructure.storage.csv_repository import CsvRepository
from infrastructure.storage.stock_repository import DuckDBStockRepository
from infrastructure.storage.stock_repository import SQLiteStockRepository
from infrastructure.storage.stock_repository import create_stock_repository


def sample_stock_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": ["2026-07-01", "2026-07-02"],
            "Open": [100.0, 102.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 101.0],
            "Close": [104.0, 103.0],
            "Volume": [1000, 1200],
        }
    )


def test_csv_repository_save_load_round_trip(tmp_path) -> None:
    repository = CsvRepository(tmp_path)
    stocks = sample_stock_frame()

    repository.save("2382", stocks)

    loaded = repository.load("2382")
    pdt.assert_frame_equal(loaded, stocks)


def test_csv_repository_sanitizes_symbol_file_name(tmp_path) -> None:
    repository = CsvRepository(tmp_path)

    repository.save("2382/QUANTA", sample_stock_frame())

    assert (tmp_path / "2382_QUANTA.csv").exists()


def test_sqlite_repository_replaces_rows_for_same_symbol(tmp_path) -> None:
    repository = SQLiteStockRepository(tmp_path / "stocks.sqlite3")
    original = sample_stock_frame()
    replacement = original.tail(1).reset_index(drop=True)

    repository.save("2382", original)
    repository.save("2382", replacement)

    loaded = repository.load("2382")
    pdt.assert_frame_equal(loaded, replacement)


def test_sqlite_repository_keeps_symbols_isolated(tmp_path) -> None:
    repository = SQLiteStockRepository(tmp_path / "stocks.sqlite3")
    first = sample_stock_frame()
    second = sample_stock_frame().assign(Close=[200.0, 201.0])

    repository.save("2382", first)
    repository.save("2330", second)

    pdt.assert_frame_equal(repository.load("2382"), first)
    pdt.assert_frame_equal(repository.load("2330"), second)


def test_sqlite_repository_missing_symbol_returns_empty_frame(tmp_path) -> None:
    repository = SQLiteStockRepository(tmp_path / "stocks.sqlite3")

    assert repository.load("9999").empty


def test_repository_factory_creates_csv_and_sqlite_backends(tmp_path) -> None:
    assert create_stock_repository("csv", tmp_path).__class__.__name__ == "CsvStockRepository"
    assert create_stock_repository("sqlite", tmp_path / "stocks.sqlite3").__class__.__name__ == "SQLiteStockRepository"


def test_duckdb_repository_is_optional_but_deterministic(tmp_path) -> None:
    try:
        repository = DuckDBStockRepository(tmp_path / "stocks.duckdb")
    except ImportError as error:
        assert "duckdb" in str(error).lower()
        return

    stocks = sample_stock_frame()
    repository.save("2382", stocks)
    pdt.assert_frame_equal(repository.load("2382"), stocks)

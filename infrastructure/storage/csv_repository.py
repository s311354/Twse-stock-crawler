"""CSV storage repository for TWSE daily and analysis outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR
from domain.models import StockRows


class CsvRepository:
    """Owns CSV filesystem writes so application code stays storage-agnostic."""

    def __init__(self, output_dir: str | Path = DATA_DIR) -> None:
        self.output_dir = Path(output_dir)

    def write_daily_rows(self, file_name: str, scheduled_time: str, rows: StockRows) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{file_name}_{scheduled_time}.csv"

        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows([row for row in rows if row])

        return output_path

    def write_analysis_dataset(self, file_name: str, dataset: pd.DataFrame) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{file_name}_analysis_dataset.csv"
        dataset.to_csv(output_path, index=False)
        return output_path


"""Deterministic CSV provider used by tests and CI."""

from datetime import date
from pathlib import Path

import pandas as pd

from trading_lab.data.base import DataProvider, DataValidationError, validate_ohlcv


class CsvDataProvider(DataProvider):
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(
        self, symbol: str, start: date | None, end: date | None, interval: str
    ) -> pd.DataFrame:
        del symbol, interval
        if not self.path.is_file():
            raise DataValidationError(f"CSV data file does not exist: {self.path}")
        frame = pd.read_csv(self.path, index_col=0, parse_dates=True)
        frame = validate_ohlcv(frame, normalize_index=True)
        if start is not None:
            frame = frame.loc[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame.index <= pd.Timestamp(end)]
        if frame.empty:
            raise DataValidationError("No CSV data remains after date filtering")
        return frame

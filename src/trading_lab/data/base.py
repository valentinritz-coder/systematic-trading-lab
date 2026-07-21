"""Data provider contracts and OHLCV validation."""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class DataValidationError(ValueError):
    """Raised when OHLCV data cannot safely be used by a backtest."""


class DataProvider(ABC):
    """Common read-only interface for market-data providers."""

    @abstractmethod
    def load(
        self, symbol: str, start: date | None, end: date | None, interval: str
    ) -> pd.DataFrame:
        """Return validated OHLCV data without side effects."""


def validate_ohlcv(frame: pd.DataFrame, *, normalize_index: bool = False) -> pd.DataFrame:
    """Validate an OHLCV frame, optionally sorting and de-duplicating timestamps."""
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required OHLCV columns: {', '.join(missing)}")
    if frame.empty:
        raise DataValidationError("OHLCV data is empty")
    result = frame.loc[:, REQUIRED_OHLCV_COLUMNS].copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if result.index.hasnans:
        raise DataValidationError("OHLCV index contains invalid timestamps")
    if normalize_index:
        result = result[~result.index.duplicated(keep="last")].sort_index()
    elif not result.index.is_monotonic_increasing:
        raise DataValidationError("OHLCV timestamps must be sorted ascending")
    elif result.index.has_duplicates:
        raise DataValidationError("OHLCV timestamps must be unique")
    if result.isna().any().any():
        raise DataValidationError("OHLCV data contains missing values")
    if (result[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise DataValidationError("OHLC prices must be positive")
    if (result["Volume"] < 0).any():
        raise DataValidationError("OHLCV volume cannot be negative")
    maximums = pd.concat([result["Open"], result["Close"], result["Low"]], axis=1).max(axis=1)
    minimums = pd.concat([result["Open"], result["Close"], result["High"]], axis=1).min(axis=1)
    if (result["High"] < maximums).any():
        raise DataValidationError("High must be at least Open, Close, and Low")
    if (result["Low"] > minimums).any():
        raise DataValidationError("Low must be at most Open, Close, and High")
    return pd.DataFrame(result.astype(float))

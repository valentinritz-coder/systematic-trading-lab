"""Data provider contracts and OHLCV validation."""

from abc import ABC, abstractmethod
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
OHLC_COLUMNS = ("Open", "High", "Low", "Close")
MAX_DIAGNOSTIC_ROWS = 5
# One IEEE-754 double-precision unit of relative rounding error.  This accepts
# Yahoo's observed one-to-two-ULP disagreements between equivalent OHLC values
# while preserving rejection of meaningful price inconsistencies.
OHLC_FLOAT_RTOL = np.finfo(float).eps
OHLC_FLOAT_ATOL = np.finfo(float).eps


class DataValidationError(ValueError):
    """Raised when OHLCV data cannot safely be used by a backtest."""


class DataProvider(ABC):
    """Common read-only interface for market-data providers."""

    @abstractmethod
    def load(
        self, symbol: str, start: date | None, end: date | None, interval: str
    ) -> pd.DataFrame:
        """Return validated OHLCV data without side effects."""


def _invalid_ohlcv_rows(frame: pd.DataFrame, invalid: pd.Series) -> str:
    """Render precise values and invariant deltas for a bounded invalid-row sample."""
    rows = frame.loc[invalid, list(OHLC_COLUMNS)].head(MAX_DIAGNOSTIC_ROWS)
    maximums = rows[["Open", "Close", "Low"]].max(axis=1)
    minimums = rows[["Open", "Close", "High"]].min(axis=1)
    diagnostic = rows.assign(
        **{
            "High - max(Open, Close, Low)": rows["High"] - maximums,
            "Low - min(Open, Close, High)": rows["Low"] - minimums,
        }
    ).reset_index(names="index")
    return diagnostic.map(
        lambda value: format(value, ".17g") if isinstance(value, float) else value
    ).to_string(index=False)


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
    open_prices = cast(pd.Series, result["Open"])
    high_prices = cast(pd.Series, result["High"])
    low_prices = cast(pd.Series, result["Low"])
    close_prices = cast(pd.Series, result["Close"])
    maximums = pd.concat([open_prices, close_prices, low_prices], axis=1).max(axis=1)
    minimums = pd.concat([open_prices, close_prices, high_prices], axis=1).min(axis=1)
    invalid_high = (high_prices < maximums) & ~np.isclose(
        high_prices,
        maximums,
        rtol=OHLC_FLOAT_RTOL,
        atol=OHLC_FLOAT_ATOL,
    )
    if invalid_high.any():
        raise DataValidationError(
            "High must be at least Open, Close, and Low. "
            f"First invalid rows (up to {MAX_DIAGNOSTIC_ROWS}):\n"
            f"{_invalid_ohlcv_rows(cast(pd.DataFrame, result), invalid_high)}"
        )
    invalid_low = (low_prices > minimums) & ~np.isclose(
        low_prices,
        minimums,
        rtol=OHLC_FLOAT_RTOL,
        atol=OHLC_FLOAT_ATOL,
    )
    if invalid_low.any():
        raise DataValidationError(
            "Low must be at most Open, Close, and High. "
            f"First invalid rows (up to {MAX_DIAGNOSTIC_ROWS}):\n"
            f"{_invalid_ohlcv_rows(cast(pd.DataFrame, result), invalid_low)}"
        )
    return pd.DataFrame(result.astype(float))

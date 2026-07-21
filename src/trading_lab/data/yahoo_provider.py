"""Exploratory Yahoo Finance data provider; never used by unit tests."""

from datetime import date

import pandas as pd

from trading_lab.data.base import DataProvider, DataValidationError, validate_ohlcv


def _describe_yfinance_frame(frame: pd.DataFrame) -> str:
    """Return schema details needed to diagnose yfinance response changes."""
    return (
        f"received_columns={frame.columns.tolist()!r}; "
        f"dtypes={frame.dtypes.astype(str).to_dict()!r}; "
        f"columns_are_multiindex={isinstance(frame.columns, pd.MultiIndex)}"
    )


def _normalize_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the MultiIndex level containing OHLCV field names, if yfinance uses one."""
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    required = {"Open", "High", "Low", "Close", "Volume"}
    matching_levels = [
        level
        for level in range(frame.columns.nlevels)
        if required.issubset(set(frame.columns.get_level_values(level)))
    ]
    if len(matching_levels) != 1:
        return frame

    result = frame.copy()
    result.columns = result.columns.get_level_values(matching_levels[0])
    return result


class YahooFinanceDataProvider(DataProvider):
    def __init__(self, *, auto_adjust: bool = False) -> None:
        self.auto_adjust = auto_adjust

    def load(
        self, symbol: str, start: date | None, end: date | None, interval: str
    ) -> pd.DataFrame:
        import yfinance as yf

        frame = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=self.auto_adjust,
            progress=False,
        )
        if frame.empty:
            raise DataValidationError(
                f"Yahoo Finance returned no data for {symbol}; "
                "service may be unavailable or dates invalid"
            )
        schema = _describe_yfinance_frame(frame)
        normalized = _normalize_yfinance_columns(frame)
        try:
            return validate_ohlcv(normalized, normalize_index=True)
        except DataValidationError as error:
            raise DataValidationError(
                f"Yahoo Finance data validation failed for {symbol}: {error}\n"
                f"yfinance response schema: {schema}"
            ) from error

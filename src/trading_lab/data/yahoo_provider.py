"""Exploratory Yahoo Finance data provider; never used by unit tests."""

from datetime import date

import pandas as pd

from trading_lab.data.base import DataProvider, DataValidationError, validate_ohlcv


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
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        if frame.empty:
            raise DataValidationError(
                f"Yahoo Finance returned no data for {symbol}; "
                "service may be unavailable or dates invalid"
            )
        return validate_ohlcv(frame, normalize_index=True)

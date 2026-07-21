"""Long-only momentum breakout strategy without look-ahead bias."""

import json

import numpy as np
import pandas as pd

from trading_lab.risk.rules import stop_and_target
from trading_lab.strategies.base import LongOnlyStrategy


def previous_high(values: pd.Series | np.ndarray, lookback: int) -> np.ndarray:
    """Highest high of fully completed prior bars; current bar is excluded."""
    return pd.Series(values).rolling(lookback).max().shift(1).to_numpy()


def sma(values: pd.Series | np.ndarray, period: int) -> np.ndarray:
    return pd.Series(values).rolling(period).mean().to_numpy()


def entry_signal(close: float, prior_high_value: float, sma_value: float) -> bool:
    """Return a deterministic long-entry decision from information available at close."""
    return close > prior_high_value and close > sma_value


class MomentumBreakoutStrategy(LongOnlyStrategy):
    """Buy close-confirmed breakouts; orders fill according to runner execution mode."""

    breakout_lookback = 20
    sma_period = 50
    stop_loss_pct = 0.02
    take_profit_pct = 0.04
    max_holding_bars: int | None = None
    execution_mode = "next_open"

    def init(self) -> None:
        self.prior_high = self.I(previous_high, self.data.High, self.breakout_lookback)
        self.sma_line = self.I(sma, self.data.Close, self.sma_period)

    def next(self) -> None:
        if self.position:
            if self.max_holding_bars is not None:
                trade = self.trades[-1]
                if len(self.data) - trade.entry_bar >= self.max_holding_bars:
                    self.position.close()
            return
        close = float(self.data.Close[-1])
        if entry_signal(close, float(self.prior_high[-1]), float(self.sma_line[-1])):
            stop, target = stop_and_target(close, self.stop_loss_pct, self.take_profit_pct)
            tag = json.dumps(
                {
                    "signal_price": close,
                    "planned_stop_price": stop,
                    "planned_target_price": target,
                }
            )
            # Native bracket orders are attached when the parent market order fills.
            self.buy(sl=stop, tp=target, tag=tag)

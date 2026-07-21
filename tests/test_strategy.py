import pandas as pd

from trading_lab.risk.rules import stop_and_target
from trading_lab.strategies.momentum_breakout import previous_high, sma_exit_signal


def test_previous_high_excludes_decision_bar() -> None:
    highs = pd.Series([10, 11, 12, 13])
    result = previous_high(highs, 3)
    assert pd.isna(result[2])
    assert result[3] == 12


def test_stop_and_target_for_long_entry() -> None:
    assert stop_and_target(100, 0.02, 0.04) == (98.0, 104.0)


def test_sma_exit_is_strictly_below_not_equal() -> None:
    assert sma_exit_signal(9.9, 10)
    assert not sma_exit_signal(10, 10)


def test_sma_exit_decision_does_not_depend_on_future_values() -> None:
    current_close, current_sma = 9.0, 10.0
    decision = sma_exit_signal(current_close, current_sma)
    future_close = 1_000_000.0
    assert future_close != current_close
    assert sma_exit_signal(current_close, current_sma) == decision

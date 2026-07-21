import pandas as pd

from trading_lab.risk.rules import stop_and_target
from trading_lab.strategies.momentum_breakout import previous_high


def test_previous_high_excludes_decision_bar() -> None:
    highs = pd.Series([10, 11, 12, 13])
    result = previous_high(highs, 3)
    assert pd.isna(result[2])
    assert result[3] == 12


def test_stop_and_target_for_long_entry() -> None:
    assert stop_and_target(100, 0.02, 0.04) == (98.0, 104.0)

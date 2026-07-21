import json

import pandas as pd
import pytest

from trading_lab.backtest.reporting import build_metrics, enrich_trades


def ohlc(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["Open", "High", "Low", "Close"],
        index=pd.date_range("2024-01-05", periods=len(rows), freq="D"),
    )


def trade(entry_bar: int, exit_bar: int, pnl: float = 1, tag: str | None = None) -> pd.DataFrame:
    dates = pd.date_range("2024-01-05", periods=6, freq="D")
    return pd.DataFrame(
        {
            "Size": [2],
            "EntryPrice": [10.0],
            "ExitPrice": [10.0 + pnl / 2],
            "EntryBar": [entry_bar],
            "ExitBar": [exit_bar],
            "EntryTime": [dates[entry_bar]],
            "ExitTime": [dates[exit_bar]],
            "PnL": [pnl],
            "ReturnPct": [pnl / 20],
            "Tag": [tag],
        }
    )


def stats_for(pnls: list[float]) -> pd.Series:
    trades = pd.concat([trade(0, 1, pnl) for pnl in pnls], ignore_index=True)
    return pd.Series(
        {
            "_trades": trades,
            "Equity Final [$]": 1000 + sum(pnls),
            "Return [%]": 0,
            "Buy & Hold Return [%]": 0,
            "Win Rate [%]": 0,
            "Max. Drawdown [%]": 0,
            "Max. Drawdown Duration": pd.Timedelta(0),
        }
    )


def test_duration_same_bar_multiple_bars_weekend_and_end_of_data() -> None:
    data = ohlc([(10, 11, 9, 10)] * 4)
    same_bar = enrich_trades(trade(0, 0), data)
    multi_bar = enrich_trades(trade(0, 3), data)
    assert same_bar.loc[0, "duration_bars"] == 0
    assert multi_bar.loc[0, "duration_bars"] == 3
    assert multi_bar.loc[0, "duration_days"] == 3  # Friday to Monday includes weekend days.
    assert multi_bar.loc[0, "exit_reason"] == "end_of_data"


def test_duration_global_average_and_maximum() -> None:
    raw = pd.concat([trade(0, 0), trade(0, 4)], ignore_index=True)
    data = ohlc([(10, 11, 9, 10)] * 5)
    stats = stats_for([1, 1])
    stats["_trades"] = raw
    metrics = build_metrics(stats, 1000, data=data)
    assert metrics["average_trade_duration_bars"] == 2
    assert metrics["max_trade_duration_bars"] == 4
    assert metrics["average_trade_duration_days"] == 2
    assert metrics["max_trade_duration_days"] == 4


@pytest.mark.parametrize(
    ("pnls", "expected"),
    [
        ([10], (100.0, 100.0, 100.0)),
        ([10, 5], (100.0 / 1.5, 100.0, 100.0)),
        ([10, -5, -5], (None, None, None)),
        ([60, 40, -50], (120.0, 100.0, 100.0)),
        ([6, 5, 4, 3, 2, 1], (100.0 / 3.5, 15 / 21 * 100, 20 / 21 * 100)),
        ([-2, -3], (40.0, 100.0, 100.0)),
    ],
)
def test_pnl_concentration_handles_trade_counts_totals_and_ranking(
    pnls: list[float], expected: tuple[float | None, float | None, float | None]
) -> None:
    metrics = build_metrics(stats_for(pnls), 1000)
    for key, value in zip(
        (
            "best_trade_contribution_pct",
            "top_3_trades_contribution_pct",
            "top_5_trades_contribution_pct",
        ),
        expected,
        strict=True,
    ):
        assert metrics[key] == pytest.approx(value) if value is not None else metrics[key] is None


@pytest.mark.parametrize(
    ("reason", "bars", "expected"),
    [
        ("end_of_data", [(10, 12, 10, 11), (11, 13, 11, 12)], (30.0, 0.0)),
        ("end_of_data", [(10, 10, 8, 9), (9, 9, 7, 8)], (0.0, -30.0)),
        ("end_of_data", [(10, 12, 9, 11), (11, 13, 8, 12)], (30.0, -20.0)),
        ("max_holding", [(10, 15, 5, 10), (11, 99, 1, 11)], (50.0, -50.0)),
    ],
)
def test_mfe_mae_conventions_and_next_open_exit_bar_exclusion(
    reason: str, bars: list[tuple[float, float, float, float]], expected: tuple[float, float]
) -> None:
    data = ohlc(bars)
    raw = trade(0, 1)
    if reason == "max_holding":
        enriched = enrich_trades(raw, data, "next_open", max_holding_bars=1)
    else:
        enriched = enrich_trades(raw, data)
    assert enriched.loc[0, "mfe_pct"] == pytest.approx(expected[0])
    assert enriched.loc[0, "mae_pct"] == pytest.approx(expected[1])


def test_sma_exit_at_next_open_excludes_future_exit_bar_range() -> None:
    data = ohlc([(10, 15, 5, 10), (10, 11, 9, 9), (8, 99, 1, 8)])
    enriched = enrich_trades(trade(0, 2), data, "next_open", exit_sma_period=2)
    assert enriched.loc[0, "exit_reason"] == "sma_exit"
    assert (enriched.loc[0, "mfe_pct"], enriched.loc[0, "mae_pct"]) == pytest.approx((50, -50))


def test_max_holding_classification_wins_over_simultaneous_sma_exit() -> None:
    data = ohlc([(10, 10, 10, 10), (9, 9, 9, 9), (8, 8, 8, 8)])
    enriched = enrich_trades(trade(0, 2), data, "next_open", max_holding_bars=2, exit_sma_period=2)
    assert enriched.loc[0, "exit_reason"] == "max_holding"


def test_intrabar_stop_and_target_classification_wins_over_time_exit() -> None:
    data = ohlc([(10, 10, 10, 10), (10, 10, 9, 10)])
    tag = json.dumps({"signal_price": 10, "planned_stop_price": 9, "planned_target_price": 11})
    stopped = enrich_trades(trade(0, 1, -2, tag), data, max_holding_bars=1, exit_sma_period=1)
    targeted = enrich_trades(trade(0, 1, 2, tag), data, max_holding_bars=1, exit_sma_period=1)
    assert stopped.loc[0, "exit_reason"] == "stop_loss"
    assert targeted.loc[0, "exit_reason"] == "take_profit"


def test_intrabar_stop_and_target_exclude_unknown_exit_candle_side() -> None:
    data = ohlc([(10, 11, 9, 10), (10, 99, 1, 10)])
    stop_tag = json.dumps({"signal_price": 10, "planned_stop_price": 9, "planned_target_price": 20})
    target_tag = json.dumps(
        {"signal_price": 10, "planned_stop_price": 1, "planned_target_price": 11}
    )
    stopped = enrich_trades(trade(0, 1, -2, stop_tag), data)
    targeted = enrich_trades(trade(0, 1, 2, target_tag), data)
    assert (stopped.loc[0, "mfe_pct"], stopped.loc[0, "mae_pct"]) == pytest.approx((10, -10))
    assert (targeted.loc[0, "mfe_pct"], targeted.loc[0, "mae_pct"]) == pytest.approx((10, -10))


def test_same_bar_excursion_uses_only_known_intrabar_stop_fill() -> None:
    data = ohlc([(10, 99, 1, 10)])
    tag = json.dumps({"signal_price": 10, "planned_stop_price": 9, "planned_target_price": 20})
    enriched = enrich_trades(trade(0, 0, -2, tag), data)
    assert (enriched.loc[0, "mfe_pct"], enriched.loc[0, "mae_pct"]) == pytest.approx((0, -10))

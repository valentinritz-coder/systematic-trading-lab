import json
from pathlib import Path

import pandas as pd
import pytest
from backtesting import Backtest

from trading_lab.backtest.reporting import enrich_trades
from trading_lab.backtest.runner import run_backtest
from trading_lab.config import load_config
from trading_lab.strategies.base import LongOnlyStrategy
from trading_lab.strategies.momentum_breakout import MomentumBreakoutStrategy, entry_signal

ROOT = Path(__file__).parents[1]


def breakout_frame(entry_open: float, entry_high: float, entry_low: float) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "Open": [10, 10, 10, 10, 10, 10, entry_open, 10],
            "High": [10, 10, 10, 10, 10, 11, entry_high, 10],
            "Low": [10, 10, 10, 10, 10, 10, entry_low, 9],
            "Close": [10, 10, 10, 10, 10, 11, 10, 10],
            "Volume": [100] * 8,
        },
        index=index,
    )


def run_frame(
    frame: pd.DataFrame,
    *,
    mode: str = "next_open",
    holding: int | None = None,
    take_profit: float | None = 0.04,
) -> pd.DataFrame:
    stats = Backtest(
        frame,
        MomentumBreakoutStrategy,
        cash=1000,
        exclusive_orders=True,
        finalize_trades=True,
        trade_on_close=mode == "signal_close",
    ).run(
        breakout_lookback=2,
        sma_period=3,
        stop_loss_pct=0.02,
        take_profit_pct=take_profit,
        max_holding_bars=holding,
        execution_mode=mode,
    )
    trades = enrich_trades(stats["_trades"], frame, mode, holding)
    assert len(trades) == 1
    return trades


def test_signal_rules_cover_valid_no_breakout_and_sma_filter() -> None:
    assert entry_signal(11, 10, 10)
    assert not entry_signal(10, 10, 9)
    assert not entry_signal(11, 10, 12)


def test_normal_next_open_entry_has_native_brackets() -> None:
    trade = run_frame(breakout_frame(11, 11.2, 10.9))
    assert trade.loc[0, "actual_entry_price"] == 11
    assert trade.loc[0, "planned_stop_price"] == pytest.approx(10.78)
    assert trade.loc[0, "planned_target_price"] == pytest.approx(11.44)
    assert trade.loc[0, "gap_status"] == "within_brackets"
    assert trade.loc[0, "actual_risk_to_planned_stop_pct"] == pytest.approx(2.0)


def test_stop_is_hit_during_entry_bar() -> None:
    trade = run_frame(breakout_frame(11, 11.2, 10.7))
    assert trade.loc[0, "ExitPrice"] == pytest.approx(10.78)
    assert trade.loc[0, "EntryBar"] == trade.loc[0, "ExitBar"]
    assert trade.loc[0, "entry_and_exit_same_bar"]


def test_target_is_hit_during_entry_bar() -> None:
    trade = run_frame(breakout_frame(11, 11.6, 10.9))
    assert trade.loc[0, "ExitPrice"] == pytest.approx(11.44)
    assert trade.loc[0, "EntryBar"] == trade.loc[0, "ExitBar"]
    assert trade.loc[0, "exit_reason"] == "take_profit"


def no_stop_or_target_frame() -> pd.DataFrame:
    frame = breakout_frame(11, 11.6, 10.9)
    frame.loc[frame.index[7:], ["High", "Low"]] = [11.2, 10.9]
    frame.loc[frame.index[7:], "Close"] = 11
    frame.loc[frame.index[7:], "Open"] = 11
    frame.loc[frame.index[7:], "Volume"] = 100
    frame = pd.concat(
        [
            frame,
            frame.iloc[[-1]].rename(
                index={frame.index[-1]: frame.index[-1] + pd.Timedelta(days=1)}
            ),
        ]
    )
    return frame


def test_disabled_take_profit_never_creates_a_target_exit() -> None:
    trade = run_frame(no_stop_or_target_frame(), take_profit=None)
    assert pd.isna(trade.loc[0, "planned_target_price"])
    assert trade.loc[0, "exit_reason"] == "end_of_data"


def test_enabled_max_holding_exits_after_configured_duration() -> None:
    trade = run_frame(no_stop_or_target_frame(), take_profit=None, holding=1)
    assert trade.loc[0, "holding_bars"] == 1
    assert trade.loc[0, "exit_reason"] == "max_holding"


def test_disabled_max_holding_keeps_position_open_until_end_of_data() -> None:
    trade = run_frame(no_stop_or_target_frame(), take_profit=None, holding=None)
    assert trade.loc[0, "holding_bars"] > 1
    assert trade.loc[0, "exit_reason"] == "end_of_data"


def test_disabling_target_and_max_holding_leaves_only_stop_and_end_of_data_exits() -> None:
    trade = run_frame(no_stop_or_target_frame(), take_profit=None, holding=None)
    assert trade.loc[0, "exit_reason"] == "end_of_data"
    assert "unknown" not in trade["exit_reason"].to_numpy()


def test_gap_below_stop_is_flagged_without_negative_risk() -> None:
    trade = run_frame(breakout_frame(10, 10.2, 9.5))
    assert trade.loc[0, "actual_entry_price"] == 10
    assert trade.loc[0, "ExitPrice"] == 10
    assert trade.loc[0, "gap_status"] == "opened_below_stop"
    assert pd.isna(trade.loc[0, "actual_risk_to_planned_stop_pct"])
    assert trade.loc[0, "exit_reason"] == "stop_loss"


def test_gap_above_target_is_flagged_at_native_engine_prices() -> None:
    trade = run_frame(breakout_frame(12, 12.2, 11.8))
    assert trade.loc[0, "actual_entry_price"] == 12
    assert trade.loc[0, "ExitPrice"] == 12
    assert trade.loc[0, "gap_status"] == "opened_above_target"
    assert pd.isna(trade.loc[0, "actual_risk_to_planned_stop_pct"])
    assert trade.loc[0, "exit_reason"] == "take_profit"


def test_signal_close_reports_close_entry_and_brackets() -> None:
    trade = run_frame(breakout_frame(11, 11.2, 10.9), mode="signal_close")
    assert trade.loc[0, "actual_entry_price"] == 11
    assert trade.loc[0, "planned_stop_price"] == pytest.approx(10.78)
    assert trade.loc[0, "planned_target_price"] == pytest.approx(11.44)


def test_holding_period_uses_actual_entry_bar_in_both_modes() -> None:
    for mode in ("next_open", "signal_close"):
        trade = run_frame(breakout_frame(11, 11.2, 10.9), mode=mode, holding=1)
        assert trade.loc[0, "ExitBar"] >= trade.loc[0, "EntryBar"]


def test_tags_are_resilient_to_missing_invalid_and_incomplete_metadata() -> None:
    raw = pd.DataFrame(
        {
            "EntryPrice": [10, 10, 10],
            "EntryBar": [1, 1, 1],
            "ExitBar": [1, 1, 1],
            "Tag": [None, "not-json", '{"signal_price": 10}'],
        }
    )
    enriched = enrich_trades(raw)
    assert not enriched["trade_metadata_valid"].any()


def test_long_only_strategy_rejects_short_orders() -> None:
    class IllegalShort(LongOnlyStrategy):
        def init(self) -> None:
            pass

        def next(self) -> None:
            self.sell()

    frame = breakout_frame(11, 11.2, 10.9)
    with pytest.raises(RuntimeError, match="Short orders are forbidden"):
        Backtest(frame, IllegalShort).run()


def test_manifest_is_reproducible_and_paths_are_config_relative(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = ROOT / "configs/momentum-demo.yml"
    monkeypatch.chdir(tmp_path)
    config = load_config(config_path)
    assert config.data.path == (ROOT / "data/samples/demo_ohlcv.csv").resolve()
    assert config.output_dir == (ROOT / "results").resolve()
    config.output_dir = tmp_path / "reports"
    manifests = []
    for _ in range(2):
        _, paths = run_backtest(config)
        manifests.append(json.loads(paths["manifest"].read_text()))
    stable = {
        "ohlcv_sha256",
        "resolved_config",
        "package_versions",
        "bar_count",
        "first_data_timestamp",
        "last_data_timestamp",
    }
    for key in stable:
        assert manifests[0][key] == manifests[1][key]
    assert manifests[0]["run_id"] != manifests[1]["run_id"]

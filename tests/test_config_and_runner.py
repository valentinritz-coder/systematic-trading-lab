import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from trading_lab.backtest.reporting import enrich_trades
from trading_lab.backtest.runner import run_backtest
from trading_lab.config import StrategyConfig, load_config
from trading_lab.data.base import DataValidationError
from trading_lab.data.yahoo_provider import YahooFinanceDataProvider

ROOT = Path(__file__).parents[1]


def test_yaml_configuration_loads() -> None:
    config = load_config(ROOT / "configs/momentum-demo.yml")
    assert config.initial_cash == 1000
    assert config.data.path is not None and config.data.path.is_file()


@pytest.mark.parametrize("field, value", [("take_profit_pct", 0), ("max_holding_bars", 0)])
def test_optional_exit_configuration_rejects_non_positive_values(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        StrategyConfig(**{field: value})


def test_optional_exit_configuration_accepts_null_values() -> None:
    strategy = StrategyConfig(take_profit_pct=None, max_holding_bars=None)
    assert strategy.take_profit_pct is None
    assert strategy.max_holding_bars is None


def test_deterministic_backtest_generates_reports(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/momentum-demo.yml")
    config.output_dir = tmp_path
    stats, paths = run_backtest(config)
    assert stats["Equity Final [$]"] > 0
    assert len(stats["_trades"]) == 4
    assert stats["Equity Final [$]"] == pytest.approx(1046.1737264)
    assert {"metrics", "trades", "equity"}.issubset(paths)
    assert all(path.is_file() for path in paths.values())


def test_strategy_never_overlaps_positions(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/momentum-demo.yml")
    config.output_dir = tmp_path
    stats, _ = run_backtest(config)
    trades = stats["_trades"]
    assert len(trades) == 4
    assert (trades["EntryBar"].iloc[1:].to_numpy() > trades["ExitBar"].iloc[:-1].to_numpy()).all()


def test_yahoo_provider_forwards_auto_adjust(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    frame = pd.DataFrame(
        {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
        index=pd.date_range("2024-01-01", periods=1),
    )

    def download(*args: object, **kwargs: object) -> pd.DataFrame:
        del args
        captured.update(kwargs)
        return frame

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    YahooFinanceDataProvider(auto_adjust=True).load("SPY", None, None, "1d")
    assert captured["auto_adjust"] is True


def test_yahoo_provider_normalizes_ticker_first_multiindex_and_reports_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = pd.MultiIndex.from_product(
        [["SPY"], ["Open", "High", "Low", "Close", "Volume"]], names=["Ticker", "Price"]
    )
    frame = pd.DataFrame(
        [[2, 1, 1, 1.5, 10]], columns=columns, index=pd.date_range("2024-01-01", periods=1)
    )

    monkeypatch.setitem(
        sys.modules, "yfinance", SimpleNamespace(download=lambda *args, **kwargs: frame)
    )

    with pytest.raises(DataValidationError, match="columns_are_multiindex=True") as error:
        YahooFinanceDataProvider(auto_adjust=True).load("SPY", None, None, "1d")

    message = str(error.value)
    assert "High must be at least" in message
    assert "2024-01-01" in message
    assert "dtypes=" in message


def test_snapshot_replay_writes_artifacts_and_never_uses_yahoo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs/momentum-demo.yml")
    config.output_dir = tmp_path / "first"
    initial_stats, initial_paths = run_backtest(config)
    snapshot = initial_paths["snapshot"]
    assert snapshot.name == "input_ohlcv.csv" and snapshot.is_file()
    assert {"summary", "monthly_returns", "yearly_returns", "drawdown", "trades_by_year"}.issubset(
        initial_paths
    )
    assert all(initial_paths[name].is_file() for name in initial_paths)

    class ForbiddenYahoo:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            raise AssertionError("Yahoo must not be called for a snapshot replay")

    monkeypatch.setattr("trading_lab.backtest.runner.YahooFinanceDataProvider", ForbiddenYahoo)
    config.data.provider = "yahoo"
    config.output_dir = tmp_path / "replay"
    replay_stats, replay_paths = run_backtest(config, data_snapshot=snapshot)
    assert replay_stats["Equity Final [$]"] == initial_stats["Equity Final [$]"]
    manifest = json.loads(replay_paths["manifest"].read_text())
    assert manifest["configured_data_provider"] == "yahoo"
    assert manifest["effective_data_provider"] == "snapshot"
    assert manifest["snapshot_path"] == str(snapshot.resolve())
    assert manifest["auto_adjust"] is False
    assert manifest["strategy_parameters"] == config.strategy.model_dump()
    assert manifest["ohlcv_sha256"]


def test_metadata_invalid_and_zero_trade_reports(tmp_path: Path) -> None:
    invalid = pd.DataFrame({"EntryPrice": [10], "EntryBar": [1], "ExitBar": [2], "Tag": [None]})
    assert enrich_trades(invalid).loc[0, "gap_status"] == "metadata_invalid"

    config = load_config(ROOT / "configs/momentum-demo.yml")
    config.output_dir = tmp_path
    config.strategy.breakout_lookback = 999
    config.strategy.sma_period = 999
    stats, paths = run_backtest(config)
    assert stats["# Trades"] == 0
    assert pd.read_csv(paths["trades_by_year"]).empty
    assert "Trades: 0" in paths["summary"].read_text()

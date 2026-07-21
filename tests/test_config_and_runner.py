from pathlib import Path

from trading_lab.backtest.runner import run_backtest
from trading_lab.config import load_config

ROOT = Path(__file__).parents[1]


def test_yaml_configuration_loads() -> None:
    config = load_config(ROOT / "configs/momentum-demo.yml")
    assert config.initial_cash == 1000
    assert config.data.path is not None and config.data.path.is_file()


def test_deterministic_backtest_generates_reports(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/momentum-demo.yml")
    config.output_dir = tmp_path
    stats, paths = run_backtest(config)
    assert stats["Equity Final [$]"] > 0
    assert len(stats["_trades"]) == 4
    assert {"metrics", "trades", "equity"}.issubset(paths)
    assert all(path.is_file() for path in paths.values())


def test_strategy_never_overlaps_positions(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/momentum-demo.yml")
    config.output_dir = tmp_path
    stats, _ = run_backtest(config)
    trades = stats["_trades"]
    assert len(trades) == 4
    assert (trades["EntryBar"].iloc[1:].to_numpy() > trades["ExitBar"].iloc[:-1].to_numpy()).all()

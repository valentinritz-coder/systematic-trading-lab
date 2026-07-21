"""Safe orchestration of deterministic, long-only backtests."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import pandas as pd
from backtesting import Backtest

from trading_lab.backtest.reporting import write_reports
from trading_lab.config import BacktestConfig
from trading_lab.data.base import DataProvider
from trading_lab.data.csv_provider import CsvDataProvider
from trading_lab.data.yahoo_provider import YahooFinanceDataProvider
from trading_lab.strategies.momentum_breakout import MomentumBreakoutStrategy


def provider_for(config: BacktestConfig) -> DataProvider:
    if config.data.provider == "csv":
        assert config.data.path is not None
        return CsvDataProvider(config.data.path)
    return YahooFinanceDataProvider()


def create_run_dir(config: BacktestConfig, run_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    safe_symbol = "".join(char if char.isalnum() else "_" for char in config.symbol)
    return config.output_dir / f"{stamp}_{safe_symbol}_{run_id}"


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def data_fingerprint(data: pd.DataFrame) -> str:
    """Hash canonical OHLCV CSV bytes, including timestamps and column order."""
    return hashlib.sha256(data.to_csv().encode("utf-8")).hexdigest()


def write_manifest(
    output_dir: Path, config: BacktestConfig, data: pd.DataFrame, run_id: str
) -> Path:
    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "python_version": sys.version,
        "operating_system": platform.platform(),
        "package_versions": {
            "pandas": version("pandas"),
            "numpy": version("numpy"),
            "backtesting": version("backtesting"),
            "trading_lab": version("systematic-trading-lab"),
        },
        "resolved_config": config.model_dump(mode="json"),
        "data_provider": config.data.provider,
        "symbol": config.symbol,
        "first_data_timestamp": data.index[0].isoformat(),
        "last_data_timestamp": data.index[-1].isoformat(),
        "bar_count": len(data),
        "ohlcv_sha256": data_fingerprint(data),
        "execution_mode": config.execution.mode,
        "run_id": run_id,
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_backtest(config: BacktestConfig) -> tuple[pd.Series, dict[str, Path]]:
    """Load data, execute a no-short/no-leverage backtest, and persist reports."""
    data = provider_for(config).load(
        config.symbol, config.data.start, config.data.end, config.data.interval
    )

    def commission(order_size: float, price: float) -> float:
        return config.commission_fixed + abs(order_size * price) * config.commission_rate

    run_id = uuid4().hex[:8]
    output_dir = create_run_dir(config, run_id)
    engine = Backtest(
        data,
        MomentumBreakoutStrategy,
        cash=config.initial_cash,
        commission=commission,
        spread=config.spread,
        margin=1.0,
        trade_on_close=config.execution.mode == "signal_close",
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = engine.run(**config.strategy.model_dump(), execution_mode=config.execution.mode)
    if (stats["_trades"]["Size"] < 0).any():
        raise RuntimeError("Backtest produced a short transaction; refusing to write reports")
    paths = write_reports(stats, config.initial_cash, output_dir)
    paths["manifest"] = write_manifest(output_dir, config, data, run_id)
    html_path = output_dir / "report.html"
    try:
        engine.plot(filename=str(html_path), open_browser=False)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        (output_dir / "html_report_unavailable.txt").write_text(str(exc), encoding="utf-8")
    else:
        paths["html"] = html_path
    return stats, paths

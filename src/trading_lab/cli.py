"""Typer command-line interface."""

from pathlib import Path
from typing import Annotated

import typer

from trading_lab.backtest.reporting import build_metrics
from trading_lab.backtest.runner import run_backtest
from trading_lab.config import load_config

app = typer.Typer(add_completion=False, help="Safe, long-only systematic trading research.")


@app.command("validate-config")
def validate_config(
    config: Annotated[Path, typer.Option(..., exists=True, readable=True)],
) -> None:
    """Validate a YAML backtest configuration without accessing the network."""
    loaded = load_config(config)
    typer.echo(f"Configuration valid: symbol={loaded.symbol}, provider={loaded.data.provider}")


@app.command("backtest")
def backtest(
    config: Annotated[Path, typer.Option(..., exists=True, readable=True)],
    data_snapshot: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
) -> None:
    """Run a simulation only; this command cannot send real orders."""
    loaded = load_config(config)
    stats, paths = run_backtest(loaded, data_snapshot=data_snapshot)
    metrics = build_metrics(stats, loaded.initial_cash)
    typer.echo("Backtest completed (simulation only; no real orders are possible).")
    for name, value in metrics.items():
        typer.echo(f"{name}: {value}")
    typer.echo("Reports:")
    for name, path in paths.items():
        typer.echo(f"  {name}: {path}")

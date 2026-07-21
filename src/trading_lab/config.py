"""Validated YAML configuration for a safe, long-only simulation."""

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    provider: Literal["csv", "yahoo"] = "csv"
    path: Path | None = None
    interval: str = "1d"
    start: date | None = None
    end: date | None = None
    auto_adjust: bool = False

    @model_validator(mode="after")
    def validate_source(self) -> "DataConfig":
        if self.provider == "csv" and self.path is None:
            raise ValueError("data.path is required when data.provider is csv")
        if self.start and self.end and self.start > self.end:
            raise ValueError("data.start must not be after data.end")
        return self


class ExecutionConfig(BaseModel):
    mode: Literal["next_open", "signal_close"] = "next_open"


class StrategyConfig(BaseModel):
    breakout_lookback: int = Field(default=20, ge=2)
    sma_period: int = Field(default=50, ge=2)
    stop_loss_pct: float = Field(default=0.02, gt=0, lt=1)
    take_profit_pct: float | None = Field(default=0.04, gt=0, lt=1)
    max_holding_bars: int | None = Field(default=None, ge=1)
    exit_sma_period: int | None = Field(default=None, ge=1)


class BacktestConfig(BaseModel):
    symbol: str = Field(min_length=1)
    data: DataConfig
    initial_cash: float = Field(default=1000.0, gt=0)
    commission_fixed: float = Field(default=0.0, ge=0)
    commission_rate: float = Field(default=0.001, ge=0, lt=1)
    spread: float = Field(default=0.0, ge=0, lt=1)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    output_dir: Path = Path("results")


def load_config(path: Path) -> BacktestConfig:
    """Load and validate configuration from YAML."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read configuration {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = BacktestConfig.model_validate(raw)
    if config.data.path is not None and not config.data.path.is_absolute():
        config.data.path = (path.parent / config.data.path).resolve()
    if not config.output_dir.is_absolute():
        config.output_dir = (path.parent / config.output_dir).resolve()
    return config

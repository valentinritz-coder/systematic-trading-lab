# systematic-trading-lab

A reproducible **research laboratory** for long-only systematic trading simulations. Version 0.1 cannot connect to a broker and cannot submit real orders. It is educational/experimental software, not investment advice; past performance does not guarantee future results.

## Safety model

- No leverage, short selling, CFDs, derivatives, options, broker integration, or real-order API exists.
- Backtests use one position at a time, estimated commissions, and spread.
- CSV data is deterministic and is the only source used by tests and the main CI workflow. Yahoo Finance is exploratory only: it can change, be unavailable, or return incomplete data. `SPY` in `momentum-yahoo.yml` is only a data example, not an investment recommendation.

A **backtest** evaluates historical data. **Paper trading** would consume current data and simulate executions without real money. **Live trading** would submit orders to a broker; it is intentionally not implemented and must remain disabled by default in any future work.

## Installation and use

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
uv run python -m trading_lab validate-config --config configs/momentum-demo.yml
uv run python -m trading_lab backtest --config configs/momentum-demo.yml
```

All relative paths, including `data.path` and `output_dir`, are resolved relative to their YAML configuration file. Each backtest creates `results/YYYYMMDD_SYMBOL_RUNID/` with `metrics.json`, `trades.csv`, `equity_curve.csv`, `run_manifest.json`, and, when plotting dependencies support it, `report.html`. The manifest records resolved configuration, data fingerprint, environment, package versions, received date range, commit when available, and run identifier.

## Execution model and limitations

`MomentumBreakoutStrategy` buys only when the close exceeds the **previous** 20-bar high and is above its 50-bar SMA. It applies a 2% planned stop-loss, a 4% planned take-profit, and optional maximum holding duration. The current bar is excluded from the breakout high, preventing that form of look-ahead bias.

### Exit configuration

The stop-loss is required. The other exit mechanisms can be disabled independently for strategy research:

```yaml
strategy:
  take_profit_pct: null    # disables the take-profit order
  max_holding_bars: null   # disables the time-based exit
```

When a numeric value is supplied, `take_profit_pct` and `max_holding_bars` must both be greater than zero. Positions that remain open at the end of the input data are closed with the `end_of_data` exit reason.

`execution.mode: next_open` (default) takes the signal after a candle closes and submits a parent order that `backtesting.py` fills at the following open. The parent is created immediately with native contingent stop-loss and take-profit brackets, so the engine can evaluate them during the entry candle. Stop and target are planned from the signal close, **not** claimed to be exact percentages of the eventual fill. The trade export exposes the signal/fill prices, entry gap, planned and actual risk, explicit gap status, and whether entry and exit occurred in one bar. `signal_close` enables `trade_on_close=True`; it is experimental and potentially optimistic because a close-price fill may not be tradable in practice.

`backtesting.py` evaluates parent/bracket execution from OHLC bars rather than tick data. When a gap crosses a planned level or multiple levels are reachable in one bar, the engine's native OHLC ordering and fill rules determine the result. Historical OHLCV, spread, fees, and stop/target execution cannot reproduce liquidity, slippage, corporate actions, or real-market execution.

Do not mass-optimize parameters on this small historical sample. Repeatedly searching parameter combinations can fit noise (overfitting) and produce misleading in-sample results. Use out-of-sample periods, realistic execution assumptions, and independent review before treating research as evidence.

## GitHub Actions

CI installs the locked dependency graph, then runs linting, type checks, tests, and the deterministic CSV backtest. From GitHub, open **Actions**, choose the workflow, then **Run workflow**. When it finishes, download the `backtest-reports` artifact from that run.

The scheduled exploratory workflow directly uses `configs/momentum-yahoo.yml`, produces only simulation reports, and never commits results or sends orders. GitHub cron is UTC: `17 22 * * 1-5` is 23:17 in Europe/Paris during standard time; daylight-saving time shifts it to 00:17 Paris. Run it manually if an exact local-time schedule is required.

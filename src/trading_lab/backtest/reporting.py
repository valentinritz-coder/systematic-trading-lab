"""Portable report generation for backtesting.py results."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MetricValue = float | int | str | None
_REQUIRED_TAG_FIELDS = ("signal_price", "planned_stop_price")
_OPTIONAL_TAG_FIELDS = ("planned_target_price",)


def _number(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return None


def enrich_trades(
    trades: pd.DataFrame,
    data: pd.DataFrame | None = None,
    execution_mode: str = "next_open",
    max_holding_bars: int | None = None,
) -> pd.DataFrame:
    """Add robust signal metadata and gap-aware risk fields to completed trades."""
    result = trades.copy()
    for column in (*_REQUIRED_TAG_FIELDS, *_OPTIONAL_TAG_FIELDS):
        result[column] = np.nan
    result["trade_metadata_valid"] = False
    for index, raw_tag in result.get("Tag", pd.Series(dtype=object)).items():
        try:
            tag = json.loads(raw_tag)
            if not isinstance(tag, dict) or not all(key in tag for key in _REQUIRED_TAG_FIELDS):
                continue
            for column in _REQUIRED_TAG_FIELDS:
                result.loc[index, column] = float(tag[column])
            target = tag.get("planned_target_price")
            if target is not None:
                result.loc[index, "planned_target_price"] = float(target)
            result.loc[index, "trade_metadata_valid"] = True
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    result["actual_entry_price"] = result.get("EntryPrice", pd.Series(dtype=float))
    result["entry_gap_pct"] = (
        (result["actual_entry_price"] - result["signal_price"]) / result["signal_price"] * 100
    )
    result["planned_risk_pct"] = (
        (result["signal_price"] - result["planned_stop_price"]) / result["signal_price"] * 100
    )
    result["gap_status"] = "within_brackets"
    result.loc[~result["trade_metadata_valid"], "gap_status"] = "metadata_invalid"
    result.loc[
        result["trade_metadata_valid"]
        & (result["actual_entry_price"] <= result["planned_stop_price"]),
        "gap_status",
    ] = "opened_below_stop"
    has_target = result["planned_target_price"].notna()
    result.loc[
        result["trade_metadata_valid"]
        & has_target
        & (result["actual_entry_price"] >= result["planned_target_price"]),
        "gap_status",
    ] = "opened_above_target"
    valid = result["gap_status"].eq("within_brackets") & result["trade_metadata_valid"]
    result["actual_risk_to_planned_stop_pct"] = np.nan
    result.loc[valid, "actual_risk_to_planned_stop_pct"] = (
        (result.loc[valid, "actual_entry_price"] - result.loc[valid, "planned_stop_price"])
        / result.loc[valid, "actual_entry_price"]
        * 100
    )
    result["entry_and_exit_same_bar"] = result.get("EntryBar", pd.Series(dtype=int)).eq(
        result.get("ExitBar", pd.Series(dtype=int))
    )
    result["entry_date"] = result.get("EntryTime", pd.Series(dtype="datetime64[ns]"))
    result["exit_date"] = result.get("ExitTime", pd.Series(dtype="datetime64[ns]"))
    result["signal_date"] = pd.NaT
    if data is not None:
        for index, trade in result.iterrows():
            entry_bar = int(trade["EntryBar"])
            signal_bar = entry_bar if execution_mode == "signal_close" else entry_bar - 1
            if 0 <= signal_bar < len(data):
                result.loc[index, "signal_date"] = data.index[signal_bar]
    result["exit_price"] = result.get("ExitPrice", pd.Series(dtype=float))
    result["return_pct"] = result.get("ReturnPct", pd.Series(dtype=float)) * 100
    result["pnl"] = result.get("PnL", pd.Series(dtype=float))
    result["holding_bars"] = result.get("ExitBar", pd.Series(dtype=float)) - result.get(
        "EntryBar", pd.Series(dtype=float)
    )
    result["exit_reason"] = "unknown"
    tolerance = 1e-8
    result.loc[result["exit_price"] <= result["planned_stop_price"] + tolerance, "exit_reason"] = (
        "stop_loss"
    )
    result.loc[
        has_target & (result["exit_price"] >= result["planned_target_price"] - tolerance),
        "exit_reason",
    ] = "take_profit"
    # A duration close is submitted once the configured holding limit is reached.
    result.loc[
        (result["exit_reason"] == "unknown")
        & (max_holding_bars is not None)
        & (result["holding_bars"] >= max_holding_bars),
        "exit_reason",
    ] = "max_holding"
    # finalize_trades closes any still-open position on the last bar.
    if data is not None:
        result.loc[
            (result["exit_reason"] == "unknown")
            & result.get("ExitBar", pd.Series(dtype=float)).eq(len(data) - 1),
            "exit_reason",
        ] = "end_of_data"
    return result


def build_metrics(
    stats: pd.Series, initial_cash: float, max_holding_bars: int | None = None
) -> dict[str, MetricValue]:
    """Normalize engine metrics and compute gap-aware statistics."""
    trades = enrich_trades(stats["_trades"], max_holding_bars=max_holding_bars)
    pnl = trades["PnL"] if not trades.empty else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit, gross_loss = float(wins.sum()), abs(float(losses.sum()))
    valid_risk = trades.loc[
        trades["gap_status"].eq("within_brackets"), "actual_risk_to_planned_stop_pct"
    ]
    return {
        "initial_capital": initial_cash,
        "final_capital": float(stats["Equity Final [$]"]),
        "total_return_pct": _number(stats.get("Return [%]")),
        "buy_and_hold_return_pct": _number(stats.get("Buy & Hold Return [%]")),
        "trade_count": int(len(trades)),
        "win_rate_pct": _number(stats.get("Win Rate [%]")),
        "average_gain": float(wins.mean()) if not wins.empty else None,
        "average_loss": float(losses.mean()) if not losses.empty else None,
        "best_trade": float(pnl.max()) if not pnl.empty else None,
        "worst_trade": float(pnl.min()) if not pnl.empty else None,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "expectancy": float(pnl.mean()) if not pnl.empty else None,
        "max_drawdown_pct": _number(stats.get("Max. Drawdown [%]")),
        "max_drawdown_duration": str(stats.get("Max. Drawdown Duration", "")),
        "estimated_commissions": float(trades["Commission"].sum())
        if "Commission" in trades
        else 0.0,
        "average_entry_gap_pct": _number(trades["entry_gap_pct"].mean()),
        "max_absolute_entry_gap_pct": _number(trades["entry_gap_pct"].abs().max()),
        "opened_below_stop_count": int(trades["gap_status"].eq("opened_below_stop").sum()),
        "opened_above_target_count": int(trades["gap_status"].eq("opened_above_target").sum()),
        "same_bar_exit_count": int(trades["entry_and_exit_same_bar"].sum()),
        "average_valid_actual_risk_pct": _number(valid_risk.mean()),
    }


def longest_loss_streak(trades: pd.DataFrame) -> int:
    """Return the longest consecutive sequence of losing completed trades."""
    longest = current = 0
    for pnl_value in trades.get("PnL", pd.Series(dtype=float)):
        current = current + 1 if pnl_value < 0 else 0
        longest = max(longest, current)
    return longest


def write_reports(
    stats: pd.Series,
    initial_cash: float,
    output_dir: Path,
    data: pd.DataFrame | None = None,
    execution_mode: str = "next_open",
    max_holding_bars: int | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(stats, initial_cash, max_holding_bars)
    metrics["generated_at"] = datetime.now(UTC).isoformat()
    metrics_path, trades_path, equity_path = (
        output_dir / "metrics.json",
        output_dir / "trades.csv",
        output_dir / "equity_curve.csv",
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    trades = enrich_trades(stats["_trades"], data, execution_mode, max_holding_bars)
    trades.to_csv(trades_path, index=False)
    equity = stats["_equity_curve"].copy()
    equity.to_csv(equity_path, index=True, index_label="timestamp")
    returns = equity["Equity"].pct_change().fillna(0)
    monthly = (1 + returns).resample("ME").prod().sub(1).mul(100).rename("return_pct")
    yearly = (1 + returns).resample("YE").prod().sub(1).mul(100).rename("return_pct")
    drawdown = (
        equity["Equity"].div(equity["Equity"].cummax()).sub(1).mul(100).rename("drawdown_pct")
    )
    monthly.to_csv(output_dir / "monthly_returns.csv", index_label="period")
    yearly.to_csv(output_dir / "yearly_returns.csv", index_label="year")
    drawdown.to_csv(output_dir / "drawdown.csv", index_label="timestamp")
    years = (
        pd.to_datetime(trades["entry_date"]).dt.year if not trades.empty else pd.Series(dtype=int)
    )
    years.value_counts().sort_index().rename_axis("year").rename("trade_count").to_csv(
        output_dir / "trades_by_year.csv"
    )
    reasons = trades["exit_reason"].value_counts().to_dict() if not trades.empty else {}
    years_count = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    final_capital = float(stats["Equity Final [$]"])
    buy_hold_return_pct = float(metrics["buy_and_hold_return_pct"] or 0)
    strategy_annualized = (final_capital / initial_cash) ** (1 / years_count) - 1
    buy_hold_final = initial_cash * (1 + buy_hold_return_pct / 100)
    buy_hold_annualized = (buy_hold_final / initial_cash) ** (1 / years_count) - 1
    buy_hold_returns = data["Close"].pct_change().fillna(0) if data is not None else returns
    buy_hold_equity = initial_cash * (1 + buy_hold_returns).cumprod()
    buy_hold_drawdown = buy_hold_equity.div(buy_hold_equity.cummax()).sub(1).mul(100)
    buy_hold_max_drawdown = float(buy_hold_drawdown.min())
    time_in_market = float(trades["holding_bars"].sum() / len(equity) * 100) if len(equity) else 0
    average_holding = trades["holding_bars"].mean() if not trades.empty else 0
    total_return = metrics["total_return_pct"]
    buy_hold_return = metrics["buy_and_hold_return_pct"]
    strategy_volatility = returns.std() * (252**0.5) * 100
    buy_hold_volatility = buy_hold_returns.std() * (252**0.5) * 100
    strategy_annualized_pct = strategy_annualized * 100
    buy_hold_annualized_pct = buy_hold_annualized * 100
    annualized_row = (
        f"| Annualized return | {strategy_annualized_pct:.2f}% | {buy_hold_annualized_pct:.2f}% |"
    )
    summary = "\n".join(
        [
            "# Backtest summary",
            "",
            f"Period: {equity.index[0].date()} to {equity.index[-1].date()}",
            f"Initial capital: {initial_cash:.2f}",
            f"Final capital: {metrics['final_capital']:.2f}",
            f"Strategy return: {metrics['total_return_pct']}%",
            f"Buy-and-hold return: {metrics['buy_and_hold_return_pct']}%",
            "",
            "## Trades",
            f"Trades: {metrics['trade_count']} | Win rate: {metrics['win_rate_pct']}%",
            f"Average gain: {metrics['average_gain']} | Average loss: {metrics['average_loss']}",
            f"Profit factor: {metrics['profit_factor']} | Expectancy: {metrics['expectancy']}",
            f"Longest loss streak: {longest_loss_streak(trades)}",
            f"Average holding bars: {average_holding}",
            "",
            f"Max drawdown: {metrics['max_drawdown_pct']}%",
            f"Average entry gap: {metrics['average_entry_gap_pct']}%",
            f"Estimated commissions: {metrics['estimated_commissions']}",
            f"Best trade: {metrics['best_trade']} | Worst trade: {metrics['worst_trade']}",
            f"Exit reasons: {reasons}",
            "",
            "## Same-data comparison",
            "| Metric | Strategy | Buy-and-hold |",
            "| --- | ---: | ---: |",
            f"| Total return | {total_return}% | {buy_hold_return}% |",
            annualized_row,
            f"| Annualized volatility | {strategy_volatility:.2f}% | {buy_hold_volatility:.2f}% |",
            f"| Max drawdown | {metrics['max_drawdown_pct']}% | {buy_hold_max_drawdown:.2f}% |",
            f"| Time in position | {time_in_market:.2f}% | 100.00% |",
            f"| Transactions | {metrics['trade_count']} | 1 |",
            f"| Final capital | {metrics['final_capital']:.2f} | {buy_hold_final:.2f} |",
            "",
            "Buy-and-hold uses the same OHLCV data but is not risk-equivalent: "
            "this strategy can remain in cash.",
            "Exit reasons identify stop-loss, take-profit, maximum-holding, and end-of-data exits.",
        ]
    )
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    return {
        "metrics": metrics_path,
        "trades": trades_path,
        "equity": equity_path,
        "summary": summary_path,
        "monthly_returns": output_dir / "monthly_returns.csv",
        "yearly_returns": output_dir / "yearly_returns.csv",
        "drawdown": output_dir / "drawdown.csv",
        "trades_by_year": output_dir / "trades_by_year.csv",
    }

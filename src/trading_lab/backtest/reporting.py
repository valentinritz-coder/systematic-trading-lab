"""Portable report generation for backtesting.py results."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MetricValue = float | int | str | None
_TAG_FIELDS = ("signal_price", "planned_stop_price", "planned_target_price")


def _number(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return None


def enrich_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Add robust signal metadata and gap-aware risk fields to completed trades."""
    result = trades.copy()
    for column in _TAG_FIELDS:
        result[column] = np.nan
    result["trade_metadata_valid"] = False
    for index, raw_tag in result.get("Tag", pd.Series(dtype=object)).items():
        try:
            tag = json.loads(raw_tag)
            if not isinstance(tag, dict) or not all(key in tag for key in _TAG_FIELDS):
                continue
            for column in _TAG_FIELDS:
                result.loc[index, column] = float(tag[column])
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
    result.loc[result["actual_entry_price"] <= result["planned_stop_price"], "gap_status"] = (
        "opened_below_stop"
    )
    result.loc[result["actual_entry_price"] >= result["planned_target_price"], "gap_status"] = (
        "opened_above_target"
    )
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
    return result


def build_metrics(stats: pd.Series, initial_cash: float) -> dict[str, MetricValue]:
    """Normalize engine metrics and compute gap-aware statistics."""
    trades = enrich_trades(stats["_trades"])
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


def write_reports(stats: pd.Series, initial_cash: float, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(stats, initial_cash)
    metrics["generated_at"] = datetime.now(UTC).isoformat()
    metrics_path, trades_path, equity_path = (
        output_dir / "metrics.json",
        output_dir / "trades.csv",
        output_dir / "equity_curve.csv",
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    enrich_trades(stats["_trades"]).to_csv(trades_path, index=False)
    stats["_equity_curve"].to_csv(equity_path, index=True, index_label="timestamp")
    return {"metrics": metrics_path, "trades": trades_path, "equity": equity_path}

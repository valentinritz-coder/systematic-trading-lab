"""Small, testable long-only risk calculations."""


def stop_and_target(
    entry_price: float, stop_loss_pct: float, take_profit_pct: float | None
) -> tuple[float, float | None]:
    """Return stop-loss and optional take-profit prices for a long position."""
    target = entry_price * (1 + take_profit_pct) if take_profit_pct is not None else None
    return entry_price * (1 - stop_loss_pct), target

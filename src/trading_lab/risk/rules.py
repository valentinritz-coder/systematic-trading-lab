"""Small, testable long-only risk calculations."""


def stop_and_target(
    entry_price: float, stop_loss_pct: float, take_profit_pct: float
) -> tuple[float, float]:
    """Return stop-loss and take-profit prices for a long position."""
    return entry_price * (1 - stop_loss_pct), entry_price * (1 + take_profit_pct)

"""Shared constraints for safe backtesting strategies."""

from typing import Any, Never

from backtesting import Strategy


class LongOnlyStrategy(Strategy):  # type: ignore[misc]
    """Strategy base that rejects every attempt to create a short order."""

    def sell(self, *args: Any, **kwargs: Any) -> Never:
        del args, kwargs
        raise RuntimeError("Short orders are forbidden: this laboratory is long-only")

# Instructions for contributors

- Code is organized under `src/trading_lab`: data providers, strategies, risk rules, backtest orchestration, and reporting remain separate.
- Before every pull request run `ruff check .`, `ruff format --check .`, `mypy src`, and `pytest`.
- Never add real trading, broker integrations, order-submission code, credentials, or secrets without an explicit user request.
- Every functional change must add or update tests.
- Keep data processing and strategy decisions deterministic, testable, and free from look-ahead bias. Unit tests must not require network access.

# Security policy

This repository is a research and simulation project. It has no broker integration and must never contain broker API keys, access tokens, passwords, or private credentials.

Do not commit secrets. If future explicitly approved integrations need credentials, use environment-provided secrets, grant the minimum permissions needed, rotate them, and keep live trading disabled by default. Report suspected secret exposure privately to the repository maintainers and rotate the affected credential immediately.

---
disclaimer: "No information in this document should be taken for granted. Any statement not backed by executable code, tests, or a verifiable reference may be incomplete, invalid, or hallucinated."
---

# Changelog

## 0.1.0

Initial backend spike promoted into a packaged SDK-first backend surface.

### Added

- Public SDK facade in [backend/src/sdk](/home/admin/spikes/red-iron-square/backend/src/sdk) for:
  - one-shot decisions
  - temporal simulations
  - self-aware simulations
- JSON-safe result models and mapping helpers for SDK consumers.
- Packaged CLI in [backend/src/sdk/cli.py](/home/admin/spikes/red-iron-square/backend/src/sdk/cli.py) with `decide` and `simulate` commands.
- Runnable example payloads in [backend/examples](/home/admin/spikes/red-iron-square/backend/examples).
- Thin FastAPI transport in [backend/src/api](/home/admin/spikes/red-iron-square/backend/src/api) with `/health`, `/decide`, and `/simulate`.
- API and CLI smoke tests in [backend/tests/test_api.py](/home/admin/spikes/red-iron-square/backend/tests/test_api.py) and [backend/tests/test_sdk_cli.py](/home/admin/spikes/red-iron-square/backend/tests/test_sdk_cli.py).

### Changed

- Backend packaging now uses `hatchling` so the `red-iron-square` console script installs correctly through `uv`.
- Development dependencies were moved to `[dependency-groups]` in [backend/pyproject.toml](/home/admin/spikes/red-iron-square/backend/pyproject.toml).
- FastAPI was moved into core backend dependencies.
- Structured logs now default to stderr so CLI stdout remains machine-readable JSON.
- README was updated to document the SDK, CLI, examples, and HTTP API.

### Verified

- `uv run pytest -q` passes with `108 passed`.

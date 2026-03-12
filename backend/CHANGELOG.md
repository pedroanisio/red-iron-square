---
disclaimer: "No information in this document should be taken for granted. Any statement not backed by executable code, tests, or a verifiable reference may be incomplete, invalid, or hallucinated."
---

# Changelog

## Unreleased

### Added

- ElevenLabs audio integration module (`src.demo.audio`) with voice-settings
  calculator, audio-tag injector/stripper, and provider adapter with env-based
  API key loading and graceful degradation.
- `.env.example` documenting all supported environment variables.
- Campaign orchestration backend in
  [backend/src/api/campaign_router.py](/home/admin/spikes/red-iron-square/backend/src/api/campaign_router.py),
  [backend/src/api/campaign_service.py](/home/admin/spikes/red-iron-square/backend/src/api/campaign_service.py),
  [backend/src/api/campaign_store.py](/home/admin/spikes/red-iron-square/backend/src/api/campaign_store.py),
  [backend/src/api/campaign_models.py](/home/admin/spikes/red-iron-square/backend/src/api/campaign_models.py),
  and
  [backend/src/api/campaign_schemas.py](/home/admin/spikes/red-iron-square/backend/src/api/campaign_schemas.py).
- New FastAPI endpoints for campaign creation, listing, detail, summary, branching, checkpoint-rule creation, and manual checkpoint evaluation.
- SQLite persistence for `campaign`, `campaign_run`, and `checkpoint_rule`.
- Campaign API, service, and store tests in
  [backend/tests/test_campaign_api.py](/home/admin/spikes/red-iron-square/backend/tests/test_campaign_api.py),
  [backend/tests/test_campaign_service.py](/home/admin/spikes/red-iron-square/backend/tests/test_campaign_service.py),
  and
  [backend/tests/test_campaign_store.py](/home/admin/spikes/red-iron-square/backend/tests/test_campaign_store.py).

### Changed

- The FastAPI app factory now wires both run and campaign routers.
- Backend verification baseline is now `457 passed` under `uv run pytest -q`.

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

---
disclaimer: "No information in this document should be taken for granted. Any statement not backed by executable code, tests, or a verifiable reference may be incomplete, invalid, or hallucinated."
last_assessed: "2026-03-11"
---

# Backlog

This backlog turns the current architecture direction into an execution sequence for the repository as it exists today:

- packaged Python backend with `uv` dependency management
- public SDK in `backend/src/sdk`
- FastAPI transport in `backend/src/api`
- Flask telemetry UI in `backend/src/ui`
- SQLite persistence in `backend/src/api/run_store.py`
- Anthropic + OpenAI adapters in `backend/src/llm`
- 127 tests, 92.5% coverage

The intended GenAI integration assumption for this backlog is the **Anthropic Python library** as the LLM client/runtime boundary. The simulation engine remains the deterministic source of truth. Anthropic models are used only for:

- scenario proposal
- narrative generation
- analysis
- intervention recommendations

Anthropic models must not mutate simulator state directly.

## Delivery Principles

1. Persist the simulation state outside model context windows.
2. Keep all simulator truth in Python domain code.
3. Make the API contract stable before adding many agent behaviors.
4. Add orchestration only after run lifecycle, persistence, and replay exist.
5. Keep LLM outputs typed and validated before they touch the simulation service.

## Phase 0: Baseline Consolidation — DONE

### Goal

Make the current backend/API layer the explicit foundation for all future work.

### Status

All tasks complete. Architecture documented in CLAUDE.md. Clear module boundaries established:

- `src/personality` — dimension registry, activation functions, decision engine
- `src/temporal` — simulator, state transitions, affective engine, memory
- `src/self_model` — self-awareness, identity drift, self-related emotions
- `src/sdk` — domain orchestration convenience layer
- `src/api` — FastAPI external service boundary
- `src/llm` — LLM integration boundary (adapters, runtime, schemas)
- `src/ui` — Flask telemetry frontend
- `src/shared` — structured logging, validators, shared types

Naming conventions frozen: simulation run, tick event, trajectory window, phase annotation, intervention decision, agent invocation.

## Phase 1: Stateful Run API — DONE

### Goal

Replace stateless single-shot simulation endpoints with a real run lifecycle.

### Status

All endpoints implemented in `src/api/run_router.py`:

- `POST /runs` — creates persisted run with config
- `GET /runs/{run_id}` — returns RunSummary with tick_count, latest_tick, phases, counts
- `POST /runs/{run_id}/tick` — executes and persists one tick
- `GET /runs/{run_id}/trajectory` — returns full trajectory with ticks, phases, invocations, interventions
- `PATCH /runs/{run_id}/params` — patches mutable config (temperature)
- `POST /runs/{run_id}/phases` — creates phase annotations

Application service: `RunService` (275 LOC) coordinates persisted runs with simulator reconstruction. Supports both temporal and self-aware runs in one contract.

Schemas: RunConfig, RunCreateRequest, RunTickRequest, RunPatchRequest, RunSummary, TrajectoryResponse, PhaseCreateRequest — all Pydantic.

Tests: `test_api_runs.py`, `test_api_basic.py` cover lifecycle, stepping, trajectory retrieval.

## Phase 2: Persistent Storage — DONE

### Goal

Persist run and tick state outside process memory.

### Status

SQLite storage via `RunStore` (291 LOC) with 5 tables:

- `simulation_run` (run_id, mode, status, config_json, parent_run_id, parent_tick, created_at, updated_at)
- `tick_event` (run_id, tick, scenario_json, requested_outcome, result_json, created_at)
- `phase_annotation` (run_id, start_tick, end_tick, label, notes, created_at)
- `agent_invocation` (invocation_id, run_id, agent_name, purpose, input_json, output_json, raw_text, metadata_json, created_at)
- `intervention_decision` (decision_id, run_id, action, reason, payload_json, applied, created_at)

Schema auto-initialized on first connection. Thread-safe via Lock(). JSON serialization for all domain objects. Schema defined in `run_store_support.py`.

All tick data persisted: tick index, scenario payload, requested outcome, full result (action, probabilities, outcome, state_before, state_after, activations, emotions, self-model fields when applicable).

Tests: RunStore exercised through API integration tests; 99% coverage.

## Phase 3: Canonical Tick Event Contract — DONE

### Goal

Make tick events the canonical substrate for analytics, orchestration, and UI.

### Status

Normalized via `TickEventRecord` dataclass. Every tick result includes:

- run_id (foreign key), tick index, UTC timestamp
- scenario payload, chosen action, outcome
- state_before, state_after, activations, emotions, probabilities
- derived fields: action effort, energy drain, stress estimate

Same payload shape consumed by UI, API clients, and LLM agents. Serialization via JSON round-trip through SQLite.

## Phase 4: Replay And Branching — DONE

### Goal

Make experiments reproducible and branchable.

### Status

Both endpoints implemented:

- `POST /runs/{run_id}/replay` — deterministic replay clone
- `POST /runs/{run_id}/branches` — branch from tick N with optional parameter patch

Lineage tracked: parent_run_id, parent_tick stored in simulation_run table. Branch cutoff validated. Phase annotations copied and trimmed to cutoff.

Tests: `test_api_runs.py::test_replay_and_branch` validates deterministic trajectory equivalence and branch provenance.

## Phase 5: Anthropic Integration Boundary — DONE

### Goal

Introduce Anthropic as a typed orchestration dependency, not as a state holder.

### Status

Complete `src/llm/` package:

- `AnthropicAdapter` — typed wrapper over Anthropic Python client, JSON extraction with markdown fence tolerance, wrapper object normalization
- `OpenAIAdapter` — same protocol, uses native structured output parser
- `AgentRuntime` — task-oriented wrapper exposing propose_scenario, summarize_window, analyze_window, recommend_intervention
- `factory.py` — provider selection via `RED_IRON_SQUARE_LLM_PROVIDER` env var

Typed outputs (all Pydantic):

- `ScenarioProposal` (name, description, values, rationale)
- `NarrativeChunk` (summary, tick_start, tick_end, evidence)
- `AnalysisReport` (dominant_regime, notable_emotions, anomalies, recommendations)
- `InterventionRecommendation` (action, reason, temperature) with constrained action set

Persistence: agent_invocation table stores raw_text, input/output JSON, metadata (model, provider, stop_reason, input_tokens, output_tokens).

No domain module imports Anthropic directly. All model outputs validated via Pydantic before use.

Tests: `test_llm.py` (9 tests) covers JSON parsing, model validation, credential checking, token capture with mocked clients.

## Phase 6: Minimum Agent Runtime — DONE

### Goal

Ship the smallest useful agent-assisted loop.

### Status

`POST /runs/{run_id}/assist/step` implements the full loop:

1. Load recent trajectory window
2. Call propose_scenario (ScenarioAgent) with current state + user goals
3. Execute tick with proposed scenario
4. Call summarize_window (ObserverAgent) on updated trajectory
5. Persist both agent invocations
6. Return AssistedStepResponse with scenario, tick, narrative, invocations

Tests: `test_api_agents.py::test_assisted_step_persists_agent_invocations` with FakeAgentRuntime.

## Phase 7: Analysis And Intervention Agents — DONE

### Goal

Add higher-order interpretation and control, still behind explicit boundaries.

### Status

`POST /runs/{run_id}/intervention` implements:

- Calls recommend_intervention (InterventionAgent) with current state, recent ticks, user goals
- Persists intervention decision (action, reason, payload, applied flag)
- Optionally applies patch_params intervention (temperature adjustment)
- Returns InterventionResponse with recommendation, invocation, decision, updated_run

7 constrained intervention actions: continue, probe, narrate, analyze, patch_params, pause, terminate. All outputs logged as explicit decisions in intervention_decision table.

Tests: `test_api_agents.py::test_intervention_endpoint_persists_and_applies_patch`.

## Phase 8: Campaign Orchestration — NOT STARTED

### Goal

Support longer-running research campaigns rather than isolated runs.

### Tasks

- Add campaign model:
  - campaign id
  - goal set
  - run set
  - branch set
- Support scheduled analysis checkpoints:
  - every N ticks
  - on threshold crossing
  - on user pause
- Add campaign-level summaries:
  - identity drift progression
  - emotion arc summaries
  - branch comparisons

### Deliverables

- campaign service
- campaign persistence model
- cross-run analytics queries

### Acceptance Criteria

- one campaign can coordinate multiple runs and branches
- campaign-level analysis is reproducible from stored artifacts

## Phase 9: Thin Telemetry Frontend — PARTIAL

### Goal

Add a frontend only after the API and event model are stable.

### Status

Flask UI implemented in `src/ui/` (app.py 186 LOC, api_client.py 70 LOC):

- Dark-themed dashboard with Inter + JetBrains Mono typography
- Run creation with JSON config editor
- AI-assisted step with goal input and lookback window
- Intervention request with auto-apply option
- Manual tick with scenario JSON editor
- Personality profile visualization (dimension bars with full names)
- Internal state visualization (mood, arousal, energy, satisfaction, frustration as colored bars)
- Emotion tag display with intensity tooltips
- Tabbed data panels: Steps (with outcome narrative), AI Calls, Interventions
- API status indicator, run ID display, toast notifications
- Plain-language explanations for every action

Tests: `test_ui.py` (2 tests) covers rendering and run view loading.

### Remaining

- Trajectory charts (emotion arcs over time, state evolution graphs)
- Phase timeline markers on trajectory view
- Identity drift visualization for self-aware runs
- Branch comparison view
- Typed frontend client models (currently uses raw dicts)

## Phase 10: Full Orchestrator — NOT STARTED

### Goal

Only after the previous phases are stable, add a manager-style orchestrator.

### Candidate Components

- `MetaController`
- `ScenarioAgent`
- `ObserverAgent`
- `AffectAnalystAgent`
- `InterventionAgent`

### Tasks

- Decide whether to use a graph orchestrator or a simpler internal loop first.
- Add human checkpoint support.
- Support pause/resume/continue/terminate decisions.
- Persist all orchestrator decisions.

### Deliverables

- orchestration runtime
- campaign execution API
- audit trail for all model-driven decisions

### Acceptance Criteria

- all orchestration behavior is reconstructible from persistence
- no hidden state lives only in prompt history

## Cross-Cutting Work

### Testing — DONE

- unit tests for schema validation: Pydantic models tested across all API tests
- repository integration tests: test_api_runs.py, test_api_agents.py exercise persistence
- deterministic replay tests: test_api_runs.py validates trajectory equivalence
- mocked Anthropic adapter tests: test_llm.py with FakeAnthropicClient, FakeOpenAIClient
- API contract tests: test_api_basic.py, test_api_runs.py, test_api_agents.py
- e2e test with real Anthropic: test_e2e.py (requires credentials)
- 127 tests total, 92.5% coverage

### Observability — DONE

- structured logs via structlog (`src/shared/logging.py`)
- agent invocation audit trail (raw_text, metadata_json per invocation)
- token/cost metadata captured (input_tokens, output_tokens, model, provider)
- remaining: explicit correlation ID propagation through request lifecycle

### Security And Safety — DONE

- Pydantic validation on all incoming payloads
- parameterized SQL queries throughout RunStore
- credentials loaded from environment only (never hardcoded)
- LLM configuration failures return HTTP 503 with actionable messages
- intervention permissions constrained to 7 explicit actions

### Configuration — DONE

- `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` for Anthropic
- `OPENAI_API_KEY` for OpenAI
- `RED_IRON_SQUARE_LLM_PROVIDER` for provider selection (anthropic|openai)
- Database path configurable (defaults to `.data/red_iron_square.sqlite3`)
- `FLASK_SECRET_KEY`, `FLASK_DEBUG` for UI
- `RED_IRON_SQUARE_API_URL` for UI-to-API connection
- Replay seed behavior documented and functional

## Summary

| Phase | Status |
|-------|--------|
| 0 — Baseline Consolidation | DONE |
| 1 — Stateful Run API | DONE |
| 2 — Persistent Storage | DONE |
| 3 — Canonical Tick Event Contract | DONE |
| 4 — Replay And Branching | DONE |
| 5 — Anthropic Integration Boundary | DONE |
| 6 — Minimum Agent Runtime | DONE |
| 7 — Analysis And Intervention Agents | DONE |
| 8 — Campaign Orchestration | NOT STARTED |
| 9 — Thin Telemetry Frontend | PARTIAL |
| 10 — Full Orchestrator | NOT STARTED |

## Recommended Next Priorities

1. **Phase 9 completion**: Add trajectory charts (emotion arcs, state evolution), phase timeline markers, and identity drift visualization to the Flask UI.
2. **Phase 8**: Introduce campaign model and cross-run analytics — the infrastructure (runs, branches, persistence) is ready to support it.
3. **Phase 10**: Full orchestrator with MetaController and human checkpoints — deferred until campaign model exists.

## Explicit Non-Goals For Now

- no frontend-local simulation engine
- no direct LLM ownership of memory or state
- no peer-to-peer freeform agent mesh before persistence exists
- no hidden state transitions outside the tick pipeline

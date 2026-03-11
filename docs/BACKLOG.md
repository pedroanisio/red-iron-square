---
disclaimer: "No information in this document should be taken for granted. Any statement not backed by executable code, tests, or a verifiable reference may be incomplete, invalid, or hallucinated."
last_assessed: "2026-03-11"
---

# Backlog

This backlog turns the current architecture direction into an execution sequence for the repository as it exists today:

- packaged Python backend with `uv` dependency management
- public SDK in `backend/src/sdk`
- public API facade in `backend/src/simulation`
- FastAPI transport in `backend/src/api`
- Flask telemetry UI in `backend/src/ui`
- SQLite persistence in `backend/src/api/run_store.py`
- Anthropic + OpenAI adapters in `backend/src/llm`
- 174 tests, 91% coverage

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
- `src/simulation` — public API facade re-exporting from all domain modules
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
- `GET /runs` — lists all persisted runs with summary
- `GET /runs/{run_id}` — returns RunSummary with tick_count, latest_tick, phases, counts
- `POST /runs/{run_id}/tick` — executes and persists one tick
- `GET /runs/{run_id}/trajectory` — returns full trajectory with ticks, phases, invocations, interventions
- `PATCH /runs/{run_id}/params` — patches mutable config (temperature)
- `POST /runs/{run_id}/phases` — creates phase annotations

Application service: `RunService` (240 LOC) coordinates persisted runs with simulator reconstruction. Supports both temporal and self-aware runs in one contract.

Schemas: RunConfig, RunCreateRequest, RunTickRequest, RunPatchRequest, RunSummary, TrajectoryResponse, PhaseCreateRequest — all Pydantic.

Tests: `test_api_runs.py`, `test_api_basic.py`, `test_api_list_runs.py` cover lifecycle, stepping, listing, trajectory retrieval.

## Phase 2: Persistent Storage — DONE

### Goal

Persist run and tick state outside process memory.

### Status

SQLite storage via `RunStore` (289 LOC) with 5 tables:

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

## Phase 8: Campaign Orchestration — DONE

### Goal

Support longer-running research campaigns rather than isolated runs.

### Status

Complete campaign infrastructure across 6 new files:

- `src/api/campaign_models.py` (40 LOC) — CampaignRecord, CampaignRunLink, CheckpointRule dataclasses
- `src/api/campaign_store.py` (179 LOC) — SQLite persistence for campaigns, run links, checkpoint rules (separate from RunStore to respect 300 LOC limit)
- `src/api/campaign_service.py` (149 LOC) — CampaignService: create_campaign, add_branch, get_campaign_summary, check_triggers, update_status
- `src/api/campaign_schemas.py` (50 LOC) — Pydantic request/response models
- `src/api/campaign_router.py` (92 LOC) — FastAPI routes registered via create_campaign_router factory
- `src/ui/templates/ui/campaigns.html` (109 LOC) — Campaign management page with sidebar list, create form, and detail view

Endpoints:

- `POST /campaigns` — create campaign (also creates and links primary run)
- `GET /campaigns` — list all campaigns
- `GET /campaigns/{id}` — campaign detail with run list
- `GET /campaigns/{id}/summary` — aggregated stats (run count, total ticks, run summaries)
- `POST /campaigns/{id}/branch` — branch a run within campaign context
- `POST /campaigns/{id}/rules` — add checkpoint trigger rule
- `POST /campaigns/{id}/checkpoint` — trigger manual checkpoint evaluation

Checkpoint trigger types: `every_n_ticks` (fires at multiples of N), `threshold` (fires once), `manual` (explicit only).

Flask UI: `/campaigns` page with campaign list sidebar, create campaign form (name, goals, JSON config template), campaign detail view with run table linking back to run dashboard.

Tests: `test_campaign_store.py` (7 tests), `test_campaign_service.py` (9 tests), `test_campaign_api.py` (8 tests), plus 2 UI tests in `test_ui.py`.

## Phase 9: Thin Telemetry Frontend — DONE

### Goal

Add a frontend only after the API and event model are stable.

### Status

Flask UI implemented in `src/ui/` (app.py 273 LOC, helpers.py 97 LOC, api_client.py 134 LOC, models.py 146 LOC):

**Visual design:** Constructivist/Suprematist aesthetic with Archivo Black + Instrument Sans + DM Mono typography. Dark theme (#0a0a0a) with red (#c8210a) accent, subtle grid background, sharp 2px border-radius. Tilted red square brand mark.

**Core features:**

- Run creation with JSON config editor (client-side validation + sessionStorage draft preservation)
- AI-assisted step with goal input and lookback window
- Intervention request with auto-apply confirmation dialog
- Manual tick with scenario JSON editor (client-side validation)
- Personality profile visualization (dimension bars with tooltip descriptions)
- Internal state visualization (mood, arousal, energy, satisfaction, frustration)
- Identity drift metrics for self-aware runs (drift, coherence, accuracy bars)
- Emotion tag display with intensity tooltips
- Trajectory sparkline chart (mood/energy/frustration) with hover tooltips and phase timeline markers
- Tabbed data panels: Steps, AI Calls, Interventions (WAI-ARIA compliant with arrow-key navigation)
- Run browser sidebar with recent runs list
- Run actions toolbar: Replay, Branch (from step N with temperature), Export JSON, Compare
- Branch comparison view (`/compare`) with side-by-side trajectory tables
- Campaign management page (`/campaigns`) with create/list/detail views
- API status indicator, run ID display

**UX infrastructure:**

- HTMX boost for smoother navigation (no full-page white flash)
- Animated progress bar during requests (htmx-indicator)
- Client-side JSON validation with inline error messages
- sessionStorage: textarea drafts, collapsible state, active tab preserved across reloads
- Human-readable error messages (maps JSONDecodeError, ConnectionError, TimeoutError)
- `_flash_on_error` decorator for DRY route error handling (extracted to helpers.py)
- Toast notifications with manual dismiss button (6s auto-dismiss)
- Confirmation dialog for destructive actions (auto-apply intervention)
- Keyboard shortcuts: Ctrl+Enter submits focused form, Escape closes dialogs/collapses cards
- Mobile responsive: main content first on small screens

**Typed client models:**

- `src/ui/models.py` — 12 TypedDict definitions: RunListItem, RunSummary, TrajectoryData, TickData, AgentInvocation, InterventionDecision, PhaseAnnotation, StateSnapshot, EmotionReading, LatestTick, RunConfig, ReplayResult, BranchResult
- `api_client.py` uses typed return annotations instead of raw `dict[str, Any]`

**Accessibility:**

- Skip-to-content link
- ARIA landmarks: banner, main, complementary
- WAI-ARIA tabs with role="tab/tablist/tabpanel", aria-selected, arrow-key navigation
- role="progressbar" on state bars with aria-valuenow/min/max
- aria-current="page" on active run and active campaign
- focus-visible outlines on all interactive elements
- WCAG AA contrast (--text-3 bumped to #8a8580 for 5.2:1 ratio)
- SVG sparkline with `<title>` and role="img"

Tests: `test_ui.py` (21 tests) covers rendering, run view, run browser, accessibility landmarks, ARIA tabs, JSON validation, sparkline with phase markers, identity drift metrics, action toolbar, replay/branch/export routes, compare view, campaign pages, HTMX boost, typed models.

## Phase 10: Full Orchestrator — NOT STARTED

### Goal

Only after the previous phases are stable, add a manager-style orchestrator.

### Plan

#### 10.1 — Orchestration Loop Design

Use an internal loop (not a graph framework) to keep dependencies minimal. The `MetaController` runs a decide-act-observe cycle:

1. **Decide**: examine latest tick + trajectory window + campaign goals → choose next action type
2. **Act**: dispatch to the appropriate agent (ScenarioAgent, ObserverAgent, AffectAnalystAgent, InterventionAgent)
3. **Observe**: read the result, update internal planner state, persist the decision
4. **Checkpoint**: evaluate campaign triggers, run analysis if needed
5. **Gate**: check termination conditions (max ticks, user pause, intervention-recommended terminate)

All decisions persisted as `orchestrator_decision` records (new table).

#### 10.2 — Agent Registry

New `src/orchestrator/agents.py`:

- `ScenarioAgent` — wraps `AgentRuntime.propose_scenario`, adds goal-directed scenario selection
- `ObserverAgent` — wraps `AgentRuntime.summarize_window`, tracks narrative continuity
- `AffectAnalystAgent` — wraps `AgentRuntime.analyze_window`, detects regime shifts
- `InterventionAgent` — wraps `AgentRuntime.recommend_intervention`, enforces action constraints

Each agent is a stateless callable: `(context: OrchestrationContext) -> AgentResult`. No agent holds state between invocations — all state flows through persistence.

#### 10.3 — Human Checkpoint Support

Orchestrator pauses at configurable points and waits for user input:

- After N ticks (configurable)
- When intervention recommends "pause"
- When affect analysis detects anomaly
- On explicit user request

Paused state persisted in `simulation_run.status = 'paused'`. Resume via `POST /runs/{run_id}/resume` with optional goal update.

#### 10.4 — Orchestration Persistence

New `orchestrator_decision` table:

- `decision_id`, `run_id`, `campaign_id` (nullable)
- `cycle` (loop iteration), `action_type` (scenario/observe/analyze/intervene/pause/terminate)
- `input_json`, `output_json`, `rationale`
- `created_at`

Every orchestrator decision is reconstructible from this table — no hidden state in prompt history.

#### 10.5 — Orchestration API

- `POST /runs/{run_id}/orchestrate` — run N orchestration cycles (default 1)
- `POST /runs/{run_id}/orchestrate/auto` — run until termination condition
- `GET /runs/{run_id}/orchestrator-log` — list all orchestrator decisions
- `POST /runs/{run_id}/resume` — resume from paused state with optional new goals

#### 10.6 — Orchestration UI

Add orchestrator controls to Flask UI:

- "Auto-run" button with cycle count input
- Orchestrator decision log (new tab in data card)
- Pause/resume controls
- Visual indicator when orchestrator is running vs. paused

### Deliverables

- `src/orchestrator/` package: MetaController, agent registry, decision persistence
- orchestrator_decision SQLite table
- orchestration API routes
- orchestrator UI controls
- human checkpoint support with pause/resume

### Acceptance Criteria

- all orchestration behavior is reconstructible from persistence
- no hidden state lives only in prompt history
- human can pause, inspect, and resume at any point
- orchestrator respects campaign goals and intervention constraints
- each cycle produces exactly one persisted decision record

## Cross-Cutting Work

### Testing — DONE

- unit tests for schema validation: Pydantic models tested across all API tests
- repository integration tests: test_api_runs.py, test_api_agents.py exercise persistence
- deterministic replay tests: test_api_runs.py validates trajectory equivalence
- mocked Anthropic adapter tests: test_llm.py with FakeAnthropicClient, FakeOpenAIClient
- API contract tests: test_api_basic.py, test_api_runs.py, test_api_list_runs.py, test_api_agents.py
- UI integration tests: test_ui.py with FakeUiClient (rendering, routes, accessibility, HTMX)
- e2e test with real Anthropic: test_e2e.py (requires credentials)
- 174 tests total, 91% coverage

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
- client-side input validation (JSON syntax check before submit)
- confirmation dialogs for destructive actions (auto-apply intervention)
- human-readable error messages (no raw exceptions shown to users)

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
| 8 — Campaign Orchestration | DONE |
| 9 — Thin Telemetry Frontend | DONE |
| 10 — Full Orchestrator | NOT STARTED |

## Recommended Next Priorities

1. **Phase 10**: Full orchestrator with MetaController, agent registry, human checkpoints, and pause/resume — the only remaining phase.

## Explicit Non-Goals For Now

- no frontend-local simulation engine
- no direct LLM ownership of memory or state
- no peer-to-peer freeform agent mesh before persistence exists
- no hidden state transitions outside the tick pipeline

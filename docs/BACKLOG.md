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
- 144 tests, 89.5% coverage

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

## Phase 8: Campaign Orchestration — NOT STARTED

### Goal

Support longer-running research campaigns rather than isolated runs.

### Plan

#### 8.1 — Campaign Domain Model

Add `CampaignRecord` dataclass and `campaign` SQLite table:

- `campaign_id` (UUID), `name` (user-facing label), `status` (active/paused/complete)
- `goals` (JSON list of strings), `config_template` (default RunConfig for child runs)
- `created_at`, `updated_at`

Add `campaign_run` join table linking campaign_id to run_id with role (primary, branch, replay).

Extend `RunStore` with campaign CRUD: `create_campaign`, `get_campaign`, `list_campaigns`, `add_run_to_campaign`.

#### 8.2 — Campaign Service

New `CampaignService` in `src/api/campaign_service.py`:

- `create_campaign(name, goals, config_template)` — creates campaign + first run
- `add_branch(campaign_id, source_run_id, tick, patch)` — branches within campaign context
- `get_campaign_summary(campaign_id)` — aggregates stats across all runs
- `get_campaign_trajectory(campaign_id)` — cross-run trajectory for comparison

#### 8.3 — Scheduled Analysis Checkpoints

Add `checkpoint_rule` table: campaign_id, trigger_type (every_n_ticks | threshold | manual), trigger_config (JSON), last_fired_at.

`CampaignService.check_triggers(campaign_id)` evaluates rules after each tick:

- `every_n_ticks`: fires analyze_window every N steps
- `threshold`: fires when a metric (mood, frustration, energy) crosses a boundary
- `manual`: fires on explicit user request

Each checkpoint produces an `AnalysisReport` persisted as an agent_invocation.

#### 8.4 — Campaign API Routes

New `campaign_router.py`:

- `POST /campaigns` — create campaign
- `GET /campaigns` — list campaigns
- `GET /campaigns/{id}` — summary with run tree
- `POST /campaigns/{id}/branch` — branch within campaign
- `POST /campaigns/{id}/checkpoint` — trigger manual checkpoint
- `GET /campaigns/{id}/analysis` — aggregated analysis across runs

#### 8.5 — Campaign UI Panel

Add campaign sidebar section to Flask UI: campaign list, create campaign form, campaign detail view showing run tree with branch lineage.

### Deliverables

- campaign domain model + SQLite tables
- CampaignService with checkpoint evaluation
- campaign API routes
- campaign UI panel
- cross-run analytics queries

### Acceptance Criteria

- one campaign can coordinate multiple runs and branches
- campaign-level analysis is reproducible from stored artifacts
- checkpoint triggers fire automatically and persist results
- UI shows campaign run tree with branch lineage

## Phase 9: Thin Telemetry Frontend — PARTIAL

### Goal

Add a frontend only after the API and event model are stable.

### Status

Flask UI implemented in `src/ui/` (app.py 274 LOC, api_client.py 104 LOC):

**Visual design:** Constructivist/Suprematist aesthetic with Archivo Black + Instrument Sans + DM Mono typography. Dark theme (#0a0a0a) with red (#c8210a) accent, subtle grid background, sharp 2px border-radius. Tilted red square brand mark.

**Core features:**

- Run creation with JSON config editor (client-side validation + sessionStorage draft preservation)
- AI-assisted step with goal input and lookback window
- Intervention request with auto-apply confirmation dialog
- Manual tick with scenario JSON editor (client-side validation)
- Personality profile visualization (dimension bars with tooltip descriptions)
- Internal state visualization (mood, arousal, energy, satisfaction, frustration)
- Emotion tag display with intensity tooltips
- Trajectory sparkline chart (mood/energy/frustration) with hover tooltips per step
- Tabbed data panels: Steps, AI Calls, Interventions (WAI-ARIA compliant with arrow-key navigation)
- Run browser sidebar with recent runs list
- Run actions toolbar: Replay, Branch (from step N with temperature), Export JSON
- API status indicator, run ID display

**UX infrastructure:**

- HTMX boost for smoother navigation (no full-page white flash)
- Animated progress bar during requests (htmx-indicator)
- Client-side JSON validation with inline error messages
- sessionStorage: textarea drafts, collapsible state, active tab preserved across reloads
- Human-readable error messages (maps JSONDecodeError, ConnectionError, TimeoutError)
- `_flash_on_error` decorator for DRY route error handling
- Toast notifications with manual dismiss button (6s auto-dismiss)
- Confirmation dialog for destructive actions (auto-apply intervention)
- Keyboard shortcuts: Ctrl+Enter submits focused form, Escape closes dialogs/collapses cards
- Mobile responsive: main content first on small screens

**Accessibility:**

- Skip-to-content link
- ARIA landmarks: banner, main, complementary
- WAI-ARIA tabs with role="tab/tablist/tabpanel", aria-selected, arrow-key navigation
- role="progressbar" on state bars with aria-valuenow/min/max
- aria-current="page" on active run
- focus-visible outlines on all interactive elements
- WCAG AA contrast (--text-3 bumped to #8a8580 for 5.2:1 ratio)
- SVG sparkline with `<title>` and role="img"

Tests: `test_ui.py` (15 tests) covers rendering, run view, run browser, accessibility landmarks, ARIA tabs, JSON validation attributes, sparkline rendering, action toolbar, replay/branch/export routes, HTMX boost.

### Remaining

- Phase timeline markers on trajectory view
- Identity drift visualization for self-aware runs
- Branch comparison view (side-by-side trajectories)
- Typed frontend client models (currently raw `dict[str, Any]` in api_client.py)

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
- 144 tests total, 89.5% coverage

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
| 8 — Campaign Orchestration | NOT STARTED |
| 9 — Thin Telemetry Frontend | PARTIAL |
| 10 — Full Orchestrator | NOT STARTED |

## Recommended Next Priorities

1. **Phase 9 completion**: Add phase timeline markers, identity drift visualization, and branch comparison view. Replace raw dict client models with typed Pydantic/TypedDict.
2. **Phase 8**: Introduce campaign model and cross-run analytics — the infrastructure (runs, branches, persistence, UI replay/branch controls) is ready.
3. **Phase 10**: Full orchestrator with MetaController and human checkpoints — deferred until campaign model exists.

## Explicit Non-Goals For Now

- no frontend-local simulation engine
- no direct LLM ownership of memory or state
- no peer-to-peer freeform agent mesh before persistence exists
- no hidden state transitions outside the tick pipeline

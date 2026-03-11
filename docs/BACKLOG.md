---
disclaimer: "No information in this document should be taken for granted. Any statement not backed by executable code, tests, or a verifiable reference may be incomplete, invalid, or hallucinated."
---

# Backlog

This backlog turns the current architecture direction into an execution sequence for the repository as it exists today:

- packaged Python backend
- public SDK in `backend/src/sdk`
- FastAPI transport in `backend/src/api`
- no real frontend yet
- no persistent simulation run store yet
- no Anthropic-powered orchestration layer yet

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

## Phase 0: Baseline Consolidation

### Goal

Make the current backend/API layer the explicit foundation for all future work.

### Tasks

- Document the current source-of-truth boundary:
  - `backend/src/personality`
  - `backend/src/temporal`
  - `backend/src/self_model`
  - `backend/src/sdk`
  - `backend/src/api`
- Freeze the public API surface for the next phase:
  - SDK remains internal orchestration convenience
  - FastAPI becomes the external service boundary
- Decide naming conventions for:
  - simulation run
  - tick event
  - trajectory window
  - phase annotation
  - intervention

### Deliverables

- updated architecture note in docs
- explicit API ownership note in backend README

### Acceptance Criteria

- there is one documented answer to “what is the authoritative state container?”
- there is one documented answer to “what can an LLM change?”

## Phase 1: Stateful Run API

### Goal

Replace stateless single-shot simulation endpoints with a real run lifecycle.

### Tasks

- Add run-scoped API routes:
  - `POST /runs`
  - `GET /runs/{run_id}`
  - `POST /runs/{run_id}/tick`
  - `GET /runs/{run_id}/trajectory`
  - `PATCH /runs/{run_id}/params`
  - `POST /runs/{run_id}/phases`
- Introduce request/response schemas:
  - `RunConfig`
  - `ScenarioInput`
  - `TickOutput`
  - `TrajectoryWindow`
  - `PhaseAnnotation`
  - `ParameterPatch`
- Add an application service layer above the simulator:
  - `RunService`
  - `RunRepository` interface
  - `TickEventRepository` interface
- Support both temporal and self-aware runs in one contract.

### Deliverables

- new API router/module for run lifecycle
- typed schemas for run creation and tick execution
- tests for create-run, step-run, fetch-trajectory

### Acceptance Criteria

- a run can be created once and stepped many times
- run state survives between API calls
- the API does not require callers to reconstruct simulator state on every tick

## Phase 2: Persistent Storage

### Goal

Persist run and tick state outside process memory.

### Tasks

- Choose initial storage:
  - SQLite for local/dev
  - Postgres-ready repository boundary for later
- Add persistence tables/models:
  - `simulation_run`
  - `tick_event`
  - `phase_annotation`
  - `intervention_decision`
- Persist at minimum:
  - run config
  - current state snapshot
  - current `psi_hat` if self-aware
  - tick-by-tick event payloads
- Define serialization format for arrays and dict-like vectors.
- Add repository tests covering create/load/append/list operations.

### Data To Persist Per Tick

- tick index
- scenario payload
- chosen action
- action probabilities
- outcome
- state before
- state after
- activations
- emotions
- self-emotions
- `psi_hat`
- behavioral evidence
- self-coherence
- self-accuracy
- identity drift
- prediction error

### Deliverables

- persistence module under backend source tree
- migration/bootstrap script
- repository integration tests

### Acceptance Criteria

- a run can be resumed after process restart
- a full trajectory can be reconstructed from storage alone
- no LLM state is required to recover run truth

## Phase 3: Canonical Tick Event Contract

### Goal

Make tick events the canonical substrate for analytics, orchestration, and UI.

### Tasks

- Normalize `TickOutput` so every downstream consumer sees the same payload.
- Add explicit event metadata:
  - `run_id`
  - timestamp
  - tick
  - event version
- Add derived fields where justified:
  - action effort
  - energy drain
  - stress estimate
  - phase labels attached to the tick window, not baked into core truth
- Keep interpretation fields clearly separated from deterministic engine outputs.

### Deliverables

- event schema versioning strategy
- serializer/deserializer for tick event storage
- contract tests for API payload compatibility

### Acceptance Criteria

- the UI, analytics jobs, and Anthropic agents can all consume the same event shape
- event payloads are replayable and versioned

## Phase 4: Replay And Branching

### Goal

Make experiments reproducible and branchable.

### Tasks

- Add replay endpoint:
  - `POST /runs/{run_id}/replay`
- Add branch endpoint:
  - `POST /runs/{run_id}/branches`
- Allow branch creation from:
  - tick N
  - alternate future scenarios
  - alternate parameter patches
- Record lineage:
  - parent run id
  - parent tick
  - branch reason

### Deliverables

- replay service
- branch lineage model
- tests for deterministic replay consistency

### Acceptance Criteria

- the same run config plus the same scenarios reproduces the same tick sequence when outcomes are deterministic or seeded
- branch provenance is queryable

## Phase 5: Anthropic Integration Boundary

### Goal

Introduce Anthropic as a typed orchestration dependency, not as a state holder.

### Tasks

- Add an `llm` package or module boundary:
  - `AnthropicClientFactory`
  - `PromptRenderer`
  - `StructuredOutputValidator`
  - `AgentRuntime`
- Use the Anthropic Python library behind a small adapter so model selection is centralized.
- Define typed outputs for each LLM role:
  - `ScenarioProposal`
  - `NarrativeChunk`
  - `AnalysisReport`
  - `InterventionRecommendation`
- Validate all model outputs before use.
- Persist:
  - raw prompt input
  - raw model output
  - parsed structured output
  - parse/validation failures

### Anthropic-Specific Notes

- Prefer one adapter layer instead of sprinkling Anthropic client calls through business code.
- Keep model identifiers configurable.
- Separate prompt templates from runtime policy.
- Record token/cost metadata if the SDK exposes it.
- Treat retries and fallback behavior as infrastructure, not domain logic.

### Deliverables

- Anthropic adapter module
- structured-output parsing flow
- persistence for agent messages and tool decisions

### Acceptance Criteria

- no domain module imports Anthropic directly
- every model-produced action is validated before execution
- failed structured outputs are observable and testable

## Phase 6: Minimum Agent Runtime

### Goal

Ship the smallest useful agent-assisted loop.

### Initial Agent Set

- `ScenarioAgent`
- `ObserverAgent`

### Tasks

- Implement `ScenarioAgent`:
  - input: recent trajectory, user goals, current run state
  - output: validated `ScenarioProposal`
- Implement `ObserverAgent`:
  - input: recent tick window
  - output: validated summary/narrative
- Add orchestration service method:
  - load recent state
  - ask scenario agent for next scenario
  - execute tick
  - persist tick
  - ask observer for summary
  - persist observer output

### Deliverables

- CLI or API endpoint to run one assisted step
- tests with mocked Anthropic adapter

### Acceptance Criteria

- the system can perform one end-to-end assisted tick without manual scenario construction
- the simulator remains authoritative for the resulting state

## Phase 7: Analysis And Intervention Agents

### Goal

Add higher-order interpretation and control, still behind explicit boundaries.

### Agents

- `AffectAnalystAgent`
- `InterventionAgent`

### Tasks

- Implement analysis reports from trajectory windows.
- Implement intervention recommendations:
  - continue
  - probe
  - narrate
  - analyze
  - patch params
  - pause
  - terminate
- Require all intervention outputs to become explicit API calls or internal service actions, never direct state mutation.

### Deliverables

- intervention schema
- analysis schema
- policy tests for allowed vs forbidden actions

### Acceptance Criteria

- interventions are logged as explicit decisions
- interventions are replayable as part of a campaign history

## Phase 8: Campaign Orchestration

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

## Phase 9: Thin Telemetry Frontend

### Goal

Add a frontend only after the API and event model are stable.

### Tasks

- Build frontend as pure API client.
- Do not embed simulator math in the browser.
- Minimum features:
  - run creation
  - tick stepping
  - trajectory charts
  - event table
  - narrative panel
  - phase markers
- Define frontend contracts from API schemas, not ad hoc objects.

### Deliverables

- frontend app consuming FastAPI endpoints
- typed client models

### Acceptance Criteria

- the frontend can be deleted and rebuilt without changing simulator truth
- the backend remains fully usable without the frontend

## Phase 10: Full Orchestrator

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

### Testing

- unit tests for schema validation
- repository integration tests
- deterministic replay tests
- mocked Anthropic adapter tests
- API contract tests

### Observability

- structured logs for run lifecycle
- structured logs for agent invocations
- correlation ids:
  - campaign id
  - run id
  - tick id
  - agent invocation id

### Security And Safety

- validate all incoming scenario values
- validate all model-produced outputs
- prevent direct mutation of persisted state from orchestration components
- keep intervention permissions explicit

### Configuration

- Anthropic API key via environment
- model names configurable by environment
- storage backend configurable by environment
- replay seed behavior documented and testable

## Recommended Execution Order

1. Phase 1: Stateful Run API
2. Phase 2: Persistent Storage
3. Phase 3: Canonical Tick Event Contract
4. Phase 4: Replay And Branching
5. Phase 5: Anthropic Integration Boundary
6. Phase 6: Minimum Agent Runtime
7. Phase 7: Analysis And Intervention Agents
8. Phase 8: Campaign Orchestration
9. Phase 9: Thin Telemetry Frontend
10. Phase 10: Full Orchestrator

## Explicit Non-Goals For Now

- no frontend-local simulation engine
- no direct LLM ownership of memory or state
- no peer-to-peer freeform agent mesh before persistence exists
- no hidden state transitions outside the tick pipeline

## Immediate Next Sprint

If only one sprint is funded, do this:

1. Implement stateful run lifecycle endpoints.
2. Persist runs and tick events in SQLite.
3. Add canonical `TickOutput` persistence and retrieval.
4. Add replay-from-run support.
5. Add Anthropic adapter interface with mocked tests only.

That yields the first architecture milestone where the simulator is a real service, trajectories are durable, and Anthropic integration can be added safely without corrupting state ownership.

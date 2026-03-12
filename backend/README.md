---
disclaimer: "No information in this document should be taken for granted. Any statement not backed by executable code, tests, or a verifiable reference may be incomplete, invalid, or hallucinated."
---

# Red Iron Square Backend

`red-iron-square` is a Python simulation library for personality-driven agents. The current public surface is the SDK under [src/sdk](/home/admin/spikes/red-iron-square/backend/src/sdk).

## Current State

The backend now ships with:
- a stable SDK facade via `src.sdk.AgentSDK`
- a packaged CLI via `red-iron-square`
- a thin FastAPI transport under `src.api`
- campaign endpoints under `src.api.campaign_*`
- orchestrator endpoints under `src.orchestrator`
- a Two Minds demo transport under `src.demo`
- a Flask + Jinja2 UI under `src.ui`
- runnable JSON examples under [backend/examples](/home/admin/spikes/red-iron-square/backend/examples)

## Public API Boundary

Use `src.sdk.AgentSDK` as the stable entrypoint for external consumers.

Public modules:
- `src.sdk`
- `src.sdk.cli`
- `src.api`
- `src.demo`
- `src.orchestrator`
- `src.ui`

Internal modules:
- `src.personality`
- `src.temporal`
- `src.precision`
- `src.efe`
- `src.constructed_emotion`
- `src.self_evidencing`
- `src.self_model`
- `src.shared`

The internal modules remain the domain implementation. They are available for advanced use, but they should be treated as unstable compared to the SDK facade.

## Install And Run

From [backend](/home/admin/spikes/red-iron-square/backend):

```bash
uv sync
uv run pytest -q
uv run red-iron-square --help
```

To run the HTTP transport:

```bash
uv sync --extra api
uv run uvicorn src.api.app:create_app --factory --reload
```

LLM-backed endpoints use `anthropic` by default. To switch to OpenAI, set:

```bash
export RED_IRON_SQUARE_LLM_PROVIDER=openai
export OPENAI_API_KEY=...
```

Anthropic remains available with:

```bash
export RED_IRON_SQUARE_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=...
```

For ElevenLabs voice synthesis in the Two Minds demo:

```bash
export ELEVENLABS_API_KEY=...
```

See [backend/.env.example](/home/admin/spikes/red-iron-square/backend/.env.example) for all supported environment variables.

To run the Flask UI:

```bash
uv sync --extra ui
uv run red-iron-square-ui
```

To run both together:

```bash
uv sync --extra api --extra ui
uv run uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000
uv run red-iron-square-ui
```

To run the Two Minds demo frontend against the API:

```bash
cd ../frontend
npm test -- --runInBand
npm run dev
```

In development the Vite app proxies `/demo` websocket and HTTP traffic to `http://127.0.0.1:8000`.

## Python SDK Example

```python
from src.sdk import AgentSDK

sdk = AgentSDK.default()

personality = sdk.personality({
    "O": 0.8, "C": 0.5, "E": 0.3, "A": 0.7,
    "N": 0.4, "R": 0.9, "I": 0.6, "T": 0.2,
})

scenario = sdk.scenario(
    {"O": 0.9, "N": 0.7},
    name="pitch_meeting",
)

actions = [
    sdk.action("bold", {"O": 1.0, "R": 0.8, "N": -0.3}),
    sdk.action("safe", {"C": 0.9, "T": 0.8}),
]

decision = sdk.decide(personality, scenario, actions)
print(decision.model_dump())
```

## Temporal Simulation Example

```python
simulator = sdk.simulator(personality, actions)
trace = simulator.run(
    [scenario],
    outcomes=[0.4],
)
print(trace.model_dump())
```

## Self-Aware Simulation Example

```python
self_aware = sdk.self_aware_simulator(
    personality,
    sdk.initial_self_model({
        "O": 0.7, "C": 0.5, "E": 0.4, "A": 0.6,
        "N": 0.4, "R": 0.8, "I": 0.6, "T": 0.3,
    }),
    actions,
)
trace = self_aware.run([scenario], outcomes=[0.6])
print(trace.model_dump())
```

## CLI Example

One-shot decision:

```bash
uv run red-iron-square decide \
  --personality '{"O":0.8,"C":0.5,"E":0.3,"A":0.7,"N":0.4,"R":0.9,"I":0.6,"T":0.2}' \
  --scenario '{"name":"pitch_meeting","values":{"O":0.9,"N":0.7}}' \
  --actions '[{"name":"bold","modifiers":{"O":1.0,"R":0.8,"N":-0.3}},{"name":"safe","modifiers":{"C":0.9,"T":0.8}}]'
```

Simulation trace:

```bash
uv run red-iron-square simulate \
  --personality @examples/personality.json \
  --actions @examples/actions.json \
  --scenarios @examples/scenarios.json \
  --outcomes '[0.4, 0.2, -0.1]'
```

Self-aware simulation trace:

```bash
uv run red-iron-square simulate \
  --personality @examples/personality.json \
  --actions @examples/actions.json \
  --scenarios @examples/scenarios.json \
  --outcomes '[0.4, 0.2, -0.1]' \
  --self-model @examples/self_model.json
```

The example payloads above are included in [backend/examples](/home/admin/spikes/red-iron-square/backend/examples).

## HTTP API

The FastAPI transport lives under [backend/src/api](/home/admin/spikes/red-iron-square/backend/src/api). It is intentionally thin and delegates all business logic to `AgentSDK`.

Available endpoints:
- `GET /health`
- `POST /decide`
- `POST /simulate`
- `POST /runs`
- `GET /runs/{run_id}`
- `POST /runs/{run_id}/tick`
- `GET /runs/{run_id}/trajectory`
- `PATCH /runs/{run_id}/params`
- `POST /runs/{run_id}/phases`
- `POST /runs/{run_id}/replay`
- `POST /runs/{run_id}/branches`
- `POST /runs/{run_id}/assist/step`
- `POST /runs/{run_id}/intervention`
- `POST /campaigns`
- `GET /campaigns`
- `GET /campaigns/{campaign_id}`
- `GET /campaigns/{campaign_id}/summary`
- `POST /campaigns/{campaign_id}/branch`
- `POST /campaigns/{campaign_id}/rules`
- `POST /campaigns/{campaign_id}/checkpoint`
- `POST /demo/sessions`
- `GET /demo/sessions/{session_id}`
- `POST /demo/sessions/{session_id}/scripted/{scenario_key}`
- `POST /demo/sessions/{session_id}/scenarios`
- `POST /demo/sessions/{session_id}/swap`
- `WS /demo/sessions/{session_id}/stream`
- `POST /runs/{run_id}/orchestrate`
- `GET /runs/{run_id}/orchestrator-log`
- `POST /runs/{run_id}/resume`

Example health check:

```bash
curl http://127.0.0.1:8000/health
```

Example decision request:

```bash
curl -X POST http://127.0.0.1:8000/decide \
  -H 'content-type: application/json' \
  -d '{
    "personality": {"O":0.8,"C":0.5,"E":0.3,"A":0.7,"N":0.4,"R":0.9,"I":0.6,"T":0.2},
    "scenario": {"name":"pitch_meeting","values":{"O":0.9,"N":0.7}},
    "actions": [
      {"name":"bold","modifiers":{"O":1.0,"R":0.8,"N":-0.3}},
      {"name":"safe","modifiers":{"C":0.9,"T":0.8}}
    ]
  }'
```

Example stateful run flow:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H 'content-type: application/json' \
  -d '{
    "personality": {"O":0.8,"C":0.5,"E":0.3,"A":0.7,"N":0.4,"R":0.9,"I":0.6,"T":0.2},
    "actions": [
      {"name":"bold","modifiers":{"O":1.0,"R":0.8,"N":-0.3}},
      {"name":"safe","modifiers":{"C":0.9,"T":0.8}}
    ],
    "temperature": 1.0,
    "seed": 42
  }'
```

Example campaign creation:

```bash
curl -X POST http://127.0.0.1:8000/campaigns \
  -H 'content-type: application/json' \
  -d '{
    "name": "baseline research",
    "goals": ["explore stable trajectories", "compare branches"],
    "config_template": {
      "personality": {"O":0.8,"C":0.5,"E":0.3,"A":0.7,"N":0.4,"R":0.9,"I":0.6,"T":0.2},
      "actions": [
        {"name":"bold","modifiers":{"O":1.0,"R":0.8,"N":-0.3}},
        {"name":"safe","modifiers":{"C":0.9,"T":0.8}}
      ],
      "temperature": 1.0,
      "seed": 42
    }
  }'
```

## Campaign API

Campaign orchestration lives in
[backend/src/api/campaign_router.py](/home/admin/spikes/red-iron-square/backend/src/api/campaign_router.py),
[backend/src/api/campaign_service.py](/home/admin/spikes/red-iron-square/backend/src/api/campaign_service.py),
and
[backend/src/api/campaign_store.py](/home/admin/spikes/red-iron-square/backend/src/api/campaign_store.py).

It currently provides:
- campaign creation plus automatic primary run creation
- campaign listing and detail retrieval
- campaign summary with run-count and total-tick aggregation
- branching an existing run within a campaign
- checkpoint rule persistence for `every_n_ticks`, `threshold`, and `manual`
- manual checkpoint evaluation via `POST /campaigns/{campaign_id}/checkpoint`

Current limitations:
- checkpoint evaluation currently returns fired rules; it does not yet persist analysis reports
- no separate campaign-specific runbook exists yet beyond the Flask UI

## Orchestrator API

The orchestrator lives under
[backend/src/orchestrator](/home/admin/spikes/red-iron-square/backend/src/orchestrator).

It currently provides:
- one-shot orchestration cycles via `POST /runs/{run_id}/orchestrate`
- multi-cycle auto-run via the same endpoint with `cycles > 1`
- persisted decision history via `GET /runs/{run_id}/orchestrator-log`
- paused-run resume with optional updated goals via `POST /runs/{run_id}/resume`

The Flask UI exposes these controls on the main run dashboard.

## Two Minds Demo API

The demo transport lives under [backend/src/demo](/home/admin/spikes/red-iron-square/backend/src/demo).

It currently provides:
- creation of a session with Luna and Marco
- scripted scenarios for the family-facing showcase
- custom audience scenarios enriched through the shared LLM runtime when configured
- websocket event streaming for state changes and transcript updates
- personality swap/reset behavior

- ElevenLabs audio integration module (`src.demo.audio`) with voice-settings calculator, audio-tag injector, and provider adapter (not yet wired into the service layer)

Current limitations:
- the service layer still emits `audio_unavailable` events; wiring `audio.py` into `service.py` is the next step
- the demo frontend is separate from the Flask UI and only uses the `/demo` API surface

## Flask UI

The Flask UI lives under [backend/src/ui](/home/admin/spikes/red-iron-square/backend/src/ui). It is a thin client over the FastAPI service and does not embed simulator logic.

It provides:
- run creation from JSON config
- manual tick execution
- assisted step execution
- intervention requests
- run summary and latest tick
- persisted call log from agent invocations
- intervention history
- tick trace display
- run replay and branching
- compare view under `/compare`
- campaign management under `/campaigns`
- orchestrator controls and decision log access on the run dashboard

The UI uses Jinja2 templates with custom CSS (constructivist aesthetic, dark/light theme toggle) and expects the API base URL from `RED_IRON_SQUARE_API_URL` if you do not want the default `http://127.0.0.1:8000`.

Backend verification command:

```bash
uv run pytest -q
```

Avoid hard-coding the pass count here; the suite is actively growing.

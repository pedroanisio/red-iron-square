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
- runnable JSON examples under [backend/examples](/home/admin/spikes/red-iron-square/backend/examples)

## Public API Boundary

Use `src.sdk.AgentSDK` as the stable entrypoint for external consumers.

Public modules:
- `src.sdk`
- `src.sdk.cli`
- `src.api`

Internal modules:
- `src.personality`
- `src.temporal`
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
uv run uvicorn src.api.app:create_app --factory --reload
```

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

The current backend test status is `108 passed` under `uv run pytest -q`.

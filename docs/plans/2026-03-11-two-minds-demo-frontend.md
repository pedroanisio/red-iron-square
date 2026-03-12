---
disclaimer: >
  No information in this document should be taken for granted.
  Any statement or premise not backed by a real logical definition
  or verifiable reference may be invalid, erroneous, or a hallucination.
title: "Two Minds Demo Frontend Implementation Plan"
date: "2026-03-11"
status: "Draft"
---

# Two Minds Demo Frontend Implementation Plan

> **Execution note:** Use the local `executing-plans` skill to implement this plan task-by-task.

**Goal:** Build a second frontend for the family-facing Two Minds demo: a React + TypeScript app in `frontend/` that talks to a demo-specific FastAPI/WebSocket surface without replacing the existing Flask UI.

**Architecture:** Keep the current backend simulation stack as the source of truth. Add a thin demo orchestration layer in FastAPI that manages Luna/Marco session state, scenario enrichment, narrative/audio events, and swap/reset behavior. Build a separate Vite-based React app in `frontend/` on Node `24` from [`frontend/.nvmrc`](/home/admin/spikes/red-iron-square/frontend/.nvmrc) and drive it over HTTP + WebSocket.

**Tech Stack:** Node 24, npm, Vite, React, TypeScript strict, Vitest, Testing Library, Playwright, FastAPI WebSockets, existing `src/llm`, existing simulator/orchestrator modules.

**LOC Budget:** Keep every new source file under ~300 LOC. Favor small modules: separate demo router/service/session/audio/frontend components instead of one large controller.

---

## Assumptions To Lock Before Execution

1. The React app lives directly in `frontend/` and is the "second frontend"; the existing Flask UI in `backend/src/ui` remains unchanged.
2. Package management uses `npm`, because the repo already has a root `package-lock.json` and no `pnpm`/`yarn` workspace metadata.
3. The first increment ships with text-first graceful degradation: demo telemetry and narrative text must work even when Anthropic or ElevenLabs are unavailable.
4. Browser-native microphone input is optional in the first delivery; typed scenarios and scripted presets are mandatory.

## Task 1: Scaffold The React Frontend

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/package-lock.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/styles/tokens.css`
- Create: `frontend/src/styles/app.css`
- Create: `frontend/src/vite-env.d.ts`
- Create: `frontend/src/test/setup.ts`

**Step 1: Write the failing bootstrap test**

Create a minimal component/render test for `App` that asserts the page shows the demo title and two agent regions.

**Step 2: Run test to verify it fails**

Run: `cd frontend && nvm use && npm test -- --runInBand`

Expected: FAIL because the React app and test runner do not exist yet.

**Step 3: Scaffold the Vite app with strict TypeScript**

Use Node `24` from [`frontend/.nvmrc`](/home/admin/spikes/red-iron-square/frontend/.nvmrc), add React/Vitest/Testing Library scripts, and create a minimal shell app with strict compiler settings.

**Step 4: Run test to verify it passes**

Run: `cd frontend && nvm use && npm test -- --runInBand`

Expected: PASS for the bootstrap render test.

**Step 5: Verify frontend build**

Run: `cd frontend && nvm use && npm run build`

Expected: production build succeeds.

**Step 6: Commit**

```bash
git add frontend
git commit -m "feat: scaffold two minds demo frontend"
```

## Task 2: Define Demo Contracts In The Backend

**Files:**
- Create: `backend/src/demo/__init__.py`
- Create: `backend/src/demo/models.py`
- Create: `backend/src/demo/schemas.py`
- Create: `backend/src/demo/personas.py`
- Create: `backend/tests/test_demo_models.py`
- Modify: `backend/src/api/app.py`

**Step 1: Write the failing model/schema tests**

Add tests for:
- the Luna and Marco default profiles
- scripted scenarios from the guideline
- swap/reset semantics
- Pydantic request/response schemas for session create, scripted step, custom scenario, and swap

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest -x -q tests/test_demo_models.py`

Expected: FAIL with missing `src.demo` module.

**Step 3: Add demo domain contracts**

Create small, typed models for:
- `DemoPersona`
- `DemoScenario`
- `DemoSessionState`
- `DemoAgentSnapshot`
- `DemoEvent`

Also codify:
- Luna and Marco trait vectors from the guideline
- the three scripted scenarios and their forced outcomes
- display-label mapping for family-facing emotions

**Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest -x -q tests/test_demo_models.py`

Expected: PASS.

**Step 5: Lint and type check**

Run: `cd backend && uv run ruff check src tests && uv run mypy src`

Expected: No errors.

**Step 6: Commit**

```bash
git add backend/src/demo backend/tests/test_demo_models.py backend/src/api/app.py
git commit -m "feat: add two minds demo contracts"
```

## Task 3: Add Demo Session Service And WebSocket Event Flow

**Files:**
- Create: `backend/src/demo/service.py`
- Create: `backend/src/demo/router.py`
- Create: `backend/src/demo/state_mapper.py`
- Create: `backend/src/demo/session_store.py`
- Create: `backend/tests/test_demo_service.py`
- Create: `backend/tests/test_demo_api.py`
- Modify: `backend/src/api/app.py`

**Step 1: Write the failing backend tests**

Cover:
- creating a fresh demo session with two agents
- running a scripted scenario against both agents
- persisting accumulated state between turns
- swapping personalities and resetting state cleanly
- emitting WebSocket-safe events in the expected order

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest -x -q tests/test_demo_service.py tests/test_demo_api.py`

Expected: FAIL because the service/router do not exist.

**Step 3: Implement the minimal service and router**

Expose:
- `POST /demo/sessions`
- `POST /demo/sessions/{session_id}/scripted/{scenario_key}`
- `POST /demo/sessions/{session_id}/scenarios`
- `POST /demo/sessions/{session_id}/swap`
- `GET /demo/sessions/{session_id}`
- `WS /demo/sessions/{session_id}/stream`

Implementation notes:
- Reuse existing simulator/SDK primitives rather than the legacy Flask UI.
- Keep session state in a dedicated demo store object first; only persist to SQLite if needed later.
- Broadcast structured events such as `session_initialized`, `scenario_received`, `agent_state_updated`, `agent_text_started`, `agent_text_completed`, `audio_started`, `audio_unavailable`, `turn_completed`, and `swap_completed`.
- Build the API so text-only mode is a first-class fallback.

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest -x -q tests/test_demo_service.py tests/test_demo_api.py`

Expected: PASS.

**Step 5: Lint and type check**

Run: `cd backend && uv run ruff check src tests && uv run mypy src`

Expected: No errors.

**Step 6: Commit**

```bash
git add backend/src/demo backend/tests/test_demo_service.py backend/tests/test_demo_api.py backend/src/api/app.py
git commit -m "feat: add two minds demo session api"
```

## Task 4: Integrate Scenario Enrichment And Narrative Generation

**Files:**
- Create: `backend/src/demo/llm_service.py`
- Create: `backend/tests/test_demo_llm_service.py`
- Modify: `backend/src/demo/service.py`

**Step 1: Write the failing LLM-integration tests**

Cover:
- free-text scenario enrichment into a validated `DemoScenario`
- fallback to neutral activations on invalid LLM output
- narrative generation per agent using recent conversation context
- heuristic display emotion fallback when the LLM is unavailable or inconsistent

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest -x -q tests/test_demo_llm_service.py`

Expected: FAIL because the adapter/service does not exist.

**Step 3: Implement the minimal LLM adapter**

Use the existing `AgentRuntime` boundary to avoid direct provider coupling. Keep prompts in small helper functions and return validated payloads only.

Required behavior:
- one enrichment call per audience scenario
- one narrative call per agent turn
- family-safe emotion label selection with valence guardrails
- deterministic fallback path with no network dependency in tests

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest -x -q tests/test_demo_llm_service.py tests/test_demo_service.py`

Expected: PASS.

**Step 5: Lint and type check**

Run: `cd backend && uv run ruff check src tests && uv run mypy src`

Expected: No errors.

**Step 6: Commit**

```bash
git add backend/src/demo/llm_service.py backend/src/demo/service.py backend/tests/test_demo_llm_service.py backend/tests/test_demo_service.py
git commit -m "feat: add llm flow for two minds demo"
```

## Task 5: Add ElevenLabs Audio Streaming With Graceful Degradation

> **Status:** Partially complete. `audio.py` and `test_demo_audio.py` are implemented (26 tests). Remaining: wire `audio.py` into `service.py` to replace `audio_unavailable` events with real TTS calls.

**Files:**
- Create: `backend/src/demo/audio.py` — **DONE**
- Create: `backend/tests/test_demo_audio.py` — **DONE**
- Modify: `backend/src/demo/service.py` — pending
- Modify: `backend/src/demo/schemas.py` — pending

**Step 1: Write the failing audio tests**

Cover:
- per-agent voice settings calculation from mood/energy/arousal
- model selection (`eleven_v3` for scripted/swap acts, `eleven_flash_v2_5` for open-floor mode)
- audio-tag injection and stripping rules
- fallback to text-only events when ElevenLabs fails or times out

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest -x -q tests/test_demo_audio.py`

Expected: FAIL because the audio module does not exist.

**Step 3: Implement the minimal audio adapter**

Keep this module pure and small:
- voice settings calculator
- audio-tag injector/stripper
- provider adapter interface
- timeout/error normalization

Do not make the core demo session service depend on raw ElevenLabs response objects.

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest -x -q tests/test_demo_audio.py tests/test_demo_service.py`

Expected: PASS.

**Step 5: Lint and type check**

Run: `cd backend && uv run ruff check src tests && uv run mypy src`

Expected: No errors.

**Step 6: Commit**

```bash
git add backend/src/demo/audio.py backend/src/demo/service.py backend/src/demo/schemas.py backend/tests/test_demo_audio.py backend/tests/test_demo_service.py
git commit -m "feat: add audio streaming for two minds demo"
```

## Task 6: Build The Demo UI Shell And State Layer

**Files:**
- Create: `frontend/src/demo/types.ts`
- Create: `frontend/src/demo/api.ts`
- Create: `frontend/src/demo/socket.ts`
- Create: `frontend/src/demo/reducer.ts`
- Create: `frontend/src/demo/useDemoSession.ts`
- Create: `frontend/src/components/layout/DemoStage.tsx`
- Create: `frontend/src/components/agents/AgentCard.tsx`
- Create: `frontend/src/components/agents/StateBar.tsx`
- Create: `frontend/src/components/scenario/ScenarioPanel.tsx`
- Create: `frontend/src/components/scenario/PresetButtons.tsx`
- Create: `frontend/src/components/system/StatusBanner.tsx`
- Create: `frontend/src/components/system/ThinkingState.tsx`
- Create: `frontend/src/assets/` placeholder portrait files or documented temp references
- Create: `frontend/src/__tests__/useDemoSession.test.tsx`
- Create: `frontend/src/__tests__/App.test.tsx`

**Step 1: Write the failing frontend behavior tests**

Cover:
- initial session bootstrap
- preset-click flow
- WebSocket event reduction into UI state
- swap button reset behavior
- text-only fallback banner

**Step 2: Run test to verify it fails**

Run: `cd frontend && nvm use && npm test -- --runInBand`

Expected: FAIL because the hooks/components do not exist.

**Step 3: Implement the UI shell**

Use the `frontend-design` skill during execution for the visual work. Required UX:
- two persistent agent cards around a central scenario card
- animated Mood / Energy / Calm bars
- visible personality one-liners
- scenario input, preset buttons, and swap button
- retained transcript text per agent
- distinct loading states for "thinking" vs "speaking"

Keep all API/WebSocket logic out of presentational components.

**Step 4: Run tests to verify they pass**

Run: `cd frontend && nvm use && npm test -- --runInBand`

Expected: PASS.

**Step 5: Verify production build**

Run: `cd frontend && nvm use && npm run build`

Expected: PASS.

**Step 6: Commit**

```bash
git add frontend
git commit -m "feat: build two minds demo interface"
```

## Task 7: Add Audio Playback And Typewriter Synchronization

**Files:**
- Create: `frontend/src/audio/player.ts`
- Create: `frontend/src/hooks/useAudioQueue.ts`
- Create: `frontend/src/hooks/useTypewriter.ts`
- Create: `frontend/src/components/agents/TranscriptPane.tsx`
- Create: `frontend/src/__tests__/audioQueue.test.ts`
- Modify: `frontend/src/demo/reducer.ts`
- Modify: `frontend/src/components/agents/AgentCard.tsx`

**Step 1: Write the failing playback tests**

Cover:
- queued sequential playback
- clean-text display with stripped audio tags
- typewriter progress tied to active response text
- graceful completion when no audio stream is available

**Step 2: Run test to verify it fails**

Run: `cd frontend && nvm use && npm test -- --runInBand`

Expected: FAIL because the playback hooks do not exist.

**Step 3: Implement minimal synchronized playback**

Requirements:
- Luna finishes before Marco starts
- text remains readable after playback ends
- no hard dependency on MSE if blob-based playback is sufficient
- text-only mode still renders as a complete turn

**Step 4: Run tests to verify they pass**

Run: `cd frontend && nvm use && npm test -- --runInBand`

Expected: PASS.

**Step 5: Verify production build**

Run: `cd frontend && nvm use && npm run build`

Expected: PASS.

**Step 6: Commit**

```bash
git add frontend
git commit -m "feat: add demo audio and transcript playback"
```

## Task 8: Wire Dev Commands, E2E Coverage, And Demo Readiness

**Files:**
- Modify: `package.json`
- Modify: `playwright.config.ts`
- Create: `e2e/two-minds-demo.spec.ts`
- Create: `frontend/.env.example`
- Create: `docs/two-minds-demo-runbook.md`
- Optionally create: `scripts/run-two-minds-demo.sh`

**Step 1: Write the failing end-to-end test**

Add one Playwright scenario that:
- starts the React app
- creates a session
- runs the first scripted scenario
- verifies both cards update their bars/text
- exercises the swap button
- verifies the text-only fallback banner when audio is disabled

**Step 2: Run test to verify it fails**

Run: `npx playwright test e2e/two-minds-demo.spec.ts`

Expected: FAIL because the app/scripts are not wired.

**Step 3: Implement the dev/test wiring**

Add root scripts for:
- frontend dev server
- backend API server
- combined demo startup
- frontend unit tests
- demo E2E tests

Document:
- required env vars
- which act uses which TTS model
- fallback behavior
- a short rehearsal checklist before showing the demo

**Step 4: Run the verification suite**

Run: `cd frontend && nvm use && npm test -- --runInBand`

Run: `cd frontend && nvm use && npm run build`

Run: `cd backend && uv run pytest -x -q tests/test_demo_models.py tests/test_demo_service.py tests/test_demo_api.py tests/test_demo_llm_service.py tests/test_demo_audio.py`

Run: `cd backend && uv run ruff check src tests && uv run mypy src`

Run: `npx playwright test e2e/two-minds-demo.spec.ts`

Expected: All pass.

**Step 5: Commit**

```bash
git add package.json playwright.config.ts e2e frontend/.env.example docs/two-minds-demo-runbook.md scripts
git commit -m "feat: finish two minds demo delivery wiring"
```

## Execution Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6
7. Task 7
8. Task 8

## Risks To Watch During Execution

1. Do not merge this demo UI into `backend/src/ui`; that would blur the "second frontend" boundary the request calls for.
2. Keep WebSocket event types stable early, or the React reducer and backend service will churn together.
3. ElevenLabs integration is the highest latency and demo-risk area; preserve a first-class text-only mode throughout.
4. The scripted scenarios should be fixtures, not inline literals spread across multiple modules.
5. Watch file size aggressively in `service.py`, `router.py`, and `App.tsx`; split early to stay under the repo’s LOC constraint.

## Suggested First Execution Batch

1. Complete Task 1 and Task 2 together to establish the frontend scaffold and backend contracts.
2. Stop and review the event schema before starting Task 3, because it is the contract between the two halves.
3. Only start audio work after the text-only experience is already end-to-end functional.

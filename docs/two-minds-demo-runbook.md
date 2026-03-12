---
disclaimer: "No information in this document should be taken for granted. Any statement not backed by executable code, tests, or a verifiable reference may be incomplete, invalid, or hallucinated."
title: "Two Minds Demo Runbook"
status: "Current as of 2026-03-11"
---

# Two Minds Demo Runbook

## Current State

The shipped demo is a separate React frontend in [frontend](/home/admin/codebases/red-iron-square/frontend) backed by the FastAPI demo routes in [backend/src/demo](/home/admin/codebases/red-iron-square/backend/src/demo).

What works now:
- session bootstrap for Luna and Marco
- scripted scenarios
- custom audience scenarios
- websocket-driven state updates
- personality swap
- text-first transcript updates
- ElevenLabs audio module (`backend/src/demo/audio.py`): voice-settings calculator, audio-tag injector/stripper, provider adapter with env-based API key and graceful degradation

What does not ship yet:
- wiring `audio.py` into `service.py` (the service still emits `audio_unavailable` events)
- browser audio playback from a live TTS pipeline
- a dedicated Playwright spec for the demo flow

## Start The Demo

Start the backend API:

```bash
cd /home/admin/codebases/red-iron-square/backend
uv sync --extra api --extra ui
uv run uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Start the frontend:

```bash
cd /home/admin/codebases/red-iron-square/frontend
npm run dev
```

In local development the frontend proxies `/demo` traffic to `http://127.0.0.1:8000`.

## Verification

Frontend:

```bash
cd /home/admin/codebases/red-iron-square/frontend
npm test -- --runInBand
npm run build
```

Backend:

```bash
cd /home/admin/codebases/red-iron-square/backend
uv run pytest -q
```

## Environment

Optional LLM-backed enrichment:

```bash
export RED_IRON_SQUARE_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=...
```

Or:

```bash
export RED_IRON_SQUARE_LLM_PROVIDER=openai
export OPENAI_API_KEY=...
```

Optional ElevenLabs voice synthesis (once audio wiring is complete):

```bash
export ELEVENLABS_API_KEY=...
```

See [backend/.env.example](/home/admin/codebases/red-iron-square/backend/.env.example) for all supported environment variables.

Without provider credentials, the demo still runs using built-in fallback behavior.

## Rehearsal Checklist

- Start backend on `127.0.0.1:8000`.
- Start frontend dev server and confirm the page loads.
- Verify the loading state resolves into Luna and Marco cards.
- Run one scripted scenario and confirm both agents update.
- Trigger a custom scenario and confirm the status banner stays healthy.
- Trigger a swap and confirm both cards reinitialize.
- Confirm text remains usable when the UI shows the audio fallback state.

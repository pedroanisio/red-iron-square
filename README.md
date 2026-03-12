---
disclaimer: "No information in this document should be taken for granted. Any statement not backed by executable code, tests, or a verifiable reference may be incomplete, invalid, or hallucinated."
---

# Red Iron Square

Red Iron Square currently has three user-facing surfaces:
- the Python SDK and FastAPI backend in [backend](/home/admin/codebases/red-iron-square/backend)
- the Flask operator UI served from the backend in [backend/src/ui](/home/admin/codebases/red-iron-square/backend/src/ui)
- the React-based Two Minds demo frontend in [frontend](/home/admin/codebases/red-iron-square/frontend)

## Quick Start

Backend API and tests:

```bash
cd backend
uv sync --extra api --extra ui
uv run pytest -q
uv run uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Flask UI:

```bash
cd backend
uv run red-iron-square-ui
```

Two Minds demo frontend:

```bash
cd frontend
npm test -- --runInBand
npm run dev
```

The Vite dev server proxies `/demo` traffic to `http://127.0.0.1:8000`. For browser use, open the frontend dev URL shown by Vite.

## Repo Notes

- Root [package.json](/home/admin/codebases/red-iron-square/package.json) only holds Playwright tooling; the real frontend scripts live in [frontend/package.json](/home/admin/codebases/red-iron-square/frontend/package.json).
- The main backend reference is [backend/README.md](/home/admin/codebases/red-iron-square/backend/README.md).
- The family-facing demo runbook is [docs/two-minds-demo-runbook.md](/home/admin/codebases/red-iron-square/docs/two-minds-demo-runbook.md).

---
name: writing-plans
description: Write a detailed implementation plan from a spec or set of requirements before changing code. Use for multi-step work that needs exact files, tests, commands, and checkpoints.
---

# Writing Plans

## Overview

Write comprehensive implementation plans assuming the engineer has zero context for our codebase and questionable taste. Document everything they need to know: which files to touch for each task, code, testing, docs they might need to check, how to test it. Give them the whole plan as bite-sized tasks. DRY. YAGNI. TDD. Frequent commits.

Assume they are a skilled developer, but know almost nothing about our toolset or problem domain. Assume they don't know good test design very well.

**Announce at start:** "I'm using the writing-plans skill to create the implementation plan."

**Context:** This should be run in a dedicated worktree (created by brainstorming skill).

**Save plans to:** `docs/plans/YYYY-MM-DD-<feature-name>.md`

## Project Conventions

All plans MUST respect these Red Iron Square constraints:

- **300 LOC limit** per source file (~10%, excluding comments). Split when approaching.
- **Python stack:** FastAPI, Pydantic, structlog, pytest, ruff, mypy, uv (not pip)
- **DDD layers:** `run_router.py` (API) → `run_service.py` (logic) → `run_store.py` (data)
- **TDD required:** Write failing test → implement → verify green
- **Test commands:** `cd backend && uv run pytest -x -q` (quick), `uv run pytest --cov` (coverage)
- **Lint:** `uv run ruff check src tests` and `uv run mypy src`
- **Coverage target:** >80%
- **UI:** Flask + Jinja2 templates, HTMX, Constructivist aesthetic (see frontend-design skill)
- **Backlog:** `docs/BACKLOG.md` tracks all phases — update after each plan

## Bite-Sized Task Granularity

**Each step is one action (2-5 minutes):**
- "Write the failing test" - step
- "Run it to make sure it fails" - step
- "Implement the minimal code to make the test pass" - step
- "Run the tests and make sure they pass" - step
- "Commit" - step

## Plan Document Header

**Every plan MUST start with this header:**

```markdown
# [Feature Name] Implementation Plan

> **Execution note:** Use the local `executing-plans` skill to implement this plan task-by-task.

**Goal:** [One sentence describing what this builds]

**Architecture:** [2-3 sentences about approach]

**Tech Stack:** [Key technologies/libraries]

**LOC Budget:** [Estimate per new/modified file, flag any near 300 LOC]

---
```

## Task Structure

```markdown
### Task N: [Component Name]

**Files:**
- Create: `backend/src/exact/path/to/file.py`
- Modify: `backend/src/exact/path/to/existing.py:123-145`
- Test: `backend/tests/exact/path/to/test.py`

**Step 1: Write the failing test**

```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/path/test.py::test_name -v`
Expected: FAIL with "function not defined"

**Step 3: Write minimal implementation**

```python
def function(input):
    return expected
```

**Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/path/test.py::test_name -v`
Expected: PASS

**Step 5: Lint and type check**

Run: `cd backend && uv run ruff check src tests && uv run mypy src`
Expected: No errors

**Step 6: Commit**

```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```
```

## Remember

- Exact file paths always (relative to `backend/`)
- Complete code in plan (not "add validation")
- Exact commands with expected output
- Reference relevant local skills by name when they should be used
- DRY, YAGNI, TDD, frequent commits
- Check 300 LOC limit before and after each file modification
- Use Pydantic models for request/response schemas
- Use structlog for logging, not stdlib logging

## Execution Handoff

After saving the plan, offer execution choice:

**"Plan complete and saved to `docs/plans/<filename>.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?"**

**If Subagent-Driven chosen:**
- Stay in this session
- Fresh subagent per task + code review

**If Parallel Session chosen:**
- Guide them to open new session in worktree
- New session uses `executing-plans`

---
name: executing-plans
description: Execute an existing step-by-step implementation plan in batches with verification and review checkpoints. Use when the user already has a written plan and wants implementation work to follow it closely.
---

# Executing Plans

## Overview

Load plan, review critically, execute tasks in batches, report for review between batches.

**Core principle:** Batch execution with checkpoints for architect review.

**Announce at start:** "I'm using the executing-plans skill to implement this plan."

## Project Conventions

All execution MUST respect these Red Iron Square constraints:

- **300 LOC limit** per source file (~10%, excluding comments). Check with `wc -l` after edits.
- **Test command:** `cd backend && uv run pytest -x -q`
- **Lint:** `cd backend && uv run ruff check src tests && uv run mypy src`
- **Coverage:** `cd backend && uv run pytest --cov --cov-report=term-missing`
- **Target:** >80% coverage, zero ruff/mypy errors
- **DDD layers:** router (API) → service (logic) → store (data). Never skip layers.
- **Pydantic** for all request/response models
- **structlog** for logging

## The Process

### Step 1: Load and Review Plan

1. Read plan file
2. Review critically — identify any questions or concerns about the plan
3. If concerns: Raise them with your human partner before starting
4. If no concerns: Create TodoWrite and proceed

### Step 2: Execute Batch

**Default: First 3 tasks**

For each task:
1. Mark as in_progress
2. Follow each step exactly (plan has bite-sized steps)
3. Run verifications as specified
4. Check LOC count on modified files (`wc -l <file>`)
5. Mark as completed

### Step 3: Report

When batch complete:
- Show what was implemented
- Show verification output (tests, lint, LOC counts)
- Say: "Ready for feedback."

### Step 4: Continue

Based on feedback:
- Apply changes if needed
- Execute next batch
- Repeat until complete

### Step 5: Complete Development

After all tasks complete and verified:
- Run full test suite: `cd backend && uv run pytest -x -q`
- Run lint: `cd backend && uv run ruff check src tests && uv run mypy src`
- Summarize results and present the completion state clearly

## When to Stop and Ask for Help

**STOP executing immediately when:**
- Hit a blocker mid-batch (missing dependency, test fails, instruction unclear)
- Plan has critical gaps preventing starting
- A file would exceed 300 LOC
- You don't understand an instruction
- Verification fails repeatedly

**Ask for clarification rather than guessing.**

## When to Revisit Earlier Steps

**Return to Review (Step 1) when:**
- Partner updates the plan based on your feedback
- Fundamental approach needs rethinking

**Don't force through blockers** — stop and ask.

## Remember

- Review plan critically first
- Follow plan steps exactly
- Don't skip verifications
- Reference skills when plan says to
- Between batches: just report and wait
- Stop when blocked, don't guess
- Always verify 300 LOC limit after file modifications

"""Pydantic schemas for orchestration API."""

from __future__ import annotations

from pydantic import BaseModel


class OrchestrateCycleRequest(BaseModel):
    """Run N orchestration cycles."""

    cycles: int = 1
    goals: list[str] = []
    campaign_id: str | None = None


class ResumeRequest(BaseModel):
    """Resume a paused run."""

    goals: list[str] = []

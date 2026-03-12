"""Hyperparameters for action space encoding and proposal."""

from pydantic import BaseModel, Field


class ActionEncoderParams(BaseModel):
    """Controls encoding behavior."""

    default_modifier: float = Field(default=0.0, ge=-1.0, le=1.0)
    heuristic_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

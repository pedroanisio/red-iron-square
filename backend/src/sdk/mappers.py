"""SDK mappers for converting internal objects to JSON-safe structures."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import Action
from src.self_model.simulator import SelfAwareTickResult
from src.temporal.emotions import EmotionReading
from src.temporal.simulator import TickResult


def vector_to_dict(values: np.ndarray, registry: DimensionRegistry) -> dict[str, float]:
    """Convert a registry-aligned vector to a JSON-safe dict."""
    return {key: float(values[i]) for i, key in enumerate(registry.keys)}


def probabilities_to_dict(
    actions: list[Action],
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Map action names to probabilities."""
    return {action.name: float(probabilities[i]) for i, action in enumerate(actions)}


def action_list_to_names(actions: list[Action]) -> list[str]:
    """Return action names in declared order."""
    return [action.name for action in actions]


def emotion_readings_to_payload(
    emotions: list[EmotionReading],
) -> list[dict[str, Any]]:
    """Serialize emotion readings to JSON-safe dicts."""
    return [
        {
            "label": reading.label.value,
            "intensity": float(reading.intensity),
            "description": reading.description,
        }
        for reading in emotions
    ]


def tick_result_to_payload(
    result: TickResult,
    registry: DimensionRegistry,
) -> dict[str, Any]:
    """Serialize a temporal tick result."""
    return {
        "tick": result.tick,
        "scenario": {
            "name": result.scenario.name,
            "description": result.scenario.description,
            "values": vector_to_dict(result.scenario.to_array(), registry),
        },
        "action": result.action.name,
        "outcome": float(result.outcome),
        "state_before": result.state_before.model_dump(),
        "state_after": result.state_after.model_dump(),
        "activations": vector_to_dict(result.activations, registry),
        "emotions": emotion_readings_to_payload(result.emotions),
        "probabilities": [float(value) for value in result.probabilities.tolist()],
    }


def self_aware_tick_result_to_payload(
    result: SelfAwareTickResult,
    registry: DimensionRegistry,
) -> dict[str, Any]:
    """Serialize a self-aware tick result."""
    payload = tick_result_to_payload(result, registry)
    payload.update(
        {
            "self_emotions": [
                {
                    "label": reading.label.value,
                    "intensity": float(reading.intensity),
                    "description": reading.description,
                }
                for reading in result.self_emotions
            ],
            "psi_hat": vector_to_dict(result.psi_hat, registry),
            "behavioral_evidence": vector_to_dict(result.behavioral_evidence, registry),
            "self_coherence": float(result.self_coherence),
            "self_accuracy": float(result.self_accuracy),
            "identity_drift": float(result.identity_drift),
            "prediction_error": float(result.prediction_error),
            "predicted_probabilities": [
                float(value) for value in result.predicted_probs.tolist()
            ],
        }
    )
    return payload

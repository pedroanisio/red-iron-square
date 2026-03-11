"""Personality bounded context: dimensions, vectors, activations, decision engine."""

from src.personality.activations import DEFAULT_ACTIVATION_REGISTRY, ActivationFunctions
from src.personality.decision import DecisionEngine, compute_activation_batch
from src.personality.dimensions import DEFAULT_DIMENSIONS, Dimension, DimensionRegistry
from src.personality.hyperparameters import HyperParameters, ResilienceMode
from src.personality.vectors import Action, PersonalityVector, Scenario

__all__ = [
    "Dimension", "DimensionRegistry", "DEFAULT_DIMENSIONS",
    "PersonalityVector", "Scenario", "Action",
    "HyperParameters", "ResilienceMode",
    "ActivationFunctions", "DEFAULT_ACTIVATION_REGISTRY",
    "DecisionEngine", "compute_activation_batch",
]

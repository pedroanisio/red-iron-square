"""Personality bounded context: dimensions, vectors, activations, decision engine."""

from src.personality.dimensions import Dimension, DimensionRegistry, DEFAULT_DIMENSIONS
from src.personality.vectors import PersonalityVector, Scenario, Action
from src.personality.hyperparameters import HyperParameters, ResilienceMode
from src.personality.activations import ActivationFunctions, DEFAULT_ACTIVATION_REGISTRY
from src.personality.decision import DecisionEngine, compute_activation_batch

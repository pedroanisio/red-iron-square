"""Temporal bounded context: state, memory, emotions, simulation loop."""

from src.temporal.state import AgentState, StateTransitionParams, update_state
from src.temporal.memory import MemoryEntry, MemoryBank
from src.temporal.emotions import EmotionLabel, EmotionReading, EmotionThresholds
from src.temporal.affective_engine import AffectiveEngine
from src.temporal.simulator import TickResult, TemporalSimulator
from src.temporal.generators import generate_scenario_sequence, generate_outcome_sequence

__all__ = [
    "AgentState", "StateTransitionParams", "update_state",
    "MemoryEntry", "MemoryBank",
    "EmotionLabel", "EmotionReading", "EmotionThresholds",
    "AffectiveEngine",
    "TickResult", "TemporalSimulator",
    "generate_scenario_sequence", "generate_outcome_sequence",
]

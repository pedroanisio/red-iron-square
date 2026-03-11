"""Facade: re-exports the full public API from all bounded contexts."""

from src.personality import (
    Dimension,
    DimensionRegistry,
    DEFAULT_DIMENSIONS,
    PersonalityVector,
    Scenario,
    Action,
    HyperParameters,
    ResilienceMode,
    ActivationFunctions,
    DEFAULT_ACTIVATION_REGISTRY,
    DecisionEngine,
    compute_activation_batch,
)
from src.temporal import (
    AgentState,
    StateTransitionParams,
    update_state,
    MemoryEntry,
    MemoryBank,
    EmotionLabel,
    EmotionReading,
    EmotionThresholds,
    AffectiveEngine,
    TickResult,
    TemporalSimulator,
    generate_scenario_sequence,
    generate_outcome_sequence,
)
from src.self_model import (
    SelfModelParams,
    SelfModel,
    SelfEmotionLabel,
    SelfEmotionReading,
    SelfEmotionDetector,
    SelfAwareTickResult,
    SelfAwareSimulator,
)

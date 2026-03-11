"""Facade: re-exports the full public API from all bounded contexts."""

from src.personality import (
    DEFAULT_ACTIVATION_REGISTRY as DEFAULT_ACTIVATION_REGISTRY,
    DEFAULT_DIMENSIONS as DEFAULT_DIMENSIONS,
    Action as Action,
    ActivationFunctions as ActivationFunctions,
    DecisionEngine as DecisionEngine,
    Dimension as Dimension,
    DimensionRegistry as DimensionRegistry,
    HyperParameters as HyperParameters,
    PersonalityVector as PersonalityVector,
    ResilienceMode as ResilienceMode,
    Scenario as Scenario,
    compute_activation_batch as compute_activation_batch,
)
from src.self_model import (
    SelfAwareSimulator as SelfAwareSimulator,
    SelfAwareTickResult as SelfAwareTickResult,
    SelfEmotionDetector as SelfEmotionDetector,
    SelfEmotionLabel as SelfEmotionLabel,
    SelfEmotionReading as SelfEmotionReading,
    SelfModel as SelfModel,
    SelfModelParams as SelfModelParams,
)
from src.temporal import (
    AffectiveEngine as AffectiveEngine,
    AgentState as AgentState,
    EmotionLabel as EmotionLabel,
    EmotionReading as EmotionReading,
    EmotionThresholds as EmotionThresholds,
    MemoryBank as MemoryBank,
    MemoryEntry as MemoryEntry,
    StateTransitionParams as StateTransitionParams,
    TemporalSimulator as TemporalSimulator,
    TickResult as TickResult,
    generate_outcome_sequence as generate_outcome_sequence,
    generate_scenario_sequence as generate_scenario_sequence,
    update_state as update_state,
)

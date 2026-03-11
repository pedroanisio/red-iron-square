"""Facade: re-exports the full public API from all bounded contexts."""

from src.personality import (
    DEFAULT_ACTIVATION_REGISTRY as DEFAULT_ACTIVATION_REGISTRY,
)
from src.personality import (
    DEFAULT_DIMENSIONS as DEFAULT_DIMENSIONS,
)
from src.personality import (
    Action as Action,
)
from src.personality import (
    ActivationFunctions as ActivationFunctions,
)
from src.personality import (
    DecisionEngine as DecisionEngine,
)
from src.personality import (
    Dimension as Dimension,
)
from src.personality import (
    DimensionRegistry as DimensionRegistry,
)
from src.personality import (
    HyperParameters as HyperParameters,
)
from src.personality import (
    PersonalityVector as PersonalityVector,
)
from src.personality import (
    ResilienceMode as ResilienceMode,
)
from src.personality import (
    Scenario as Scenario,
)
from src.personality import (
    compute_activation_batch as compute_activation_batch,
)
from src.self_model import (
    SelfAwareSimulator as SelfAwareSimulator,
)
from src.self_model import (
    SelfAwareTickResult as SelfAwareTickResult,
)
from src.self_model import (
    SelfEmotionDetector as SelfEmotionDetector,
)
from src.self_model import (
    SelfEmotionLabel as SelfEmotionLabel,
)
from src.self_model import (
    SelfEmotionReading as SelfEmotionReading,
)
from src.self_model import (
    SelfModel as SelfModel,
)
from src.self_model import (
    SelfModelParams as SelfModelParams,
)
from src.temporal import (
    AffectiveEngine as AffectiveEngine,
)
from src.temporal import (
    AgentState as AgentState,
)
from src.temporal import (
    EmotionLabel as EmotionLabel,
)
from src.temporal import (
    EmotionReading as EmotionReading,
)
from src.temporal import (
    EmotionThresholds as EmotionThresholds,
)
from src.temporal import (
    MemoryBank as MemoryBank,
)
from src.temporal import (
    MemoryEntry as MemoryEntry,
)
from src.temporal import (
    StateTransitionParams as StateTransitionParams,
)
from src.temporal import (
    TemporalSimulator as TemporalSimulator,
)
from src.temporal import (
    TickResult as TickResult,
)
from src.temporal import (
    generate_outcome_sequence as generate_outcome_sequence,
)
from src.temporal import (
    generate_scenario_sequence as generate_scenario_sequence,
)
from src.temporal import (
    update_state as update_state,
)

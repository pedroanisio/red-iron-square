"""SDK facade for constructing and running agent simulations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.constructed_emotion.affect import ConstructedAffectiveEngine, EmotionCallback
from src.constructed_emotion.params import ConstructedEmotionParams
from src.efe.engine import EFEEngine
from src.efe.params import EFEParams
from src.narrative.model import NarrativeGenerativeModel
from src.personality.decision import DecisionEngine
from src.personality.dimensions import DEFAULT_DIMENSIONS, Dimension, DimensionRegistry
from src.personality.hyperparameters import HyperParameters, ResilienceMode
from src.personality.vectors import Action, PersonalityVector, Scenario
from src.precision.engine import PrecisionEngine
from src.precision.params import PrecisionParams
from src.sdk.builders import (
    build_action,
    build_initial_self_model,
    build_personality,
    build_registry,
    build_scenario,
)
from src.sdk.decision_client import DecisionClient
from src.sdk.self_model_client import SelfModelSimulationClient
from src.sdk.simulation_client import TemporalSimulationClient
from src.sdk.types import (
    DecisionResult,
    SelfAwareSimulationTrace,
    SelfAwareTickRecord,
    SimulationTrace,
    TickRecord,
)
from src.self_evidencing.modulator import SelfEvidencingModulator
from src.self_evidencing.params import SelfEvidencingParams
from src.shared.protocols import DecisionEngineProtocol


class AgentSDK:
    """Public SDK facade over the domain-level simulation components."""

    def __init__(
        self,
        registry: DimensionRegistry | None = None,
        *,
        hyperparameters: HyperParameters | None = None,
        resilience_mode: ResilienceMode = ResilienceMode.ACTIVATION,
        precision_engine: PrecisionEngine | None = None,
        efe_params: EFEParams | None = None,
        emotion_params: ConstructedEmotionParams | None = None,
        self_evidencing_params: SelfEvidencingParams | None = None,
    ) -> None:
        self.registry = registry or build_registry()
        self.engine = DecisionEngine(
            registry=self.registry,
            hyperparameters=hyperparameters or HyperParameters(),
            resilience_mode=resilience_mode,
        )
        self._precision_engine = precision_engine
        self._efe_params = efe_params
        self._emotion_params = emotion_params
        self._self_evidencing_params = self_evidencing_params
        self._emotion_callback: EmotionCallback | None = None

    @classmethod
    def default(cls) -> AgentSDK:
        """Create the default OCEAN+RIT SDK."""
        return cls()

    @classmethod
    def with_precision(cls, params: PrecisionParams | None = None) -> AgentSDK:
        """Create the default SDK with precision tracking enabled."""
        return cls(precision_engine=PrecisionEngine(params))

    @classmethod
    def with_efe(
        cls,
        efe_params: EFEParams | None = None,
        precision_params: PrecisionParams | None = None,
    ) -> AgentSDK:
        """Create the SDK with EFE decision engine and precision tracking."""
        return cls(
            precision_engine=PrecisionEngine(precision_params),
            efe_params=efe_params or EFEParams(),
        )

    @classmethod
    def with_constructed_emotion(
        cls,
        emotion_params: ConstructedEmotionParams | None = None,
        efe_params: EFEParams | None = None,
        precision_params: PrecisionParams | None = None,
    ) -> AgentSDK:
        """Create SDK with EFE + precision + constructed emotion (Phase C1)."""
        return cls(
            precision_engine=PrecisionEngine(precision_params),
            efe_params=efe_params or EFEParams(),
            emotion_params=emotion_params or ConstructedEmotionParams(),
        )

    @classmethod
    def with_self_evidencing(
        cls,
        self_evidencing_params: SelfEvidencingParams | None = None,
        emotion_params: ConstructedEmotionParams | None = None,
        efe_params: EFEParams | None = None,
        precision_params: PrecisionParams | None = None,
    ) -> AgentSDK:
        """Create SDK with full Phase C: EFE + precision + emotion + self-evidencing."""
        return cls(
            precision_engine=PrecisionEngine(precision_params),
            efe_params=efe_params or EFEParams(),
            emotion_params=emotion_params or ConstructedEmotionParams(),
            self_evidencing_params=self_evidencing_params or SelfEvidencingParams(),
        )

    def set_emotion_callback(self, callback: EmotionCallback) -> None:
        """Register an LLM emotion callback for System 2 surprise spikes."""
        self._emotion_callback = callback

    @classmethod
    def from_dimensions(cls, dimensions: Sequence[Dimension]) -> AgentSDK:
        """Create an SDK for a custom dimension registry."""
        return cls(registry=build_registry(dimensions))

    def personality(self, values: Mapping[str, float]) -> PersonalityVector:
        """Build a personality vector from sparse values."""
        return build_personality(values, self.registry)

    def scenario(
        self,
        values: Mapping[str, float],
        *,
        name: str = "",
        description: str = "",
    ) -> Scenario:
        """Build a scenario from sparse values."""
        return build_scenario(
            values,
            self.registry,
            name=name,
            description=description,
        )

    def action(
        self,
        name: str,
        modifiers: Mapping[str, float],
        *,
        description: str = "",
    ) -> Action:
        """Build an action from sparse modifiers."""
        return build_action(
            name=name,
            modifiers=modifiers,
            registry=self.registry,
            description=description,
        )

    def initial_self_model(self, values: Mapping[str, float]) -> np.ndarray:
        """Build an initial self-model vector from sparse values."""
        return build_initial_self_model(values, self.registry)

    def decide(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
        actions: Sequence[Action],
        *,
        temperature: float = 1.0,
        bias: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> DecisionResult:
        """Run a one-shot decision through the SDK."""
        engine = self._resolve_engine(personality)
        return DecisionClient(engine, self.registry).decide(
            personality,
            scenario,
            actions,
            temperature=temperature,
            bias=bias,
            rng=rng,
        )

    def _resolve_engine(
        self,
        personality: PersonalityVector,
    ) -> DecisionEngineProtocol:
        """Return EFEEngine if EFE mode is active, else base DecisionEngine."""
        if self._efe_params is not None:
            return EFEEngine(self.engine, personality, self._efe_params)
        return self.engine

    def simulator(
        self,
        personality: PersonalityVector,
        actions: Sequence[Action],
        **simulator_kwargs: object,
    ) -> TemporalSimulationClient:
        """Create a temporal simulation client."""
        if self._precision_engine and "precision_engine" not in simulator_kwargs:
            simulator_kwargs["precision_engine"] = self._precision_engine
        if self._emotion_params and "constructed_affect" not in simulator_kwargs:
            simulator_kwargs["constructed_affect"] = ConstructedAffectiveEngine(
                self._emotion_params,
                emotion_callback=self._emotion_callback,
            )
        if self._emotion_params and "narrative_model" not in simulator_kwargs:
            pvals = dict(zip(personality.registry.keys, personality.to_array()))
            simulator_kwargs["narrative_model"] = NarrativeGenerativeModel(pvals)
        engine = self._resolve_engine(personality)
        return TemporalSimulationClient(
            personality,
            actions,
            engine,
            self.registry,
            **simulator_kwargs,
        )

    def self_aware_simulator(
        self,
        personality: PersonalityVector,
        initial_self_model: np.ndarray,
        actions: Sequence[Action],
        **simulator_kwargs: object,
    ) -> SelfModelSimulationClient:
        """Create a self-aware simulation client."""
        if self._precision_engine and "precision_engine" not in simulator_kwargs:
            simulator_kwargs["precision_engine"] = self._precision_engine
        if self._emotion_params and "constructed_affect" not in simulator_kwargs:
            simulator_kwargs["constructed_affect"] = ConstructedAffectiveEngine(
                self._emotion_params,
                emotion_callback=self._emotion_callback,
            )
        if self._emotion_params and "narrative_model" not in simulator_kwargs:
            pvals = dict(zip(personality.registry.keys, personality.to_array()))
            simulator_kwargs["narrative_model"] = NarrativeGenerativeModel(pvals)
        if self._self_evidencing_params and "self_evidencing" not in simulator_kwargs:
            simulator_kwargs["self_evidencing"] = SelfEvidencingModulator(
                self._self_evidencing_params,
            )
        engine = self._resolve_engine(personality)
        return SelfModelSimulationClient(
            personality,
            initial_self_model,
            actions,
            engine,
            self.registry,
            **simulator_kwargs,
        )


__all__ = [
    "AgentSDK",
    "ConstructedAffectiveEngine",
    "ConstructedEmotionParams",
    "DEFAULT_DIMENSIONS",
    "DecisionResult",
    "EFEEngine",
    "EFEParams",
    "EmotionCallback",
    "NarrativeGenerativeModel",
    "PrecisionEngine",
    "PrecisionParams",
    "SelfAwareSimulationTrace",
    "SelfAwareTickRecord",
    "SelfEvidencingModulator",
    "SelfEvidencingParams",
    "SimulationTrace",
    "TickRecord",
]

"""AblationRunner: run simulations with systematically disabled components.

Implements the ten ablation conditions from the research protocol,
each removing or altering one architectural component to measure its
contribution.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from src.personality.vectors import Action, PersonalityVector
from src.precision.params import PrecisionParams
from src.sdk import AgentSDK
from src.sdk.self_model_client import SelfModelSimulationClient
from src.sdk.simulation_client import TemporalSimulationClient
from src.sdk.types import SelfAwareTickRecord
from src.self_evidencing.params import SelfEvidencingParams
from src.shared.entropy import compute_action_entropy


class AblationConfig(BaseModel):
    """Configuration for one ablation condition."""

    name: str
    description: str
    sdk_mode: str
    use_self_model: bool = True
    n_ticks: int = 100
    precision_params: PrecisionParams | None = None
    self_evidencing_params: SelfEvidencingParams | None = None


class AblationResult(BaseModel):
    """Summary metrics from one ablation run."""

    config_name: str
    mean_entropy: float
    mean_mood: float
    mean_coherence: float | None = None
    n_ticks: int


def _build_sdk(
    sdk_mode: str,
    precision_params: PrecisionParams | None = None,
    self_evidencing_params: SelfEvidencingParams | None = None,
) -> AgentSDK:
    """Construct an AgentSDK from the named mode string.

    Modes mirror the layered SDK factory hierarchy:
      - default: no precision, no EFE, no emotion, no SE
      - precision: precision engine only
      - efe: precision + EFE
      - constructed_emotion: precision + EFE + constructed emotion
      - self_evidencing: full stack (precision + EFE + emotion + SE)

    Optional ``precision_params`` and ``self_evidencing_params`` are
    forwarded to factories that accept them.
    """
    valid_modes = {
        "default",
        "precision",
        "efe",
        "constructed_emotion",
        "self_evidencing",
    }
    if sdk_mode not in valid_modes:
        msg = f"Unknown sdk_mode: {sdk_mode!r}. Valid: {sorted(valid_modes)}"
        raise ValueError(msg)

    if sdk_mode == "default":
        return AgentSDK.default()
    if sdk_mode == "precision":
        return AgentSDK.with_precision(precision_params)
    if sdk_mode == "efe":
        return AgentSDK.with_efe(precision_params=precision_params)
    if sdk_mode == "constructed_emotion":
        return AgentSDK.with_constructed_emotion(precision_params=precision_params)
    return AgentSDK.with_self_evidencing(
        self_evidencing_params=self_evidencing_params,
        precision_params=precision_params,
    )


ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig(
        name="full_model",
        description="All components enabled",
        sdk_mode="self_evidencing",
        use_self_model=True,
    ),
    AblationConfig(
        name="no_precision",
        description="Flat precision PI=1",
        sdk_mode="default",
        use_self_model=True,
    ),
    AblationConfig(
        name="no_efe",
        description="Base utility engine (no EFE)",
        sdk_mode="precision",
        use_self_model=True,
    ),
    AblationConfig(
        name="no_constructed_emotion",
        description="No constructed affect engine",
        sdk_mode="efe",
        use_self_model=True,
    ),
    AblationConfig(
        name="no_self_model",
        description="Temporal simulator only (no self-model)",
        sdk_mode="self_evidencing",
        use_self_model=False,
    ),
    AblationConfig(
        name="no_self_evidencing",
        description="Self-model without SE modulation",
        sdk_mode="constructed_emotion",
        use_self_model=True,
    ),
    AblationConfig(
        name="no_narrative",
        description="Fixed narrative precision L2=1",
        sdk_mode="precision",
        use_self_model=True,
    ),
    AblationConfig(
        name="no_allostatic_setpoints",
        description="Full stack without precision-weighted state transitions",
        sdk_mode="self_evidencing",
        use_self_model=True,
    ),
    AblationConfig(
        name="learned_vs_fixed_precision",
        description="Fixed hand-tuned precision params vs learned defaults",
        sdk_mode="self_evidencing",
        use_self_model=True,
        precision_params=PrecisionParams(
            n_mood_precision_weight=0.3,
            e_arousal_precision_weight=0.3,
            r_frustration_precision_weight=0.3,
            default_bias=0.54,
        ),
    ),
    AblationConfig(
        name="no_se_cap",
        description="Remove stability cap (pi_max very high)",
        sdk_mode="self_evidencing",
        use_self_model=True,
        self_evidencing_params=SelfEvidencingParams(pi_max=100.0),
    ),
]


def _default_personality() -> dict[str, float]:
    """Return a balanced personality vector for ablation runs."""
    return {k: 0.5 for k in "OCEANRIT"}


class AblationRunner:
    """Run ablation protocol across all ten configurations.

    Each configuration disables or alters one architectural component
    so that the resulting behavioral metrics reveal its contribution.
    """

    def __init__(
        self,
        personality: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        self._personality = personality or _default_personality()
        self._seed = seed

    def run(self, config: AblationConfig) -> AblationResult:
        """Run one ablation condition and return summary metrics."""
        sdk = _build_sdk(
            config.sdk_mode,
            precision_params=config.precision_params,
            self_evidencing_params=config.self_evidencing_params,
        )
        personality = sdk.personality(self._personality)
        scenario = sdk.scenario(
            {k: 0.5 for k in "OCEANRIT"},
            name="ablation",
        )
        actions = [
            sdk.action("Engage", {"O": 0.5, "C": 0.3, "E": 0.3}),
            sdk.action("Reflect", {"O": 0.2, "C": 0.5, "E": -0.1}),
            sdk.action("Wait", {"O": -0.1, "C": 0.1, "E": -0.2}),
        ]
        rng = np.random.default_rng(self._seed)

        sim = self._build_simulator(
            sdk,
            personality,
            actions,
            config,
            rng,
        )

        action_counts: dict[str, int] = {}
        moods: list[float] = []
        coherences: list[float] = []

        for _ in range(config.n_ticks):
            rec = sim.tick(scenario)
            action_counts[rec.action] = action_counts.get(rec.action, 0) + 1
            moods.append(rec.state_after["mood"])
            if isinstance(rec, SelfAwareTickRecord):
                coherences.append(rec.self_coherence)

        return AblationResult(
            config_name=config.name,
            mean_entropy=compute_action_entropy(action_counts),
            mean_mood=float(np.mean(moods)),
            mean_coherence=(float(np.mean(coherences)) if coherences else None),
            n_ticks=config.n_ticks,
        )

    def run_all(self) -> list[AblationResult]:
        """Run all ten ablation configurations and return results."""
        return [self.run(cfg) for cfg in ABLATION_CONFIGS]

    @staticmethod
    def _build_simulator(
        sdk: AgentSDK,
        personality: PersonalityVector,
        actions: list[Action],
        config: AblationConfig,
        rng: np.random.Generator,
    ) -> SelfModelSimulationClient | TemporalSimulationClient:
        """Build the appropriate simulator based on config flags."""
        if config.use_self_model:
            psi_hat = sdk.initial_self_model(
                {k: 0.5 for k in "OCEANRIT"},
            )
            return sdk.self_aware_simulator(
                personality,
                psi_hat,
                actions,
                rng=rng,
            )
        return sdk.simulator(personality, actions, rng=rng)

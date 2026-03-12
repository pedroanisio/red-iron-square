"""EFE decision engine: Expected Free Energy surrogate for action selection.

Replaces the utility dot-product with a decomposition into pragmatic value
(alignment with personality-derived C-vector preferences) and epistemic
value (outcome uncertainty from memory). Policy precision gamma from the
precision engine modulates action selection temperature.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.efe.c_vector import N_INTEROCEPTIVE, CVector
from src.efe.epistemic import compute_epistemic_value
from src.efe.params import EFEParams
from src.personality.decision import DecisionEngine
from src.personality.vectors import Action, PersonalityVector, Scenario
from src.shared.logging import get_logger

if TYPE_CHECKING:
    from src.temporal.memory import MemoryBank

_log = get_logger(module="efe.engine")

BIN_CENTERS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


class EFEEngine:
    """EFE-based decision engine composing over DecisionEngine.

    Delegates activation computation and outcome resolution to the
    base engine. Replaces utility with negative expected free energy
    and decide with gamma-modulated Boltzmann selection.
    """

    def __init__(
        self,
        base_engine: DecisionEngine,
        personality: PersonalityVector,
        params: EFEParams | None = None,
    ) -> None:
        self._base = base_engine
        self._personality = personality
        self._params = params or EFEParams()
        self._c_vector = CVector(personality, self._params)
        self._memory: MemoryBank | None = None
        self.registry = base_engine.registry
        self.hp = base_engine.hp
        self.resilience_mode = base_engine.resilience_mode
        self.activations = base_engine.activations

        keys = set(personality.registry.keys)
        self._O = personality["O"] if "O" in keys else 0.5
        self._C_trait = personality["C"] if "C" in keys else 0.5

    def bind_memory(self, memory: MemoryBank) -> None:
        """Bind a memory bank for epistemic value computation."""
        self._memory = memory

    def compute_activations(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
    ) -> np.ndarray:
        """Delegate activation computation to base engine."""
        return self._base.compute_activations(personality, scenario)

    def utility(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
        action: Action,
        bias: float = 0.0,
        activations_override: np.ndarray | None = None,
    ) -> float:
        """Compute negative EFE as utility: higher is better.

        ``utility = -G(pi) = w_pragmatic * (-pragmatic) + w_O * epistemic``
        """
        pragmatic = self._pragmatic_value(
            personality,
            scenario,
            action,
            activations_override,
        )
        epistemic = self._epistemic_value(action)

        w_O = self._O * self._params.w_base
        c_scale = 1.0 + self._params.c_pragmatic_scale * self._C_trait
        return bias + c_scale * (1.0 - w_O) * (-pragmatic) + w_O * epistemic

    def decide(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
        actions: Sequence[Action],
        temperature: float = 1.0,
        bias: float = 0.0,
        rng: np.random.Generator | None = None,
        activations_override: np.ndarray | None = None,
    ) -> tuple[Action, np.ndarray]:
        """Boltzmann selection over negative EFE scores."""
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if len(actions) == 0:
            raise ValueError("Must provide at least one action.")

        rng = rng or np.random.default_rng()
        utilities = np.array(
            [
                self.utility(
                    personality,
                    scenario,
                    a,
                    bias=bias,
                    activations_override=activations_override,
                )
                for a in actions
            ]
        )

        effective_temp = self._modulate_temperature(temperature)
        logits = utilities / effective_temp
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        chosen_idx = rng.choice(len(actions), p=probs)
        _log.debug(
            "efe_action_selected",
            action=actions[chosen_idx].name,
            temperature=round(temperature, 4),
            efe_spread=round(float(utilities.max() - utilities.min()), 4),
        )
        return actions[chosen_idx], probs

    def _modulate_temperature(self, base_temperature: float) -> float:
        """Apply O/C personality modulation to temperature.

        Uses exponential scaling: temp * exp(scale * (O - C)).
        High-O increases temperature (exploration), high-C decreases it
        (exploitation). Implements the exploration-exploitation tradeoff.
        """
        oc_diff = self._O - self._C_trait
        return base_temperature * float(
            np.exp(self._params.oc_temperature_scale * oc_diff)
        )

    def _pragmatic_value(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
        action: Action,
        activations_override: np.ndarray | None = None,
    ) -> float:
        """KL divergence between predicted outcome and C-vector preferences.

        Uses the base utility as a proxy for expected outcome quality,
        then maps to predicted interoceptive changes and soft-bins them.
        """
        base_u = self._base.utility(
            personality,
            scenario,
            action,
            activations_override=activations_override,
        )
        predicted_changes = self._predict_interoceptive_changes(
            base_u,
            personality,
            action,
        )
        q = self._soft_bin(predicted_changes)
        return self._kl_divergence(q, self._c_vector.log_preferences)

    def _predict_interoceptive_changes(
        self,
        base_utility: float,
        personality: PersonalityVector,
        action: Action,
    ) -> np.ndarray:
        """Predict 5D interoceptive changes from scalar utility."""
        u = float(np.clip(base_utility, -1.0, 1.0))
        keys = set(personality.registry.keys)
        N = personality["N"] if "N" in keys else 0.5
        R = personality["R"] if "R" in keys else 0.5

        mood = u * (0.25 if u > 0 else 0.35 * (1.0 + 0.5 * N))
        arousal = abs(u) * 0.15
        effort = float(np.linalg.norm(action.modifiers))
        energy = -0.015 * effort
        satisfaction = max(u, 0.0) * 0.20
        frustration = max(-u, 0.0) * 0.25 * (1.0 - 0.4 * R)
        return np.array([mood, arousal, energy, satisfaction, frustration])

    def _soft_bin(self, changes: np.ndarray) -> np.ndarray:
        """Map 5D predicted changes to bin distributions via softmax."""
        sharpness = self._params.bin_sharpness
        q = np.zeros((N_INTEROCEPTIVE, self._params.n_bins))
        for j in range(N_INTEROCEPTIVE):
            logits = -sharpness * (BIN_CENTERS - changes[j]) ** 2
            logits -= logits.max()
            exp_l = np.exp(logits)
            q[j] = exp_l / exp_l.sum()
        return q

    def _epistemic_value(self, action: Action) -> float:
        """Outcome variance for this action from memory."""
        if self._memory is None:
            return self._params.default_epistemic
        return compute_epistemic_value(
            action.name,
            self._memory,
            self._params.memory_window,
            default=self._params.default_epistemic,
        )

    def compute_efe_breakdown(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
        actions: Sequence[Action],
        activations_override: np.ndarray | None = None,
    ) -> dict[str, dict[str, float]]:
        """Return per-action pragmatic, epistemic, and total EFE values."""
        breakdown: dict[str, dict[str, float]] = {}
        for action in actions:
            pragmatic = self._pragmatic_value(
                personality, scenario, action, activations_override,
            )
            epistemic = self._epistemic_value(action)
            w_O = self._O * self._params.w_base
            c_scale = 1.0 + self._params.c_pragmatic_scale * self._C_trait
            total = c_scale * (1.0 - w_O) * (-pragmatic) + w_O * epistemic
            breakdown[action.name] = {
                "pragmatic": round(pragmatic, 6),
                "epistemic": round(epistemic, 6),
                "total": round(total, 6),
            }
        return breakdown

    @staticmethod
    def _kl_divergence(q: np.ndarray, log_p: np.ndarray) -> float:
        """Sum of per-dimension KL(q || p) with numerical safety."""
        eps = 1e-10
        log_q = np.log(q + eps)
        kl = np.sum(q * (log_q - log_p))
        return max(0.0, float(kl))

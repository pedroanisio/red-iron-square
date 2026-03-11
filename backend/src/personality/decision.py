"""Decision engine: utility computation and Boltzmann action selection."""

from typing import Optional, Callable, Sequence

import numpy as np

from src.personality.dimensions import DimensionRegistry
from src.personality.vectors import PersonalityVector, Scenario, Action
from src.personality.hyperparameters import HyperParameters, ResilienceMode
from src.personality.activations import DEFAULT_ACTIVATION_REGISTRY
from src.shared.logging import get_logger

_log = get_logger(module="personality.decision")


class DecisionEngine:
    """
    Computes utility scores for actions and selects via Boltzmann sampling.

    U(psi, s, a) = bias + sum_i f_i(s_i, psi_i) * a_modifiers_i

    P(a | psi, s, tau) = exp(U(a)/tau) / sum exp(U(a')/tau)
    """

    def __init__(
        self,
        registry: DimensionRegistry = DimensionRegistry(),
        activation_registry: Optional[dict[str, Callable[..., float]]] = None,
        hyperparameters: HyperParameters = HyperParameters(),
        resilience_mode: ResilienceMode = ResilienceMode.ACTIVATION,
    ) -> None:
        self.registry = registry
        self.hp = hyperparameters
        self.resilience_mode = resilience_mode
        self.activations = activation_registry or {
            k: v for k, v in DEFAULT_ACTIVATION_REGISTRY.items()
            if k in set(registry.keys)
        }

        missing = set(registry.keys) - set(self.activations.keys())
        if missing:
            raise ValueError(
                f"No activation function registered for dimensions: {missing}. "
                f"Provide them via `activation_registry`."
            )

    def compute_activations(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
    ) -> np.ndarray:
        """Compute the N-dimensional activation vector, each entry in [0, 1]."""
        psi = personality.to_array()
        s = scenario.to_array()
        activations = np.zeros(self.registry.size)

        for i, key in enumerate(self.registry.keys):
            fn = self.activations[key]
            if key == "R":
                activations[i] = fn(s[i], psi[i], self.hp, mode=self.resilience_mode)
            else:
                activations[i] = fn(s[i], psi[i], self.hp)

        return activations

    def utility(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
        action: Action,
        bias: float = 0.0,
        activations_override: Optional[np.ndarray] = None,
    ) -> float:
        """U(psi, s, a) = bias + activations dot modifiers."""
        if activations_override is not None:
            activations = activations_override
        else:
            activations = self.compute_activations(personality, scenario)
        return bias + float(np.dot(activations, action.modifiers))

    def decide(
        self,
        personality: PersonalityVector,
        scenario: Scenario,
        actions: Sequence[Action],
        temperature: float = 1.0,
        bias: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        activations_override: Optional[np.ndarray] = None,
    ) -> tuple[Action, np.ndarray]:
        """Boltzmann (softmax) action selection. Returns (chosen_action, probs)."""
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if len(actions) == 0:
            raise ValueError("Must provide at least one action.")

        rng = rng or np.random.default_rng()

        utilities = np.array([
            self.utility(personality, scenario, a, bias=bias,
                         activations_override=activations_override)
            for a in actions
        ])

        logits = utilities / temperature
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        chosen_idx = rng.choice(len(actions), p=probs)
        _log.debug(
            "action_selected",
            action=actions[chosen_idx].name,
            temperature=round(temperature, 4),
            utility_spread=round(float(utilities.max() - utilities.min()), 4),
            num_actions=len(actions),
        )
        return actions[chosen_idx], probs


def compute_activation_batch(
    personalities: np.ndarray,
    scenarios: np.ndarray,
    hp: HyperParameters = HyperParameters(),
) -> np.ndarray:
    """
    Vectorized activation computation for batch simulation.

    Assumes standard 8-dimensional OCEAN+RIT layout.
    For custom registries, use DecisionEngine.compute_activations instead.
    """
    psi = np.asarray(personalities)
    s = np.asarray(scenarios)
    if psi.shape != s.shape or psi.ndim != 2:
        raise ValueError(f"Shape mismatch: personalities {psi.shape} vs scenarios {s.shape}")

    n_dim = psi.shape[1]
    act = np.zeros_like(psi)

    act[:, 0] = psi[:, 0] * np.tanh(hp.alpha * s[:, 0])

    if n_dim > 1:
        tc = 2.0 * psi[:, 1] - 1.0
        act[:, 1] = 1.0 / (1.0 + np.exp(-hp.beta * tc * (s[:, 1] - hp.c_threshold)))
    if n_dim > 2:
        te = 2.0 * psi[:, 2] - 1.0
        act[:, 2] = 1.0 / (1.0 + np.exp(-hp.gamma * te * (s[:, 2] - 0.5)))
    if n_dim > 3:
        act[:, 3] = psi[:, 3] * s[:, 3] + (1.0 - psi[:, 3]) * (1.0 - s[:, 3])
    if n_dim > 4:
        act[:, 4] = np.exp(-hp.delta * psi[:, 4] * s[:, 4] ** 2)
    if n_dim > 5:
        act[:, 5] = psi[:, 5] * (1.0 - np.exp(-hp.rho * s[:, 5]))
    if n_dim > 6:
        act[:, 6] = psi[:, 6] * s[:, 6] + (1.0 - psi[:, 6]) * (1.0 - s[:, 6])
    if n_dim > 7:
        act[:, 7] = psi[:, 7] * s[:, 7] + (1.0 - psi[:, 7]) * (1.0 - s[:, 7])

    return act

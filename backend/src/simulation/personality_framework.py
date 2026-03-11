"""
Personality Vector Simulation Framework (v3)
=============================================

A general-purpose, extensible framework for modeling personality-driven
decision-making under multidimensional scenarios.

Core model:
    - An N-dimensional personality vector ψ ∈ [0, 1]^N
    - An N-dimensional scenario stimulus vector s ∈ [0, 1]^N
    - A set of activation functions f_i(s_i, ψ_i) → [0, 1] (normalized)
    - A utility function U(ψ, s, a) → ℝ that scores actions
    - A decision rule (softmax/Boltzmann) parameterized by temperature τ

Default dimensions include the Big Five (OCEAN) plus three optional
cultural/contextual dimensions (Resilience, Idealism, Tradition), but
the framework supports arbitrary dimension registries.

Changelog (v3):
    - Added `activations_override` to utility() and decide() so higher
      layers can inject state-modulated activations into the choice pipeline.

Changelog (v2):
    - Fixed dead `practical_value` parameter in f_idealism
    - Fixed sigmoid polarity for f_extraversion and f_conscientiousness
      using (2·trait - 1) centering so both poles of each trait are active
    - Added [0, 1] validation on all bounded fields
    - Normalized all activation functions to output [0, 1]
    - Added Action.modifiers shape validation
    - Added utility function and Boltzmann decision rule
    - Made dimension set configurable via DimensionRegistry
    - Removed unused imports
    - Added resilience mode toggle (activation vs. buffer)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Sequence
from enum import Enum


# =============================================================================
# 0. VALIDATION UTILITIES
# =============================================================================

def _validate_unit_interval(name: str, value: float) -> None:
    """Raise ValueError if value is not in [0, 1]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name}={value} is outside the required [0, 1] interval.")


def _validate_real(name: str, value: float) -> None:
    """Raise ValueError if value is not finite."""
    if not np.isfinite(value):
        raise ValueError(f"{name}={value} is not finite.")


# =============================================================================
# 1. DIMENSION REGISTRY
# =============================================================================

@dataclass(frozen=True)
class Dimension:
    """
    A single personality/scenario dimension.

    Attributes:
        key:         Short identifier (e.g. 'O', 'C', 'R').
        name:        Human-readable name.
        description: What this dimension measures.
    """
    key: str
    name: str
    description: str


# Default OCEAN + RIT dimensions.  Users can define any set.
DEFAULT_DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("O", "Openness", "Receptivity to novel experience and creative stimuli."),
    Dimension("C", "Conscientiousness", "Preference for structure, planning, and order."),
    Dimension("E", "Extraversion", "Energy gain/drain from social interaction."),
    Dimension("A", "Agreeableness", "Preference for cooperation over competition."),
    Dimension("N", "Neuroticism", "Sensitivity to stress (high N = fragile under stress)."),
    Dimension("R", "Resilience", "Capacity to mobilize or endure under adversity."),
    Dimension("I", "Idealism", "Priority of ideal outcomes over pragmatic ones."),
    Dimension("T", "Tradition", "Preference for tradition-aligned over novel approaches."),
)


class DimensionRegistry:
    """
    Maintains an ordered set of dimensions and provides index lookups.

    This decouples the framework from a hardcoded 8-dimensional structure:
    you can register 5 dimensions (pure OCEAN), 8 (OCEAN + RIT), or any
    custom set.
    """

    def __init__(self, dimensions: Sequence[Dimension] = DEFAULT_DIMENSIONS):
        self._dims = tuple(dimensions)
        self._index = {d.key: i for i, d in enumerate(self._dims)}
        if len(self._index) != len(self._dims):
            dupes = [d.key for d in self._dims]
            raise ValueError(f"Duplicate dimension keys: {dupes}")

    @property
    def size(self) -> int:
        return len(self._dims)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(d.key for d in self._dims)

    def index(self, key: str) -> int:
        return self._index[key]

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"DimensionRegistry({self.keys})"


# =============================================================================
# 2. CORE DATA STRUCTURES
# =============================================================================

@dataclass
class PersonalityVector:
    """
    N-dimensional personality vector.  All values must lie in [0, 1].

    Construction:
        PersonalityVector(values={'O': 0.8, 'C': 0.6, ...}, registry=reg)
        PersonalityVector(array=np.array([0.8, 0.6, ...]), registry=reg)
    """
    _array: np.ndarray
    _registry: DimensionRegistry

    def __init__(
        self,
        *,
        values: Optional[dict[str, float]] = None,
        array: Optional[np.ndarray] = None,
        registry: DimensionRegistry = DimensionRegistry(),
    ):
        object.__setattr__(self, '_registry', registry)

        if values is not None and array is not None:
            raise ValueError("Provide either `values` or `array`, not both.")

        if values is not None:
            arr = np.zeros(registry.size)
            for key, val in values.items():
                _validate_unit_interval(key, val)
                arr[registry.index(key)] = val
            object.__setattr__(self, '_array', arr)

        elif array is not None:
            arr = np.asarray(array, dtype=float)
            if arr.shape != (registry.size,):
                raise ValueError(
                    f"Expected array of shape ({registry.size},), got {arr.shape}"
                )
            for i, key in enumerate(registry.keys):
                _validate_unit_interval(key, arr[i])
            object.__setattr__(self, '_array', arr)

        else:
            raise ValueError("Must provide either `values` or `array`.")

    def __getitem__(self, key: str) -> float:
        return float(self._array[self._registry.index(key)])

    def to_array(self) -> np.ndarray:
        return self._array.copy()

    @property
    def registry(self) -> DimensionRegistry:
        return self._registry

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{k}={self._array[i]:.2f}" for i, k in enumerate(self._registry.keys)
        )
        return f"ψ({pairs})"


@dataclass
class Scenario:
    """
    N-dimensional scenario stimulus vector.  All values in [0, 1].

    Each component is the stimulus intensity for the corresponding
    personality dimension (e.g., index 0 = creative stimulus if
    dimension 0 is Openness).
    """
    _array: np.ndarray
    _registry: DimensionRegistry
    name: str = ""
    description: str = ""

    def __init__(
        self,
        *,
        values: Optional[dict[str, float]] = None,
        array: Optional[np.ndarray] = None,
        registry: DimensionRegistry = DimensionRegistry(),
        name: str = "",
        description: str = "",
    ):
        object.__setattr__(self, '_registry', registry)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'description', description)

        if values is not None and array is not None:
            raise ValueError("Provide either `values` or `array`, not both.")

        if values is not None:
            arr = np.zeros(registry.size)
            for key, val in values.items():
                _validate_unit_interval(key, val)
                arr[registry.index(key)] = val
            object.__setattr__(self, '_array', arr)

        elif array is not None:
            arr = np.asarray(array, dtype=float)
            if arr.shape != (registry.size,):
                raise ValueError(
                    f"Expected array of shape ({registry.size},), got {arr.shape}"
                )
            for i, key in enumerate(registry.keys):
                _validate_unit_interval(key, arr[i])
            object.__setattr__(self, '_array', arr)

        else:
            raise ValueError("Must provide either `values` or `array`.")

    def __getitem__(self, key: str) -> float:
        return float(self._array[self._registry.index(key)])

    def to_array(self) -> np.ndarray:
        return self._array.copy()

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{k}={self._array[i]:.2f}" for i, k in enumerate(self._registry.keys)
        )
        return f"S({pairs}; '{self.name}')"


@dataclass
class Action:
    """
    A possible action in response to a scenario.

    `modifiers` is an N-dimensional vector of the same shape as the
    personality/scenario vectors.  It represents how well-suited this
    action is along each dimension (values in [-1, 1]).
    """
    name: str
    description: str
    _modifiers: np.ndarray
    _registry: DimensionRegistry

    def __init__(
        self,
        name: str,
        description: str,
        modifiers: np.ndarray,
        registry: DimensionRegistry = DimensionRegistry(),
    ):
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'description', description)
        object.__setattr__(self, '_registry', registry)

        m = np.asarray(modifiers, dtype=float)
        if m.shape != (registry.size,):
            raise ValueError(
                f"modifiers must have shape ({registry.size},), got {m.shape}"
            )
        object.__setattr__(self, '_modifiers', m)

    @property
    def modifiers(self) -> np.ndarray:
        return self._modifiers.copy()

    def __repr__(self) -> str:
        return f"Action('{self.name}')"


# =============================================================================
# 3. HYPERPARAMETERS
# =============================================================================

@dataclass
class HyperParameters:
    """
    Tunable constants for the activation functions.

    These are separated from the functions so they can be configured
    per simulation run, per personality, or per scenario without
    modifying the activation logic.
    """
    alpha: float = 3.0       # Openness: tanh steepness
    beta: float = 5.0        # Conscientiousness: sigmoid steepness
    c_threshold: float = 0.3  # Conscientiousness: structure threshold
    gamma: float = 4.0       # Extraversion: sigmoid steepness
    delta: float = 3.0       # Neuroticism: Gaussian width
    rho: float = 4.0         # Resilience: saturation rate

    def __post_init__(self):
        for name, val in vars(self).items():
            _validate_real(name, val)


class ResilienceMode(Enum):
    """
    Controls how the resilience activation function interprets adversity.

    ACTIVATION: Adversity mobilizes the resilient person (output rises
                with adversity).  Metaphor: "a muscle that fires under load."
    BUFFER:     Adversity is a penalty; resilience reduces the penalty.
                Output = 1 - adversity * (1 - R).  Metaphor: "armor."
    """
    ACTIVATION = "activation"
    BUFFER = "buffer"


# =============================================================================
# 4. ACTIVATION FUNCTIONS
# =============================================================================

class ActivationFunctions:
    """
    Activation functions f_i : [0,1] × [0,1] → [0,1].

    CONTRACT: every function in this class maps (stimulus, trait) to
    a value in [0, 1].  Downstream code (utility computation, action
    scoring) depends on this guarantee.

    Design note on centering
    -------------------------
    For dimensions where the trait represents a bipolar continuum
    (e.g., Extraversion: introversion ↔ extraversion), we center the
    trait at 0.5 via  t_centered = 2·trait - 1  ∈ [-1, 1].

    This ensures:
        trait < 0.5  →  negative slope  →  aversion to high stimulus
        trait = 0.5  →  flat            →  indifference
        trait > 0.5  →  positive slope  →  attraction to high stimulus

    For dimensions where the trait is a *magnitude* (e.g., Openness:
    zero means "closed", one means "very open", there is no meaningful
    "anti-openness"), the trait scales the amplitude directly.
    """

    @staticmethod
    def f_openness(stimulus: float, trait: float, hp: HyperParameters) -> float:
        """
        f_O(s, O) = O · tanh(α · s)

        Output range: [0, O] ⊂ [0, 1].
        Rationale: open individuals respond more strongly to creative
        stimuli, with saturation (no infinite creativity).
        O is a magnitude (no meaningful "anti-openness"), so no centering.
        """
        return trait * np.tanh(hp.alpha * stimulus)

    @staticmethod
    def f_conscientiousness(stimulus: float, trait: float, hp: HyperParameters) -> float:
        """
        f_C(s, C) = σ( β · (2C−1) · (s − θ) )

        Output range: (0, 1).
        Rationale: the trait is bipolar (low-C people actively prefer
        unstructured environments; high-C people prefer structure).
        Centering at 0.5 via (2C−1) gives:
            C < 0.5, high structure → output < 0.5  (discomfort)
            C > 0.5, high structure → output > 0.5  (comfort)
        """
        t_centered = 2.0 * trait - 1.0
        return 1.0 / (1.0 + np.exp(-hp.beta * t_centered * (stimulus - hp.c_threshold)))

    @staticmethod
    def f_extraversion(stimulus: float, trait: float, hp: HyperParameters) -> float:
        """
        f_E(s, E) = σ( γ · (2E−1) · (s − 0.5) )

        Output range: (0, 1).
        Rationale: bipolar trait.  Introverts (E < 0.5) are drained by
        high social stimulus; extraverts (E > 0.5) are energized.
        The inflection point is at s = 0.5 (moderate social demand).
        """
        t_centered = 2.0 * trait - 1.0
        return 1.0 / (1.0 + np.exp(-hp.gamma * t_centered * (stimulus - 0.5)))

    @staticmethod
    def f_agreeableness(stimulus: float, trait: float, _hp: HyperParameters) -> float:
        """
        f_A(s, A) = A·s + (1−A)·(1−s) = (2A−1)·s + (1−A)

        Output range: [0, 1].
        Rationale: linear interpolation.  High A prefers cooperation (high s),
        low A prefers competition (low s).  Algebraically this is a linear
        ramp whose direction flips at A = 0.5.
        """
        return trait * stimulus + (1.0 - trait) * (1.0 - stimulus)

    @staticmethod
    def f_neuroticism(stimulus: float, trait: float, hp: HyperParameters) -> float:
        """
        f_N(s, N) = exp(−δ · N · s²)

        Output range: (0, 1].
        Rationale: Gaussian-like decay under stress.  High N (neurotic)
        causes steep performance collapse as stress rises.
        Low N (emotionally stable) produces near-flat response.

        NOTE ON POLARITY: This function returns HIGH values when stress
        is LOW, i.e., it measures effective functioning / emotional
        stability, not raw neuroticism.  The trait label N follows the
        standard OCEAN naming, but the *output* should be read as
        "how well the person is functioning under this stress level."
        """
        return np.exp(-hp.delta * trait * stimulus ** 2)

    @staticmethod
    def f_resilience(
        stimulus: float,
        trait: float,
        hp: HyperParameters,
        mode: ResilienceMode = ResilienceMode.ACTIVATION,
    ) -> float:
        """
        ACTIVATION mode:
            f_R(s, R) = R · (1 − exp(−ρ · s))
            Output range: [0, R] ⊂ [0, 1].
            Metaphor: adversity mobilizes the resilient person.

        BUFFER mode:
            f_R(s, R) = 1 − s · (1 − R)
            Output range: [R, 1] ⊂ [0, 1].
            Metaphor: adversity is a penalty; resilience reduces it.

        Both modes satisfy f ∈ [0, 1] for inputs in [0, 1].
        Choose the mode that fits your domain semantics.
        """
        if mode == ResilienceMode.ACTIVATION:
            return trait * (1.0 - np.exp(-hp.rho * stimulus))
        else:
            return 1.0 - stimulus * (1.0 - trait)

    @staticmethod
    def f_idealism(stimulus: float, trait: float, _hp: HyperParameters) -> float:
        """
        f_I(s, I) = I·s + (1−I)·(1−s)

        Output range: [0, 1].
        Rationale: identical form to agreeableness/tradition.
        High I prioritizes the ideal value of the scenario;
        low I (pragmatist) prioritizes the practical complement.

        v2 FIX: removed the dead `practical_value` parameter.
        If you need an independent practical dimension, add it as
        a separate dimension in the registry.
        """
        return trait * stimulus + (1.0 - trait) * (1.0 - stimulus)

    @staticmethod
    def f_tradition(stimulus: float, trait: float, _hp: HyperParameters) -> float:
        """
        f_T(s, T) = T·s + (1−T)·(1−s)

        Output range: [0, 1].
        Rationale: identical form to agreeableness/idealism.
        High T prefers tradition-aligned scenarios;
        low T (innovator) prefers novelty.
        """
        return trait * stimulus + (1.0 - trait) * (1.0 - stimulus)


# Default activation function registry, keyed by dimension key.
# Users can override individual functions or supply entirely new ones.
DEFAULT_ACTIVATION_REGISTRY: dict[str, Callable[..., float]] = {
    "O": ActivationFunctions.f_openness,
    "C": ActivationFunctions.f_conscientiousness,
    "E": ActivationFunctions.f_extraversion,
    "A": ActivationFunctions.f_agreeableness,
    "N": ActivationFunctions.f_neuroticism,
    "R": ActivationFunctions.f_resilience,
    "I": ActivationFunctions.f_idealism,
    "T": ActivationFunctions.f_tradition,
}


# =============================================================================
# 5. DECISION ENGINE
# =============================================================================

class DecisionEngine:
    """
    Computes utility scores for actions and selects via Boltzmann sampling.

    The utility of an action a given personality ψ and scenario s is:

        U(ψ, s, a) = bias + Σ_i  f_i(s_i, ψ_i) · a_modifiers_i

    where f_i is the activation function for dimension i.

    The probability of choosing action a from a set A is:

        P(a | ψ, s, τ) = exp(U(a) / τ) / Σ_{a' ∈ A} exp(U(a') / τ)

    where τ (temperature) controls exploration:
        τ → 0:   deterministic (argmax)
        τ → ∞:   uniform random
    """

    def __init__(
        self,
        registry: DimensionRegistry = DimensionRegistry(),
        activation_registry: Optional[dict[str, Callable[..., float]]] = None,
        hyperparameters: HyperParameters = HyperParameters(),
        resilience_mode: ResilienceMode = ResilienceMode.ACTIVATION,
    ):
        self.registry = registry
        self.hp = hyperparameters
        self.resilience_mode = resilience_mode
        self.activations = activation_registry or {
            k: v for k, v in DEFAULT_ACTIVATION_REGISTRY.items()
            if k in set(registry.keys)
        }

        # Validate that every dimension has an activation function.
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
        """
        Compute the N-dimensional activation vector.

        Returns:
            np.ndarray of shape (N,) with each entry in [0, 1].
        """
        psi = personality.to_array()
        s = scenario.to_array()
        n = self.registry.size
        activations = np.zeros(n)

        for i, key in enumerate(self.registry.keys):
            fn = self.activations[key]
            # Resilience gets the extra `mode` kwarg.
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
        """
        U(ψ, s, a) = bias + activations · modifiers

        Args:
            activations_override: If provided, use these pre-computed
                activations instead of computing from personality × scenario.
                This allows external layers (e.g., temporal state modulation)
                to inject state-modified activations into the choice pipeline.

        Returns a scalar utility score (unbounded).
        """
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
        """
        Boltzmann (softmax) action selection.

        Args:
            temperature: τ > 0.  Lower = more deterministic.
            bias:        Additive bias on all utilities.
            rng:         NumPy random generator (for reproducibility).
            activations_override: If provided, use these instead of
                recomputing from personality × scenario.

        Returns:
            (chosen_action, probability_vector)
        """
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

        # Numerically stable softmax.
        logits = utilities / temperature
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        chosen_idx = rng.choice(len(actions), p=probs)
        return actions[chosen_idx], probs


# =============================================================================
# 6. CONVENIENCE / BATCH OPERATIONS
# =============================================================================

def compute_activation_batch(
    personalities: np.ndarray,
    scenarios: np.ndarray,
    hp: HyperParameters = HyperParameters(),
) -> np.ndarray:
    """
    Vectorized activation computation for batch simulation.

    Args:
        personalities: (batch, N) array of personality vectors.
        scenarios:     (batch, N) array of scenario stimuli.
        hp:            Hyperparameters.

    Returns:
        (batch, N) array of activations.

    NOTE: This uses the default OCEAN+RIT activation forms and assumes
    the standard 8-dimensional layout.  For custom registries, iterate
    through DecisionEngine.compute_activations instead.
    """
    psi = np.asarray(personalities)
    s = np.asarray(scenarios)
    if psi.shape != s.shape or psi.ndim != 2:
        raise ValueError(f"Shape mismatch: personalities {psi.shape} vs scenarios {s.shape}")

    n_dim = psi.shape[1]
    act = np.zeros_like(psi)

    # O: magnitude scaling, tanh
    act[:, 0] = psi[:, 0] * np.tanh(hp.alpha * s[:, 0])

    # C: centered sigmoid
    if n_dim > 1:
        tc = 2.0 * psi[:, 1] - 1.0
        act[:, 1] = 1.0 / (1.0 + np.exp(-hp.beta * tc * (s[:, 1] - hp.c_threshold)))

    # E: centered sigmoid
    if n_dim > 2:
        te = 2.0 * psi[:, 2] - 1.0
        act[:, 2] = 1.0 / (1.0 + np.exp(-hp.gamma * te * (s[:, 2] - 0.5)))

    # A: linear interpolation
    if n_dim > 3:
        act[:, 3] = psi[:, 3] * s[:, 3] + (1.0 - psi[:, 3]) * (1.0 - s[:, 3])

    # N: Gaussian decay
    if n_dim > 4:
        act[:, 4] = np.exp(-hp.delta * psi[:, 4] * s[:, 4] ** 2)

    # R: saturation (activation mode)
    if n_dim > 5:
        act[:, 5] = psi[:, 5] * (1.0 - np.exp(-hp.rho * s[:, 5]))

    # I: linear interpolation
    if n_dim > 6:
        act[:, 6] = psi[:, 6] * s[:, 6] + (1.0 - psi[:, 6]) * (1.0 - s[:, 6])

    # T: linear interpolation
    if n_dim > 7:
        act[:, 7] = psi[:, 7] * s[:, 7] + (1.0 - psi[:, 7]) * (1.0 - s[:, 7])

    return act


# =============================================================================
# 7. DEMO / SMOKE TEST
# =============================================================================

def _demo() -> None:
    """Run a minimal smoke test."""
    reg = DimensionRegistry()
    engine = DecisionEngine(registry=reg)

    # A sample personality.
    psi = PersonalityVector(
        values={"O": 0.85, "C": 0.40, "E": 0.25, "A": 0.70,
                "N": 0.60, "R": 0.90, "I": 0.75, "T": 0.30},
        registry=reg,
    )

    # A high-stress, high-adversity, creative scenario.
    scenario = Scenario(
        values={"O": 0.9, "C": 0.2, "E": 0.7, "A": 0.5,
                "N": 0.8, "R": 0.9, "I": 0.6, "T": 0.3},
        registry=reg,
        name="Crisis requiring creative leadership",
    )

    # Two possible actions.
    act_bold = Action(
        "Bold move", "Take a risky creative approach",
        modifiers=np.array([1.0, -0.5, 0.5, 0.3, -0.3, 0.8, 0.7, -0.5]),
        registry=reg,
    )
    act_safe = Action(
        "Safe move", "Follow established procedure",
        modifiers=np.array([0.2, 0.9, 0.1, 0.5, 0.5, 0.1, 0.2, 0.8]),
        registry=reg,
    )

    print(f"Personality: {psi}")
    print(f"Scenario:    {scenario}")
    print()

    activations = engine.compute_activations(psi, scenario)
    print(f"Activations: {np.round(activations, 3)}")
    print()

    u_bold = engine.utility(psi, scenario, act_bold)
    u_safe = engine.utility(psi, scenario, act_safe)
    print(f"U(bold)  = {u_bold:.4f}")
    print(f"U(safe)  = {u_safe:.4f}")
    print()

    chosen, probs = engine.decide(
        psi, scenario, [act_bold, act_safe],
        temperature=0.5, rng=np.random.default_rng(42),
    )
    print(f"P(bold)  = {probs[0]:.4f}")
    print(f"P(safe)  = {probs[1]:.4f}")
    print(f"Chosen:  {chosen}")

    # Validation smoke test.
    print("\n--- Validation tests ---")
    try:
        PersonalityVector(values={"O": 1.5}, registry=reg)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        Action("bad", "bad", modifiers=np.array([1, 2, 3]), registry=reg)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    _demo()

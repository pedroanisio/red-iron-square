"""Precision and prediction error computation engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.personality.vectors import PersonalityVector, Scenario
from src.precision.params import PrecisionParams
from src.precision.setpoints import AllostaticSetPoints
from src.precision.state import PrecisionState, PredictionErrors

if TYPE_CHECKING:
    from src.temporal.state import AgentState


class PredictionErrorEngine:
    """Compute Level-0 allostatic prediction errors.

    ``epsilon_0_i(t) = s_i(t) - s_hat_i(theta)``

    where ``s_hat`` is the personality-dependent allostatic set-point.
    """

    def __init__(self) -> None:
        self._setpoints = AllostaticSetPoints()

    def compute(
        self,
        state: AgentState,
        personality: PersonalityVector,
    ) -> PredictionErrors:
        """Compute prediction errors at Level 0."""
        observed = state.to_array()
        predicted = self._setpoints.compute(personality)
        return PredictionErrors(level_0=observed - predicted)


class PrecisionEngine:
    """Compute precision from personality, state, and context.

    ``Pi_l(t) = softplus(W_l . sigma(theta) + V_l . s(t) + U_l . c(t) + b_l)``

    Shadow-only in Phase A: computed and tracked but does not
    influence decisions.
    """

    def __init__(self, params: PrecisionParams | None = None) -> None:
        self._params = params or PrecisionParams()
        self._error_engine = PredictionErrorEngine()
        self._W: dict[int, np.ndarray] = {}
        self._V: dict[int, np.ndarray] = {}
        self._U: dict[int, np.ndarray] = {}
        self._b: dict[int, np.ndarray] = {}
        self._init_parameters()

    @property
    def error_engine(self) -> PredictionErrorEngine:
        """Expose the prediction error engine for external use."""
        return self._error_engine

    def _init_parameters(self) -> None:
        """Supervise initialization from handcrafted coefficients."""
        p = self._params
        n_p, n_s, n_c = p.n_personality, p.n_state, p.n_context
        n_i = p.n_interoceptive

        self._init_level_0(n_i, n_p, n_s, n_c, p)
        self._init_level_1(n_p, n_s, n_c, p)
        self._init_level_2(n_p, n_s, n_c, p)

    def _init_level_0(
        self, n_out: int, n_p: int, n_s: int, n_c: int, p: PrecisionParams
    ) -> None:
        """Initialize Level 0 — interoceptive precision."""
        W = np.zeros((n_out, n_p))
        V = np.zeros((n_out, n_s))
        U = np.zeros((n_out, n_c))
        b = np.full(n_out, p.default_bias)

        # Personality -> interoceptive precision mappings
        W[0, 4] = p.n_mood_precision_weight  # N -> mood precision
        W[1, 2] = p.e_arousal_precision_weight  # E -> arousal precision
        W[4, 5] = p.r_frustration_precision_weight  # R -> frustration precision
        W[3, 4] = p.n_satisfaction_precision_weight  # N -> satisfaction precision

        # State modulation: arousal/frustration amplify own precision
        V[1, 1] = 0.15  # arousal state -> arousal precision
        V[4, 4] = 0.15  # frustration state -> frustration precision
        V[0, 0] = -0.10  # negative mood -> higher mood precision

        # Context modulation: scenario stress raises precision
        U[0, 4] = 0.10  # scenario N -> mood precision
        U[4, 4] = 0.10  # scenario N -> frustration precision

        self._W[0] = W
        self._V[0] = V
        self._U[0] = U
        self._b[0] = b

    def _init_level_1(self, n_p: int, n_s: int, n_c: int, p: PrecisionParams) -> None:
        """Initialize Level 1 — policy precision."""
        W = np.zeros((1, n_p))
        V = np.zeros((1, n_s))
        U = np.zeros((1, n_c))
        b = np.full(1, p.default_bias)

        W[0, 2] = p.e_policy_precision_weight  # E -> policy precision

        self._W[1] = W
        self._V[1] = V
        self._U[1] = U
        self._b[1] = b

    def _init_level_2(self, n_p: int, n_s: int, n_c: int, p: PrecisionParams) -> None:
        """Initialize Level 2 — narrative precision."""
        W = np.zeros((1, n_p))
        V = np.zeros((1, n_s))
        U = np.zeros((1, n_c))
        b = np.full(1, p.default_bias)

        W[0, 7] = p.t_narrative_precision_weight  # T -> narrative precision

        self._W[2] = W
        self._V[2] = V
        self._U[2] = U
        self._b[2] = b

    def compute(
        self,
        personality: PersonalityVector,
        state: AgentState,
        scenario: Scenario,
    ) -> PrecisionState:
        """Compute precision at all hierarchy levels."""
        theta = self._sigmoid(personality.to_array())
        s = state.to_array()
        c = scenario.to_array()

        l0 = self._W[0] @ theta + self._V[0] @ s + self._U[0] @ c + self._b[0]
        l1 = self._W[1] @ theta + self._V[1] @ s + self._U[1] @ c + self._b[1]
        l2 = self._W[2] @ theta + self._V[2] @ s + self._U[2] @ c + self._b[2]

        return PrecisionState(
            level_0=self._softplus(l0),
            level_1=float(self._softplus(l1)[0]),
            level_2=float(self._softplus(l2)[0]),
        )

    def compute_errors(
        self,
        state: AgentState,
        personality: PersonalityVector,
    ) -> PredictionErrors:
        """Compute Level-0 prediction errors (delegates to error engine)."""
        return self._error_engine.compute(state, personality)

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        """Compute numerically stable softplus: log(1 + exp(x))."""
        return np.where(x > 20, x, np.log1p(np.exp(x)))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Compute element-wise sigmoid: 1 / (1 + exp(-x))."""
        result: np.ndarray = np.where(
            x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))
        )
        return result

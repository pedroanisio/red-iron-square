"""Information-gain epistemic value from generative model matrices (§10).

Replaces the memory-variance approximation with proper Bayesian
information gain computed from the narrative generative model's
A-matrix (observation likelihood).

Information gain for action a:
    G(a) = sum_o p(o|s,a) * [H[p(s|o,a)] - H[p(s|a)]]

This is the expected reduction in state uncertainty from observing
outcomes under each action.
"""

from __future__ import annotations

import numpy as np


def compute_info_gain(
    a_matrix: np.ndarray,
    belief: np.ndarray,
    action_idx: int,
) -> float:
    """Compute expected information gain for a single action.

    Args:
        a_matrix: Observation likelihood p(o|s,a), shape (n_obs, n_states, n_actions).
        belief: Current state belief distribution, shape (n_states,).
        action_idx: Index of the action to evaluate.

    Returns:
        Expected information gain in nats (non-negative).
    """
    # p(o|a) = sum_s p(o|s,a) * p(s)  -- predicted observation distribution
    likelihood = a_matrix[:, :, action_idx]  # (n_obs, n_states)
    p_o_given_a = likelihood @ belief  # (n_obs,)
    p_o_given_a = np.maximum(p_o_given_a, 1e-16)

    # H[p(s|a)] = H[belief] -- prior entropy (constant across observations)
    h_prior = _entropy(belief)

    # Expected posterior entropy: sum_o p(o|a) * H[p(s|o,a)]
    expected_h_posterior = 0.0
    for o_idx in range(likelihood.shape[0]):
        p_s_given_o = likelihood[o_idx, :] * belief
        total = p_s_given_o.sum()
        if total < 1e-16:
            continue
        p_s_given_o = p_s_given_o / total
        expected_h_posterior += p_o_given_a[o_idx] * _entropy(p_s_given_o)

    # Information gain = prior entropy - expected posterior entropy
    return float(max(0.0, h_prior - expected_h_posterior))


def compute_all_info_gains(
    a_matrix: np.ndarray,
    belief: np.ndarray,
) -> np.ndarray:
    """Compute information gain for all actions.

    Args:
        a_matrix: Observation likelihood, shape (n_obs, n_states, n_actions).
        belief: Current state belief, shape (n_states,).

    Returns:
        Array of information gains, shape (n_actions,).
    """
    n_actions = a_matrix.shape[2]
    gains = np.zeros(n_actions)
    for a in range(n_actions):
        gains[a] = compute_info_gain(a_matrix, belief, a)
    return gains


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats."""
    p = p[p > 1e-16]
    return float(-np.sum(p * np.log(p)))

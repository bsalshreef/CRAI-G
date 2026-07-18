"""
optimizer.py
------------
Adaptive weight optimization for CRAI-G using a log-transformed max-margin
program with L2 regularization, solved via projected gradient ascent.
Guarantees a unique global optimum under KKT conditions (Boyd & Vandenberghe, 2004).
"""

import numpy as np
from scipy.optimize import minimize


def _objective(w: np.ndarray, V_safe: np.ndarray, V_unsafe: np.ndarray, gamma: float) -> float:
    """
    Strictly concave objective J(w):
      J(w) = min_{V in M_safe} sum(w * log(V)) - max_{V in M_unsafe} sum(w * log(V)) - gamma * ||w||^2

    Returns the NEGATIVE (for minimization with scipy).
    """
    log_safe = np.log(np.clip(V_safe, 1e-9, None))   # shape (n_safe, 5)
    log_unsafe = np.log(np.clip(V_unsafe, 1e-9, None))  # shape (n_unsafe, 5)

    scores_safe = log_safe @ w       # shape (n_safe,)
    scores_unsafe = log_unsafe @ w   # shape (n_unsafe,)

    J = scores_safe.min() - scores_unsafe.max() - gamma * np.dot(w, w)
    return -J  # negate for minimization


def _gradient(w: np.ndarray, V_safe: np.ndarray, V_unsafe: np.ndarray, gamma: float) -> np.ndarray:
    """
    Subgradient of J(w) (negated for minimization).
    """
    log_safe = np.log(np.clip(V_safe, 1e-9, None))
    log_unsafe = np.log(np.clip(V_unsafe, 1e-9, None))

    scores_safe = log_safe @ w
    scores_unsafe = log_unsafe @ w

    idx_min_safe = np.argmin(scores_safe)
    idx_max_unsafe = np.argmax(scores_unsafe)

    grad = log_safe[idx_min_safe] - log_unsafe[idx_max_unsafe] - 2 * gamma * w
    return -grad  # negate for minimization


def optimize_weights(
    V_safe: np.ndarray,
    V_unsafe: np.ndarray,
    gamma: float = 0.01,
    epsilon_w: float = 0.05,
    n_restarts: int = 5,
    seed: int = 42
) -> np.ndarray:
    """
    Solve the adaptive weight optimization program.

    Parameters
    ----------
    V_safe : np.ndarray, shape (n_safe, 5)
        Readiness vectors of historically safe models.
    V_unsafe : np.ndarray, shape (n_unsafe, 5)
        Readiness vectors of historically unsafe models.
    gamma : float
        L2 regularization strength (ensures strict concavity).
    epsilon_w : float
        Minimum weight per dimension (prevents ignoring any pillar).
    n_restarts : int
        Number of random restarts to ensure global optimum.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    w_star : np.ndarray, shape (5,)
        Optimal weight vector satisfying sum(w) = 1, w_i >= epsilon_w.
    """
    rng = np.random.default_rng(seed)
    n_dims = 5

    # Constraints: sum(w) = 1, w_i >= epsilon_w
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(epsilon_w, 1.0 - (n_dims - 1) * epsilon_w)] * n_dims

    best_val = np.inf
    best_w = None

    for _ in range(n_restarts):
        # Random initialization on the simplex
        w0 = rng.dirichlet(np.ones(n_dims))
        w0 = np.clip(w0, epsilon_w, None)
        w0 /= w0.sum()

        result = minimize(
            fun=_objective,
            x0=w0,
            args=(V_safe, V_unsafe, gamma),
            jac=_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000}
        )

        if result.fun < best_val:
            best_val = result.fun
            best_w = result.x

    # Final projection onto simplex with epsilon_w floor
    best_w = np.clip(best_w, epsilon_w, None)
    best_w /= best_w.sum()
    return best_w


def default_equal_weights(n_dims: int = 5) -> np.ndarray:
    """Return equal weights (1/n_dims each) as a fallback."""
    return np.ones(n_dims) / n_dims

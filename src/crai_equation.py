"""
crai_equation.py
----------------
Implements the unified CRAI-G equation:

  CRAI-G(V) = Phi(V) * exp(-alpha * V^T L V) * exp(-beta * [ln(5) - H(V_tilde)]) * prod(V_i^w_i)

where:
  Phi(V)    = differentiable sigmoid regulatory gate
  L         = graph Laplacian
  H(V_tilde)= Shannon entropy of L1-normalized V
  w         = adaptive weights
"""

import numpy as np
from scipy.special import expit  # sigmoid


def sigmoid_gate(V: np.ndarray, tau: float = 50.0, epsilon_v: float = 0.10) -> float:
    """
    Differentiable regulatory gate: Phi(V) = prod_i sigmoid(tau * (V_i - epsilon_v))

    Approaches 0 if any V_i < epsilon_v, approaches 1 if all V_i >> epsilon_v.
    """
    return float(np.prod(expit(tau * (V - epsilon_v))))


def laplacian_penalty(V: np.ndarray, L: np.ndarray, alpha: float = 1.0) -> float:
    """
    Spectral graph penalty: exp(-alpha * V^T L V)
    Penalizes inconsistencies between dependent dimensions.
    """
    quad = float(V @ L @ V)
    return float(np.exp(-alpha * quad))


def entropy_penalty(V: np.ndarray, beta: float = 0.5) -> float:
    """
    Information entropy penalty: exp(-beta * [ln(5) - H(V_tilde)])
    Rewards balanced models (maximum entropy = ln(5)).
    """
    V_tilde = V / (V.sum() + 1e-12)
    V_tilde = np.clip(V_tilde, 1e-12, None)
    H = -np.sum(V_tilde * np.log(V_tilde))
    ln5 = np.log(5)
    return float(np.exp(-beta * (ln5 - H)))


def geometric_mean(V: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted geometric mean: prod(V_i^w_i)
    """
    V_clipped = np.clip(V, 1e-12, None)
    return float(np.prod(V_clipped ** w))


def crai_g_score(
    V: np.ndarray,
    L: np.ndarray,
    w: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 50.0,
    epsilon_v: float = 0.10
) -> float:
    """
    Compute the CRAI-G score for a single model.

    Parameters
    ----------
    V : np.ndarray, shape (5,)
        Readiness vector [V1, V2, V3, V4, V5], each in [0, 1].
    L : np.ndarray, shape (5, 5)
        Graph Laplacian matrix.
    w : np.ndarray, shape (5,)
        Adaptive weight vector (sums to 1).
    alpha : float
        Laplacian penalty weight.
    beta : float
        Entropy penalty weight.
    tau : float
        Sigmoid gate temperature.
    epsilon_v : float
        Minimum acceptable pillar score.

    Returns
    -------
    score : float
        CRAI-G score in [0, 1].
    """
    V = np.asarray(V, dtype=float)
    if V.shape != (5,):
        raise ValueError("V must be a 1D array of length 5.")
    if not np.all((V >= 0) & (V <= 1)):
        raise ValueError("All pillar scores must be in [0, 1].")

    phi = sigmoid_gate(V, tau=tau, epsilon_v=epsilon_v)
    lap = laplacian_penalty(V, L, alpha=alpha)
    ent = entropy_penalty(V, beta=beta)
    geo = geometric_mean(V, w)

    score = phi * lap * ent * geo
    return float(np.clip(score, 0.0, 1.0))


def crai_g_batch(
    V_matrix: np.ndarray,
    L: np.ndarray,
    w: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 50.0,
    epsilon_v: float = 0.10
) -> np.ndarray:
    """
    Compute CRAI-G scores for a batch of models.

    Parameters
    ----------
    V_matrix : np.ndarray, shape (M, 5)
        Each row is a model's readiness vector.

    Returns
    -------
    scores : np.ndarray, shape (M,)
    """
    return np.array([
        crai_g_score(V_matrix[i], L, w, alpha, beta, tau, epsilon_v)
        for i in range(V_matrix.shape[0])
    ])

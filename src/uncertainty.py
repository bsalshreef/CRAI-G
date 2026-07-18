"""
uncertainty.py
--------------
First-order uncertainty propagation for CRAI-G using the Delta Method.

sigma_CRAI^2 = (grad f)^T * Sigma_V * (grad f)

where grad f is the gradient of CRAI-G with respect to V,
and Sigma_V is the bootstrap covariance matrix of the pillar estimates.
"""

import numpy as np
from src.crai_equation import crai_g_score


def numerical_gradient(
    V: np.ndarray,
    L: np.ndarray,
    w: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 50.0,
    epsilon_v: float = 0.10,
    h: float = 1e-5
) -> np.ndarray:
    """
    Compute the numerical gradient of CRAI-G with respect to V using central differences.

    Parameters
    ----------
    V : np.ndarray, shape (5,)
    h : float
        Step size for finite differences.

    Returns
    -------
    grad : np.ndarray, shape (5,)
    """
    grad = np.zeros(5)
    for i in range(5):
        V_plus = V.copy()
        V_minus = V.copy()
        V_plus[i] = min(1.0, V[i] + h)
        V_minus[i] = max(0.0, V[i] - h)
        f_plus = crai_g_score(V_plus, L, w, alpha, beta, tau, epsilon_v)
        f_minus = crai_g_score(V_minus, L, w, alpha, beta, tau, epsilon_v)
        grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


def bootstrap_covariance(
    V_samples: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> np.ndarray:
    """
    Estimate the covariance matrix Sigma_V of pillar estimates via percentile bootstrap.

    Parameters
    ----------
    V_samples : np.ndarray, shape (N, 5)
        Raw per-sample pillar scores (e.g., per-patient or per-image).
    n_bootstrap : int
        Number of bootstrap iterations.
    seed : int
        Random seed.

    Returns
    -------
    Sigma_V : np.ndarray, shape (5, 5)
        Bootstrap covariance matrix of the mean pillar vector.
    """
    rng = np.random.default_rng(seed)
    N = V_samples.shape[0]
    boot_means = np.zeros((n_bootstrap, 5))

    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        boot_means[b] = V_samples[idx].mean(axis=0)

    Sigma_V = np.cov(boot_means.T)
    return Sigma_V


def delta_method_ci(
    V: np.ndarray,
    Sigma_V: np.ndarray,
    L: np.ndarray,
    w: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
    tau: float = 50.0,
    epsilon_v: float = 0.10,
    z: float = 1.96
) -> dict:
    """
    Compute the CRAI-G score and its 95% confidence interval via the Delta Method.

    Returns
    -------
    result : dict with keys:
        'score'   : float — point estimate
        'sigma'   : float — standard deviation
        'ci_lower': float — lower 95% CI bound
        'ci_upper': float — upper 95% CI bound
    """
    score = crai_g_score(V, L, w, alpha, beta, tau, epsilon_v)
    grad = numerical_gradient(V, L, w, alpha, beta, tau, epsilon_v)
    var = float(grad @ Sigma_V @ grad)
    sigma = float(np.sqrt(max(var, 0.0)))

    return {
        "score": score,
        "sigma": sigma,
        "ci_lower": max(0.0, score - z * sigma),
        "ci_upper": min(1.0, score + z * sigma),
    }

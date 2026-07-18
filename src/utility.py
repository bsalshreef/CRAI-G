"""
utility.py
----------
Computes V5 = normalized integrated Net Benefit over clinically relevant thresholds.
Implements Decision Curve Analysis (Vickers & Elkin, 2006).
"""

import numpy as np


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    """
    Compute Net Benefit at a given decision threshold pt.

    NB(pt) = TP/N - FP/N * pt/(1-pt)
    """
    N = len(y_true)
    y_pred = (y_prob >= threshold).astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    if threshold >= 1.0:
        return 0.0

    nb = tp / N - fp / N * (threshold / (1.0 - threshold))
    return float(nb)


def integrated_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pt_min: float = 0.1,
    pt_max: float = 0.5,
    n_thresholds: int = 50
) -> float:
    """
    Compute the integrated Net Benefit over a range of clinically relevant thresholds.

    Returns the area under the NB curve, normalized by the maximum possible NB
    (treating all positives as true positives at pt_min).
    """
    thresholds = np.linspace(pt_min, pt_max, n_thresholds)
    nbs = np.array([net_benefit(y_true, y_prob, pt) for pt in thresholds])
    nbs = np.clip(nbs, 0.0, None)  # Negative NB = worse than treating none

    # Integrate using trapezoidal rule
    integrated = np.trapz(nbs, thresholds)

    # Normalize: maximum NB at pt_min is prevalence / N
    prevalence = y_true.mean()
    max_nb = prevalence * (1.0 - pt_min) / pt_min if pt_min > 0 else prevalence
    max_integrated = max_nb * (pt_max - pt_min)

    if max_integrated <= 0:
        return 0.0

    return float(np.clip(integrated / max_integrated, 0.0, 1.0))


def utility_pillar(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pt_min: float = 0.1,
    pt_max: float = 0.5
) -> float:
    """Return V5 = normalized integrated Net Benefit."""
    return integrated_net_benefit(y_true, y_prob, pt_min, pt_max)

"""
robustness.py
-------------
Computes V3 = distributional robustness as 1 - normalized Brier score degradation
under temporal/domain shift.
"""

import numpy as np
from sklearn.metrics import brier_score_loss


def brier_score_degradation(
    y_true_source: np.ndarray,
    y_prob_source: np.ndarray,
    y_true_shifted: np.ndarray,
    y_prob_shifted: np.ndarray
) -> float:
    """
    Compute the Brier score degradation under distribution shift.

    degradation = BS_shifted - BS_source  (higher = worse robustness)
    """
    bs_source = brier_score_loss(y_true_source, y_prob_source)
    bs_shifted = brier_score_loss(y_true_shifted, y_prob_shifted)
    return float(max(0.0, bs_shifted - bs_source))


def robustness_pillar(
    y_true_source: np.ndarray,
    y_prob_source: np.ndarray,
    y_true_shifted: np.ndarray,
    y_prob_shifted: np.ndarray,
    max_degradation: float = 0.25
) -> float:
    """
    Return V3 = 1 - (degradation / max_degradation), clipped to [0, 1].

    Parameters
    ----------
    max_degradation : float
        Maximum expected degradation for normalization (default 0.25).
    """
    deg = brier_score_degradation(
        y_true_source, y_prob_source,
        y_true_shifted, y_prob_shifted
    )
    return float(np.clip(1.0 - deg / max_degradation, 0.0, 1.0))

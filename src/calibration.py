"""
calibration.py
--------------
Computes V2 = 1 - ECE (Expected Calibration Error).
"""

import numpy as np


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute the Expected Calibration Error (ECE).

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Binary ground-truth labels.
    y_prob : np.ndarray, shape (N,)
        Predicted probabilities.
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    ece : float in [0, 1]
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / N) * abs(acc - conf)

    return float(ece)


def calibration_pillar(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Return V2 = 1 - ECE, clipped to [0, 1].
    """
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    return float(np.clip(1.0 - ece, 0.0, 1.0))

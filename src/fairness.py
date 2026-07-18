"""
fairness.py
-----------
Computes V4 = 1 - L_inf norm of FPR/FNR disparities across protected groups.
"""

import numpy as np


def group_fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> dict:
    """
    Compute per-group FPR and FNR.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_pred : np.ndarray, shape (N,) — binary predictions (0/1)
    group  : np.ndarray, shape (N,) — group labels

    Returns
    -------
    dict mapping group_label -> {'fpr': float, 'fnr': float}
    """
    results = {}
    for g in np.unique(group):
        mask = group == g
        yt = y_true[mask]
        yp = y_pred[mask]

        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()

        fpr = fp / (fp + tn + 1e-12)
        fnr = fn / (fn + tp + 1e-12)
        results[g] = {"fpr": float(fpr), "fnr": float(fnr)}

    return results


def fairness_pillar(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> float:
    """
    Return V4 = 1 - max(|FPR_i - FPR_j|, |FNR_i - FNR_j|) across all group pairs.

    A value of 1.0 indicates perfect fairness; lower values indicate disparate impact.
    """
    stats = group_fpr_fnr(y_true, y_pred, group)
    groups = list(stats.keys())

    if len(groups) < 2:
        return 1.0  # No disparity possible with one group

    max_disp = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            fpr_diff = abs(stats[groups[i]]["fpr"] - stats[groups[j]]["fpr"])
            fnr_diff = abs(stats[groups[i]]["fnr"] - stats[groups[j]]["fnr"])
            max_disp = max(max_disp, fpr_diff, fnr_diff)

    return float(np.clip(1.0 - max_disp, 0.0, 1.0))

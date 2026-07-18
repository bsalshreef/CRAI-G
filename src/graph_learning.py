"""
graph_learning.py
-----------------
Estimates the inter-pillar dependency graph for CRAI-G using Graphical Lasso
(Friedman et al., 2008). Returns the adjacency matrix A and Laplacian L.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from sklearn.preprocessing import StandardScaler


def estimate_precision_matrix(V_matrix: np.ndarray, lambda_lasso: float = None) -> np.ndarray:
    """
    Estimate the precision matrix Theta from a matrix of pillar scores.

    Parameters
    ----------
    V_matrix : np.ndarray, shape (M, 5)
        Each row is a model's readiness vector [V1, V2, V3, V4, V5].
    lambda_lasso : float or None
        L1 regularization strength. If None, uses cross-validation to select.

    Returns
    -------
    Theta_hat : np.ndarray, shape (5, 5)
        Estimated precision matrix (inverse covariance).
    """
    if V_matrix.shape[1] != 5:
        raise ValueError("V_matrix must have exactly 5 columns (one per pillar).")

    scaler = StandardScaler()
    V_scaled = scaler.fit_transform(V_matrix)

    if lambda_lasso is None:
        model = GraphicalLassoCV(cv=5, max_iter=500)
    else:
        model = GraphicalLasso(alpha=lambda_lasso, max_iter=500)

    model.fit(V_scaled)
    Theta_hat = model.precision_
    return Theta_hat


def build_adjacency_matrix(Theta_hat: np.ndarray) -> np.ndarray:
    """
    Build the adjacency matrix A from normalized partial correlations.

    A_ij = |Theta_ij| / sqrt(Theta_ii * Theta_jj)  for i != j
    A_ii = 0

    Parameters
    ----------
    Theta_hat : np.ndarray, shape (5, 5)
        Estimated precision matrix.

    Returns
    -------
    A : np.ndarray, shape (5, 5)
        Symmetric adjacency matrix with values in [0, 1].
    """
    n = Theta_hat.shape[0]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                denom = np.sqrt(Theta_hat[i, i] * Theta_hat[j, j])
                if denom > 1e-12:
                    A[i, j] = abs(Theta_hat[i, j]) / denom
    return A


def build_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Build the graph Laplacian L = D - A, where D is the degree matrix.

    Parameters
    ----------
    A : np.ndarray, shape (5, 5)
        Adjacency matrix.

    Returns
    -------
    L : np.ndarray, shape (5, 5)
        Graph Laplacian (positive semi-definite).
    """
    D = np.diag(A.sum(axis=1))
    L = D - A
    return L


def get_fiedler_value(L: np.ndarray) -> float:
    """
    Return the Fiedler value (second smallest eigenvalue of L).
    Measures algebraic connectivity of the graph.
    """
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues_sorted = np.sort(eigenvalues)
    return float(eigenvalues_sorted[1])


def load_meta_dataset(filepath: str) -> np.ndarray:
    """
    Load the meta-dataset CSV and return the V matrix.

    Parameters
    ----------
    filepath : str
        Path to CSV with columns: V1, V2, V3, V4, V5.

    Returns
    -------
    V_matrix : np.ndarray, shape (M, 5)
    """
    df = pd.read_csv(filepath)
    required_cols = ["V1", "V2", "V3", "V4", "V5"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from meta-dataset.")
    return df[required_cols].values

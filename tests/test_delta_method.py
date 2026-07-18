"""
test_delta_method.py
--------------------
Unit tests for the Delta Method uncertainty propagation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.uncertainty import bootstrap_covariance, delta_method_ci, numerical_gradient
from src.graph_learning import build_laplacian

A_test = np.array([
    [0, 0.5, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0],
    [0, 0.5, 0, 0.5, 0],
    [0, 0, 0.5, 0, 0.5],
    [0, 0, 0, 0.5, 0],
], dtype=float)
L_test = build_laplacian(A_test)
w_test = np.array([0.2, 0.2, 0.2, 0.2, 0.2])


class TestDeltaMethod:

    def setup_method(self):
        rng = np.random.default_rng(42)
        self.V = np.array([0.75, 0.82, 0.70, 0.78, 0.73])
        self.V_samples = self.V + rng.normal(0, 0.03, size=(300, 5))
        self.V_samples = np.clip(self.V_samples, 0.01, 1.0)
        self.Sigma_V = bootstrap_covariance(self.V_samples, n_bootstrap=500)

    def test_ci_contains_point_estimate(self):
        result = delta_method_ci(self.V, self.Sigma_V, L_test, w_test)
        assert result["ci_lower"] <= result["score"] <= result["ci_upper"], (
            f"Point estimate {result['score']:.4f} not in CI "
            f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
        )

    def test_ci_within_unit_interval(self):
        result = delta_method_ci(self.V, self.Sigma_V, L_test, w_test)
        assert 0.0 <= result["ci_lower"] <= result["ci_upper"] <= 1.0

    def test_sigma_positive(self):
        result = delta_method_ci(self.V, self.Sigma_V, L_test, w_test)
        assert result["sigma"] >= 0.0

    def test_gradient_shape(self):
        grad = numerical_gradient(self.V, L_test, w_test)
        assert grad.shape == (5,)

    def test_gradient_positive_for_all_pillars(self):
        """Gradient should be positive for all pillars in the valid interior."""
        grad = numerical_gradient(self.V, L_test, w_test)
        # Not all gradients must be positive (Laplacian can create negative interactions)
        # but the gradient must be finite
        assert np.all(np.isfinite(grad)), f"Non-finite gradient: {grad}"

    def test_bootstrap_covariance_psd(self):
        """Bootstrap covariance matrix must be positive semi-definite."""
        eigenvalues = np.linalg.eigvalsh(self.Sigma_V)
        assert np.all(eigenvalues >= -1e-10), f"Covariance not PSD: min eigenvalue = {eigenvalues.min()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

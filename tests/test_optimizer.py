"""
test_optimizer.py
-----------------
Unit tests for the adaptive weight optimizer.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.optimizer import optimize_weights, default_equal_weights


class TestOptimizer:

    def setup_method(self):
        rng = np.random.default_rng(42)
        # Safe models: all pillars high
        self.V_safe = rng.uniform(0.7, 1.0, size=(15, 5))
        # Unsafe models: at least one pillar low
        V_unsafe = rng.uniform(0.5, 0.9, size=(15, 5))
        V_unsafe[:, 3] = rng.uniform(0.1, 0.3, size=15)  # Low fairness
        self.V_unsafe = V_unsafe

    def test_weights_sum_to_one(self):
        w = optimize_weights(self.V_safe, self.V_unsafe)
        assert abs(w.sum() - 1.0) < 1e-6, f"Weights do not sum to 1: {w.sum()}"

    def test_weights_above_epsilon(self):
        epsilon_w = 0.05
        w = optimize_weights(self.V_safe, self.V_unsafe, epsilon_w=epsilon_w)
        assert np.all(w >= epsilon_w - 1e-8), f"Some weights below epsilon_w: {w}"

    def test_weights_shape(self):
        w = optimize_weights(self.V_safe, self.V_unsafe)
        assert w.shape == (5,), f"Expected shape (5,), got {w.shape}"

    def test_default_equal_weights(self):
        w = default_equal_weights()
        assert abs(w.sum() - 1.0) < 1e-10
        assert np.allclose(w, 0.2)

    def test_fairness_gets_high_weight_when_unsafe_low_fairness(self):
        """When unsafe models have low V4, optimizer should assign higher weight to V4."""
        w = optimize_weights(self.V_safe, self.V_unsafe)
        # V4 (index 3) should have above-average weight
        assert w[3] >= 0.15, f"Expected higher weight on V4, got {w[3]:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

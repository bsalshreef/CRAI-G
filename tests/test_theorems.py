"""
test_theorems.py
----------------
Unit tests for the three formal theorems of CRAI-G:
  Theorem 1: Monotonicity under Consistency
  Theorem 2: Strict Non-compensation
  Theorem 3: Lipschitz Stability
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.crai_equation import crai_g_score
from src.graph_learning import build_laplacian

# Fixed test Laplacian (simple chain graph)
A_test = np.array([
    [0, 0.5, 0, 0, 0],
    [0.5, 0, 0.5, 0, 0],
    [0, 0.5, 0, 0.5, 0],
    [0, 0, 0.5, 0, 0.5],
    [0, 0, 0, 0.5, 0],
], dtype=float)
L_test = build_laplacian(A_test)
w_test = np.array([0.2, 0.2, 0.2, 0.2, 0.2])


class TestTheorem1Monotonicity:
    """Theorem 1: Increasing a pillar score (without increasing Laplacian penalty) increases CRAI-G."""

    def test_increasing_v1_increases_score(self):
        V_base = np.array([0.5, 0.7, 0.6, 0.8, 0.7])
        V_higher = V_base.copy()
        V_higher[0] = 0.8  # Increase V1

        score_base = crai_g_score(V_base, L_test, w_test)
        score_higher = crai_g_score(V_higher, L_test, w_test)
        assert score_higher > score_base, (
            f"Expected score to increase when V1 increases: {score_base:.4f} -> {score_higher:.4f}"
        )

    def test_increasing_v4_increases_score(self):
        V_base = np.array([0.7, 0.7, 0.7, 0.5, 0.7])
        V_higher = V_base.copy()
        V_higher[3] = 0.9

        score_base = crai_g_score(V_base, L_test, w_test)
        score_higher = crai_g_score(V_higher, L_test, w_test)
        assert score_higher > score_base


class TestTheorem2NonCompensation:
    """Theorem 2: If any pillar -> 0, CRAI-G -> 0 regardless of other pillars."""

    def test_zero_v1_collapses_score(self):
        V = np.array([0.001, 0.95, 0.95, 0.95, 0.95])
        score = crai_g_score(V, L_test, w_test)
        assert score < 0.05, f"Expected near-zero score when V1 ≈ 0, got {score:.4f}"

    def test_zero_v4_collapses_score(self):
        V = np.array([0.95, 0.95, 0.95, 0.001, 0.95])
        score = crai_g_score(V, L_test, w_test)
        assert score < 0.05, f"Expected near-zero score when V4 ≈ 0, got {score:.4f}"

    def test_all_high_except_one_collapses(self):
        for k in range(5):
            V = np.full(5, 0.95)
            V[k] = 0.001
            score = crai_g_score(V, L_test, w_test)
            assert score < 0.05, f"Pillar {k+1} = 0.001 should collapse score, got {score:.4f}"


class TestTheorem3LipschitzStability:
    """Theorem 3: CRAI-G is Lipschitz continuous on the valid domain."""

    def test_small_perturbation_small_change(self):
        V = np.array([0.7, 0.8, 0.75, 0.85, 0.72])
        delta = 0.01
        for k in range(5):
            V_perturbed = V.copy()
            V_perturbed[k] = min(1.0, V[k] + delta)
            score_base = crai_g_score(V, L_test, w_test)
            score_perturbed = crai_g_score(V_perturbed, L_test, w_test)
            change = abs(score_perturbed - score_base)
            # Lipschitz: change should be bounded (empirically << 1 for delta=0.01)
            assert change < 0.5, f"Change too large for pillar {k+1}: {change:.4f}"

    def test_score_in_unit_interval(self):
        for _ in range(100):
            V = np.random.default_rng(99).uniform(0.15, 1.0, size=5)
            score = crai_g_score(V, L_test, w_test)
            assert 0.0 <= score <= 1.0, f"Score out of [0,1]: {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

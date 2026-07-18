"""
monte_carlo.py
--------------
Monte Carlo stability analysis: adds Gaussian noise N(0, 0.05^2) to pillar
scores across 500 iterations to confirm CRAI-G stability under perturbation.

Usage:
    python scripts/monte_carlo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.graph_learning import load_meta_dataset, estimate_precision_matrix, build_adjacency_matrix, build_laplacian
from src.optimizer import optimize_weights, default_equal_weights
from src.crai_equation import crai_g_score

os.makedirs("results", exist_ok=True)

V_meta = load_meta_dataset("data/meta_dataset_covariance.csv")
Theta_hat = estimate_precision_matrix(V_meta, lambda_lasso=0.1)
A = build_adjacency_matrix(Theta_hat)
L = build_laplacian(A)

mean_scores = V_meta.mean(axis=1)
V_safe = V_meta[mean_scores >= np.median(mean_scores)]
V_unsafe = V_meta[mean_scores < np.median(mean_scores)]
w_star = optimize_weights(V_safe, V_unsafe) if len(V_safe) >= 2 and len(V_unsafe) >= 2 else default_equal_weights()

example_df = pd.read_csv("data/example_input.csv")
V_example = example_df[["V1", "V2", "V3", "V4", "V5"]].values

rng = np.random.default_rng(42)
n_iter = 500
noise_std = 0.05

fig, axes = plt.subplots(1, len(example_df), figsize=(14, 4))
for i, row in example_df.iterrows():
    V = V_example[i]
    perturbed_scores = []
    for _ in range(n_iter):
        V_noisy = np.clip(V + rng.normal(0, noise_std, size=5), 0.01, 1.0)
        perturbed_scores.append(crai_g_score(V_noisy, L, w_star))
    perturbed_scores = np.array(perturbed_scores)
    axes[i].hist(perturbed_scores, bins=30, color="#2C7BB6", edgecolor="black", alpha=0.8)
    axes[i].axvline(np.mean(perturbed_scores), color="red", linestyle="--", label=f"Mean={np.mean(perturbed_scores):.3f}")
    axes[i].set_title(row["model_id"], fontsize=10)
    axes[i].set_xlabel("CRAI-G Score")
    axes[i].legend(fontsize=8)

fig.suptitle(f"Monte Carlo Stability (n={n_iter}, noise σ={noise_std})", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("results/monte_carlo_stability.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Monte Carlo stability analysis complete. Saved to results/monte_carlo_stability.png")

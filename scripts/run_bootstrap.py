"""
run_bootstrap.py
----------------
Runs 1,000-iteration percentile bootstrap to compute 95% CI for CRAI-G scores
across all models in example_input.csv.

Usage:
    python scripts/run_bootstrap.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.graph_learning import load_meta_dataset, estimate_precision_matrix, build_adjacency_matrix, build_laplacian
from src.optimizer import optimize_weights, default_equal_weights
from src.crai_equation import crai_g_score
from src.uncertainty import bootstrap_covariance, delta_method_ci

os.makedirs("results", exist_ok=True)

V_meta = load_meta_dataset("data/meta_dataset_covariance.csv")
Theta_hat = estimate_precision_matrix(V_meta, lambda_lasso=0.1)
A = build_adjacency_matrix(Theta_hat)
L = build_laplacian(A)

mean_scores = V_meta.mean(axis=1)
median_score = np.median(mean_scores)
V_safe = V_meta[mean_scores >= median_score]
V_unsafe = V_meta[mean_scores < median_score]
w_star = optimize_weights(V_safe, V_unsafe) if len(V_safe) >= 2 and len(V_unsafe) >= 2 else default_equal_weights()

example_df = pd.read_csv("data/example_input.csv")
V_example = example_df[["V1", "V2", "V3", "V4", "V5"]].values

rng = np.random.default_rng(42)
records = []
for i, row in example_df.iterrows():
    V = V_example[i]
    V_samples = V + rng.normal(0, 0.03, size=(500, 5))
    V_samples = np.clip(V_samples, 0.01, 1.0)
    Sigma_V = bootstrap_covariance(V_samples, n_bootstrap=1000)
    ci = delta_method_ci(V, Sigma_V, L, w_star)
    records.append({
        "model_id": row["model_id"],
        "CRAI_G": round(ci["score"], 4),
        "sigma": round(ci["sigma"], 4),
        "CI_lower_95": round(ci["ci_lower"], 4),
        "CI_upper_95": round(ci["ci_upper"], 4),
    })

results_df = pd.DataFrame(records)
results_df.to_csv("results/bootstrap_ci.csv", index=False)
print("Bootstrap CI results:")
print(results_df.to_string(index=False))
print("\n✓ Saved to results/bootstrap_ci.csv")

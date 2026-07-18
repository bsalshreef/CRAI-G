"""
generate_data.py
----------------
Generates the demonstration meta-dataset (meta_dataset_covariance.csv)
and example_input.csv for the CRAI-G repository.

These are synthetic demonstration records, NOT the full M=250 literature-derived
meta-dataset described in the manuscript. See Supplementary Table S1 for the
documentation of the full dataset, which will be released upon publication.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Generate 30 synthetic model evaluations with realistic correlations
# V1 (discrimination) and V2 (calibration) are positively correlated
# V3 (robustness) and V4 (fairness) are positively correlated
# V5 (utility) correlates with V1 and V2
n = 30
V1 = rng.beta(8, 2, n)  # High discrimination (skewed toward 1)
V2 = np.clip(V1 + rng.normal(0, 0.08, n), 0.3, 1.0)
V3 = rng.beta(5, 3, n)
V4 = np.clip(V3 + rng.normal(0, 0.1, n), 0.2, 1.0)
V5 = np.clip((V1 + V2) / 2 + rng.normal(0, 0.07, n), 0.2, 1.0)

meta_df = pd.DataFrame({
    "model_id": [f"M_{i+1:03d}" for i in range(n)],
    "V1": np.round(V1, 4),
    "V2": np.round(V2, 4),
    "V3": np.round(V3, 4),
    "V4": np.round(V4, 4),
    "V5": np.round(V5, 4),
})
meta_df.to_csv("data/meta_dataset_covariance.csv", index=False)
print(f"Saved meta_dataset_covariance.csv with {n} demonstration records.")

# Generate example_input.csv: 5 models to score
example_data = {
    "model_id": ["ModelA", "ModelB", "ModelC", "ModelD", "ModelE"],
    "V1": [0.91, 0.85, 0.78, 0.93, 0.70],
    "V2": [0.88, 0.72, 0.90, 0.65, 0.85],
    "V3": [0.80, 0.60, 0.75, 0.45, 0.82],
    "V4": [0.85, 0.55, 0.80, 0.40, 0.88],
    "V5": [0.76, 0.68, 0.72, 0.80, 0.70],
}
example_df = pd.DataFrame(example_data)
example_df.to_csv("data/example_input.csv", index=False)
print("Saved example_input.csv with 5 example models.")

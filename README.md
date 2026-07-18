# CRAI-G: Clinical Readiness of AI Index (Graph-Regularized)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/bsalshreef/CRAI-G/actions/workflows/tests.yml/badge.svg)](https://github.com/bsalshreef/CRAI-G/actions)

**CRAI-G** is a mathematically rigorous framework for evaluating the multidimensional clinical readiness of medical AI models. It integrates five performance pillars — discrimination, calibration, robustness, fairness, and clinical utility — into a single, formally guaranteed composite score using spectral graph theory, convex optimization, and information entropy.

> **Note:** This is the companion code repository for the manuscript:
> *"A Unified Mathematical Theory for the Clinical Readiness of Medical Artificial Intelligence: The CRAI-G Framework"*
> (Under review — code available to reviewers upon request)

---

## The CRAI-G Equation

$$\text{CRAI-G}(\mathbf{V}) = \Phi(\mathbf{V}) \cdot \exp(-\alpha \mathbf{V}^T \mathbf{L} \mathbf{V}) \cdot \exp(-\beta [\ln(5) - H(\tilde{\mathbf{V}})]) \cdot \prod_{i=1}^5 V_i^{w_i^*}$$

Where:
- **Φ(V)** — Differentiable sigmoid regulatory gate
- **exp(−α V^T L V)** — Spectral graph Laplacian penalty (penalizes inter-dimensional inconsistencies)
- **exp(−β[ln(5) − H(Ṽ)])** — Information entropy penalty (rewards balanced models)
- **∏ V_i^{w_i*}** — Weighted geometric mean with KKT-optimal adaptive weights

---

## Repository Structure

```
CRAI-G/
├── README.md
├── LICENSE
├── CITATION.cff
├── environment.yml
├── requirements.txt
├── data/
│   ├── meta_dataset_covariance.csv   # Demonstration meta-dataset (30 records)
│   └── example_input.csv             # Example models to score
├── src/
│   ├── crai_equation.py              # Unified CRAI-G score computation
│   ├── graph_learning.py             # Graphical Lasso graph estimation
│   ├── optimizer.py                  # KKT adaptive weight optimization
│   ├── uncertainty.py                # Delta Method CI propagation
│   ├── calibration.py                # V2: Calibration pillar
│   ├── fairness.py                   # V4: Fairness pillar
│   ├── robustness.py                 # V3: Robustness pillar
│   ├── utility.py                    # V5: Clinical utility pillar
│   └── visualization.py              # Figure generation
├── scripts/
│   ├── reproduce_all.py              # Master reproduction script
│   ├── run_bootstrap.py              # Bootstrap CI computation
│   └── monte_carlo.py                # Monte Carlo stability analysis
├── tests/
│   ├── test_theorems.py              # Tests for Theorems 1, 2, 3
│   ├── test_optimizer.py             # Tests for weight optimizer
│   └── test_delta_method.py          # Tests for Delta Method CI
└── examples/
    └── example_notebook.ipynb        # Interactive walkthrough
```

---

## Installation

```bash
git clone https://github.com/bsalshreef/CRAI-G.git
cd CRAI-G
pip install -r requirements.txt
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate crai-g
```

---

## Quick Start

```python
import numpy as np
from src.graph_learning import load_meta_dataset, estimate_precision_matrix, build_adjacency_matrix, build_laplacian
from src.optimizer import optimize_weights
from src.crai_equation import crai_g_score

# Load demonstration meta-dataset and learn graph
V_meta = load_meta_dataset("data/meta_dataset_covariance.csv")
Theta = estimate_precision_matrix(V_meta, lambda_lasso=0.1)
A = build_adjacency_matrix(Theta)
L = build_laplacian(A)

# Optimize weights (or use equal weights as default)
mean_scores = V_meta.mean(axis=1)
V_safe = V_meta[mean_scores >= np.median(mean_scores)]
V_unsafe = V_meta[mean_scores < np.median(mean_scores)]
w = optimize_weights(V_safe, V_unsafe)

# Score a model
V = np.array([0.85, 0.88, 0.75, 0.80, 0.72])
score = crai_g_score(V, L, w)
print(f"CRAI-G Score: {score:.4f}")
```

---

## Reproduce All Results

```bash
python scripts/reproduce_all.py
python scripts/run_bootstrap.py
python scripts/monte_carlo.py
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Important Notes on the Meta-Dataset

The `data/meta_dataset_covariance.csv` file contains **30 synthetic demonstration records** generated for methodological validation. This is **not** the full M=250 literature-derived meta-dataset described in the manuscript. The full dataset will be released upon publication. Reviewers may request access to the complete dataset during peer review.

---

## Citation

If you use CRAI-G in your research, please cite:

```bibtex
@article{crai_g_2025,
  title   = {A Unified Mathematical Theory for the Clinical Readiness of Medical Artificial Intelligence: The CRAI-G Framework},
  journal = {Under Review},
  year    = {2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

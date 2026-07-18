"""
visualization.py
----------------
Generates publication-quality figures for CRAI-G analysis.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_radar_chart(V: np.ndarray, labels=None, title="CRAI-G Pillar Scores", save_path=None):
    """
    Plot a radar chart of the five CRAI-G pillar scores.
    """
    if labels is None:
        labels = ["V1\nDiscrimination", "V2\nCalibration", "V3\nRobustness",
                  "V4\nFairness", "V5\nUtility"]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    values = V.tolist() + V[:1].tolist()

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2, color="#2C7BB6")
    ax.fill(angles, values, alpha=0.25, color="#2C7BB6")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_ablation_bar(results: dict, save_path=None):
    """
    Plot ablation bar chart comparing CRAI-G variants and baseline metrics.

    Parameters
    ----------
    results : dict
        Keys are metric names, values are dicts with 'rho', 'ci_lower', 'ci_upper'.
    """
    metrics = list(results.keys())
    rhos = [results[m]["rho"] for m in metrics]
    ci_lower = [results[m]["ci_lower"] for m in metrics]
    ci_upper = [results[m]["ci_upper"] for m in metrics]
    errors = [[rhos[i] - ci_lower[i] for i in range(len(metrics))],
              [ci_upper[i] - rhos[i] for i in range(len(metrics))]]

    colors = ["#D73027" if m in ["AUROC", "Linear Mean"] else "#2C7BB6" for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(metrics, rhos, color=colors, edgecolor="black", linewidth=0.8)
    ax.errorbar(metrics, rhos, yerr=errors, fmt="none", color="black",
                capsize=5, linewidth=1.5)

    ax.set_ylabel("Spearman ρ with SCRI", fontsize=12)
    ax.set_title("Ablation Study: Correlation with Simulated Clinical Risk Index (SCRI)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-0.5, 1.0)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=15)

    legend_patches = [
        mpatches.Patch(color="#D73027", label="Baseline metrics"),
        mpatches.Patch(color="#2C7BB6", label="CRAI-G variants"),
    ]
    ax.legend(handles=legend_patches, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_score_distribution(scores: np.ndarray, labels: np.ndarray = None, save_path=None):
    """
    Plot distribution of CRAI-G scores, optionally colored by safe/unsafe label.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    if labels is not None:
        safe_scores = scores[labels == 1]
        unsafe_scores = scores[labels == 0]
        ax.hist(safe_scores, bins=20, alpha=0.6, color="#2C7BB6", label="Safe models")
        ax.hist(unsafe_scores, bins=20, alpha=0.6, color="#D73027", label="Unsafe models")
        ax.legend(fontsize=10)
    else:
        ax.hist(scores, bins=20, color="#2C7BB6", edgecolor="black", alpha=0.8)

    ax.set_xlabel("CRAI-G Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of CRAI-G Scores", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig

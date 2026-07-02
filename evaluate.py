"""
evaluate.py
===========
Feature importance extraction and plotting utilities used by the notebook.
Kept separate from models.py so plotting/formatting concerns don't clutter
the modeling logic.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.inspection import permutation_importance


def get_feature_importance(model, X_test, y_test, feature_names, n_repeats: int = 20,
                            random_state: int = 42) -> pd.DataFrame:
    """
    Uses permutation importance rather than each model's built-in
    importance attribute (e.g. RandomForest.feature_importances_) because
    permutation importance is comparable ACROSS model types (tree-based
    impurity importance and linear-model coefficients aren't on the same
    scale or even measuring the same thing), which matters since this
    project explicitly wants to compare feature importance across several
    different model families.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats,
        random_state=random_state, scoring="f1",
    )
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return imp


def plot_model_comparison(sup_results: dict, save_path: str | None = None):
    names = list(sup_results.keys())
    f1s = [sup_results[n]["holdout"]["f1"] for n in names]
    aucs = [sup_results[n]["holdout"].get("roc_auc", np.nan) for n in names]
    cv_f1 = [sup_results[n]["cv_f1_mean"] for n in names]
    cv_std = [sup_results[n]["cv_f1_std"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(names))

    axes[0].bar(x - 0.2, f1s, width=0.4, label="Holdout F1")
    axes[0].bar(x + 0.2, aucs, width=0.4, label="Holdout ROC-AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Holdout performance by model")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, cv_f1, yerr=cv_std, capsize=4, color="steelblue")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_title("Time-series cross-validated F1 (mean +/- std)")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_behavioral_trends(feat: pd.DataFrame, labels: pd.DataFrame, save_path: str | None = None):
    """Long-term trend plot: total activity + flagged deviation days overlaid."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(feat.index, feat["total_events"], color="steelblue", lw=1, alpha=0.7, label="Daily total events")
    axes[0].plot(feat.index, feat["total_events_roll14_mean"], color="darkorange", lw=2, label="14-day rolling mean")
    deviant = labels[labels["label_statistical"] == 1].index
    deviant = deviant.intersection(feat.index)
    axes[0].scatter(deviant, feat.loc[deviant, "total_events"], color="red", zorder=5, s=25, label="Flagged deviation day")
    axes[0].set_title("Daily total sensor activity with flagged behavioral deviations")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(feat.index, feat["routine_similarity_cos"], color="seagreen", lw=1)
    axes[1].set_title("Routine similarity to recent baseline (cosine similarity, 1 = identical rhythm)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(feat.index, feat["n_long_inactivity_gaps"], color="purple", lw=1)
    axes[2].set_title("Long daytime inactivity gaps per day (>= 60 min, waking hours)")
    axes[2].grid(alpha=0.3)

    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrices(sup_results: dict, save_path: str | None = None):
    n = len(sup_results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, r) in zip(axes, sup_results.items()):
        cm = np.array(r["holdout"]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Deviant"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Deviant"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

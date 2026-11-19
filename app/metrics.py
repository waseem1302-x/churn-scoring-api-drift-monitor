"""Evaluation metrics. Accuracy is deliberately absent: on a 26.5%
churn rate it rewards the majority class, and predicting 'no churn'
for everyone already scores 0.7346. The metrics here are ranking
(ROC-AUC, PR-AUC), calibration (Brier, reliability curve), and the
decision (precision at k, precision and recall at the operating
threshold). The do-nothing baseline is reported next to every number
that could be mistaken for accuracy.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app import config


def roc_auc(y_true, y_proba) -> float:
    return float(roc_auc_score(y_true, y_proba))


def pr_auc(y_true, y_proba) -> float:
    return float(average_precision_score(y_true, y_proba))


def brier_score(y_true, y_proba) -> float:
    return float(brier_score_loss(y_true, y_proba))


def precision_at_k(y_true, y_proba, k: int) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if k < 1 or k > len(y_true):
        raise ValueError(f"k must be between 1 and {len(y_true)}, got {k}")
    top = np.argsort(y_proba)[::-1][:k]
    return float(y_true[top].mean())


def reliability_curve(y_true, y_proba, n_bins: int = 10) -> list:
    """Ten equal-width probability bins: mean predicted vs fraction
    positive in each bin. A calibrated model sits on the diagonal."""
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    # Bin by floor(p * n_bins), clipped to the last bin. This avoids the
    # float edge problem of linspace comparisons (0.6 vs 0.6000000001).
    bin_idx = np.minimum((y_proba * n_bins).astype(int), n_bins - 1)
    curve = []
    for i in range(n_bins):
        mask = bin_idx == i
        count = int(mask.sum())
        lo, hi = i / n_bins, (i + 1) / n_bins
        curve.append({
            "bin": [round(lo, 2), round(hi, 2)],
            "count": count,
            "mean_predicted": round(float(y_proba[mask].mean()), 4) if count else None,
            "fraction_positive": round(float(y_true[mask].mean()), 4) if count else None,
        })
    return curve


def evaluate_model(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Full metric set for one model on one scored set. k=100 is one
    retention team's daily call list; k=10% is the fixed-share view."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    k_abs = min(config.PRECISION_K_ABSOLUTE, n)
    k_frac = max(1, int(round(config.PRECISION_K_FRACTION * n)))
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "n": n,
        "churn_rate": round(float(y_true.mean()), 4),
        "roc_auc": round(roc_auc(y_true, y_proba), 4),
        "pr_auc": round(pr_auc(y_true, y_proba), 4),
        "brier": round(brier_score(y_true, y_proba), 4),
        "precision_at_100": round(precision_at_k(y_true, y_proba, k_abs), 4),
        "precision_at_10pct": round(precision_at_k(y_true, y_proba, k_frac), 4),
        "precision_at_threshold": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_at_threshold": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "threshold": threshold,
        "do_nothing_accuracy": round(float((y_true == 0).mean()), 4),
    }

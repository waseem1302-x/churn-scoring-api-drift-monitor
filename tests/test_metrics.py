"""Metric tests: hand-computed toy cases, no circularity.

TDD: these tests fail until app/metrics.py exists.
"""

import pytest

from app import metrics as m

# y = [0, 0, 1, 1], p = [0.1, 0.4, 0.35, 0.8]
# ROC-AUC by pairs: (0.1,0.35) win, (0.1,0.8) win, (0.4,0.35) loss,
# (0.4,0.8) win -> 3/4 = 0.75.
# Brier: (0.01 + 0.16 + 0.4225 + 0.04) / 4 = 0.158125.
Y = [0, 0, 1, 1]
P = [0.1, 0.4, 0.35, 0.8]


def test_roc_auc_hand_computed():
    assert m.roc_auc(Y, P) == pytest.approx(0.75)


def test_pr_auc_is_between_0_and_1():
    value = m.pr_auc(Y, P)
    assert 0.0 < value <= 1.0


def test_brier_hand_computed():
    assert m.brier_score(Y, P) == pytest.approx(0.158125)


def test_brier_perfect_is_zero():
    assert m.brier_score([0, 1], [0.0, 1.0]) == pytest.approx(0.0)


def test_precision_at_k_hand_computed():
    # Top 2 by score: 0.8 (y=1), 0.4 (y=0) -> 1/2.
    assert m.precision_at_k(Y, P, 2) == pytest.approx(0.5)
    # Top 1: 0.8 (y=1) -> 1.0.
    assert m.precision_at_k(Y, P, 1) == pytest.approx(1.0)
    # Top 4: all rows -> base rate 0.5.
    assert m.precision_at_k(Y, P, 4) == pytest.approx(0.5)


def test_precision_at_k_rejects_bad_k():
    with pytest.raises(ValueError):
        m.precision_at_k(Y, P, 0)
    with pytest.raises(ValueError):
        m.precision_at_k(Y, P, 5)


def test_evaluate_model_returns_all_keys():
    result = m.evaluate_model(Y, P)
    for key in ("roc_auc", "pr_auc", "brier", "precision_at_100",
                "precision_at_10pct", "precision_at_threshold",
                "recall_at_threshold", "do_nothing_accuracy"):
        assert key in result, key
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["brier"] == pytest.approx(0.158125, abs=1e-4)


def test_reliability_curve_bins_cover_unit_interval():
    y = [0, 1] * 50
    p = [i / 100 for i in range(100)]
    curve = m.reliability_curve(y, p, n_bins=5)
    assert len(curve) == 5
    for b in curve:
        assert 0.0 <= b["mean_predicted"] <= 1.0
        assert 0.0 <= b["fraction_positive"] <= 1.0
        assert b["count"] == 20
    # First bin holds the lowest scores, last bin the highest.
    assert curve[0]["mean_predicted"] < curve[-1]["mean_predicted"]


def test_reliability_curve_perfect_calibration():
    y = [0, 0, 1, 1]
    p = [0.0, 0.0, 1.0, 1.0]
    curve = m.reliability_curve(y, p, n_bins=2)
    assert curve[0]["fraction_positive"] == pytest.approx(0.0)
    assert curve[1]["fraction_positive"] == pytest.approx(1.0)

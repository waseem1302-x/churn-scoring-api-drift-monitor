"""Profit (expected maximum profit) tests: hand-computed toy cases.

TDD: these tests fail until app/profit.py exists.
"""

import pytest

from app import profit as pr

# y = [1, 1, 0, 0], p = [0.9, 0.8, 0.3, 0.2], offer cost 50, saved value 500.
# Threshold 0.5: TP=2, FP=0, profit = 2 * (500 - 50) = 900.
# Threshold 0.25: TP=2, FP=1, profit = 900 - 50 = 850.
Y = [1, 1, 0, 0]
P = [0.9, 0.8, 0.3, 0.2]


def test_profit_at_threshold_hand_computed():
    assert pr.profit_at_threshold(Y, P, 0.5, 50.0, 500.0) == pytest.approx(900.0)
    assert pr.profit_at_threshold(Y, P, 0.25, 50.0, 500.0) == pytest.approx(850.0)


def test_profit_at_threshold_1_is_zero():
    # Threshold 1.0 contacts nobody: no cost, no saved customers.
    assert pr.profit_at_threshold(Y, P, 1.0, 50.0, 500.0) == pytest.approx(0.0)


def test_profit_at_threshold_0_contacts_everyone():
    # Threshold 0.0: TP=2, FP=2 -> 2*450 - 2*50 = 800.
    assert pr.profit_at_threshold(Y, P, 0.0, 50.0, 500.0) == pytest.approx(800.0)


def test_best_threshold_is_profit_maximising():
    best = pr.best_threshold(Y, P, 50.0, 500.0)
    assert best["profit"] == pytest.approx(900.0)
    assert best["tp"] == 2
    assert best["fp"] == 0
    # Any threshold in (0.3, 0.8] contacts exactly the two churners.
    assert 0.3 < best["threshold"] <= 0.8


def test_profit_per_1000_scales():
    best = pr.best_threshold(Y, P, 50.0, 500.0)
    # 900 dollars on 4 customers -> 225,000 per 1,000 customers.
    assert best["profit_per_1000"] == pytest.approx(225000.0)


def test_profit_curve_is_sorted_by_threshold():
    curve = pr.profit_curve(Y, P, 50.0, 500.0)
    thresholds = [row["threshold"] for row in curve]
    assert thresholds == sorted(thresholds)
    assert all("tp" in row and "fp" in row and "profit" in row for row in curve)


def test_default_threshold_profit_is_lower_than_best():
    default = pr.profit_at_threshold(Y, P, 0.5, 50.0, 500.0)
    best = pr.best_threshold(Y, P, 50.0, 500.0)
    assert best["profit"] >= default

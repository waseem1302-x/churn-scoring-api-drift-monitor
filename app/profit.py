"""Profit-based threshold selection (expected maximum profit).

A threshold is a business decision, not a default 0.5. Profit at a
threshold is: every true positive contacted saves the customer (value
minus offer cost), every false positive contacted wastes the offer
cost, everyone below the threshold is left alone.

    profit(t) = TP(t) * (SAVED_VALUE - OFFER_COST) - FP(t) * OFFER_COST

SAVED_VALUE and OFFER_COST are the explicit assumptions in
app/config.py ($500 and $50, illustrative, not measured business
figures). Every profit number in the repo comes from this one formula.
"""

import numpy as np


def profit_at_threshold(y_true, y_proba, threshold, offer_cost, saved_value) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    contacted = y_proba >= threshold
    tp = int(((y_true == 1) & contacted).sum())
    fp = int(((y_true == 0) & contacted).sum())
    return tp * (saved_value - offer_cost) - fp * offer_cost


def profit_curve(y_true, y_proba, offer_cost, saved_value) -> list:
    """Profit at every distinct score threshold, plus 0.0 and 1.0."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = sorted(set([0.0, 1.0] + [round(float(p), 4) for p in y_proba]))
    curve = []
    for t in thresholds:
        contacted = y_proba >= t
        tp = int(((y_true == 1) & contacted).sum())
        fp = int(((y_true == 0) & contacted).sum())
        curve.append({
            "threshold": t,
            "tp": tp,
            "fp": fp,
            "profit": tp * (saved_value - offer_cost) - fp * offer_cost,
        })
    return curve


def best_threshold(y_true, y_proba, offer_cost, saved_value) -> dict:
    """The profit-maximising threshold. Ties go to the higher threshold
    (contact fewer customers for the same profit)."""
    y_true = np.asarray(y_true)
    curve = profit_curve(y_true, y_proba, offer_cost, saved_value)
    best = max(curve, key=lambda r: (r["profit"], r["threshold"]))
    n = len(y_true)
    return {
        "threshold": best["threshold"],
        "profit": float(best["profit"]),
        "profit_per_1000": round(float(best["profit"]) / n * 1000, 2),
        "tp": best["tp"],
        "fp": best["fp"],
    }

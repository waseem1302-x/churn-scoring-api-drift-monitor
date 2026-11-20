"""Profit sensitivity: what the frozen threshold costs when the
assumptions change.

Every profit number in this repo rests on two assumed business
figures: a $50 retention offer cost and a $500 saved value
(app/config.py). This script recomputes, from the champion model's
recorded test profit curve (benchmarks/results/profit_curve.json,
true and false positives per threshold, which do not depend on the
costs), the profit-maximising threshold and the test profit per
1,000 customers for a 3x3 grid of assumptions: offer cost $25, $50,
$100 crossed with saved value $250, $500, $1,000.

For each cell it records the optimal threshold, the profit per
1,000 at that threshold, and the profit per 1,000 at the frozen
0.0934 threshold, so the reader can see what keeping the frozen
threshold costs under other assumptions. The frozen threshold was
chosen on out-of-fold train predictions at $50/$500, never on the
test set, so the test-optimal threshold in the base cell may differ
from it; both numbers are reported, and the gap is the cost of
keeping the frozen threshold.

Output: benchmarks/results/profit_sensitivity.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from app import config

RESULTS = config.ROOT / "benchmarks" / "results"
OFFER_COSTS = [25.0, 50.0, 100.0]
SAVED_VALUES = [250.0, 500.0, 1000.0]


def _profit(tp, fp, offer_cost, saved_value):
    return tp * (saved_value - offer_cost) - fp * offer_cost


def cell_for(curve, offer_cost, saved_value, frozen_threshold, n):
    """One grid cell from tp/fp per threshold.

    Ties on profit go to the higher threshold (contact fewer
    customers for the same profit), the same rule as
    app.profit.best_threshold.
    """
    scored = []
    for row in curve:
        profit = _profit(row["tp"], row["fp"], offer_cost, saved_value)
        scored.append((row["threshold"], profit, row["tp"] + row["fp"]))
    optimal_threshold, optimal_profit, optimal_contacted = max(
        scored, key=lambda r: (r[1], r[0]))
    frozen = [r for r in scored if r[0] == frozen_threshold]
    if not frozen:
        raise ValueError(f"frozen threshold {frozen_threshold} not in curve")
    _, frozen_profit, frozen_contacted = frozen[0]
    optimal_per_1000 = round(optimal_profit / n * 1000, 2)
    frozen_per_1000 = round(frozen_profit / n * 1000, 2)
    return {
        "offer_cost": offer_cost,
        "saved_value": saved_value,
        "optimal_threshold": optimal_threshold,
        "optimal_profit": float(optimal_profit),
        "optimal_profit_per_1000": optimal_per_1000,
        "optimal_contacted": optimal_contacted,
        "frozen_profit": float(frozen_profit),
        "frozen_profit_per_1000": frozen_per_1000,
        "frozen_contacted": frozen_contacted,
        "gap_per_1000": round(optimal_per_1000 - frozen_per_1000, 2),
    }


def main():
    curve = json.loads((RESULTS / "profit_curve.json").read_text())
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    frozen = metrics["champion"]["threshold_frozen"]
    n = metrics["dataset"]["test_rows"]
    cells = [cell_for(curve, cost, saved, frozen, n)
             for cost in OFFER_COSTS for saved in SAVED_VALUES]
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": metrics["model_version"],
        "frozen_threshold": frozen,
        "n_test": n,
        "source": ("benchmarks/results/profit_curve.json: true and false "
                   "positives per threshold from the champion's test "
                   "predictions; profit recomputed per cell, never refit."),
        "offer_costs": OFFER_COSTS,
        "saved_values": SAVED_VALUES,
        "cells": cells,
    }
    (RESULTS / "profit_sensitivity.json").write_text(json.dumps(data, indent=2))
    print(f"{'Offer':>7} {'Saved':>7} {'Best thr':>8} {'Best $/1k':>11} "
          f"{'Frozen $/1k':>12} {'Gap $/1k':>9}")
    for c in cells:
        print(f"${c['offer_cost']:>6.0f} ${c['saved_value']:>6.0f} "
              f"{c['optimal_threshold']:>8.4f} "
              f"${c['optimal_profit_per_1000']:>10,.2f} "
              f"${c['frozen_profit_per_1000']:>11,.2f} "
              f"${c['gap_per_1000']:>8,.2f}")


if __name__ == "__main__":
    main()

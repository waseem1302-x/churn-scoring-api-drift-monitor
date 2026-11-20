"""Tests for the profit sensitivity analysis.

The profit model rests on two assumed numbers ($50 offer cost, $500
saved value). scripts/profit_sensitivity.py recomputes the optimal
threshold and the test profit per 1,000 customers for a 3x3 grid of
assumptions from the champion's recorded test profit curve
(benchmarks/results/profit_curve.json, tp and fp per threshold), so
the reader can see what the frozen threshold costs when the
assumptions change. If the JSON is missing, the message says to run
the script first.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
SENSITIVITY = RESULTS / "profit_sensitivity.json"


def test_cell_math_on_a_small_curve():
    from scripts.profit_sensitivity import cell_for

    # profit(t) = tp * (saved - cost) - fp * cost, cost 50, saved 500:
    # t=0.0 -> 10*450 - 90*50 = 0; t=0.5 -> 8*450 - 10*50 = 3100;
    # t=0.9 -> 2*450 - 0 = 900. Best is 0.5.
    curve = [
        {"threshold": 0.0, "tp": 10, "fp": 90},
        {"threshold": 0.5, "tp": 8, "fp": 10},
        {"threshold": 0.9, "tp": 2, "fp": 0},
    ]
    cell = cell_for(curve, offer_cost=50.0, saved_value=500.0,
                    frozen_threshold=0.5, n=100)
    assert cell["optimal_threshold"] == 0.5
    assert cell["optimal_profit"] == 3100.0
    assert cell["optimal_profit_per_1000"] == 31000.0
    assert cell["frozen_profit"] == 3100.0
    assert cell["frozen_profit_per_1000"] == 31000.0
    assert cell["gap_per_1000"] == 0.0
    assert cell["optimal_contacted"] == 18
    assert cell["frozen_contacted"] == 18


def test_cell_math_when_frozen_threshold_is_not_optimal():
    from scripts.profit_sensitivity import cell_for

    curve = [
        {"threshold": 0.0, "tp": 10, "fp": 90},
        {"threshold": 0.5, "tp": 8, "fp": 10},
        {"threshold": 0.9, "tp": 2, "fp": 0},
    ]
    cell = cell_for(curve, offer_cost=50.0, saved_value=500.0,
                    frozen_threshold=0.0, n=100)
    assert cell["optimal_threshold"] == 0.5
    assert cell["frozen_profit"] == 0.0
    assert cell["gap_per_1000"] == 31000.0


def test_cell_ties_go_to_the_higher_threshold():
    # Same rule as app.profit.best_threshold: contact fewer
    # customers for the same profit.
    from scripts.profit_sensitivity import cell_for

    curve = [
        {"threshold": 0.2, "tp": 4, "fp": 0},
        {"threshold": 0.8, "tp": 4, "fp": 0},
    ]
    cell = cell_for(curve, offer_cost=50.0, saved_value=500.0,
                    frozen_threshold=0.2, n=100)
    assert cell["optimal_threshold"] == 0.8


def test_sensitivity_json_has_nine_cells():
    if not SENSITIVITY.exists():
        pytest.fail("profit_sensitivity.json missing: "
                    "run scripts/profit_sensitivity.py first")
    data = json.loads(SENSITIVITY.read_text())
    assert data["model_version"] == "churn-xgboost-smote-isotonic-v1"
    assert data["frozen_threshold"] == 0.0934
    assert data["n_test"] == 1409
    assert sorted(set(c["offer_cost"] for c in data["cells"])) == [25.0, 50.0, 100.0]
    assert sorted(set(c["saved_value"] for c in data["cells"])) == [250.0, 500.0, 1000.0]
    assert len(data["cells"]) == 9
    keys = {"offer_cost", "saved_value", "optimal_threshold",
            "optimal_profit", "optimal_profit_per_1000",
            "optimal_contacted", "frozen_profit",
            "frozen_profit_per_1000", "frozen_contacted", "gap_per_1000"}
    for cell in data["cells"]:
        assert keys <= set(cell), cell


def test_base_cell_reproduces_the_known_phase_1_numbers():
    data = json.loads(SENSITIVITY.read_text())
    base = [c for c in data["cells"]
            if c["offer_cost"] == 50.0 and c["saved_value"] == 500.0][0]
    # Phase 1 froze 0.0934 from out-of-fold train predictions; on the
    # test set at $50/$500 the frozen threshold earns $132,150.
    assert base["frozen_profit"] == 132150.0
    assert base["frozen_profit_per_1000"] == 93789.92
    assert base["frozen_contacted"] == 927


def _money(s):
    return float(s.replace("$", "").replace(",", ""))


def test_docs_sensitivity_table_matches_json():
    data = json.loads(SENSITIVITY.read_text())
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    start = text.index("## Profit sensitivity")
    end = text.index("## ", start + 3)
    section = text[start:end]
    rows = [line for line in section.splitlines()
            if line.startswith("| $")]
    assert len(rows) == 9, f"expected 9 table rows, got {len(rows)}"
    by_key = {(c["offer_cost"], c["saved_value"]): c for c in data["cells"]}
    for line in rows:
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        offer, saved = _money(parts[0]), _money(parts[1])
        cell = by_key[(offer, saved)]
        assert float(parts[2]) == cell["optimal_threshold"], line
        assert _money(parts[3]) == cell["optimal_profit_per_1000"], line
        assert _money(parts[4]) == cell["frozen_profit_per_1000"], line
        assert _money(parts[5]) == cell["gap_per_1000"], line


def test_architecture_points_at_the_sensitivity_table():
    text = (ROOT / "ARCHITECTURE.md").read_text()
    assert "profit_sensitivity.json" in text
    assert "Profit sensitivity" in text

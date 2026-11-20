"""Promotion margin sensitivity tests.

The $2,500 margin is a judgment call. These tests pin down what
the retraining log actually proves about it: the two recorded
decisions are the same at $1,000, $2,500 and $5,000, and the
noise floor behind the margin is recomputed from the champion's
own per-week profit, not quoted from a docstring.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def _load():
    path = RESULTS / "margin_sensitivity.json"
    if not path.exists():
        pytest.fail("margin_sensitivity.json missing: run "
                    "python -m scripts.margin_sensitivity first")
    return json.loads(path.read_text())


def test_decisions_identical_at_1000_2500_5000():
    data = _load()
    by_margin = data["decisions_by_margin"]
    assert by_margin["1000"] == by_margin["2500"] == by_margin["5000"]
    assert by_margin["2500"] == {"TRG-08-1": "promote",
                                 "TRG-11-2": "refuse"}


def test_flip_points_are_the_observed_margins():
    data = _load()
    assert data["flip_points_per_trigger"] == {"TRG-08-1": 9490.0,
                                               "TRG-11-2": 130.0}
    assert data["margin_range_with_same_decisions"] == [130.0, 9490.0]


def test_noise_floor_recomputed_from_manifest():
    data = _load()
    noise = data["noise_floor"]
    assert noise["clean_weeks"] == [1, 2, 3, 4]
    assert len(noise["champion_profit_per_1000_by_week"]) == 4
    # Pooled two-week std is the single-week std over sqrt(2).
    assert abs(noise["two_week_pooled_std_sample"] * 2 ** 0.5
               - noise["single_week_std_sample"]) < 0.01
    # $2,500 is 0.82x the pooled std, not two: the false-promotion
    # rate at $2,500 must be well above the rate at $5,000.
    rates = data["false_promotion_rate"]
    assert rates["2500"] > rates["5000"]
    assert 0.15 < rates["2500"] < 0.25
    assert 0.04 < rates["5000"] < 0.06


def test_docs_margin_section_matches_json():
    data = _load()
    text = (ROOT / "MONITORING.md").read_text()
    assert "3,054.42" in text
    assert "130" in text and "9,490" in text
    assert "0.82" in text and "pooled" in text.lower()

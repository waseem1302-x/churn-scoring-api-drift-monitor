"""Tests for the concurrency record.

benchmarks/results/concurrency.json holds the 200-call parallel
load measurement, and the docs section must match it exactly.
If the JSON is missing, the message says to run
scripts/measure_concurrency.py first.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
CONCURRENCY = RESULTS / "concurrency.json"


def _load():
    if not CONCURRENCY.exists():
        pytest.fail("concurrency.json missing: run "
                    "scripts/measure_concurrency.py first")
    return json.loads(CONCURRENCY.read_text())


def test_concurrency_all_calls_succeeded():
    data = _load()
    assert data["calls"] == 200
    assert data["succeeded"] == 200


def test_docs_concurrency_section_matches_json():
    data = _load()
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    start = text.index("### Concurrency: 200 parallel scores")
    end = text.index("## ", start + 3)
    section = text[start:end]
    for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
        assert f"{data[key]:.2f} ms" in section, key
    assert str(data["calls"]) in section
    assert str(data["succeeded"]) in section
    assert f"{data['wall_s']:.2f} s" in section

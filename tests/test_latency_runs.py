"""Tests for the latency re-measurement record.

Phase 1 measured p50 64.07 ms / p95 77.47 ms and Phase 2 measured
p50 63.52 ms / p95 99.10 ms. Two fresh re-measurements disagreed
with both, so benchmarks/results/latency_runs.json records all
four runs and the docs table must match it exactly. latency.json
itself always holds the latest single run. If the JSON is missing,
the message says to run scripts/measure_latency.py first.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
RUNS = RESULTS / "latency_runs.json"
LATENCY = RESULTS / "latency.json"


def _load_runs():
    if not RUNS.exists():
        pytest.fail("latency_runs.json missing: record the four runs first")
    return json.loads(RUNS.read_text())


def test_four_runs_recorded():
    data = _load_runs()
    assert [r["label"] for r in data["runs"]] == [
        "Phase 1", "Phase 2", "Re-measure A", "Re-measure B"]
    for run in data["runs"]:
        for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
            assert isinstance(run[key], (int, float)), (run["label"], key)


def test_latency_json_is_the_latest_run():
    data = _load_runs()
    latest = data["runs"][-1]
    current = json.loads(LATENCY.read_text())
    for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
        assert current[key] == latest[key], key


def _ms(cell):
    return float(cell.replace("ms", "").strip())


def test_docs_latency_table_matches_runs():
    data = _load_runs()
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    start = text.index("## Latency and cost")
    end = text.index("## ", start + 3)
    section = text[start:end]
    rows = {}
    for line in section.splitlines():
        if line.startswith("| Mean") or line.startswith("| p50") \
                or line.startswith("| p95") or line.startswith("| p99"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            rows[parts[0]] = [_ms(p) for p in parts[1:]]
    assert set(rows) == {"Mean", "p50", "p95", "p99"}, rows.keys()
    keys = {"Mean": "mean_ms", "p50": "p50_ms", "p95": "p95_ms", "p99": "p99_ms"}
    for name, key in keys.items():
        assert rows[name] == [r[key] for r in data["runs"]], name


def test_docs_state_the_jitter_conclusion():
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    start = text.index("## Latency and cost")
    end = text.index("## ", start + 3)
    section = text[start:end]
    assert "jitter" in section.lower()
    assert "77.47" in section and "176.09" in section
    assert "re-measure" in section.lower()

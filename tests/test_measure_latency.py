"""Tests for the latency-runs log scripts/measure_latency.py keeps.

measure_latency.py writes latency.json with the latest measurement,
and it must append that same run to benchmarks/results/
latency_runs.json, so latency.json always equals the latest run in
the log on any machine. Before this, the script never touched the
log: CI measured a fast 14 ms run, wrote latency.json, and the
committed log still ended at the 103 ms VM run, so
tests/test_latency_runs.py failed in CI while passing locally.
"""

import json
from pathlib import Path

from scripts import measure_latency

ROOT = Path(__file__).resolve().parents[1]


def _result():
    return {
        "model_version": "churn-xgboost-smote-isotonic-v1",
        "n": 200,
        "mean_ms": 21.5,
        "p50_ms": 19.8,
        "p95_ms": 33.1,
        "p99_ms": 41.0,
        "note": "test run",
    }


def _runs_file(tmp_path, labels=("Phase 1", "Phase 2")):
    path = tmp_path / "latency_runs.json"
    path.write_text(json.dumps({
        "generated_at": "2026-11-20T01:49:08+00:00",
        "model_version": "churn-xgboost-smote-isotonic-v1",
        "runs": [
            {"label": label, "mean_ms": 66.04, "p50_ms": 64.07,
             "p95_ms": 77.47, "p99_ms": 93.4}
            for label in labels
        ],
    }, indent=2))
    return path


def test_append_adds_the_new_run_last(tmp_path):
    path = _runs_file(tmp_path)
    measure_latency._append_run(_result(), path)
    data = json.loads(path.read_text())
    assert [r["label"] for r in data["runs"][:2]] == ["Phase 1", "Phase 2"]
    latest = data["runs"][-1]
    assert latest["mean_ms"] == 21.5
    assert latest["p50_ms"] == 19.8
    assert latest["p95_ms"] == 33.1
    assert latest["p99_ms"] == 41.0
    assert data["model_version"] == "churn-xgboost-smote-isotonic-v1"


def test_append_creates_the_log_when_missing(tmp_path):
    path = tmp_path / "latency_runs.json"
    measure_latency._append_run(_result(), path)
    data = json.loads(path.read_text())
    assert len(data["runs"]) == 1
    assert data["runs"][0]["mean_ms"] == 21.5

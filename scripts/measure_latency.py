"""Measure scoring latency of the shipped artifact through the API.

Loads the artifact into the FastAPI app (in-process TestClient, so the
number includes HTTP handling and the SQLite log write), warms up,
scores 200 held-out test rows one at a time, and writes
benchmarks/results/latency.json with p50/p95/p99 and the mean.
Run after scripts/train.py.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app import api as api_mod
from app import config, data, scoring

RESULTS = config.ROOT / "benchmarks" / "results"
RUNS_PATH = RESULTS / "latency_runs.json"
N_ROWS = 200


def _headers():
    """The API needs a key since Phase 3. The script reads it from the
    environment, same as a real caller, and fails clearly without it."""
    key = os.environ.get(api_mod.API_KEY_ENV)
    if not key:
        raise SystemExit(f"set {api_mod.API_KEY_ENV} before measuring latency")
    return {api_mod.API_KEY_HEADER: key}


def _row_to_payload(row) -> dict:
    payload = {}
    for f in config.FEATURES:
        v = row[f]
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and v != v:
            v = None
        payload[f] = v
    payload["customerID"] = str(row[config.ID_COLUMN])
    return payload


def _append_run(result, runs_path=RUNS_PATH):
    """Append this measurement to the latency-runs log, so
    latency.json always equals the latest run in the log on any
    machine. The log is a history: earlier runs stay in place, the
    new run goes last, labelled with the UTC date. If the log does
    not exist yet, it is created with just this run."""
    run = {
        "label": datetime.now(timezone.utc).date().isoformat(),
        "mean_ms": result["mean_ms"],
        "p50_ms": result["p50_ms"],
        "p95_ms": result["p95_ms"],
        "p99_ms": result["p99_ms"],
    }
    if runs_path.exists():
        try:
            data = json.loads(runs_path.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}
    runs = data.get("runs") or []
    runs.append(run)
    data["runs"] = runs
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    data["model_version"] = result["model_version"]
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    runs_path.write_text(json.dumps(data, indent=2))
    return run


def main():
    model, meta = scoring.load_artifact()
    import tempfile
    db_path = str(Path(tempfile.mkdtemp()) / "latency.db")
    api_mod.set_model(model, meta["model_version"], db_path)

    df = data.load_telco()
    _, X_test, _, _ = data.split(df)
    rows = [_row_to_payload(X_test.iloc[i]) for i in range(N_ROWS)]

    with TestClient(api_mod.app) as client:
        headers = _headers()
        # Warm-up: 10 scores, then the timed run.
        for row in rows[:10]:
            assert client.post("/score", json=row, headers=headers).status_code == 200
        latencies = []
        for row in rows:
            t0 = time.perf_counter()
            resp = client.post("/score", json=row, headers=headers)
            assert resp.status_code == 200
            latencies.append((time.perf_counter() - t0) * 1000)
    lat = np.array(latencies)
    result = {
        "model_version": meta["model_version"],
        "n": N_ROWS,
        "mean_ms": round(float(lat.mean()), 2),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "note": "In-process TestClient on this VM (1.5 shared CPUs). "
                "Includes HTTP handling and the SQLite log write.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "latency.json").write_text(json.dumps(result, indent=2))
    _append_run(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

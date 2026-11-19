"""Measure scoring latency of the shipped artifact through the API.

Loads the artifact into the FastAPI app (in-process TestClient, so the
number includes HTTP handling and the SQLite log write), warms up,
scores 200 held-out test rows one at a time, and writes
benchmarks/results/latency.json with p50/p95/p99 and the mean.
Run after scripts/train.py.
"""

import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app import api as api_mod
from app import config, data, scoring

RESULTS = config.ROOT / "benchmarks" / "results"
N_ROWS = 200


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


def main():
    model, meta = scoring.load_artifact()
    import tempfile
    db_path = str(Path(tempfile.mkdtemp()) / "latency.db")
    api_mod.set_model(model, meta["model_version"], db_path)

    df = data.load_telco()
    _, X_test, _, _ = data.split(df)
    rows = [_row_to_payload(X_test.iloc[i]) for i in range(N_ROWS)]

    with TestClient(api_mod.app) as client:
        # Warm-up: 10 scores, then the timed run.
        for row in rows[:10]:
            assert client.post("/score", json=row).status_code == 200
        latencies = []
        for row in rows:
            t0 = time.perf_counter()
            resp = client.post("/score", json=row)
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

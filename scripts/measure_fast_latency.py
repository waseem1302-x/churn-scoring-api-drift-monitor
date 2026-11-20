"""Measure the fast model's latency through the API.

Same method as scripts/measure_latency.py (200 held-out test rows,
one at a time, in-process TestClient, includes HTTP handling and
the SQLite log write), but with the fast artifact loaded instead
of the champion. Writes benchmarks/results/latency_fast.json.
Run with CHURN_API_KEY set, same as a real caller.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app import api as api_mod
from app import config, data, scoring
from scripts.measure_latency import _headers, _row_to_payload

RESULTS = config.ROOT / "benchmarks" / "results"
N_ROWS = 200


def main():
    model, meta = scoring.load_artifact(str(config.FAST_ARTIFACT_PATH))
    import tempfile
    db_path = str(Path(tempfile.mkdtemp()) / "fast-latency.db")
    api_mod.set_model(model, meta["model_version"], db_path,
                      threshold=meta["threshold"])

    df = data.load_telco()
    _, X_test, _, _ = data.split(df)
    rows = [_row_to_payload(X_test.iloc[i]) for i in range(N_ROWS)]

    with TestClient(api_mod.app) as client:
        headers = _headers()
        for row in rows[:10]:
            assert client.post("/score", json=row,
                               headers=headers).status_code == 200
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
                "Includes HTTP handling and the SQLite log write. "
                "Same method as measure_latency.py, fast artifact.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "latency_fast.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

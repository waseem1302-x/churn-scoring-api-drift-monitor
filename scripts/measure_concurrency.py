"""Measure scoring latency under concurrent load.

Scores 200 held-out test rows from 8 threads at once through the API
(the shipped champion, in-process TestClient, so the number includes
HTTP handling and the SQLite log write) and writes
benchmarks/results/concurrency.json with the latency percentiles,
the success count, and the conditions. The point is twofold: no call
fails with "database is locked" (WAL plus the busy timeout), and the
latency under load is recorded, not guessed.

Run with CHURN_API_KEY set, same as a real caller.
"""

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app import api as api_mod
from app import config, data, scoring
from scripts.measure_latency import _row_to_payload

RESULTS = config.ROOT / "benchmarks" / "results"
N_WORKERS = 8
CALLS_PER_WORKER = 25


def _score_some(args):
    client_rows, headers = args
    latencies = []
    ok = 0
    with TestClient(api_mod.app) as client:
        for row in client_rows:
            t0 = time.perf_counter()
            resp = client.post("/score", json=row, headers=headers)
            latencies.append((time.perf_counter() - t0) * 1000)
            if resp.status_code == 200:
                ok += 1
    return ok, latencies


def main():
    key = os.environ.get(api_mod.API_KEY_ENV)
    if not key:
        raise SystemExit(f"set {api_mod.API_KEY_ENV} before measuring")
    headers = {api_mod.API_KEY_HEADER: key}

    model, meta = scoring.load_artifact()
    db_path = str(Path(tempfile.mkdtemp()) / "concurrency.db")
    api_mod.set_model(model, meta["model_version"], db_path)

    df = data.load_telco()
    _, X_test, _, _ = data.split(df)
    rows = [_row_to_payload(X_test.iloc[i])
            for i in range(N_WORKERS * CALLS_PER_WORKER)]
    chunks = [rows[i * CALLS_PER_WORKER:(i + 1) * CALLS_PER_WORKER]
              for i in range(N_WORKERS)]

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_score_some,
                                [(chunk, headers) for chunk in chunks]))
    wall_s = time.perf_counter() - started
    ok = sum(r[0] for r in results)
    lat = np.array([ms for _, lats in results for ms in lats])
    result = {
        "model_version": meta["model_version"],
        "workers": N_WORKERS,
        "calls": N_WORKERS * CALLS_PER_WORKER,
        "succeeded": ok,
        "wall_s": round(wall_s, 2),
        "mean_ms": round(float(lat.mean()), 2),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "note": "In-process TestClient on this VM (1.5 shared CPUs), "
                "8 threads scoring at once, includes HTTP handling and "
                "the SQLite log write. Measured while the S6E3 27-config "
                "sweep was running on the same VM, so the tail is "
                "heavier than a quiet run would be.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "concurrency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Concurrency tests: parallel /score calls must not hit
"database is locked".

TDD: these tests fail until app/db.py enables WAL mode and a busy
timeout. Every score writes one row to SQLite, and a production
caller scores many customers at once, so 8 threads scoring in
parallel must all succeed and all 200 rows must land in the log.
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from app import api as api_mod
from app import data as d
from app import features as f
from tests.conftest import AUTH_HEADERS
from tests.test_api import VALID_ROW

N_WORKERS = 8
CALLS_PER_WORKER = 25


def _score_some(n):
    """One thread's share: its own TestClient, n scores, the count
    of 200s and any error text seen."""
    ok = 0
    errors = []
    with TestClient(api_mod.app) as c:
        for i in range(n):
            row = dict(VALID_ROW, customerID=f"CONC-{i:04d}")
            resp = c.post("/score", json=row, headers=AUTH_HEADERS)
            if resp.status_code == 200:
                ok += 1
            else:
                errors.append(resp.text[:200])
    return ok, errors


def test_parallel_scores_all_succeed(tmp_path):
    df = d.load_telco(d.DATA_PATH)
    X_train, _, y_train, _ = d.split(df)
    pipe = f.build_pipeline("logistic", handling="none", calibration=None)
    pipe.fit(X_train.head(300), y_train.head(300))
    db_path = str(tmp_path / "concurrent.db")
    api_mod.set_model(pipe, "test-logistic-v0", db_path)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_score_some,
                                [CALLS_PER_WORKER] * N_WORKERS))
    total_ok = sum(ok for ok, _ in results)
    all_errors = [e for _, errs in results for e in errs]
    assert "database is locked" not in " ".join(all_errors).lower()
    assert total_ok == N_WORKERS * CALLS_PER_WORKER, all_errors[:3]
    rows = sqlite3.connect(db_path).execute(
        "SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert rows == N_WORKERS * CALLS_PER_WORKER


def test_write_waits_out_a_held_lock(tmp_path):
    """A write must wait for a held write lock instead of failing.

    One connection holds the write lock for 7 seconds. The default
    sqlite3 timeout is 5 seconds, so an unconfigured write fails
    with "database is locked"; with the 30-second busy timeout the
    same write waits and succeeds. This is the test that proves the
    timeout, not the parallel smoke test above."""
    import threading
    import time

    from app import db as db_mod

    path = str(tmp_path / "locked.db")
    db_mod.init_db(path)

    def hold_lock():
        conn = sqlite3.connect(path)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO predictions (timestamp, customer_id, "
            "inputs_json, probability, tier, model_version, latency_ms) "
            "VALUES ('t', 'LOCK', '{}', 0.5, 'LOW', 'lock', 0.0)")
        time.sleep(7)
        conn.commit()
        conn.close()

    thread = threading.Thread(target=hold_lock)
    thread.start()
    time.sleep(0.5)  # let the holder take the lock first
    row_id = db_mod.log_prediction(
        path, "t", "WAITED", {}, 0.5, "LOW", "test", 1.0)
    thread.join()
    assert row_id is not None
    rows = sqlite3.connect(path).execute(
        "SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert rows == 2


def test_db_uses_wal_mode(tmp_path):
    """The log database is in WAL mode, so readers do not block the
    writer and concurrent writers queue instead of failing."""
    from app import db as db_mod

    path = str(tmp_path / "wal.db")
    db_mod.init_db(path)
    mode = sqlite3.connect(path).execute(
        "PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_concurrent_first_calls_build_the_reference_once(tmp_path):
    """Eight threads scoring at once must build the drift reference
    once, not eight times. The reference build loads the training
    data and scores it, so a thundering herd wastes real CPU."""
    from concurrent.futures import ThreadPoolExecutor

    from app import data as data_mod

    df = d.load_telco(d.DATA_PATH)
    X_train, _, y_train, _ = d.split(df)
    pipe = f.build_pipeline("logistic", handling="none", calibration=None)
    pipe.fit(X_train.head(300), y_train.head(300))
    db_path = str(tmp_path / "ref.db")
    api_mod.set_model(pipe, "test-logistic-v0", db_path)
    api_mod.reset_drift_state()

    calls = {"n": 0}
    real_load = data_mod.load_telco

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return real_load(*args, **kwargs)

    data_mod.load_telco = counting_load
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: api_mod.get_reference(), range(8)))
    finally:
        data_mod.load_telco = real_load
    assert calls["n"] == 1

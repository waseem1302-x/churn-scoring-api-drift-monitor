"""SQLite prediction log. Every score is recorded with its inputs,
probability, tier, model version and timestamp, so any past decision
can be replayed. The database file is gitignored and never committed.
No drift fields in Phase 1; that is Phase 2.
"""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    customer_id TEXT,
    inputs_json TEXT NOT NULL,
    probability REAL NOT NULL,
    tier TEXT NOT NULL,
    model_version TEXT NOT NULL,
    latency_ms REAL NOT NULL
);
"""


def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)


def log_prediction(path: str, timestamp: str, customer_id, inputs: dict,
                   probability: float, tier: str, model_version: str,
                   latency_ms: float) -> int:
    init_db(path)
    with sqlite3.connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO predictions "
            "(timestamp, customer_id, inputs_json, probability, tier, "
            "model_version, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp, customer_id, json.dumps(inputs), probability, tier,
             model_version, latency_ms),
        )
        return cur.lastrowid

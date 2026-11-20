"""SQLite prediction log. Every score is recorded with its inputs,
probability, tier, model version and timestamp, so any past decision
can be replayed. The database file is gitignored and never committed.
Phase 2 adds drift_flags: the per-row quality flags and the latest
window alerts at the time the row was scored.
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
    latency_ms REAL NOT NULL,
    drift_flags TEXT
);
"""


def init_db(path: str) -> None:
    """Create the database and table if they do not exist yet, and add
    the drift_flags column to a database created by Phase 1, which did
    not have it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(predictions)")]
        if "drift_flags" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN drift_flags TEXT")


def log_prediction(path: str, timestamp: str, customer_id, inputs: dict,
                   probability: float, tier: str, model_version: str,
                   latency_ms: float, drift_flags: dict = None) -> int:
    """Write one prediction row. drift_flags holds the per-row quality
    flags and the latest window alerts, or None before any window has
    been evaluated."""
    init_db(path)
    with sqlite3.connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO predictions "
            "(timestamp, customer_id, inputs_json, probability, tier, "
            "model_version, latency_ms, drift_flags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, customer_id, json.dumps(inputs), probability, tier,
             model_version, latency_ms,
             json.dumps(drift_flags) if drift_flags is not None else None),
        )
        return cur.lastrowid


def recent_predictions(path: str, limit: int = 500) -> list:
    """The most recent logged predictions, oldest first, as dicts with
    inputs_json, probability and timestamp. The drift endpoint scores
    these against the training reference."""
    init_db(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT inputs_json, probability, timestamp FROM predictions"
            " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in reversed(rows)]

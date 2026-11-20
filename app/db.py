"""SQLite prediction log. Every score is recorded with its inputs,
probability, tier, model version and timestamp, so any past decision
can be replayed. The database file is gitignored and never committed.
Phase 2 adds drift_flags: the per-row quality flags and the latest
window alerts at the time the row was scored.
"""

import json
import sqlite3
from pathlib import Path

# Concurrency: every connection waits up to 30 seconds for a held
# write lock instead of failing after the sqlite3 default of 5, and
# the log runs in WAL mode so readers never block the writer. Without
# this, parallel /score calls can fail with "database is locked":
# tests/test_concurrency.py holds a write lock for 7 seconds and
# requires the next write to wait it out and succeed.
BUSY_TIMEOUT_MS = 30000


def _connect(path: str) -> sqlite3.Connection:
    """One connection with the concurrency settings applied. WAL is
    persistent per database, but setting it on every connection is
    harmless and keeps a fresh copy in the right mode."""
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

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
    drift_flags TEXT,
    label REAL
);
"""

# Labels a backfill CSV may carry, normalised to 0.0 / 1.0.
LABEL_VALUES = {
    "0": 0.0, "1": 1.0, "0.0": 0.0, "1.0": 1.0,
    "no": 0.0, "yes": 1.0, "false": 0.0, "true": 1.0,
}


def init_db(path: str) -> None:
    """Create the database and table if they do not exist yet, and add
    the drift_flags and label columns to a database created by an
    earlier phase, which did not have them."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(SCHEMA)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(predictions)")]
        if "drift_flags" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN drift_flags TEXT")
        if "label" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN label REAL")


def log_prediction(path: str, timestamp: str, customer_id, inputs: dict,
                   probability: float, tier: str, model_version: str,
                   latency_ms: float, drift_flags: dict = None) -> int:
    """Write one prediction row. drift_flags holds the per-row quality
    flags and the latest window alerts, or None before any window has
    been evaluated."""
    init_db(path)
    with _connect(path) as conn:
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
    inputs_json, probability, timestamp, customer_id and label. The
    drift endpoint scores these against the training reference, and
    uses the labelled ones for realised performance."""
    init_db(path)
    with _connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT inputs_json, probability, timestamp, customer_id,"
            " label FROM predictions"
            " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in reversed(rows)]


def set_label(path: str, customer_id: str, label: float):
    """Set the label on the most recent unlabelled row for one
    customer, and return its id, or None when no such row exists.

    Only the latest row is labelled: one row per customer keeps the
    realised metrics on /drift to independent observations instead
    of counting one customer three times for three scores."""
    init_db(path)
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT id FROM predictions WHERE customer_id = ?"
            " AND label IS NULL ORDER BY id DESC LIMIT 1",
            (customer_id,)).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE predictions SET label = ? WHERE id = ?",
                     (label, row[0]))
        return row[0]


def backfill_labels(path: str, csv_path: str, customer_column: str = "customer_id",
                    label_column: str = "label") -> dict:
    """Backfill labels from a labelled CSV into the prediction log.

    Each CSV row carries a customer id and a label (0/1, Yes/No,
    True/False). Returns how many rows were matched to an unlabelled
    prediction, how many customers had no such row, and how many
    labels could not be read. A label is never invented: unmatched
    and invalid rows are counted and left alone."""
    import csv

    init_db(path)
    matched = unmatched = invalid = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            customer = (row.get(customer_column) or "").strip()
            raw = (row.get(label_column) or "").strip().lower()
            if not customer or raw not in LABEL_VALUES:
                invalid += 1
                continue
            if set_label(path, customer, LABEL_VALUES[raw]) is None:
                unmatched += 1
            else:
                matched += 1
    return {"matched": matched, "unmatched": unmatched, "invalid": invalid}

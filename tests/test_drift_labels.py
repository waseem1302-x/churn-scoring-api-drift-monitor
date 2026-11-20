"""Labels in the prediction log, and realised performance on /drift.

Labels arrive about 30 days after the score, so the log stores
them late: a `label` column on the predictions table (added by
migration to existing databases), a backfill script that reads a
labelled CSV, and /drift reporting realised performance whenever
labelled rows exist. Without labels the realised channel stays
null, exactly as before.
"""

import csv
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import api as api_mod
from app import data as d
from app import db as db_mod
from tests.conftest import AUTH_HEADERS
from tests.test_drift_api import _score_rows, _telco_rows

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def champion():
    # The real champion artifact, loaded once. Same as the drift
    # API tests: /drift serves the champion, not a toy.
    from app import scoring

    model, meta = scoring.load_artifact()
    return model, meta


@pytest.fixture()
def client(tmp_path, champion):
    # A fresh log per test, so no test depends on another test
    # having run first.
    model, meta = champion
    db_path = tmp_path / "drift_predictions.db"
    api_mod.set_model(model, meta["model_version"], str(db_path),
                      threshold=meta["threshold"])
    api_mod.reset_drift_state()
    with TestClient(api_mod.app) as c:
        yield c, str(db_path)


def test_init_db_migrates_old_schema_without_label(tmp_path):
    # A database created before this change has no label column.
    # Opening it must add the column instead of failing on the
    # next score, and existing rows must survive.
    path = str(tmp_path / "old.db")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " timestamp TEXT NOT NULL, customer_id TEXT, inputs_json TEXT"
            " NOT NULL, probability REAL NOT NULL, tier TEXT NOT NULL,"
            " model_version TEXT NOT NULL, latency_ms REAL NOT NULL,"
            " drift_flags TEXT)")
        conn.execute(
            "INSERT INTO predictions (timestamp, customer_id, inputs_json,"
            " probability, tier, model_version, latency_ms) VALUES"
            " ('2026-11-20T00:00:00', 'C1', '{}', 0.5, 'LOW', 'v1', 1.0)")
    db_mod.init_db(path)
    cols = [r[1] for r in sqlite3.connect(path).execute(
        "PRAGMA table_info(predictions)")]
    assert "label" in cols
    assert sqlite3.connect(path).execute(
        "SELECT COUNT(*) FROM predictions").fetchone()[0] == 1


def test_backfill_labels_latest_row_only(tmp_path):
    path = str(tmp_path / "labels.db")
    db_mod.init_db(path)
    for customer, prob in [("A", 0.9), ("A", 0.2), ("B", 0.8)]:
        db_mod.log_prediction(path, "2026-11-20T00:00:00", customer, {},
                              prob, "LOW", "v1", 1.0)
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["customer_id", "label"])
        writer.writerow(["A", 1])
        writer.writerow(["B", 0])
        writer.writerow(["C", 1])
        writer.writerow(["B", "maybe"])
    result = db_mod.backfill_labels(path, str(csv_path))
    assert result == {"matched": 2, "unmatched": 1, "invalid": 1}
    rows = sqlite3.connect(path).execute(
        "SELECT customer_id, label FROM predictions ORDER BY id").fetchall()
    # Only the latest row per customer is labelled: one row per
    # customer keeps the realised metrics to independent rows.
    assert rows == [("A", None), ("A", 1.0), ("B", 0.0)]


def test_drift_realised_is_null_without_labels(client):
    c, _ = client
    _score_rows(c, _telco_rows(200))
    body = c.get("/drift", headers=AUTH_HEADERS).json()
    assert body["status"] == "ok"
    assert body["realised_performance"] is None
    assert body["n_labelled"] == 0


def test_drift_reports_realised_when_labels_exist(client, tmp_path):
    c, db_path = client
    rows = _telco_rows(200)
    _score_rows(c, rows)
    # The true labels come from the Telco test split, in the same
    # order the rows were scored, customer DRIFT-0000 onwards.
    # y_test is already encoded 0/1 by the split.
    df = d.load_telco(d.DATA_PATH)
    _, _, _, y_test = d.split(df)
    labels = [int(v) for v in y_test.head(200)]
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["customer_id", "label"])
        for i, label in enumerate(labels):
            writer.writerow([f"DRIFT-{i:04d}", label])
    result = db_mod.backfill_labels(db_path, str(csv_path))
    assert result == {"matched": 200, "unmatched": 0, "invalid": 0}
    body = c.get("/drift", headers=AUTH_HEADERS).json()
    assert body["status"] == "ok"
    assert body["n_labelled"] == 200
    realised = body["realised_performance"]
    assert realised is not None
    assert 0.5 < realised["roc_auc"] < 1.0
    assert 0.0 < realised["precision_at_threshold"] < 1.0
    assert 0.0 < realised["recall_at_threshold"] < 1.0
    assert realised["churn_rate"] == pytest.approx(
        sum(labels) / len(labels), abs=1e-4)
    # Labels do not move the input detectors: the same clean rows
    # still show no feature or prediction drift.
    assert body["alerts"]["feature_drift"] is False
    assert body["alerts"]["prediction_drift"] is False


def test_drift_single_class_labels_do_not_break_json(client, tmp_path):
    # ROC-AUC is undefined when the labelled rows hold only one
    # class. The endpoint must report the channel with a null
    # ROC-AUC, not fail to serialise.
    c, db_path = client
    _score_rows(c, _telco_rows(200))
    csv_path = tmp_path / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["customer_id", "label"])
        for i in range(200):
            writer.writerow([f"DRIFT-{i:04d}", 0])
    assert db_mod.backfill_labels(db_path, str(csv_path))["matched"] == 200
    resp = c.get("/drift", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_labelled"] == 200
    assert body["realised_performance"]["roc_auc"] is None
    assert body["realised_performance"]["churn_rate"] == 0.0


def test_docs_labels_section():
    text = (ROOT / "MONITORING.md").read_text()
    assert "backfill_labels" in text
    assert "30 days" in text
    assert "realised_performance" in text

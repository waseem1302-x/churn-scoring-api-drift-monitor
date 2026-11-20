"""API drift tests: GET /drift and drift flags in the prediction log.

GET /drift compares the recently logged scores against the Phase 1
training reference, using the same monitor the offline experiments
use. The contract test seeds drifted rows through /score, then checks
the endpoint reports the same numbers the monitor computes offline
on the same frame.
"""

import json
import sqlite3

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import api as api_mod
from app import data as d
from app import monitor
from tests.test_api import VALID_ROW

OUT_OF_RANGE_ROW = dict(VALID_ROW, MonthlyCharges=200.0)


@pytest.fixture(scope="module")
def champion():
    # The real champion artifact, loaded once. /drift serves the
    # champion, and a 300-row logistic has a different predicted-rate
    # behaviour (its rate moves 24 percent relative between train and
    # test splits), which would test the toy, not the endpoint.
    from app import scoring

    model, meta = scoring.load_artifact()
    return model, meta


@pytest.fixture()
def client(tmp_path, champion):
    # A fresh log per test, so no test depends on another test having
    # run first.
    model, meta = champion
    db_path = tmp_path / "drift_predictions.db"
    api_mod.set_model(model, meta["model_version"], str(db_path),
                      threshold=meta["threshold"])
    api_mod.reset_drift_state()
    with TestClient(api_mod.app) as c:
        yield c, str(db_path)


def _telco_rows(n, monthly_delta=0.0):
    """n varied Telco test-split rows as /score payloads, optionally
    shifted in MonthlyCharges. Varied rows matter: 60 identical rows
    would look drifted to any distribution detector."""
    from app import config

    df = d.load_telco(d.DATA_PATH)
    _, X_test, _, _ = d.split(df)
    rows = []
    for _, r in X_test.head(n).iterrows():
        row = {col: (None if pd.isna(r[col]) else r[col])
               for col in config.FEATURES}
        row["MonthlyCharges"] = float(row["MonthlyCharges"]) + monthly_delta
        row["TotalCharges"] = (None if row["TotalCharges"] is None
                               else float(row["TotalCharges"]))
        row["tenure"] = int(row["tenure"])
        row["SeniorCitizen"] = int(row["SeniorCitizen"])
        rows.append(row)
    return rows


def _score_rows(c, rows):
    for i, row in enumerate(rows):
        payload = dict(row, customerID=f"DRIFT-{i:04d}")
        resp = c.post("/score", json=payload)
        assert resp.status_code == 200, resp.text


def test_drift_needs_a_minimum_window(client):
    c, _ = client
    resp = c.get("/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_enough_data"
    assert body["window_rows"] < 200


def test_drift_quiet_on_clean_rows(client):
    c, _ = client
    _score_rows(c, _telco_rows(200))
    resp = c.get("/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["window_rows"] == 200
    assert body["alerts"]["feature_drift"] is False
    assert body["alerts"]["prediction_drift"] is False


def test_drift_matches_offline_monitor_on_drifted_rows(client):
    c, db_path = client
    _score_rows(c, _telco_rows(200, monthly_delta=25.0))
    resp = c.get("/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    monthly = body["feature_drift"]["MonthlyCharges"]
    assert monthly["drifted"] is True
    assert monthly["psi"] > 0.25
    # The endpoint must report the same numbers the monitor computes
    # offline on the same logged frame.
    from app import db as db_mod

    rows = db_mod.recent_predictions(db_path, limit=500)
    frame = pd.DataFrame([json.loads(r["inputs_json"]) for r in rows])
    scores = pd.Series([r["probability"] for r in rows]).to_numpy()
    ref_frame, ref_scores = api_mod.get_reference()
    offline = monitor.evaluate_window(
        ref_frame, frame, ref_scores, scores, api_mod.get_threshold())
    assert offline["feature_drift"]["MonthlyCharges"]["psi"] == pytest.approx(
        monthly["psi"], abs=1e-3)
    assert offline["drift_share"] == pytest.approx(body["drift_share"], abs=1e-3)


def test_score_stores_drift_flags_in_log(client):
    c, db_path = client
    # A /drift call first, so the window alerts exist to be attached.
    _score_rows(c, _telco_rows(200))
    assert c.get("/drift").json()["status"] == "ok"
    c.post("/score", json=OUT_OF_RANGE_ROW)
    row = sqlite3.connect(db_path).execute(
        "SELECT drift_flags FROM predictions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    flags = json.loads(row[0])
    assert "MonthlyCharges" in flags["row"]["out_of_range"]
    # The test scored 200 rows and called /drift first, so the window
    # alerts are attached to the row too.
    assert flags["window"] is not None
    assert set(flags["window"]) >= {"feature_drift", "prediction_drift",
                                    "data_quality", "performance_drop",
                                    "label_shift"}


def test_score_row_with_null_total_charges_flags_it(client):
    c, db_path = client
    row = dict(VALID_ROW)
    row["TotalCharges"] = None
    assert c.post("/score", json=row).status_code == 200
    flags = json.loads(sqlite3.connect(db_path).execute(
        "SELECT drift_flags FROM predictions ORDER BY id DESC LIMIT 1"
    ).fetchone()[0])
    assert flags["row"]["null_total_charges"] is True


def test_init_db_migrates_old_schema_without_drift_flags(tmp_path):
    # A database created by Phase 1 has no drift_flags column. Opening
    # it must add the column instead of failing on the next score.
    import sqlite3

    from app import db as db_mod

    path = str(tmp_path / "old.db")
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " timestamp TEXT NOT NULL, customer_id TEXT, inputs_json TEXT"
            " NOT NULL, probability REAL NOT NULL, tier TEXT NOT NULL,"
            " model_version TEXT NOT NULL, latency_ms REAL NOT NULL)")
    db_mod.init_db(path)
    cols = [r[1] for r in sqlite3.connect(path).execute(
        "PRAGMA table_info(predictions)")]
    assert "drift_flags" in cols

"""API tests: contract, validation, logging.

TDD: these tests fail until app/api.py, app/scoring.py and app/db.py exist.
The fixture trains a small logistic pipeline on 300 rows so tests stay fast
and never load the shipped artifact.
"""

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import api as api_mod
from app import data as d
from app import features as f
from tests.conftest import AUTH_HEADERS

VALID_ROW = {
    "customerID": "TEST-0001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 358.2,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    df = d.load_telco(d.DATA_PATH)
    X_train, _, y_train, _ = d.split(df)
    pipe = f.build_pipeline("logistic", handling="none", calibration=None)
    pipe.fit(X_train.head(300), y_train.head(300))
    db_path = tmp_path_factory.mktemp("db") / "test_predictions.db"
    api_mod.set_model(pipe, "test-logistic-v0", str(db_path))
    with TestClient(api_mod.app) as c:
        yield c, str(db_path)


def test_health_ok(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test-logistic-v0"


def test_score_returns_probability_and_tier(client):
    c, _ = client
    resp = c.post("/score", json=VALID_ROW, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["tier"] in ("LOW", "MEDIUM", "HIGH")
    assert body["model_version"] == "test-logistic-v0"
    assert body["customer_id"] == "TEST-0001"


def test_score_tier_matches_probability(client):
    c, _ = client
    resp = c.post("/score", json=VALID_ROW, headers=AUTH_HEADERS)
    p = resp.json()["probability"]
    tier = resp.json()["tier"]
    if p < 0.30:
        assert tier == "LOW"
    elif p < 0.60:
        assert tier == "MEDIUM"
    else:
        assert tier == "HIGH"


def test_score_rejects_unknown_category(client):
    c, _ = client
    bad = dict(VALID_ROW, InternetService="Satellite")
    assert c.post("/score", json=bad, headers=AUTH_HEADERS).status_code == 422


def test_score_rejects_missing_field(client):
    c, _ = client
    bad = {k: v for k, v in VALID_ROW.items() if k != "tenure"}
    assert c.post("/score", json=bad, headers=AUTH_HEADERS).status_code == 422


def test_score_rejects_negative_tenure(client):
    c, _ = client
    bad = dict(VALID_ROW, tenure=-1)
    assert c.post("/score", json=bad, headers=AUTH_HEADERS).status_code == 422


def test_score_rejects_bad_senior_citizen(client):
    c, _ = client
    bad = dict(VALID_ROW, SeniorCitizen=2)
    assert c.post("/score", json=bad, headers=AUTH_HEADERS).status_code == 422


def test_score_without_customer_id_still_works(client):
    c, _ = client
    row = {k: v for k, v in VALID_ROW.items() if k != "customerID"}
    resp = c.post("/score", json=row, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["customer_id"] is None


def test_score_writes_prediction_log_row(client):
    c, db_path = client
    before = sqlite3.connect(db_path).execute(
        "SELECT COUNT(*) FROM predictions").fetchone()[0]
    c.post("/score", json=VALID_ROW, headers=AUTH_HEADERS)
    after = sqlite3.connect(db_path).execute(
        "SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert after == before + 1
    row = sqlite3.connect(db_path).execute(
        "SELECT probability, tier, model_version FROM predictions "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert 0.0 <= row[0] <= 1.0
    assert row[1] in ("LOW", "MEDIUM", "HIGH")
    assert row[2] == "test-logistic-v0"


def test_metrics_endpoint_returns_json(client):
    c, _ = client
    resp = c.get("/metrics")
    assert resp.status_code == 200
    assert "model_version" in resp.json()

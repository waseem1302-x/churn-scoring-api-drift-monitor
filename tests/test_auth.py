"""API key tests: /score, /drift and /reload need the key.

TDD: these tests fail until app/api.py checks the X-API-Key header
against the CHURN_API_KEY environment variable. /health stays open,
and /metrics stays open: it serves the published benchmark file.
/reload is covered by tests/test_model_store.py once it exists; the
key checks here apply to it too.
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import api as api_mod
from app import data as d
from app import features as f
from tests.conftest import AUTH_HEADERS, TEST_API_KEY
from tests.test_api import VALID_ROW


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    df = d.load_telco(d.DATA_PATH)
    X_train, _, y_train, _ = d.split(df)
    pipe = f.build_pipeline("logistic", handling="none", calibration=None)
    pipe.fit(X_train.head(300), y_train.head(300))
    db_path = tmp_path_factory.mktemp("db") / "auth_predictions.db"
    api_mod.set_model(pipe, "test-logistic-v0", str(db_path))
    with TestClient(api_mod.app) as c:
        yield c


def test_health_open_without_key(client):
    assert client.get("/health").status_code == 200


def test_metrics_open_without_key(client):
    assert client.get("/metrics").status_code == 200


def test_score_without_key_is_401(client):
    assert client.post("/score", json=VALID_ROW).status_code == 401


def test_score_with_wrong_key_is_401(client):
    headers = {"X-API-Key": "not-the-key"}
    assert client.post("/score", json=VALID_ROW,
                       headers=headers).status_code == 401


def test_score_with_right_key_is_200(client):
    resp = client.post("/score", json=VALID_ROW, headers=AUTH_HEADERS)
    assert resp.status_code == 200


def test_drift_without_key_is_401(client):
    assert client.get("/drift").status_code == 401


def test_drift_with_right_key_is_not_401(client):
    resp = client.get("/drift", headers=AUTH_HEADERS)
    assert resp.status_code in (200,)


def test_score_when_key_not_configured_is_503(client, monkeypatch):
    monkeypatch.delenv("CHURN_API_KEY")
    assert client.post("/score", json=VALID_ROW,
                       headers=AUTH_HEADERS).status_code == 503


def test_env_var_name_is_documented():
    root = Path(__file__).resolve().parents[1]
    text = (root / "README.md").read_text()
    assert "CHURN_API_KEY" in text
    assert "X-API-Key" in text

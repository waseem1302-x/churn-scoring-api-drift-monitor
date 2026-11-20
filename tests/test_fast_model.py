"""Fast-model tests: the logistic alternative as an opt-in.

TDD: these tests fail until scripts/train_fast.py saves
artifacts/model-fast.joblib and app/api.py honours
CHURN_FAST_MODEL=1. The champion stays the default and
artifacts/model.joblib is never overwritten: the fast model is a
separate file, and the opt-in is an environment variable, not a
code change.
"""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import api as api_mod
from app import config, data, metrics, scoring
from tests.conftest import AUTH_HEADERS
from tests.test_api import VALID_ROW

ROOT = Path(__file__).resolve().parents[1]
FAST_PATH = ROOT / "artifacts" / "model-fast.joblib"


@pytest.fixture(autouse=True)
def _reset_model_after_test():
    """Leave no fast model cached for the next test file: the next
    _ensure_model call reloads from disk (champion by default)."""
    yield
    api_mod._MODEL = None
    api_mod.reset_drift_state()


def _load_fast():
    if not FAST_PATH.exists():
        pytest.fail("model-fast.joblib missing: run "
                    "scripts/train_fast.py first")
    return scoring.load_artifact(str(FAST_PATH))


def test_fast_artifact_reproduces_phase1_logistic():
    """The fast artifact is the Phase 1 logistic-none-platt config,
    refit the same way. Its test metrics must match the recorded
    Phase 1 numbers, or the file is not that config."""
    _, meta = _load_fast()
    assert meta["model_version"] == "churn-logistic-none-platt-v1"
    assert meta["threshold"] == pytest.approx(0.1272, abs=0.0001)
    all_configs = json.loads(
        (ROOT / "benchmarks" / "results" / "all_configs.json").read_text())
    recorded = next(c for c in all_configs
                    if c["model_version"] == "churn-logistic-none-platt-v1")
    model, _ = _load_fast()
    df = data.load_telco()
    _, X_test, _, y_test = data.split(df)
    X_test_f = X_test.drop(columns=[config.ID_COLUMN])
    proba = model.predict_proba(X_test_f)[:, 1]
    measured = metrics.evaluate_model(y_test, proba,
                                      threshold=meta["threshold"])
    for key in ("roc_auc", "pr_auc", "brier", "precision_at_100"):
        assert measured[key] == pytest.approx(
            recorded["test_metrics"][key], abs=0.002), key


def test_champion_artifact_still_the_champion():
    model, meta = scoring.load_artifact()
    summary = json.loads(
        (ROOT / "benchmarks" / "results" / "metrics.json").read_text())
    assert meta["model_version"] == summary["champion"]["model_version"]
    assert meta["threshold"] == summary["champion"]["threshold_frozen"]


def test_api_serves_fast_model_when_opted_in(monkeypatch, tmp_path):
    monkeypatch.setenv("CHURN_FAST_MODEL", "1")
    api_mod.set_model(None, "not-loaded", str(tmp_path / "fast.db"))
    with TestClient(api_mod.app) as c:
        body = c.get("/health").json()
        assert body["model_version"] == "churn-logistic-none-platt-v1"
        resp = c.post("/score", json=VALID_ROW, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["model_version"] == "churn-logistic-none-platt-v1"


def test_api_serves_champion_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("CHURN_FAST_MODEL", raising=False)
    api_mod.set_model(None, "not-loaded", str(tmp_path / "default.db"))
    with TestClient(api_mod.app) as c:
        body = c.get("/health").json()
        assert body["model_version"] == "churn-xgboost-smote-isotonic-v1"


def test_docs_fast_section_matches_json():
    path = ROOT / "benchmarks" / "results" / "latency_fast.json"
    if not path.exists():
        pytest.fail("latency_fast.json missing: run "
                    "scripts/measure_fast_latency.py first")
    fast = json.loads(path.read_text())
    assert fast["model_version"] == "churn-logistic-none-platt-v1"
    quiet = json.loads(
        (ROOT / "benchmarks" / "results"
         / "latency_champion_quiet.json").read_text())
    assert quiet["model_version"] == "churn-xgboost-smote-isotonic-v1"
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    start = text.index("### Fast alternative")
    end = text.index("## ", start + 3)
    section = text[start:end]
    for key in ("mean_ms", "p50_ms", "p95_ms"):
        assert f"{fast[key]:.2f} ms" in section, key
    for key in ("mean_ms", "p50_ms", "p95_ms"):
        assert f"{quiet[key]:.2f} ms" in section, key
    assert "0.6348" in section
    assert "0.6395" in section

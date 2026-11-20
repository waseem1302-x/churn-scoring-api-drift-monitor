"""Model store tests: versioned artifacts, pointer, reload endpoint.

TDD: these tests fail until app/model_store.py exists and
POST /reload serves the pointed model. The Phase 1 artifact
(artifacts/model.joblib) is never overwritten: a promotion saves a
new versioned file and moves a pointer, and /reload serves whatever
the pointer says.
"""

import joblib
import pytest
from fastapi.testclient import TestClient

from app import api as api_mod
from app import data as d
from app import features as f
from app import model_store
from tests.conftest import AUTH_HEADERS
from tests.test_api import VALID_ROW


def _bundle(tmp_path, name, version, threshold=0.0934):
    """A tiny fitted pipeline saved as a joblib bundle, like training
    saves. Returns the bundle path."""
    df = d.load_telco(d.DATA_PATH)
    X_train, _, y_train, _ = d.split(df)
    pipe = f.build_pipeline("logistic", handling="none", calibration=None)
    pipe.fit(X_train.head(300), y_train.head(300))
    path = tmp_path / name
    joblib.dump({"model": pipe,
                 "metadata": {"model_version": version,
                              "threshold": threshold}}, path)
    return path


def _use_tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(model_store, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(model_store, "POINTER_PATH",
                        tmp_path / "current_model.json")


def test_resolve_without_pointer_returns_default(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    path = model_store.resolve_current_artifact()
    assert path.name == "model.joblib"


def test_save_promotion_writes_versioned_file_and_pointer(
        monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    (tmp_path / "model.joblib").write_bytes(b"phase1")
    bundle = joblib.load(_bundle(tmp_path, "challenger.joblib",
                                 "churn-x-challenger-1"))
    saved = model_store.save_promotion(
        bundle["model"], "churn-x-challenger-1", 0.0934,
        source="TRG-08-1")
    assert saved.name == "model-churn-x-challenger-1.joblib"
    assert saved.exists()
    # The Phase 1 file is untouched.
    assert (tmp_path / "model.joblib").read_bytes() == b"phase1"
    pointer = model_store.read_pointer()
    assert pointer["model_file"] == saved.name
    assert pointer["model_version"] == "churn-x-challenger-1"
    assert pointer["source"] == "TRG-08-1"
    assert model_store.resolve_current_artifact() == saved


def test_save_promotion_rejects_bad_version(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    bundle = joblib.load(_bundle(tmp_path, "c.joblib", "v1"))
    with pytest.raises(ValueError):
        model_store.save_promotion(bundle["model"], "../evil", 0.0934)


def test_save_promotion_refuses_to_overwrite(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    bundle = joblib.load(_bundle(tmp_path, "c.joblib", "v1"))
    model_store.save_promotion(bundle["model"], "v1", 0.0934)
    with pytest.raises(FileExistsError):
        model_store.save_promotion(bundle["model"], "v1", 0.0934)


def test_reload_serves_the_pointed_model(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    _bundle(tmp_path, "model.joblib", "model-a-v1")
    bundle_b = joblib.load(_bundle(tmp_path, "challenger.joblib",
                                   "model-b-v1"))
    model_store.save_promotion(bundle_b["model"], "model-b-v1", 0.0934)
    api_mod.set_model(None, "not-loaded", str(tmp_path / "db.db"))
    with TestClient(api_mod.app) as c:
        assert c.get("/health").json()["model_version"] == "model-b-v1"
        resp = c.post("/score", json=VALID_ROW, headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["model_version"] == "model-b-v1"


def test_reload_endpoint_swaps_the_served_model(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    _bundle(tmp_path, "model.joblib", "model-a-v1")
    api_mod.set_model(None, "not-loaded", str(tmp_path / "db.db"))
    with TestClient(api_mod.app) as c:
        assert c.get("/health").json()["model_version"] == "model-a-v1"
        bundle_b = joblib.load(_bundle(tmp_path, "challenger.joblib",
                                       "model-b-v1"))
        model_store.save_promotion(bundle_b["model"], "model-b-v1", 0.0934)
        # Still serving A until /reload is called.
        assert c.get("/health").json()["model_version"] == "model-a-v1"
        resp = c.post("/reload", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["model_version"] == "model-b-v1"
        assert c.get("/health").json()["model_version"] == "model-b-v1"
        scored = c.post("/score", json=VALID_ROW, headers=AUTH_HEADERS)
        assert scored.json()["model_version"] == "model-b-v1"


def test_reload_without_key_is_401(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    _bundle(tmp_path, "model.joblib", "model-a-v1")
    api_mod.set_model(None, "not-loaded", str(tmp_path / "db.db"))
    with TestClient(api_mod.app) as c:
        assert c.post("/reload").status_code == 401
        assert c.post("/reload",
                      headers={"X-API-Key": "wrong"}).status_code == 401


def test_promote_cli_moves_the_pointer(monkeypatch, tmp_path):
    # scripts/promote.py end to end, against the tmp store: the real
    # artifacts directory is never touched by this test.
    import sys

    from scripts import promote

    _use_tmp_store(monkeypatch, tmp_path)
    challenger = _bundle(tmp_path, "challenger.joblib", "model-b-v1")
    monkeypatch.setattr(sys, "argv", [
        "promote", "--artifact", str(challenger),
        "--version", "model-b-v1", "--threshold", "0.0934",
        "--source", "TRG-08-1"])
    promote.main()
    pointer = model_store.read_pointer()
    assert pointer["model_version"] == "model-b-v1"
    assert pointer["source"] == "TRG-08-1"
    assert (tmp_path / "model-model-b-v1.joblib").exists()

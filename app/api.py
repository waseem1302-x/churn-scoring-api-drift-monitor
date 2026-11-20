"""FastAPI scoring service.

POST /score: one customer in, churn probability and risk tier out.
GET /health: liveness plus the loaded model version.
GET /metrics: the measured Phase 1 metrics (from benchmarks/results/).

Auth: /score, /drift and /reload need the API key from the
CHURN_API_KEY environment variable, sent as the X-API-Key header.
Missing or wrong key is 401; a server with no key configured is 503
on those endpoints. /health and /metrics stay open: /health is what
a load balancer calls, and /metrics serves the published benchmark
file. The key is never committed and never logged.

Every score is logged to SQLite with its inputs, probability, tier,
model version and latency, plus Phase 2 drift flags: the per-row
quality check, and the alerts from the latest evaluated window.
GET /drift returns that window's drift summary, computed by the same
monitor the offline experiments use, against the Phase 1 training
reference. The served model is the artifact the current-model
pointer names (artifacts/current_model.json), or the Phase 1
artifact when no promotion has happened. POST /reload re-reads the
pointer and serves the pointed model; there is no separate serving
code path.
"""

import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app import config, data, db, model_store, monitor, scoring

app = FastAPI(title="Churn Scoring API")

API_KEY_ENV = "CHURN_API_KEY"
API_KEY_HEADER = "X-API-Key"


def require_api_key(request: Request) -> None:
    """Check the X-API-Key header against CHURN_API_KEY.

    Read at request time, not import time, so the key can be set after
    the process starts and tests can swap it. A missing server key is
    a 503: the endpoint is misconfigured, not unauthorised."""
    expected = os.environ.get(API_KEY_ENV)
    if not expected:
        raise HTTPException(status_code=503,
                            detail="API key not configured on the server")
    provided = request.headers.get(API_KEY_HEADER)
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401,
                            detail="Invalid or missing API key")

_MODEL = None
_MODEL_VERSION = "not-loaded"
_DB_PATH = str(config.DB_PATH)
_THRESHOLD = 0.5
# The drift reference (training frame plus the champion's scores on it)
# and the latest evaluated window, cached in process. The reference is
# built once, on first use, and reset by set_model. The lock stops
# concurrent first calls from each building it: without it, 8 threads
# scoring at once would build the same reference 8 times.
_REFERENCE = None
_REFERENCE_LOCK = threading.Lock()
_LATEST_DRIFT = None
# A drift window needs this many logged rows before it means anything.
# 200, not 50: measured with the champion, a clean 60-row window has
# a MonthlyCharges Wasserstein distance of 0.105, over the 0.1
# threshold, while a clean 200-row window sits at 0.065. Windows under
# 200 rows are too small for the monitor's thresholds, which were
# validated on 2,000-row windows.
DRIFT_MIN_ROWS = 200
DRIFT_WINDOW_ROWS = 500


class ScoreRequest(BaseModel):
    customerID: Optional[str] = None
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    # None means no bill yet (new customer, tenure 0); the pipeline
    # imputes the training median, exactly as it does for the 11 blank
    # TotalCharges rows in the training data.
    TotalCharges: Optional[float] = Field(default=None, ge=0)


def set_model(model, model_version: str, db_path: str,
              threshold: float = 0.5) -> None:
    """Test hook and manual wiring: install an already-fitted model."""
    global _MODEL, _MODEL_VERSION, _DB_PATH, _THRESHOLD
    _MODEL = model
    _MODEL_VERSION = model_version
    _DB_PATH = db_path
    _THRESHOLD = threshold
    reset_drift_state()


def reset_drift_state() -> None:
    """Clear the cached drift reference and latest window. Used when a
    different model is installed, and by the drift tests."""
    global _REFERENCE, _LATEST_DRIFT
    with _REFERENCE_LOCK:
        _REFERENCE = None
    _LATEST_DRIFT = None


def get_threshold() -> float:
    return _THRESHOLD


def get_reference():
    """The drift reference: the Phase 1 training frame and the loaded
    model's scores on it. Built once, cached, cheap after the first
    call. The frame is what row-quality flags and the /drift window
    are compared against."""
    global _REFERENCE
    _ensure_model()
    if _REFERENCE is None:
        with _REFERENCE_LOCK:
            if _REFERENCE is None:
                df = data.load_telco(data.DATA_PATH)
                X_train, _, _, _ = data.split(df)
                ref_frame = X_train.drop(columns=[config.ID_COLUMN])
                ref_scores = _MODEL.predict_proba(ref_frame[config.FEATURES])[:, 1]
                _REFERENCE = (ref_frame, ref_scores)
    return _REFERENCE


def _ensure_model():
    global _MODEL, _MODEL_VERSION, _THRESHOLD
    if _MODEL is None:
        # Opt-in fast model: one logistic pipeline instead of the
        # three-model calibrated champion. Set CHURN_FAST_MODEL=1 at
        # startup; the default is the champion, and the pointer still
        # decides which champion artifact serves.
        if os.environ.get("CHURN_FAST_MODEL") == "1":
            path = config.FAST_ARTIFACT_PATH
            if not path.exists():
                raise HTTPException(
                    status_code=503,
                    detail="fast model artifact not found: "
                           "run scripts/train_fast.py first")
            _MODEL, meta = scoring.load_artifact(str(path))
            _MODEL_VERSION = meta["model_version"]
            _THRESHOLD = meta["threshold"]
            return
        try:
            path = model_store.resolve_current_artifact()
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        if not path.exists():
            raise HTTPException(status_code=503, detail="model artifact not found")
        _MODEL, meta = scoring.load_artifact(str(path))
        _MODEL_VERSION = meta["model_version"]
        _THRESHOLD = meta["threshold"]


@app.get("/health")
def health():
    # Load the model here so /health reports the real version (and 503s
    # when the artifact is missing, which is what a health check is for).
    _ensure_model()
    return {
        "status": "ok",
        "model_version": _MODEL_VERSION,
        "features": len(config.FEATURES),
    }


@app.post("/score")
def score(req: ScoreRequest, _key: None = Depends(require_api_key)):
    _ensure_model()
    started = time.perf_counter()
    inputs = req.model_dump()
    customer_id = inputs.pop("customerID")
    probability = scoring.score_row(_MODEL, inputs)
    tier = scoring.risk_tier(probability)
    latency_ms = (time.perf_counter() - started) * 1000
    # Drift flags: the per-row quality check, plus the alerts from the
    # latest evaluated window, if any window has been evaluated yet.
    ref_frame, _ = get_reference()
    drift_flags = {
        "row": monitor.row_quality_flags(inputs, ref_frame),
        "window": _LATEST_DRIFT["alerts"] if _LATEST_DRIFT is not None else None,
    }
    db.log_prediction(
        _DB_PATH,
        datetime.now(timezone.utc).isoformat(),
        customer_id, inputs, probability, tier, _MODEL_VERSION, latency_ms,
        drift_flags=drift_flags,
    )
    return {
        "customer_id": customer_id,
        "probability": round(probability, 4),
        "tier": tier,
        "model_version": _MODEL_VERSION,
    }


@app.get("/drift")
def drift(_key: None = Depends(require_api_key)):
    """The latest window's drift summary: the most recent logged scores
    compared against the Phase 1 training reference, by the same
    monitor the offline experiments use. Rows that have a label
    (backfilled late, about 30 days after the score, with
    scripts/backfill_labels.py) add the realised-performance channel;
    without any labelled rows that channel stays null, and the window
    is inputs and predictions only."""
    global _LATEST_DRIFT
    _ensure_model()
    rows = db.recent_predictions(_DB_PATH, limit=DRIFT_WINDOW_ROWS)
    if len(rows) < DRIFT_MIN_ROWS:
        return {
            "status": "not_enough_data",
            "window_rows": len(rows),
            "minimum_rows": DRIFT_MIN_ROWS,
            "model_version": _MODEL_VERSION,
        }
    frame = pd.DataFrame([json.loads(r["inputs_json"]) for r in rows])
    scores = np.array([r["probability"] for r in rows])
    labelled = [(r["label"], r["probability"]) for r in rows
                if r.get("label") is not None]
    y_true = y_proba = None
    reference_metrics = None
    if labelled:
        y_true = np.array([label for label, _ in labelled])
        y_proba = np.array([prob for _, prob in labelled])
        reference_metrics = _reference_metrics()
    ref_frame, ref_scores = get_reference()
    summary = monitor.evaluate_window(
        ref_frame, frame, ref_scores, scores, _THRESHOLD,
        y_true=y_true, y_proba=y_proba,
        reference_metrics=reference_metrics)
    summary["status"] = "ok"
    summary["model_version"] = _MODEL_VERSION
    summary["window_rows"] = len(rows)
    summary["n_labelled"] = len(labelled)
    _LATEST_DRIFT = summary
    return _json_safe(summary)


def _json_safe(value):
    """Replace non-finite floats with None, recursively. The realised
    channel can produce NaN when the labelled rows hold only one
    class (ROC-AUC is undefined then), and NaN is not valid JSON:
    without this the endpoint fails to serialise instead of
    reporting the channel."""
    import math

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _reference_metrics():
    """The Phase 1 test metrics for the champion, used as the
    baseline for the realised-performance drops on /drift. None
    when the metrics file is missing or has no champion block."""
    path = Path(config.METRICS_PATH)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    champion = data.get("champion") or {}
    metrics = champion.get("metrics") or {}
    return metrics or None


@app.post("/reload")
def reload_model(_key: None = Depends(require_api_key)):
    """Re-read the current-model pointer and serve the pointed model.

    Called after a promotion moves the pointer. The in-process model
    is dropped, the drift reference is rebuilt from the new model,
    and /health afterwards reports the new version. A missing or
    corrupt pointer is a 503, not a silent fall back to the old model.
    """
    global _MODEL
    _MODEL = None
    reset_drift_state()
    _ensure_model()
    return {"status": "reloaded", "model_version": _MODEL_VERSION}


@app.get("/metrics")
def metrics():
    path = Path(config.METRICS_PATH)
    if path.exists():
        return json.loads(path.read_text())
    return {"model_version": _MODEL_VERSION, "metrics": "not recorded yet"}

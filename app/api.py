"""FastAPI scoring service.

POST /score: one customer in, churn probability and risk tier out.
GET /health: liveness plus the loaded model version.
GET /metrics: the measured Phase 1 metrics (from benchmarks/results/).

Every score is logged to SQLite with its inputs, probability, tier,
model version and latency. The served model is the artifact training
saved; there is no separate serving code path.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import config, db, scoring

app = FastAPI(title="Churn Scoring API")

_MODEL = None
_MODEL_VERSION = "not-loaded"
_DB_PATH = str(config.DB_PATH)


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


def set_model(model, model_version: str, db_path: str) -> None:
    """Test hook and manual wiring: install an already-fitted model."""
    global _MODEL, _MODEL_VERSION, _DB_PATH
    _MODEL = model
    _MODEL_VERSION = model_version
    _DB_PATH = db_path


def _ensure_model():
    global _MODEL, _MODEL_VERSION
    if _MODEL is None:
        path = Path(config.ARTIFACT_PATH)
        if not path.exists():
            raise HTTPException(status_code=503, detail="model artifact not found")
        _MODEL, meta = scoring.load_artifact(str(path))
        _MODEL_VERSION = meta["model_version"]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": _MODEL_VERSION,
        "features": len(config.FEATURES),
    }


@app.post("/score")
def score(req: ScoreRequest):
    _ensure_model()
    started = time.perf_counter()
    inputs = req.model_dump()
    customer_id = inputs.pop("customerID")
    probability = scoring.score_row(_MODEL, inputs)
    tier = scoring.risk_tier(probability)
    latency_ms = (time.perf_counter() - started) * 1000
    db.log_prediction(
        _DB_PATH,
        datetime.now(timezone.utc).isoformat(),
        customer_id, inputs, probability, tier, _MODEL_VERSION, latency_ms,
    )
    return {
        "customer_id": customer_id,
        "probability": round(probability, 4),
        "tier": tier,
        "model_version": _MODEL_VERSION,
    }


@app.get("/metrics")
def metrics():
    path = Path(config.METRICS_PATH)
    if path.exists():
        return json.loads(path.read_text())
    return {"model_version": _MODEL_VERSION, "metrics": "not recorded yet"}

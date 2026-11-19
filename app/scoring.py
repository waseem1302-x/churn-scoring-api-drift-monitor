"""Serving-side scoring: load the shipped artifact and score one row.

The served model is the exact object training saved, and the API scores
through the same pipeline training used, so there is no separate serving
code path to drift from training. customerID is accepted for the audit
log only and is never fed to the model.
"""

import joblib
import pandas as pd

from app import config


def risk_tier(probability: float) -> str:
    if probability < config.TIER_LOW_MAX:
        return "LOW"
    if probability < config.TIER_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def load_artifact(path: str = str(config.ARTIFACT_PATH)):
    bundle = joblib.load(path)
    return bundle["model"], bundle["metadata"]


def score_row(model, inputs: dict) -> float:
    """Score one customer. inputs holds the 19 model features; an extra
    customerID key is ignored by the model, never used as a feature."""
    row = {k: v for k, v in inputs.items() if k in config.FEATURES}
    missing = [f for f in config.FEATURES if f not in row]
    if missing:
        raise ValueError(f"missing features: {missing}")
    frame = pd.DataFrame([row], columns=config.FEATURES)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    return float(model.predict_proba(frame)[0, 1])

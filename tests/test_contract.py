"""Contract test: the shipped artifact must reproduce its recorded scores.

TDD: fails until scripts/train.py has written artifacts/model.joblib and
benchmarks/results/contract_rows.json. Run scripts/train.py first; the
README documents that order. If the artifact is missing the test fails
loudly instead of skipping, so a checkout without a trained model cannot
pass CI by accident.
"""

import json
from pathlib import Path

import pytest

from app import scoring

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "model.joblib"
CONTRACT = ROOT / "benchmarks" / "results" / "contract_rows.json"


def test_artifact_and_contract_rows_exist():
    assert ARTIFACT.exists(), "run scripts/train.py first: artifacts/model.joblib missing"
    assert CONTRACT.exists(), "run scripts/train.py first: contract rows missing"


def test_shipped_model_reproduces_recorded_probabilities():
    model, meta = scoring.load_artifact(str(ARTIFACT))
    rows = json.loads(CONTRACT.read_text())
    assert len(rows) >= 20
    for row in rows:
        proba = scoring.score_row(model, row["inputs"])
        assert proba == pytest.approx(row["probability"], abs=1e-9), row["id"]
        assert scoring.risk_tier(proba) == row["tier"]


def test_artifact_metadata_matches_metrics_file():
    _, meta = scoring.load_artifact(str(ARTIFACT))
    metrics_path = ROOT / "benchmarks" / "results" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["champion"]["model_version"] == meta["model_version"]
    assert meta["model_name"] in ("logistic", "random_forest", "xgboost")
    assert meta["handling"] in ("none", "class_weight", "smote")
    assert meta["calibration"] in (None, "platt", "isotonic")

"""CBPE tests: the prepared frames and the estimate record.

The estimation itself runs under NannyML in a separate venv
(.venv-cbpe), because NannyML 0.6.0 cannot install on Python 3.12
and 0.13.1 would downgrade the repo's pinned xgboost. These tests
run in the repo venv and never import nannyml: they check the
prepared frames, and they check that the estimate JSON and the
docs agree. If a file is missing, the message says which script
to run first.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def _load(name):
    path = RESULTS / name
    if not path.exists():
        pytest.fail(f"{name} missing: run scripts/cbpe_prepare.py first")
    return path


def test_prepare_frame_shape_and_columns():
    _load("cbpe_frames.csv")
    _load("cbpe_manifest.json")
    frame = pd.read_csv(RESULTS / "cbpe_frames.csv")
    assert list(frame.columns) == [
        "week", "date", "y_true", "y_pred_proba", "y_pred"]
    assert len(frame) == 60000
    assert sorted(frame["week"].unique()) == list(range(1, 13))
    assert frame.groupby("week").size().eq(5000).all()
    assert frame["y_pred_proba"].between(0, 1).all()
    assert set(frame["y_true"].unique()) <= {0, 1}
    assert set(frame["y_pred"].unique()) <= {0, 1}
    manifest = json.loads((RESULTS / "cbpe_manifest.json").read_text())
    assert manifest["reference_weeks"] == [1, 4]
    assert manifest["analysis_weeks"] == list(range(5, 13))
    assert manifest["n_customers_per_week"] == 5000


def test_estimate_json_exists_and_covers_all_analysis_weeks():
    path = RESULTS / "cbpe_estimate.json"
    if not path.exists():
        pytest.fail("cbpe_estimate.json missing: run "
                    "scripts/cbpe_estimate.py with the CBPE venv first")
    data = json.loads(path.read_text())
    assert data["analysis_weeks"] == list(range(5, 13))
    assert data["nannyml_version"] == "0.13.1"
    metrics = {m["metric"]: m for m in data["metrics"]}
    assert set(metrics) == {"roc_auc", "precision", "recall"}
    for metric, record in metrics.items():
        assert [p["week"] for p in record["per_week"]] == list(range(5, 13))
        for p in record["per_week"]:
            assert 0.0 <= p["estimated"] <= 1.0
            assert 0.0 <= p["realised"] <= 1.0
            assert p["absolute_error"] >= 0.0
        assert 0.0 <= record["mae"] <= 1.0


def test_docs_cbpe_section_matches_json():
    path = RESULTS / "cbpe_estimate.json"
    if not path.exists():
        pytest.fail("cbpe_estimate.json missing: run "
                    "scripts/cbpe_estimate.py with the CBPE venv first")
    data = json.loads(path.read_text())
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    start = text.index("## CBPE: performance without labels")
    end = text.index("## ", start + 3)
    section = text[start:end]
    roc = next(m for m in data["metrics"] if m["metric"] == "roc_auc")
    assert f"{roc['mae']:.4f}" in section
    assert f"{data['nannyml_version']}" in section
    assert "0.6.0" in section  # the blocked version, stated honestly
    assert f"{data['max_weekly_move_estimated_roc_auc']:.4f}" in section

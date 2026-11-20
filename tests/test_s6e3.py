"""Tests for the S6E3 scale run.

The Kaggle Playground Series S6E3 synthetic train set (594,194 rows)
is the scale complement to Telco: same schema, about 85 copies of
each original row. The loader must map it onto the Telco shape
(id becomes customerID, Churn Yes/No becomes 1/0) so the same
pipeline runs unchanged. The scale numbers live in
benchmarks/results/scale_s6e3.json and the docs table must match
that file exactly. If the downloaded file is missing, the message
says where it came from.
"""

import json
from pathlib import Path

import pytest

from app import config, data

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
SCALE = RESULTS / "scale_s6e3.json"

S6E3_PATH = Path(
    "/home/hatch/workspace/browser_downloads/sess-3804851174/"
    "browser-download-20261120T014700.434094123Z-0-train.csv")

EXPECTED_ROWS = 594194
EXPECTED_CHURN_RATE = 0.2252


def _load():
    if not S6E3_PATH.exists():
        pytest.fail(f"S6E3 file missing: {S6E3_PATH} (Kaggle download)")
    return data.load_s6e3(S6E3_PATH)


def test_load_s6e3_row_count_and_churn_rate():
    df = _load()
    assert len(df) == EXPECTED_ROWS, len(df)
    assert abs(float(df["Churn"].mean()) - EXPECTED_CHURN_RATE) < 0.001


def test_load_s6e3_schema_matches_telco():
    df = _load()
    assert "id" not in df.columns
    assert config.ID_COLUMN in df.columns
    assert set(df["Churn"].unique()) == {0, 1}
    assert sorted(df.columns) == sorted(config.FEATURES + [config.ID_COLUMN, "Churn"])
    assert not data.has_timestamps(df)
    # Same category values as Telco: the pipeline's one-hot encoder
    # sees nothing new.
    assert sorted(df["InternetService"].unique()) == ["DSL", "Fiber optic", "No"]
    assert sorted(df["Contract"].unique()) == [
        "Month-to-month", "One year", "Two year"]


def test_verify_s6e3():
    df = _load()
    report = data.verify_s6e3(df)
    assert report["rows"] == EXPECTED_ROWS
    assert abs(report["churn_rate"] - EXPECTED_CHURN_RATE) < 0.001
    assert report["duplicate_ids"] == 0
    assert report["blank_total_charges"] == 0


def test_s6e3_split_sizes():
    df = _load()
    X_train, X_test, _, _ = data.split(df)
    assert len(X_train) == 475355, len(X_train)
    assert len(X_test) == 118839, len(X_test)


def _load_scale():
    if not SCALE.exists():
        pytest.fail("scale_s6e3.json missing: run scripts/scale_s6e3.py first")
    return json.loads(SCALE.read_text())


def test_scale_results_shape():
    result = _load_scale()
    assert result["dataset"]["rows"] == EXPECTED_ROWS
    assert abs(result["dataset"]["churn_rate"] - EXPECTED_CHURN_RATE) < 0.001
    assert result["dataset"]["train_rows"] == 475355
    assert result["dataset"]["test_rows"] == 118839
    assert set(result["configs"]) == {"champion", "logistic"}
    for name in ("champion", "logistic"):
        cfg = result["configs"][name]
        assert cfg["train_time_s"] > 0
        for key in ("roc_auc", "pr_auc", "brier", "precision_at_100",
                    "precision_at_10pct"):
            assert isinstance(cfg["metrics_frozen_threshold"][key], float), (name, key)
        assert cfg["metrics_reoptimized_threshold"]["threshold"] > 0
    assert result["latency"]["n"] == 200
    for key in ("mean_ms", "p50_ms", "p95_ms"):
        assert result["latency"][key] > 0


def _table_rows(section):
    rows = {}
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) == 4 and parts[0] not in ("Metric", "") and not set(parts[0]) <= {"-", " "}:
            rows[parts[0]] = parts[1:]
    return rows


def test_docs_scale_table_matches_json():
    result = _load_scale()
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    start = text.index("## S6E3 scale run")
    end = text.index("## ", start + 3)
    section = text[start:end]
    rows = _table_rows(section)
    champ = result["configs"]["champion"]
    logi = result["configs"]["logistic"]
    telco = {
        "ROC-AUC": "0.8366",
        "PR-AUC": "0.6395",
        "Brier": "0.1392",
        "Precision at 100": "0.76",
        "Precision at 10%": "0.7589",
    }
    key_map = {
        "ROC-AUC": "roc_auc",
        "PR-AUC": "pr_auc",
        "Brier": "brier",
        "Precision at 100": "precision_at_100",
        "Precision at 10%": "precision_at_10pct",
    }
    for metric, telco_value in telco.items():
        assert metric in rows, metric
        assert rows[metric][0] == telco_value, metric
        key = key_map[metric]
        # Compare as numbers: the docs write 0.97 where the JSON
        # holds 0.9700, same value, different trailing zeros.
        assert float(rows[metric][1]) == champ["metrics_frozen_threshold"][key], metric
        assert float(rows[metric][2]) == logi["metrics_frozen_threshold"][key], metric
    assert "Train time" in rows
    assert rows["Train time"][1] == f"{champ['train_time_s']:.2f} s"
    assert rows["Train time"][2] == f"{logi['train_time_s']:.2f} s"
    assert "Latency p50" in rows
    assert rows["Latency p50"][1] == f"{result['latency']['p50_ms']:.2f} ms"
    assert "594,194" in section
    assert "22.52%" in section


def test_docs_state_the_honesty_limits():
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    start = text.index("## S6E3 scale run")
    end = text.index("## ", start + 3)
    section = text[start:end].lower()
    assert "near-duplicate" in section
    assert "not independent evidence" in section
    assert "22.52%" in section or "22.52" in section

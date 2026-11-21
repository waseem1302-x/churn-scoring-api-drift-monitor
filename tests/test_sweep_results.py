"""Cross-checks for the S6E3 27-config sweep results.

The sweep ran on Colab, so its numbers arrive as a results file,
not as a local run. These tests tie the file to itself (champion
rule, train-time sum, calibration values) and tie the docs table
to the file, so a wrong number in either place fails loudly.
"""

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "benchmarks" / "results" / "scale_s6e3_sweep.json"


def _load():
    return json.loads(SWEEP_PATH.read_text())


def _section():
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    start = text.index("## S6E3 27-config sweep")
    end = text.index("## ", start + 3)
    return text[start:end]


def _calibration(config):
    return config["calibration"] or "none"


def test_sweep_file_is_complete():
    data = _load()
    assert data["complete"] is True
    assert data["resumed_from_partial"] is True
    assert len(data["configs"]) == 27
    versions = [c["model_version"] for c in data["configs"]]
    assert len(set(versions)) == 27
    for model in ("logistic", "random_forest", "xgboost"):
        assert sum(1 for c in data["configs"] if c["model"] == model) == 9
    assert data["champion"]["model_version"] in versions


def test_champion_is_max_oof_profit():
    data = _load()
    configs = data["configs"]
    best = max(configs, key=lambda c: (c["oof_profit_per_1000"],
                                       c["test_metrics"]["pr_auc"]))
    assert data["champion"]["model_version"] == best["model_version"]
    assert data["champion"]["model_version"] == "churn-xgboost-none-raw-v1"
    assert data["champion"]["oof_profit_per_1000"] == best["oof_profit_per_1000"]


def test_total_train_time_is_sum_not_wall_clock():
    data = _load()
    total = round(sum(c["train_time_s"] for c in data["configs"]), 2)
    assert data["total_train_time_s"] == total
    assert data["total_train_time_s"] == 19829.52


def test_calibration_values_normalise():
    data = _load()
    values = {_calibration(c) for c in data["configs"]}
    assert values == {"none", "platt", "isotonic"}
    raw = [c for c in data["configs"] if _calibration(c) == "none"]
    assert len(raw) == 9
    # The merged file mixes null and "none" for uncalibrated configs:
    # the first 10 (VM run) carry null, the rest carry the string.
    # Both mean the same thing; the mix is recorded in the docs.
    assert sum(1 for c in data["configs"] if c["calibration"] is None) == 4
    assert sum(1 for c in data["configs"] if c["calibration"] == "none") == 5


def test_docs_sweep_table_matches_json():
    data = _load()
    section = _section()
    champion = data["champion"]
    assert champion["model_version"] in section
    assert f"{champion['test_metrics']['pr_auc']:.4f}" in section
    assert f"{data['total_train_time_s']:,.2f}" in section
    for model in ("logistic", "random_forest", "xgboost"):
        mean_pr = mean(c["test_metrics"]["pr_auc"]
                       for c in data["configs"] if c["model"] == model)
        assert f"{mean_pr:.4f}" in section, model
    for handling in ("none", "class_weight", "smote"):
        mean_pr = mean(c["test_metrics"]["pr_auc"]
                       for c in data["configs"] if c["handling"] == handling)
        assert f"{mean_pr:.4f}" in section, handling
    for calibration in ("none", "platt", "isotonic"):
        mean_brier = mean(c["test_metrics"]["brier"]
                          for c in data["configs"]
                          if _calibration(c) == calibration)
        assert f"{mean_brier:.4f}" in section, calibration


def test_disk_backed_configs_recorded():
    section = _section()
    assert "disk-backed" in section
    assert "6,844" in section
    assert "6,855" in section
    assert "0.6977" in section
    assert "0.6998" in section

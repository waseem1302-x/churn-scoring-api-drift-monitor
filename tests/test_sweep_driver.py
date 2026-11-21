"""Driver merge tests.

The driver crashed on Colab with KeyError: 'model' while merging
the VM-written class_weight-raw per-config file. That file uses an
older schema with no model/handling/calibration keys. The merge
must take identity from the driver's own loop, not from the file.
"""

from scripts.sweep_driver import _merge_entry

STALE_RESULT = {
    "model_version": "churn-random_forest-class_weight-raw-v1",
    "train_time_s": 617.08,
    "oof_threshold": 0.06,
    "oof_profit_per_1000": 82047.42,
    "test_metrics": {"pr_auc": 0.7, "roc_auc": 0.9},
    "test_profit": 1000.0,
    "test_profit_per_1000": 10.0,
}


def test_merge_stale_schema_uses_loop_identity():
    entry = _merge_entry(
        "random_forest", "class_weight", None,
        "churn-random_forest-class_weight-raw-v1", STALE_RESULT)
    assert entry["model"] == "random_forest"
    assert entry["handling"] == "class_weight"
    assert entry["calibration"] == "none"
    assert entry["model_version"] == (
        "churn-random_forest-class_weight-raw-v1")
    assert entry["train_time_s"] == 617.08
    assert entry["oof_threshold"] == 0.06
    assert entry["test_metrics"] == {"pr_auc": 0.7, "roc_auc": 0.9}


def test_merge_current_schema_matches():
    current = dict(
        STALE_RESULT, model="random_forest",
        handling="class_weight", calibration="none")
    entry = _merge_entry(
        "random_forest", "class_weight", None,
        "churn-random_forest-class_weight-raw-v1", current)
    assert entry["model"] == "random_forest"
    assert entry["handling"] == "class_weight"
    assert entry["calibration"] == "none"
    assert entry["model_version"] == (
        "churn-random_forest-class_weight-raw-v1")


def test_merge_calibrated_keeps_string():
    entry = _merge_entry(
        "random_forest", "none", "platt",
        "churn-random_forest-none-platt-v1", STALE_RESULT)
    assert entry["calibration"] == "platt"
    assert entry["model_version"] == "churn-random_forest-none-platt-v1"

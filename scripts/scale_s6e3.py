"""S6E3 scale experiment: same pipeline, 594k-row synthetic set.

A measurement experiment, NOT a replacement of the shipped model.
Trains the champion config (XGBoost, SMOTE, isotonic) and the best
logistic config (logistic, no handling, Platt) on a stratified 80/20
split of the S6E3 synthetic train set, with the same hyperparameters
as Phase 1, and measures train time, held-out metrics, and latency.

The S6E3 rows are near-duplicates of Telco (about 85 copies of each
original row), so these metrics are NOT independent evidence about
new customers. They measure scale (train time, latency) and pipeline
stability on a larger file, not generalisation. The shipped artifact
(artifacts/model.joblib) and the Phase 1 champion numbers are never
touched: this script writes only benchmarks/results/scale_s6e3.json.
"""

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from sklearn.model_selection import cross_val_predict

from app import api as api_mod, config, data, features, metrics, profit
from scripts.measure_latency import _row_to_payload

RESULTS = config.ROOT / "benchmarks" / "results"
FROZEN_THRESHOLD = 0.0934
N_LATENCY_ROWS = 200

# The two configs measured. The full 27-config sweep is not run here:
# at this scale it would take hours for numbers that add nothing to
# the scale question, which is train time and latency of the pipeline.
CONFIGS = [
    ("champion", "xgboost", "smote", "isotonic"),
    ("logistic", "logistic", "none", "platt"),
]


def _run_config(label, name, handling, calibration, X_train_f, y_train,
                X_test_f, y_test, scale_pos_weight):
    """Fit one config, time it, score test at the frozen Telco threshold
    and at the threshold re-optimised on out-of-fold train predictions."""
    version = features.model_version(name, handling, calibration)
    model = features.build_pipeline(
        name, handling=handling, calibration=calibration,
        scale_pos_weight=scale_pos_weight)
    started = time.perf_counter()
    oof_proba = cross_val_predict(model, X_train_f, y_train, cv=3,
                                  method="predict_proba")[:, 1]
    reopt = profit.best_threshold(y_train, oof_proba,
                                  config.OFFER_COST, config.SAVED_VALUE)
    model.fit(X_train_f, y_train)
    train_time = time.perf_counter() - started
    test_proba = model.predict_proba(X_test_f)[:, 1]
    frozen_metrics = metrics.evaluate_model(y_test, test_proba,
                                            threshold=FROZEN_THRESHOLD)
    frozen_profit = profit.profit_at_threshold(
        y_test, test_proba, FROZEN_THRESHOLD,
        config.OFFER_COST, config.SAVED_VALUE)
    reopt_metrics = metrics.evaluate_model(y_test, test_proba,
                                           threshold=reopt["threshold"])
    reopt_profit = profit.profit_at_threshold(
        y_test, test_proba, reopt["threshold"],
        config.OFFER_COST, config.SAVED_VALUE)
    n_test = len(y_test)
    print(f"{label} ({version}): train {train_time:.1f}s, "
          f"ROC-AUC {frozen_metrics['roc_auc']:.4f}, "
          f"PR-AUC {frozen_metrics['pr_auc']:.4f}, "
          f"reopt threshold {reopt['threshold']}", flush=True)
    return model, {
        "model": name,
        "handling": handling,
        "calibration": calibration,
        "model_version": version,
        "train_time_s": round(train_time, 2),
        "oof_threshold": reopt["threshold"],
        "oof_profit_per_1000": reopt["profit_per_1000"],
        "metrics_frozen_threshold": frozen_metrics,
        "test_profit_frozen": round(float(frozen_profit), 2),
        "test_profit_frozen_per_1000": round(float(frozen_profit) / n_test * 1000, 2),
        "metrics_reoptimized_threshold": reopt_metrics,
        "test_profit_reoptimized": round(float(reopt_profit), 2),
        "test_profit_reoptimized_per_1000": round(float(reopt_profit) / n_test * 1000, 2),
    }


def _measure_latency(model, version, X_test):
    """Score 200 held-out S6E3 rows through the API code path with the
    S6E3-trained model loaded in-process. Same method as
    scripts/measure_latency.py. The shipped artifact is not touched."""
    db_path = str(Path(tempfile.mkdtemp()) / "s6e3-latency.db")
    api_mod.set_model(model, version, db_path)
    rows = [_row_to_payload(X_test.iloc[i]) for i in range(N_LATENCY_ROWS)]
    with TestClient(api_mod.app) as client:
        for row in rows[:10]:
            assert client.post("/score", json=row).status_code == 200
        latencies = []
        for row in rows:
            t0 = time.perf_counter()
            resp = client.post("/score", json=row)
            assert resp.status_code == 200
            latencies.append((time.perf_counter() - t0) * 1000)
    lat = np.array(latencies)
    return {
        "model_version": version,
        "n": N_LATENCY_ROWS,
        "mean_ms": round(float(lat.mean()), 2),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "note": "In-process TestClient on this VM (1.5 shared CPUs), "
                "S6E3-trained champion, same method as measure_latency.py.",
    }


def main():
    df = data.load_s6e3()
    report = data.verify_s6e3(df)
    print(f"S6E3 verified: {report['rows']} rows, "
          f"churn rate {report['churn_rate']:.4f}", flush=True)
    X_train, X_test, y_train, y_test = data.split(df)
    X_train_f = X_train.drop(columns=[config.ID_COLUMN])
    X_test_f = X_test.drop(columns=[config.ID_COLUMN])
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    configs = {}
    champion_model = None
    for label, name, handling, calibration in CONFIGS:
        model, result = _run_config(
            label, name, handling, calibration,
            X_train_f, y_train, X_test_f, y_test, scale_pos_weight)
        configs[label] = result
        if label == "champion":
            champion_model = model

    latency = _measure_latency(
        champion_model,
        configs["champion"]["model_version"] + "-s6e3", X_test)
    print(json.dumps(latency, indent=2), flush=True)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Kaggle Playground Series S6E3 synthetic train set",
            "source_file": str(data.S6E3_PATH),
            "rows": report["rows"],
            "columns": report["columns"],
            "churn_rate": round(report["churn_rate"], 4),
            "blank_total_charges": report["blank_total_charges"],
            "duplicate_ids": report["duplicate_ids"],
            "train_rows": len(X_train_f),
            "test_rows": len(X_test_f),
            "do_nothing_accuracy": round(1 - report["churn_rate"], 4),
        },
        "frozen_threshold": FROZEN_THRESHOLD,
        "configs": configs,
        "latency": latency,
        "honesty_limits": (
            "S6E3 rows are near-duplicates of Telco (about 85 copies of "
            "each original row), so these metrics are NOT independent "
            "evidence about new customers. They measure scale (train time, "
            "latency) and pipeline stability on a larger file, not "
            "generalisation. Churn rate is 22.52%, not Telco's 26.54%."),
        "full_sweep": "not run: at this scale the 27-config sweep would "
                      "take hours for numbers that add nothing to the "
                      "scale question.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "scale_s6e3.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {RESULTS / 'scale_s6e3.json'}", flush=True)


if __name__ == "__main__":
    main()

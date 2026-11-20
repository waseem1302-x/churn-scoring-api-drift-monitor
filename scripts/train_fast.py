"""Train the fast alternative: the best logistic config as an opt-in.

Refits churn-logistic-none-platt-v1 on the Telco train split, the
same way scripts/train.py did (threshold from out-of-fold train
predictions), and saves it as artifacts/model-fast.joblib. The
Phase 1 artifact is never touched: the fast model is a separate
file, and the API serves it only when CHURN_FAST_MODEL=1.

The script refuses to save a model that does not reproduce the
Phase 1 numbers for this config (ROC-AUC 0.8420, PR-AUC 0.6348,
threshold 0.1272): a refit that drifts from the recorded config
is a bug, not a new model.
"""

import json
from datetime import datetime, timezone

import joblib
from sklearn.model_selection import cross_val_predict

from app import config, data, features, metrics, profit

RESULTS = config.ROOT / "benchmarks" / "results"
ARTIFACTS = config.ROOT / "artifacts"

MODEL = "logistic"
HANDLING = "none"
CALIBRATION = "platt"
TOLERANCE = 0.002


def main():
    df = data.load_telco()
    report = data.verify_dataset(df)
    X_train, X_test, y_train, y_test = data.split(df)
    X_train_f = X_train.drop(columns=[config.ID_COLUMN])
    X_test_f = X_test.drop(columns=[config.ID_COLUMN])
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = features.build_pipeline(
        MODEL, handling=HANDLING, calibration=CALIBRATION,
        scale_pos_weight=scale_pos_weight)
    oof_proba = cross_val_predict(model, X_train_f, y_train, cv=3,
                                  method="predict_proba")[:, 1]
    oof_best = profit.best_threshold(y_train, oof_proba,
                                     config.OFFER_COST, config.SAVED_VALUE)
    model.fit(X_train_f, y_train)
    test_proba = model.predict_proba(X_test_f)[:, 1]
    measured = metrics.evaluate_model(y_test, test_proba,
                                      threshold=oof_best["threshold"])
    recorded = next(
        c for c in json.loads((RESULTS / "all_configs.json").read_text())
        if c["model_version"] == "churn-logistic-none-platt-v1")
    for key in ("roc_auc", "pr_auc", "brier", "precision_at_100"):
        if abs(measured[key] - recorded["test_metrics"][key]) > TOLERANCE:
            raise SystemExit(
                f"refit does not reproduce Phase 1 {key}: "
                f"{measured[key]} vs recorded "
                f"{recorded['test_metrics'][key]}")
    if abs(oof_best["threshold"]
           - recorded["oof_threshold"]) > 0.0001:
        raise SystemExit(
            f"refit threshold {oof_best['threshold']} does not match "
            f"recorded {recorded['oof_threshold']}")
    metadata = {
        "model_name": MODEL,
        "handling": HANDLING,
        "calibration": CALIBRATION,
        "model_version": recorded["model_version"],
        "threshold": oof_best["threshold"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(X_train_f),
        "test_rows": len(X_test_f),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata},
                config.FAST_ARTIFACT_PATH)
    print(f"Saved {config.FAST_ARTIFACT_PATH}")
    print(f"Test metrics: {measured}")


if __name__ == "__main__":
    main()

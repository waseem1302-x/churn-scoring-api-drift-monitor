"""Train all Phase 1 configurations and pick the champion.

27 configs: 3 models (logistic, random_forest, xgboost) x 3 class
handlings (none, class_weight, smote) x 3 calibrations (raw, platt,
isotonic). Every config is fitted on the training split only and
scored on the held-out test split.

The operating threshold is chosen on out-of-fold train predictions
(3-fold cross_val_predict), never on the test set, then frozen and
applied to test. The champion is the config with the highest
out-of-fold expected maximum profit, because profit is the decision
metric; ranking metrics are reported for every config alongside.

Outputs (all under benchmarks/results/ unless noted):
- all_configs.json: every config with test metrics, OOF threshold,
  test profit at the frozen threshold, and train time.
- metrics.json: champion summary plus the gate thresholds.
- reliability.json: the champion's reliability curve on test.
- profit_curve.json: the champion's profit curve on test.
- contract_rows.json: 25 test rows with recorded probabilities, for
  tests/test_contract.py.
- artifacts/model.joblib: the champion pipeline plus metadata.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import cross_val_predict

from app import config, data, features, metrics, profit, scoring

ROOT = config.ROOT
RESULTS = ROOT / "benchmarks" / "results"
ARTIFACTS = ROOT / "artifacts"


def _oof_threshold(model, X_train, y_train):
    """Profit-maximising threshold from out-of-fold train predictions."""
    oof_proba = cross_val_predict(model, X_train, y_train, cv=3,
                                  method="predict_proba")[:, 1]
    best = profit.best_threshold(y_train, oof_proba,
                                 config.OFFER_COST, config.SAVED_VALUE)
    return best, oof_proba


def main():
    df = data.load_telco()
    report = data.verify_dataset(df)
    X_train, X_test, y_train, y_test = data.split(df)
    X_train_f = X_train.drop(columns=[config.ID_COLUMN])
    X_test_f = X_test.drop(columns=[config.ID_COLUMN])
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    results = []
    for model_name in config.MODELS:
        for handling in config.HANDLINGS:
            for calibration in config.CALIBRATIONS:
                version = features.model_version(model_name, handling, calibration)
                model = features.build_pipeline(
                    model_name, handling=handling, calibration=calibration,
                    scale_pos_weight=scale_pos_weight)
                started = time.perf_counter()
                oof_best, _ = _oof_threshold(model, X_train_f, y_train)
                model.fit(X_train_f, y_train)
                train_time = time.perf_counter() - started
                test_proba = model.predict_proba(X_test_f)[:, 1]
                frozen = oof_best["threshold"]
                test_metrics = metrics.evaluate_model(y_test, test_proba, threshold=frozen)
                test_profit = profit.profit_at_threshold(
                    y_test, test_proba, frozen,
                    config.OFFER_COST, config.SAVED_VALUE)
                results.append({
                    "model": model_name,
                    "handling": handling,
                    "calibration": calibration,
                    "model_version": version,
                    "train_time_s": round(train_time, 2),
                    "oof_threshold": frozen,
                    "oof_profit_per_1000": oof_best["profit_per_1000"],
                    "test_metrics": test_metrics,
                    "test_profit": round(float(test_profit), 2),
                    "test_profit_per_1000": round(float(test_profit) / len(y_test) * 1000, 2),
                    "reliability_curve": metrics.reliability_curve(y_test, test_proba),
                })
                print(f"{version}: PR-AUC {test_metrics['pr_auc']:.4f} "
                      f"ROC-AUC {test_metrics['roc_auc']:.4f} "
                      f"Brier {test_metrics['brier']:.4f} "
                      f"P@100 {test_metrics['precision_at_100']:.4f} "
                      f"EMP/test ${test_profit:.0f} "
                      f"({train_time:.1f}s)", flush=True)

    champion = max(results, key=lambda r: (r["oof_profit_per_1000"],
                                           r["test_metrics"]["pr_auc"]))
    print(f"\nChampion (highest out-of-fold profit): {champion['model_version']}")

    # Refit the champion fresh so the artifact is exactly the model
    # whose numbers are recorded (same seed, same data, same result).
    champ_model = features.build_pipeline(
        champion["model"], handling=champion["handling"],
        calibration=champion["calibration"], scale_pos_weight=scale_pos_weight)
    champ_model.fit(X_train_f, y_train)
    metadata = {
        "model_name": champion["model"],
        "handling": champion["handling"],
        "calibration": champion["calibration"],
        "model_version": champion["model_version"],
        "threshold": champion["oof_threshold"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(X_train_f),
        "test_rows": len(X_test_f),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": champ_model, "metadata": metadata}, config.ARTIFACT_PATH)

    # Contract rows: 25 test rows, recorded probabilities from the
    # shipped artifact, for tests/test_contract.py.
    test_proba = champ_model.predict_proba(X_test_f)[:, 1]
    contract = []
    for i in range(25):
        row = X_test.iloc[i]
        inputs = {}
        for f in config.FEATURES:
            v = row[f]
            if hasattr(v, "item"):
                v = v.item()
            # JSON has no NaN: a blank TotalCharges becomes null, and
            # the API treats null as "no bill yet", same as training.
            if isinstance(v, float) and v != v:
                v = None
            inputs[f] = v
        proba = float(test_proba[i])
        contract.append({
            "id": str(row[config.ID_COLUMN]),
            "inputs": inputs,
            "probability": proba,
            "tier": scoring.risk_tier(proba),
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "all_configs.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "reliability.json").write_text(
        json.dumps(champion["reliability_curve"], indent=2))
    test_proba_all = champ_model.predict_proba(X_test_f)[:, 1]
    (RESULTS / "profit_curve.json").write_text(json.dumps(
        profit.profit_curve(y_test, test_proba_all,
                            config.OFFER_COST, config.SAVED_VALUE), indent=2))
    (RESULTS / "contract_rows.json").write_text(json.dumps(contract, indent=2))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rows": report["rows"],
            "churn_rate": report["churn_rate"],
            "train_rows": len(X_train_f),
            "test_rows": len(X_test_f),
            "do_nothing_accuracy": round(1 - report["churn_rate"], 4),
        },
        "champion": {
            "model": champion["model"],
            "handling": champion["handling"],
            "calibration": champion["calibration"],
            "model_version": champion["model_version"],
            "threshold_frozen": champion["oof_threshold"],
            "metrics": champion["test_metrics"],
            "test_profit": champion["test_profit"],
            "test_profit_per_1000": champion["test_profit_per_1000"],
            "oof_profit_per_1000": champion["oof_profit_per_1000"],
            "train_time_s": champion["train_time_s"],
        },
        "best_pr_auc": max(results, key=lambda r: r["test_metrics"]["pr_auc"])["model_version"],
        "gate_thresholds": config.GATE_THRESHOLDS,
        "s6e3_scale_run": "pending",
        "s6e3": {"train_time_s": None, "note": "Kaggle credentials not present; scale run pending"},
    }
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS / 'metrics.json'}")
    print(f"Champion: {champion['model_version']}")
    print(f"Test metrics: {champion['test_metrics']}")


if __name__ == "__main__":
    main()

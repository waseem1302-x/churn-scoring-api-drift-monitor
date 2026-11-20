"""Bank generalisation check: the same pipeline on the Bank
Customer Churn dataset.

Same loop as scripts/train.py (3 models x 3 handlings x 3
calibrations, threshold from out-of-fold train predictions,
champion by the pre-registered rule: highest out-of-fold profit),
but retrained on the 10,000-row Bank set. The result is reported
separately in benchmarks/results/bank_generalisation.json, never
pooled with Telco: a different business (retail bank vs telecom)
and a different schema (10 features vs 19).

Two extra comparisons, both honest about what they are:
- the Telco champion configuration (XGBoost, SMOTE, isotonic)
  retrained on Bank: does the same config win on a new schema?
- that config scored at the Telco frozen threshold (0.0934):
  a Telco threshold does not transfer, and the gap shows it.

Profit uses the same illustrative $50 offer / $500 saved
assumptions as Telco, so the numbers compare; a bank's real
retention economics are not a telecom's, and the docs say so.
"""

import json
import time
from datetime import datetime, timezone

from sklearn.model_selection import cross_val_predict

from app import config, data, features, metrics, profit

RESULTS = config.ROOT / "benchmarks" / "results"
OUT_PATH = RESULTS / "bank_generalisation.json"
TELCO_FROZEN_THRESHOLD = 0.0934


def main():
    df = data.load_bank()
    report = data.verify_bank(df)
    print(f"Bank verified: {report['rows']} rows, "
          f"churn rate {report['churn_rate']:.4f}", flush=True)
    X_train, X_test, y_train, y_test = data.split_bank(df)
    X_train_f = X_train.drop(columns=[data.BANK_ID_COLUMN])
    X_test_f = X_test.drop(columns=[data.BANK_ID_COLUMN])
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    results = []
    for model_name in config.MODELS:
        for handling in config.HANDLINGS:
            for calibration in config.CALIBRATIONS:
                version = features.model_version(model_name, handling, calibration)
                model = features.build_pipeline(
                    model_name, handling=handling, calibration=calibration,
                    scale_pos_weight=scale_pos_weight,
                    numeric=data.BANK_NUMERIC_FEATURES,
                    categorical=data.BANK_CATEGORICAL_FEATURES)
                started = time.perf_counter()
                oof_proba = cross_val_predict(
                    model, X_train_f, y_train, cv=3,
                    method="predict_proba")[:, 1]
                oof_best = profit.best_threshold(
                    y_train, oof_proba, config.OFFER_COST, config.SAVED_VALUE)
                model.fit(X_train_f, y_train)
                train_time = time.perf_counter() - started
                test_proba = model.predict_proba(X_test_f)[:, 1]
                frozen = oof_best["threshold"]
                test_metrics = metrics.evaluate_model(
                    y_test, test_proba, threshold=frozen)
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
                })
                print(f"{version}: PR-AUC {test_metrics['pr_auc']:.4f} "
                      f"ROC-AUC {test_metrics['roc_auc']:.4f} "
                      f"Brier {test_metrics['brier']:.4f} "
                      f"P@100 {test_metrics['precision_at_100']:.4f} "
                      f"EMP/test ${test_profit:.0f} "
                      f"({train_time:.1f}s)", flush=True)

    champion = max(results, key=lambda r: (r["oof_profit_per_1000"],
                                           r["test_metrics"]["pr_auc"]))
    telco_config = next(
        r for r in results
        if r["model_version"] == "churn-xgboost-smote-isotonic-v1")
    # The Telco champion config, retrained on Bank, scored at the
    # Telco frozen threshold: the transfer test.
    telco_config_model = features.build_pipeline(
        "xgboost", handling="smote", calibration="isotonic",
        scale_pos_weight=scale_pos_weight,
        numeric=data.BANK_NUMERIC_FEATURES,
        categorical=data.BANK_CATEGORICAL_FEATURES)
    telco_config_model.fit(X_train_f, y_train)
    telco_proba = telco_config_model.predict_proba(X_test_f)[:, 1]
    telco_threshold_metrics = metrics.evaluate_model(
        y_test, telco_proba, threshold=TELCO_FROZEN_THRESHOLD)
    telco_threshold_profit = profit.profit_at_threshold(
        y_test, telco_proba, TELCO_FROZEN_THRESHOLD,
        config.OFFER_COST, config.SAVED_VALUE)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Bank Customer Churn (Kaggle Churn_Modelling.csv)",
            "source_file": str(data.BANK_PATH),
            "rows": report["rows"],
            "columns": report["columns"],
            "churn_rate": round(report["churn_rate"], 4),
            "train_rows": len(X_train_f),
            "test_rows": len(X_test_f),
            "do_nothing_accuracy": round(1 - report["churn_rate"], 4),
        },
        "configs": results,
        "champion": {
            "model": champion["model"],
            "handling": champion["handling"],
            "calibration": champion["calibration"],
            "model_version": champion["model_version"],
            "threshold": champion["oof_threshold"],
            "test_metrics": champion["test_metrics"],
            "test_profit": champion["test_profit"],
            "test_profit_per_1000": champion["test_profit_per_1000"],
            "oof_profit_per_1000": champion["oof_profit_per_1000"],
            "train_time_s": champion["train_time_s"],
        },
        "telco_champion_config_on_bank": {
            "model_version": telco_config["model_version"],
            "oof_threshold": telco_config["oof_threshold"],
            "test_metrics": telco_config["test_metrics"],
            "test_profit": telco_config["test_profit"],
            "test_profit_per_1000": telco_config["test_profit_per_1000"],
            "at_telco_frozen_threshold": {
                "threshold": TELCO_FROZEN_THRESHOLD,
                "test_metrics": telco_threshold_metrics,
                "test_profit": round(float(telco_threshold_profit), 2),
                "test_profit_per_1000": round(
                    float(telco_threshold_profit) / len(y_test) * 1000, 2),
            },
        },
        "honesty_limits": (
            "Bank is a different business and schema from Telco, so "
            "this is a generalisation check of the pipeline, not of "
            "the Telco model: the Telco artifact cannot score Bank "
            "rows, and no number here is pooled with Telco. Profit "
            "uses the same illustrative $50/$500 assumptions for "
            "comparability; a bank's real retention economics differ."),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nBank champion: {champion['model_version']}", flush=True)
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

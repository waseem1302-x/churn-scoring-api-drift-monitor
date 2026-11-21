"""Diagnostic: run ONE S6E3 sweep config alone, in a fresh process.

The full sweep loop died twice on the random-forest configs, both
times silently mid-config with no traceback. A single RF config run
alone peaks at 4,725 MB RSS, just under the 6 GB cell limit, so the
loop's death is memory: the long-lived process accumulates enough
across configs that the next RF peak tips it over. Running each
config in its own fresh process keeps the peak to one config.

Takes --model/--handling/--calibration (calibration "none" means
uncalibrated, the "-raw" version). Writes its result to
benchmarks/results/scale_s6e3_sweep_<version>.json in this copy.
The driver (scripts/sweep_driver.py) merges those files into the
partial sweep file. This does not touch the partial sweep file.

Configs 17 and 18 (random forest + SMOTE + Platt / isotonic) use
the disk-backed calibration in scripts/disk_backed_calibration.py:
the standard wrapper holds three 300-tree forests in RAM at once,
about 15 GB on the SMOTE-grown 736k rows, and both configs died
twice on Colab with exit -9 against 12 GB. The disk-backed path
fits the same three folds and averages the same three calibrators,
so its predictions are identical; only the storage changes.
"""

import argparse
import json
import resource
import time
from datetime import datetime, timezone

from sklearn.model_selection import cross_val_predict

from app import config, data, features, metrics, profit
from scripts.disk_backed_calibration import DiskBackedCalibratedClassifier


def build_model_for_config(model, handling, calibration, scale_pos_weight):
    """Build one sweep config. Random forest + SMOTE + calibration
    (configs 17 and 18) cannot run the standard CalibratedClassifierCV
    on Colab: three 300-tree forests on the SMOTE-grown 736k rows peak
    near 15 GB against 12 GB, and config 17 died twice with exit -9.
    Those two use the disk-backed wrapper, which fits the same three
    folds and averages the same three calibrators but keeps only one
    forest in RAM at a time, so the predictions are identical. Every
    other config uses the standard path."""
    if (model == "random_forest" and handling == "smote"
            and calibration in ("platt", "isotonic")):
        pipe = features.build_pipeline(
            model, handling=handling, calibration=None,
            scale_pos_weight=scale_pos_weight)
        method = "sigmoid" if calibration == "platt" else "isotonic"
        return DiskBackedCalibratedClassifier(pipe, method=method, cv=3)
    return features.build_pipeline(
        model, handling=handling, calibration=calibration,
        scale_pos_weight=scale_pos_weight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--handling", required=True)
    parser.add_argument("--calibration", required=True,
                        help="'none', 'platt' or 'isotonic'")
    args = parser.parse_args()
    calibration = None if args.calibration == "none" else args.calibration
    df = data.load_s6e3()
    report = data.verify_s6e3(df)
    print(f"S6E3 verified: {report['rows']} rows", flush=True)
    X_train, X_test, y_train, y_test = data.split(df)
    X_train_f = X_train.drop(columns=[config.ID_COLUMN])
    X_test_f = X_test.drop(columns=[config.ID_COLUMN])
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    version = features.model_version(args.model, args.handling, calibration)
    model = build_model_for_config(
        args.model, args.handling, calibration, scale_pos_weight)
    try:
        started = time.perf_counter()
        oof_proba = cross_val_predict(
            model, X_train_f, y_train, cv=3, method="predict_proba")[:, 1]
        oof_best = profit.best_threshold(
            y_train, oof_proba, config.OFFER_COST, config.SAVED_VALUE)
        model.fit(X_train_f, y_train)
        train_time = time.perf_counter() - started
        test_proba = model.predict_proba(X_test_f)[:, 1]
        test_metrics = metrics.evaluate_model(
            y_test, test_proba, threshold=oof_best["threshold"])
        test_profit = profit.profit_at_threshold(
            y_test, test_proba, oof_best["threshold"],
            config.OFFER_COST, config.SAVED_VALUE)
    finally:
        if hasattr(model, "close"):
            model.close()
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result = {
        "model": args.model,
        "handling": args.handling,
        "calibration": args.calibration,
        "model_version": version,
        "train_time_s": round(train_time, 2),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "oof_threshold": oof_best["threshold"],
        "oof_profit_per_1000": oof_best["profit_per_1000"],
        "test_metrics": test_metrics,
        "test_profit": round(float(test_profit), 2),
        "test_profit_per_1000": round(
            float(test_profit) / len(y_test) * 1000, 2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out = (config.ROOT / "benchmarks" / "results"
           / f"scale_s6e3_sweep_{version}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"{version}: PR-AUC {test_metrics['pr_auc']:.4f} "
          f"ROC-AUC {test_metrics['roc_auc']:.4f} "
          f"train {train_time:.1f}s peak RSS {peak_rss_mb:.0f} MB",
          flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()

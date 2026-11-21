"""Full 27-config sweep on the S6E3 synthetic train set.

Same loop as scripts/train.py (3 models x 3 handlings x 3
calibrations, threshold from out-of-fold train predictions, champion
by the pre-registered rule: highest out-of-fold profit), but on the
594,194-row S6E3 set instead of Telco, and writing only
benchmarks/results/scale_s6e3_sweep.json. It never touches the shipped
artifact, metrics.json, or any Phase 1 result file.

This run takes a couple of hours at this scale, so it is launched
from an isolated copy of the repo and writes its result file after
every config: if the run is interrupted, the completed configs are
still recorded.

The S6E3 rows are near-duplicates of Telco (about 85 copies of each
original row), so these metrics are NOT independent evidence about
new customers. They measure scale (train time at 594k rows) and
whether the Phase 1 ranking of configs holds on a larger file.
"""

import json
import time
from datetime import datetime, timezone

from sklearn.model_selection import cross_val_predict

from app import config, data, features, metrics, profit

ROOT = config.ROOT
RESULTS = ROOT / "benchmarks" / "results"
OUT_PATH = RESULTS / "scale_s6e3_sweep.json"


def _write_partial(payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))


def main():
    started_all = time.perf_counter()
    df = data.load_s6e3()
    report = data.verify_s6e3(df)
    print(f"S6E3 verified: {report['rows']} rows, "
          f"churn rate {report['churn_rate']:.4f}", flush=True)
    X_train, X_test, y_train, y_test = data.split(df)
    X_train_f = X_train.drop(columns=[config.ID_COLUMN])
    X_test_f = X_test.drop(columns=[config.ID_COLUMN])
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Kaggle Playground Series S6E3 synthetic train set",
            "source_file": str(data.S6E3_PATH),
            "rows": report["rows"],
            "columns": report["columns"],
            "churn_rate": round(report["churn_rate"], 4),
            "train_rows": len(X_train_f),
            "test_rows": len(X_test_f),
            "do_nothing_accuracy": round(1 - report["churn_rate"], 4),
        },
        "configs": [],
        "champion": None,
        "total_train_time_s": None,
        "complete": False,
        "resumed_from_partial": False,
        "honesty_limits": (
            "S6E3 rows are near-duplicates of Telco (about 85 copies of "
            "each original row), so these metrics are NOT independent "
            "evidence about new customers. They measure scale (train time "
            "at 594k rows) and whether the Phase 1 ranking of configs "
            "holds on a larger file, not generalisation. Churn rate is "
            "22.52%, not Telco's 26.54%."),
    }

    # Resume: if a previous run was interrupted, its partial file holds
    # the completed configs. Skip those instead of redoing them; the
    # run only counts as complete when all 27 are present. The first
    # launch of this sweep died after 10 configs (no traceback, the
    # process was gone; the image build was running at the same time),
    # so the resumed run finishes the remaining 17.
    results = []
    if OUT_PATH.exists():
        try:
            previous = json.loads(OUT_PATH.read_text())
            results = previous.get("configs", [])
            if results:
                payload["resumed_from_partial"] = True
                print(f"Resuming: {len(results)} configs already done, "
                      f"skipping them", flush=True)
        except (json.JSONDecodeError, KeyError):
            results = []
    done = {r["model_version"] for r in results}
    payload["configs"] = results
    _write_partial(payload)

    for model_name in config.MODELS:
        for handling in config.HANDLINGS:
            for calibration in config.CALIBRATIONS:
                version = features.model_version(model_name, handling, calibration)
                if version in done:
                    continue
                model = features.build_pipeline(
                    model_name, handling=handling, calibration=calibration,
                    scale_pos_weight=scale_pos_weight)
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
                })
                payload["configs"] = results
                _write_partial(payload)
                print(f"{version}: PR-AUC {test_metrics['pr_auc']:.4f} "
                      f"ROC-AUC {test_metrics['roc_auc']:.4f} "
                      f"Brier {test_metrics['brier']:.4f} "
                      f"P@100 {test_metrics['precision_at_100']:.4f} "
                      f"EMP/test ${test_profit:.0f} "
                      f"({train_time:.1f}s)", flush=True)

    champion = max(results, key=lambda r: (r["oof_profit_per_1000"],
                                           r["test_metrics"]["pr_auc"]))
    payload["champion"] = {
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
    }
    payload["total_train_time_s"] = round(time.perf_counter() - started_all, 2)
    payload["complete"] = True
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    _write_partial(payload)
    print(f"\nChampion: {champion['model_version']}", flush=True)
    print(f"Total: {payload['total_train_time_s']}s for 27 configs", flush=True)
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

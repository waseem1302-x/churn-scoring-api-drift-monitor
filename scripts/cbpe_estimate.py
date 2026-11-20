"""Estimate performance without labels with NannyML CBPE.

Runs in the CBPE venv (.venv-cbpe), NOT the repo venv: NannyML
0.6.0 cannot install on Python 3.12 (it needs Python <3.11), and
NannyML 0.13.1 would downgrade the repo's pinned xgboost 3.4.1 to
2.1.4, which would break the bit-for-bit Phase 1 reproduction.
So NannyML 0.13.1 lives in its own venv, and this script reads
only the prepared frame CSV from scripts/cbpe_prepare.py: CBPE
works on predictions, not on model objects, so no artifact ever
enters the CBPE venv.

    .venv-cbpe/bin/python scripts/cbpe_estimate.py

Reference weeks 1 to 4 (clean, labelled) fit the estimator;
analysis weeks 5 to 12 are estimated chunk by chunk, one chunk
per week (5,000 rows). The realised values come from
cbpe_manifest.json, which the prepare step computed from the
labels: the comparison is the check on the estimator, and the
labels are used only here, after the fact, exactly as they would
be when they arrive about 30 days late in production.

Output: benchmarks/results/cbpe_estimate.json, with the per-week
estimated vs realised ROC-AUC, precision and recall, the mean
absolute error per metric, and the first week the estimated
ROC-AUC falls below the lowest reference-week estimate. That
week is the answer to whether CBPE catches the concept drift
(weeks 6 to 9) earlier than the retraining loop's realised
trigger in week 8.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import nannyml as nml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
CHUNK_SIZE = 5000


def _metric_values(result_df, metric, period):
    """Pull (value, sampling_error) per chunk out of a CBPE result
    frame, for one period only.

    NannyML returns the reference chunks AND the analysis chunks in
    every result, so the rows must be filtered by period: reading
    the first rows without filtering mixes reference estimates into
    the analysis weeks. Column names differ slightly across NannyML
    versions, so match on the metric name and the 'value' /
    'sampling_error' suffixes instead of assuming one layout."""
    cols = result_df.columns
    flat = ["_".join(map(str, c)) if isinstance(c, tuple) else str(c)
            for c in cols]
    result_df = result_df.copy()
    result_df.columns = flat
    period_col = next(c for c in flat if c.endswith("period"))
    result_df = result_df[result_df[period_col] == period]
    value_col = next(c for c in flat
                     if c.startswith(metric) and c.endswith("value"))
    err_col = next((c for c in flat
                    if c.startswith(metric) and "sampling_error" in c),
                   None)
    alert_col = next((c for c in flat if c.endswith("alert")), None)
    values = result_df[value_col].tolist()
    errors = result_df[err_col].tolist() if err_col else [None] * len(values)
    alerts = (result_df[alert_col].tolist() if alert_col
              else [None] * len(values))
    return values, errors, alerts


def main():
    manifest = json.loads((RESULTS / "cbpe_manifest.json").read_text())
    frame = pd.read_csv(RESULTS / "cbpe_frames.csv")
    ref_weeks = manifest["reference_weeks"]
    analysis_weeks = manifest["analysis_weeks"]
    reference = frame[frame["week"].between(ref_weeks[0], ref_weeks[1])]
    analysis = frame[frame["week"].isin(analysis_weeks)]

    estimator = nml.CBPE(
        problem_type="classification_binary",
        y_pred_proba="y_pred_proba",
        y_pred="y_pred",
        y_true="y_true",
        metrics=["roc_auc", "precision", "recall"],
        chunk_size=CHUNK_SIZE,
    )
    estimator.fit(reference)
    ref_results = estimator.estimate(reference)
    results = estimator.estimate(analysis)

    realised = manifest["realised_per_week"]
    weeks = []
    for metric in ("roc_auc", "precision", "recall"):
        ref_values, _, _ = _metric_values(ref_results.data, metric,
                                          "reference")
        est_values, est_errors, est_alerts = _metric_values(
            results.data, metric, "analysis")
        realised_key = {
            "roc_auc": "roc_auc",
            "precision": "precision_at_threshold",
            "recall": "recall_at_threshold",
        }[metric]
        per_week = []
        for i, week in enumerate(analysis_weeks):
            real = realised[str(week)][realised_key]
            est = float(est_values[i])
            per_week.append({
                "week": week,
                "estimated": round(est, 4),
                "realised": real,
                "absolute_error": round(abs(est - real), 4),
                "sampling_error": (round(float(est_errors[i]), 4)
                                   if est_errors[i] is not None else None),
                "nannyml_alert": bool(est_alerts[i])
                                 if est_alerts[i] is not None else None,
            })
        weeks.append({
            "metric": metric,
            "reference_estimated": [round(float(v), 4) for v in ref_values],
            "per_week": per_week,
            "mae": round(sum(p["absolute_error"] for p in per_week)
                         / len(per_week), 4),
        })

    roc = next(w for w in weeks if w["metric"] == "roc_auc")
    ref_low = min(roc["reference_estimated"])
    # The estimate never steps at the concept flip: the largest
    # week-to-week move is the honest signal, not a threshold on a
    # slowly drifting point estimate.
    ests = [p["estimated"] for p in roc["per_week"]]
    max_weekly_move = round(max(abs(ests[i + 1] - ests[i])
                                for i in range(len(ests) - 1)), 4)
    any_alert = any(p["nannyml_alert"] for p in roc["per_week"])
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nannyml_version": nml.__version__,
        "model_version": manifest["model_version"],
        "threshold": manifest["threshold"],
        "reference_weeks": ref_weeks,
        "analysis_weeks": analysis_weeks,
        "chunk_size": CHUNK_SIZE,
        "metrics": weeks,
        "reference_low_estimated_roc_auc": ref_low,
        "max_weekly_move_estimated_roc_auc": max_weekly_move,
        "nannyml_alert_on_any_analysis_week": any_alert,
        "retraining_loop_trigger_week": 8,
        "honesty_limits": (
            "CBPE estimates performance from the model's own "
            "confidence, not from labels. It is checked here "
            "against realised performance on the same simulator "
            "weeks, which is the only way to know if the estimate "
            "is any good. On real production data there is no such "
            "check until the labels arrive about 30 days later."),
    }
    (RESULTS / "cbpe_estimate.json").write_text(
        json.dumps(result, indent=2))
    for w in weeks:
        print(f"{w['metric']}: MAE {w['mae']}, "
              f"estimated {[p['estimated'] for p in w['per_week']]}")
    print(f"Reference low estimated ROC-AUC: {ref_low}")
    print(f"Max week-to-week move in estimated ROC-AUC: "
          f"{max_weekly_move}")
    print(f"NannyML alert on any analysis week: {any_alert}")
    print(f"Wrote {RESULTS / 'cbpe_estimate.json'}")


if __name__ == "__main__":
    main()

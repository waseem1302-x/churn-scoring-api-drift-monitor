"""Prepare the CBPE experiment frames (repo venv).

Scores the 12-week simulator run (5,000 customers a week, the same
run the retraining loop used, seed 42) with the shipped champion and
writes benchmarks/results/cbpe_frames.csv: one row per customer-week
with week, date, y_true, y_pred_proba and y_pred at the frozen
threshold. No NannyML here: this step only needs the champion
(xgboost 3.4.1), which the CBPE venv must not touch.

Reference weeks 1 to 4 are clean and labelled: CBPE fits on them.
Analysis weeks 5 to 12 carry the drift schedule (price rise, Fiber
growth, concept flip weeks 6 to 9, prevalence shift from week 9,
TotalCharges breakage weeks 10 to 11, stale tenure week 12).
In production the labels for those weeks arrive about 30 days late,
so the CBPE estimates are what you would have during that window;
the comparison against realised performance happens only after the
labels arrive. In this experiment the labels exist immediately, so
the estimator can be checked: the schedule and the decisions are
the same, only the calendar is compressed.

Run with the repo venv, then scripts/cbpe_estimate.py with the
CBPE venv (see cbpe_venv.md).
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app import config, scoring
from scripts import simulate

RESULTS = config.ROOT / "benchmarks" / "results"
REFERENCE_WEEKS = (1, 4)
N_CUSTOMERS = 5000
N_WEEKS = 12


def main():
    model, meta = scoring.load_artifact()
    threshold = meta["threshold"]
    run = simulate.simulate(n_customers=N_CUSTOMERS, n_weeks=N_WEEKS,
                            seed=42)
    proba = model.predict_proba(run[config.FEATURES])[:, 1]
    frame = pd.DataFrame({
        "week": run["week"].to_numpy(),
        "date": run["date"].astype(str),
        "y_true": run["Churn"].to_numpy(),
        "y_pred_proba": proba,
        "y_pred": (proba >= threshold).astype(int),
    })
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "cbpe_frames.csv"
    frame.to_csv(out, index=False)

    realised = {}
    for week in range(1, N_WEEKS + 1):
        w = frame[frame["week"] == week]
        realised[str(week)] = {
            "rows": int(len(w)),
            "churn_rate": round(float(w["y_true"].mean()), 4),
            "roc_auc": round(float(roc_auc_score(w["y_true"],
                                                 w["y_pred_proba"])), 4),
            "precision_at_threshold": round(
                float((w["y_true"][w["y_pred"] == 1]).mean())
                if (w["y_pred"] == 1).any() else 0.0, 4),
            "recall_at_threshold": round(
                float((w["y_pred"][w["y_true"] == 1]).mean())
                if (w["y_true"] == 1).any() else 0.0, 4),
        }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": meta["model_version"],
        "threshold": threshold,
        "n_customers_per_week": N_CUSTOMERS,
        "n_weeks": N_WEEKS,
        "reference_weeks": list(REFERENCE_WEEKS),
        "analysis_weeks": list(range(REFERENCE_WEEKS[1] + 1, N_WEEKS + 1)),
        "frame_file": "cbpe_frames.csv",
        "rows": len(frame),
        "realised_per_week": realised,
    }
    (RESULTS / "cbpe_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"Wrote {out} ({len(frame)} rows)")
    print(f"Realised ROC-AUC by week: "
          f"{[realised[str(w)]['roc_auc'] for w in range(1, N_WEEKS + 1)]}")


if __name__ == "__main__":
    main()

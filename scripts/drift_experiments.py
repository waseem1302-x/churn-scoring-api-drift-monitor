"""Controlled drift injection experiments.

Design: the reference is the Phase 1 training split (5,634 rows) scored
by the champion. Clean windows come from the simulator with
schedule=False (drift-free weeks). Each drift type is injected into a
clean window on its own, so the detector table measures one thing at a
time. Detection delay is counted in windows: 0 means the detector
fired in the injected window itself.

What is measured, per drift type and per detector:
- delay: first window where the detector's verdict fired on the target
  feature (or on the right channel for prevalence and quality events),
  or null if it never fired inside the horizon.
- false alarms: how many of the 6 clean windows each detector flagged.

Thresholds are NOT chosen here. The script uses
monitor.DEFAULT_THRESHOLDS and reports what they do. If the table
shows a threshold is wrong (a detector crying wolf on clean windows,
or missing a real drift), the threshold changes in app/monitor.py with
a reason written in MONITORING.md, and the script is re-run. The
measurements decide; the table is the evidence.

Drift types and their target:
- abrupt_price: +$25 MonthlyCharges. Target feature MonthlyCharges.
- gradual_fiber: Fiber share grows 5 points per window. Target
  feature InternetService. Windows carry 5, 10, 15, 20, 25 points.
- prevalence: labels redrawn with the churn intercept +0.6, no feature
  change. Input detectors must NOT fire (that is the honest result);
  the realised-performance channel (precision at 100, PR-AUC) is the
  detector.
- breakage_total_charges: TotalCharges := MonthlyCharges. Target
  feature TotalCharges.
- null_storm: 40% of TotalCharges becomes null. The data-quality
  channel (null_rate) is the detector.
- stale_tenure: tenure frozen at the previous week's values. The
  data-quality channel (stale_feature) is the detector.

Output: benchmarks/results/drift_experiments.json, plus a printed
summary table. The CI gate reads the JSON.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app import config, data, monitor, scoring
from scripts import simulate

RESULTS = config.ROOT / "benchmarks" / "results"
N_CUSTOMERS = 2000
N_CLEAN_WEEKS = 6
GRADUAL_STEPS = [0.05, 0.10, 0.15, 0.20, 0.25]
HORIZON = 5

DETECTOR_KEYS = {
    "psi": "psi_drifted",
    "ks": "ks_drifted",
    "jensen_shannon": "js_drifted",
    "wasserstein": "wasserstein_drifted",
}


def _score(model, frame):
    return model.predict_proba(frame[config.FEATURES])[:, 1]


def _window_summary(ref_frame, ref_scores, frame, model, threshold,
                    ref_metrics, previous_frame=None):
    scores = _score(model, frame)
    y_true = frame["Churn"].to_numpy() if "Churn" in frame.columns else None
    return monitor.evaluate_window(
        ref_frame, frame, ref_scores, scores, threshold,
        y_true=y_true, y_proba=scores,
        previous_frame=previous_frame,
        id_column="customerID" if "customerID" in frame.columns else None,
        reference_metrics=ref_metrics)


def _first_firing_window(summaries, target_feature, detector_key):
    """First window index where the detector fired on the target
    feature, or None inside the horizon."""
    for i, summary in enumerate(summaries):
        result = summary["feature_drift"].get(target_feature)
        if result and result.get(detector_key):
            return i
    return None


def _clean_false_alarms(summaries):
    """Per detector: how many clean windows it flagged on any feature,
    and how many feature-level flags it raised in total."""
    out = {}
    for name, key in DETECTOR_KEYS.items():
        windows_flagged = 0
        feature_flags = 0
        for summary in summaries:
            flagged = [f for f, r in summary["feature_drift"].items()
                       if r.get(key)]
            feature_flags += len(flagged)
            if flagged:
                windows_flagged += 1
        out[name] = {
            "windows_flagged": windows_flagged,
            "windows_total": len(summaries),
            "feature_flags_total": feature_flags,
        }
    return out


def main():
    model, meta = scoring.load_artifact()
    threshold = meta["threshold"]
    df = data.load_telco()
    X_train, X_test, y_train, y_test = data.split(df)
    ref_frame = X_train.drop(columns=[config.ID_COLUMN])
    ref_scores = _score(model, ref_frame)
    ref_metrics = json.loads((RESULTS / "metrics.json").read_text())["champion"]["metrics"]
    print(f"Reference: {len(ref_frame)} train rows, "
          f"model {meta['model_version']}, threshold {threshold}")

    top_features = monitor.top_features_by_permutation(
        model, X_test.drop(columns=[config.ID_COLUMN]), y_test, n=5)
    print(f"Top-5 features by permutation importance: {top_features}")

    clean = simulate.simulate(n_customers=N_CUSTOMERS, n_weeks=N_CLEAN_WEEKS,
                              seed=123, schedule=False)
    clean_weeks = [clean[clean["week"] == w].reset_index(drop=True)
                   for w in range(1, N_CLEAN_WEEKS + 1)]
    clean_summaries = []
    for i, week_frame in enumerate(clean_weeks):
        prev = clean_weeks[i - 1] if i > 0 else None
        summary = _window_summary(ref_frame, ref_scores, week_frame, model,
                                  threshold, ref_metrics, previous_frame=prev)
        clean_summaries.append(summary)
        print(f"clean week {i + 1}: drift_share {summary['drift_share']}, "
              f"alerts {summary['alerts']}")
    false_alarms = _clean_false_alarms(clean_summaries)

    base = clean_weeks[0]
    injections = {}

    # Abrupt price rise: one window, fires or not.
    abrupt = simulate.inject_drift(base, "abrupt_price", seed=11)
    abrupt_summaries = [_window_summary(ref_frame, ref_scores, abrupt, model,
                                        threshold, ref_metrics)]
    injections["abrupt_price"] = {
        "target": "MonthlyCharges",
        "delays": {name: _first_firing_window(abrupt_summaries, "MonthlyCharges", key)
                   for name, key in DETECTOR_KEYS.items()},
        "values_at_detection": abrupt_summaries[0]["feature_drift"]["MonthlyCharges"],
    }

    # Gradual fiber growth: five windows, 5 to 25 points.
    gradual_summaries = []
    for step in GRADUAL_STEPS:
        injected = simulate.inject_drift(base, "gradual_fiber", seed=12,
                                         severity=step)
        gradual_summaries.append(_window_summary(
            ref_frame, ref_scores, injected, model, threshold, ref_metrics))
    injections["gradual_fiber"] = {
        "target": "InternetService",
        "steps": GRADUAL_STEPS,
        "delays": {name: _first_firing_window(gradual_summaries, "InternetService", key)
                   for name, key in DETECTOR_KEYS.items()},
        "values_by_step": [s["feature_drift"]["InternetService"]
                           for s in gradual_summaries],
    }

    # Prevalence shift: no feature changes, so input detectors must stay
    # quiet. The performance channel is the detector.
    prevalence = simulate.inject_drift(base, "prevalence", seed=13)
    prev_summary = _window_summary(ref_frame, ref_scores, prevalence, model,
                                   threshold, ref_metrics)
    perf = prev_summary["realised_performance"]
    injections["prevalence"] = {
        "target": "realised_performance",
        "input_detectors_fired": any(
            r["drifted"] for r in prev_summary["feature_drift"].values()),
        "performance_drop_flagged": bool(prev_summary["alerts"]["performance_drop"]),
        "label_shift_flagged": bool(prev_summary["alerts"]["label_shift"]),
        "realised_churn_rate": perf["churn_rate"],
        "churn_rate_change_rel": perf.get("churn_rate_change_rel"),
        "precision_at_100": perf["precision_at_100"],
        "precision_at_100_drop": perf["precision_at_100_drop"],
        "pr_auc": perf["pr_auc"],
        "pr_auc_drop": perf["pr_auc_drop"],
        "reference_precision_at_100": ref_metrics["precision_at_100"],
        "reference_pr_auc": ref_metrics["pr_auc"],
        "reference_churn_rate": ref_metrics["churn_rate"],
    }

    # Single-feature breakage: TotalCharges holds MonthlyCharges.
    broken = simulate.inject_drift(base, "breakage_total_charges", seed=14)
    broken_summaries = [_window_summary(ref_frame, ref_scores, broken, model,
                                        threshold, ref_metrics)]
    injections["breakage_total_charges"] = {
        "target": "TotalCharges",
        "delays": {name: _first_firing_window(broken_summaries, "TotalCharges", key)
                   for name, key in DETECTOR_KEYS.items()},
        "values_at_detection": broken_summaries[0]["feature_drift"]["TotalCharges"],
        "quality_issues": broken_summaries[0]["data_quality"],
    }

    # Null storm: 40% of TotalCharges null. The quality channel is the
    # detector; the distribution detectors see the surviving values.
    nulled = simulate.inject_drift(base, "null_storm", seed=15)
    null_summaries = [_window_summary(ref_frame, ref_scores, nulled, model,
                                      threshold, ref_metrics)]
    null_kinds = [i["kind"] for i in null_summaries[0]["data_quality"]]
    injections["null_storm"] = {
        "target": "data_quality",
        "quality_flagged": "null_rate" in null_kinds,
        "quality_issues": null_summaries[0]["data_quality"],
        "feature_delays": {name: _first_firing_window(null_summaries, "TotalCharges", key)
                           for name, key in DETECTOR_KEYS.items()},
    }

    # Stale tenure: this week's tenure equals last week's per customer.
    stale = simulate.inject_drift(base, "stale_tenure", seed=16,
                                  previous_frame=clean_weeks[0])
    stale_summaries = [_window_summary(ref_frame, ref_scores, stale, model,
                                       threshold, ref_metrics,
                                       previous_frame=clean_weeks[0])]
    stale_kinds = [i["kind"] for i in stale_summaries[0]["data_quality"]]
    injections["stale_tenure"] = {
        "target": "data_quality",
        "quality_flagged": "stale_feature" in stale_kinds,
        "quality_issues": stale_summaries[0]["data_quality"],
        "feature_drifted": any(
            r["drifted"] for r in stale_summaries[0]["feature_drift"].values()),
    }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": meta["model_version"],
        "threshold": threshold,
        "n_ref": len(ref_frame),
        "n_window": N_CUSTOMERS,
        "top_features": top_features,
        "thresholds_used": monitor.DEFAULT_THRESHOLDS,
        "clean_windows": {
            "n_windows": N_CLEAN_WEEKS,
            "false_alarms_per_detector": false_alarms,
            "drift_share_by_week": [s["drift_share"] for s in clean_summaries],
            "alerts_by_week": [s["alerts"] for s in clean_summaries],
        },
        "injections": injections,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "drift_experiments.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {RESULTS / 'drift_experiments.json'}")
    print(f"\nFalse alarms on {N_CLEAN_WEEKS} clean windows:")
    for name, fa in false_alarms.items():
        print(f"  {name}: {fa['windows_flagged']}/{fa['windows_total']} windows, "
              f"{fa['feature_flags_total']} feature flags")
    print("\nDetection delays (windows; 0 means the injected window itself):")
    for drift_type, info in injections.items():
        if "delays" in info:
            print(f"  {drift_type}: {info['delays']}")
        elif drift_type == "prevalence":
            print(f"  prevalence: input detectors fired: "
                  f"{info['input_detectors_fired']}, performance drop flagged: "
                  f"{info['performance_drop_flagged']}, label shift flagged: "
                  f"{info['label_shift_flagged']}, "
                  f"realised churn {info['realised_churn_rate']} "
                  f"(rel change {info['churn_rate_change_rel']}), "
                  f"P@100 drop {info['precision_at_100_drop']}")
        else:
            print(f"  {drift_type}: quality flagged: {info['quality_flagged']}")


if __name__ == "__main__":
    main()

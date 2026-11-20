"""Retraining loop with shadow promotion.

Trigger rules (from the research report, section 5.4). A trigger
starts a shadow run, never a promotion:
- top_feature_drift: PSI above 0.25 on any top-5 monitored feature,
  sustained for TWO consecutive windows.
- dataset_drift: drift_share above 0.5, sustained for TWO windows.
- prediction_drift: predicted churn rate moved more than 10% relative
  to baseline, sustained for TWO windows.
- performance_drop: realised precision at 100 fell more than 5 points
  against the simulator baseline on a labelled window. One window is
  enough here: a labelled window is already a month of evidence,
  because labels lag.

Two consecutive windows is the anti-fatigue rule: one noisy window is
not a retrain. Data-quality issues (null storm, breakage, stale
features) are deliberately NOT triggers: retraining on broken data
launders a data failure into the weights, after which the monitors go
quiet. A trigger caused by broken data must end in a refusal, which
is the point of the shadow run.

Shadow: the challenger is the champion configuration refit on the
last four labelled weeks. It scores the next two labelled weeks
alongside the champion, and promotion happens ONLY when the
challenger beats the champion on profit at the frozen threshold on
those weeks by more than the promotion margin. Two weeks, not one,
because a single 2,000-row week is too noisy to promote on, and the
margin exists for the same reason: the champion's own profit per
1,000 fluctuates with a standard deviation of $3,010 across clean
weeks (measured over 6 clean simulator weeks), so a win smaller than
the noise floor is not evidence. The margin is $2,500 per 1,000,
just above the two-week shadow's noise floor. A tie is not a win,
and neither is a $130 win. On promotion the challenger becomes the
current model for later weeks; the Phase 1 artifact file is never
overwritten.

Label lag, stated honestly: in production the labels for a week take
about 30 days to exist, so a shadow decision waits for them. In this
simulation the labels are available immediately, so the loop runs
week by week. The schedule and the decisions are the same; only the
calendar is compressed.

Outputs: benchmarks/results/retraining_log.jsonl (one line per
trigger, with the numbers that decided it) and
benchmarks/results/retraining_run.json (weekly summaries plus the
trigger list). The CI gate reads the log.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app import config, data, features, model_store, monitor, profit, scoring
from scripts import simulate

RESULTS = config.ROOT / "benchmarks" / "results"
N_CUSTOMERS = 5000
N_WEEKS = 12
TRAIN_WEEKS = 4  # challenger trains on the last four labelled weeks
SHADOW_WEEKS = 2  # shadow scores the next two labelled weeks combined

# Trigger rules: (streak needed, threshold). performance_drop needs a
# streak of 1 by design; see the module docstring.
TRIGGER_RULES = {
    "top_feature_drift": {"streak": 2, "psi_act": 0.25},
    "dataset_drift": {"streak": 2, "drift_share": 0.5},
    "prediction_drift": {"streak": 2, "rate_rel_change": 0.10},
    "performance_drop": {"streak": 1, "precision_at_k_drop": 0.05,
                         "pr_auc_drop": 0.03},
}

# Promotion margin: a challenger must beat the champion by more than
# this, in profit per 1,000 customers on the shadow weeks. Measured
# noise floor: the champion's own profit per 1,000 has a standard
# deviation of $3,010 across 6 clean simulator weeks (range $8,870),
# so a win of a few hundred dollars is noise, not evidence.
PROMOTION_MARGIN_PER_1000 = 2500.0


def empty_streaks():
    return {rule: 0 for rule in TRIGGER_RULES}


def check_triggers(summary, streaks, top_monitored, rules=None):
    """Evaluate the trigger rules against one window summary. Returns
    (fired_rules, updated_streaks). A rule fires when its condition
    holds for its streak length; a clean week resets its streak."""
    rules = rules or TRIGGER_RULES
    conditions = {}
    feature_results = summary.get("feature_drift", {})
    conditions["top_feature_drift"] = any(
        feature_results.get(f, {}).get("psi_drifted")
        and "skipped" not in feature_results.get(f, {})
        for f in top_monitored)
    conditions["dataset_drift"] = (
        summary.get("drift_share", 0) > rules["dataset_drift"]["drift_share"])
    conditions["prediction_drift"] = bool(
        summary.get("prediction_drift", {}).get("rate_drifted"))
    realised = summary.get("realised_performance") or {}
    conditions["performance_drop"] = (
        realised.get("precision_at_100_drop", 0)
        > rules["performance_drop"]["precision_at_k_drop"]
        or realised.get("pr_auc_drop", 0)
        > rules["performance_drop"].get("pr_auc_drop", 0.03))
    fired = []
    new_streaks = dict(streaks)
    for rule, holds in conditions.items():
        new_streaks[rule] = streaks.get(rule, 0) + 1 if holds else 0
        if holds and new_streaks[rule] >= rules[rule]["streak"]:
            fired.append(rule)
    return fired, new_streaks


def shadow_decision(champion_profit_per_1000, challenger_profit_per_1000,
                    margin=PROMOTION_MARGIN_PER_1000):
    """Promotion needs a win bigger than the noise floor, not just any
    win. A tie is not a win, and neither is a win smaller than the
    margin."""
    if challenger_profit_per_1000 - champion_profit_per_1000 > margin:
        return "promote"
    return "refuse"


def train_challenger(train_frame, champion_meta):
    """Refit the champion configuration on recent labelled weeks. The
    configuration is the champion's; only the data is newer."""
    y = train_frame["Churn"].to_numpy()
    X = train_frame[config.FEATURES]
    scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    model = features.build_pipeline(
        champion_meta["model_name"], handling=champion_meta["handling"],
        calibration=champion_meta["calibration"],
        scale_pos_weight=scale_pos_weight)
    model.fit(X, y)
    return model


def _profit_per_1000(y_true, y_proba, threshold):
    total = profit.profit_at_threshold(
        y_true, y_proba, threshold, config.OFFER_COST, config.SAVED_VALUE)
    return round(float(total) / len(y_true) * 1000, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persist-promotions", action="store_true",
        help="save each promoted challenger as a versioned artifact "
             "and move the current-model pointer to it (default off: "
             "the simulation only records the decision)")
    args = parser.parse_args()

    model, meta = scoring.load_artifact()
    threshold = meta["threshold"]
    experiments_path = RESULTS / "drift_experiments.json"
    if not experiments_path.exists():
        raise SystemExit("drift_experiments.json missing: "
                         "run scripts/drift_experiments.py first")
    experiments = json.loads(experiments_path.read_text())
    baseline = experiments["simulator_baseline"]
    top_monitored = experiments["top_monitored_features"]
    print(f"Champion: {meta['model_version']}, threshold {threshold}")
    print(f"Baseline (clean simulator weeks): {baseline}")
    print(f"Top monitored features: {top_monitored}")

    df = data.load_telco()
    X_train, _, _, _ = data.split(df)
    ref_frame = X_train.drop(columns=[config.ID_COLUMN])
    ref_scores = model.predict_proba(ref_frame[config.FEATURES])[:, 1]

    run = simulate.simulate(n_customers=N_CUSTOMERS, n_weeks=N_WEEKS, seed=42)
    weeks = [run[run["week"] == w].reset_index(drop=True)
             for w in range(1, N_WEEKS + 1)]

    current_model = model
    current_version = meta["model_version"]
    challenger_n = 0
    streaks = empty_streaks()
    weekly = []
    log_entries = []

    for w, frame in enumerate(weeks, start=1):
        prev = weeks[w - 2] if w > 1 else None
        scores = current_model.predict_proba(frame[config.FEATURES])[:, 1]
        summary = monitor.evaluate_window(
            ref_frame, frame, ref_scores, scores, threshold,
            y_true=frame["Churn"].to_numpy(), y_proba=scores,
            previous_frame=prev,
            id_column="customerID" if "customerID" in frame.columns else None,
            reference_metrics=baseline)
        summary["week"] = w
        summary["model_version"] = current_version
        weekly.append(summary)
        fired, streaks = check_triggers(summary, streaks, top_monitored)
        print(f"week {w}: drift_share {summary['drift_share']}, "
              f"alerts {summary['alerts']}, fired {fired}")
        if not fired:
            continue
        trigger_id = f"TRG-{w:02d}-{len(log_entries) + 1}"
        shadow_weeks = weeks[w:w + SHADOW_WEEKS]
        if not shadow_weeks:
            log_entries.append({
                "trigger_id": trigger_id, "week": w, "rules": fired,
                "decision": "deferred",
                "reason": "no future labelled week in this run",
                "champion_profit_per_1000": None,
                "challenger_profit_per_1000": None,
                "challenger_version": None,
                "model_version": current_version,
            })
            print(f"  {trigger_id}: deferred, no future labelled week")
            continue
        train_frame = pd.concat(weeks[max(0, w - TRAIN_WEEKS):w],
                                ignore_index=True)
        challenger = train_challenger(train_frame, meta)
        challenger_n += 1
        challenger_version = f"{meta['model_version']}-challenger-{challenger_n}"
        shadow = pd.concat(shadow_weeks, ignore_index=True)
        shadow_week_numbers = [w + 1 + i for i in range(len(shadow_weeks))]
        y_shadow = shadow["Churn"].to_numpy()
        champ_proba = current_model.predict_proba(shadow[config.FEATURES])[:, 1]
        chal_proba = challenger.predict_proba(shadow[config.FEATURES])[:, 1]
        champ_profit = _profit_per_1000(y_shadow, champ_proba, threshold)
        chal_profit = _profit_per_1000(y_shadow, chal_proba, threshold)
        decision = shadow_decision(champ_profit, chal_profit)
        entry = {
            "trigger_id": trigger_id,
            "week": w,
            "rules": fired,
            "decision": decision,
            "champion_profit_per_1000": champ_profit,
            "challenger_profit_per_1000": chal_profit,
            "profit_margin_per_1000": round(chal_profit - champ_profit, 2),
            "promotion_margin_per_1000": PROMOTION_MARGIN_PER_1000,
            "challenger_version": challenger_version,
            "challenger_trained_on_weeks": [int(x) for x in
                                           range(max(1, w - TRAIN_WEEKS + 1), w + 1)],
            "shadow_weeks": shadow_week_numbers,
            "model_version": current_version,
        }
        log_entries.append(entry)
        print(f"  {trigger_id}: {decision} "
              f"(champion ${champ_profit:,.0f} vs challenger ${chal_profit:,.0f} "
              f"per 1,000 on weeks {shadow_week_numbers})")
        if decision == "promote":
            current_model = challenger
            current_version = challenger_version
            if args.persist_promotions:
                saved = model_store.save_promotion(
                    challenger, challenger_version, threshold,
                    source=trigger_id)
                entry["artifact_file"] = saved.name
                print(f"  promoted artifact saved: {saved.name}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "retraining_log.jsonl", "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")
    run_record = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "champion_version": meta["model_version"],
        "threshold": threshold,
        "baseline": baseline,
        "top_monitored_features": top_monitored,
        "weekly_summaries": weekly,
        "triggers": log_entries,
    }
    (RESULTS / "retraining_run.json").write_text(json.dumps(run_record, indent=2))
    decisions = [e["decision"] for e in log_entries]
    print(f"\nTriggers: {len(log_entries)}, "
          f"promotions: {decisions.count('promote')}, "
          f"refusals: {decisions.count('refuse')}, "
          f"deferred: {decisions.count('deferred')}")
    print(f"Wrote {RESULTS / 'retraining_log.jsonl'}")


if __name__ == "__main__":
    main()

"""CI gate: fail the build if the champion regresses below its thresholds.

Phase 1 gate: reads benchmarks/results/metrics.json (written by
scripts/train.py) and checks the champion's ROC-AUC, PR-AUC, Brier
and precision at 100 against app/config.py GATE_THRESHOLDS.
Thresholds sit below the measured Phase 1 numbers so jitter does not
fail the build; lowering a threshold needs a reason in the commit
message.

Phase 2 drift gate: reads benchmarks/results/drift_experiments.json
(written by scripts/drift_experiments.py) and
benchmarks/results/retraining_log.jsonl (written by scripts/retrain.py).
It fails the build if injected drift is not caught within the stated
windows, if clean windows alert, or if the retraining log holds no
real promotion and no real refusal. The numbers in those files are
the evidence; the gate only checks them.
"""

import json
import sys
from pathlib import Path

from app import config

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def check_drift_gate(drift_path=RESULTS_DIR / "drift_experiments.json",
                     log_path=RESULTS_DIR / "retraining_log.jsonl"):
    """The drift half of the gate. Returns (passed, failures)."""
    failures = []
    if not Path(drift_path).exists():
        failures.append(f"drift experiments file missing: {drift_path}, "
                        "run scripts/drift_experiments.py first")
    if not Path(log_path).exists():
        failures.append(f"retraining log missing: {log_path}, "
                        "run scripts/retrain.py first")
    if failures:
        return False, failures

    data = json.loads(Path(drift_path).read_text())
    clean = data["clean_windows"]
    for detector, stats in clean["false_alarms_per_detector"].items():
        if stats["windows_flagged"] != 0:
            failures.append(
                f"clean windows: {detector} flagged "
                f"{stats['windows_flagged']}/{stats['windows_total']} "
                "clean windows, expected 0")
    if any(share != 0.0 for share in clean["drift_share_by_week"]):
        failures.append(
            f"clean windows: drift_share {clean['drift_share_by_week']}, "
            "expected 0.0 every week")

    injections = data["injections"]
    abrupt = injections["abrupt_price"]["delays"]
    for detector in ("psi", "ks", "jensen_shannon", "wasserstein"):
        if abrupt[detector] != 0:
            failures.append(
                f"abrupt_price: {detector} delay {abrupt[detector]}, "
                "expected 0 (the price rise must be caught in its own week)")
    gradual = injections["gradual_fiber"]["delays"]
    for detector, limit in (("psi", 4), ("jensen_shannon", 4)):
        delay = gradual[detector]
        if delay is None or delay > limit:
            failures.append(
                f"gradual_fiber: {detector} delay {delay}, expected at "
                f"most {limit} windows (25 points of Fiber growth)")

    prevalence = injections["prevalence"]
    if prevalence["input_detectors_fired"]:
        failures.append(
            "prevalence: input detectors fired, but a prevalence shift "
            "changes no feature, so they must stay quiet")
    if not prevalence["label_shift_flagged"]:
        failures.append(
            "prevalence: label shift not flagged, the churn rate moved "
            "from 0.2658 to 0.3875 and the monitor must see it")

    breakage = injections["breakage_total_charges"]
    if not breakage["quality_flagged"]:
        failures.append(
            "breakage_total_charges: data quality did not flag it")
    elif not any(issue["kind"] == "accumulation_broken"
                 for issue in breakage["quality_issues"]):
        failures.append(
            "breakage_total_charges: flagged, but not as "
            "accumulation_broken")

    if not injections["null_storm"]["quality_flagged"]:
        failures.append("null_storm: data quality did not flag it")
    stale = injections["stale_tenure"]
    if not stale["quality_flagged"]:
        failures.append("stale_tenure: data quality did not flag it")
    if stale["feature_drifted"]:
        failures.append(
            "stale_tenure: feature drift fired, but a frozen feature "
            "is a quality event, not a distribution drift")

    decisions = [json.loads(line)["decision"]
                 for line in Path(log_path).read_text().splitlines()
                 if line.strip()]
    if "promote" not in decisions:
        failures.append(
            "retraining log: no promotion, the loop must promote at "
            "least once on a labelled window")
    if "refuse" not in decisions:
        failures.append(
            "retraining log: no refusal, the loop must refuse at least "
            "once (a win below the promotion margin, or a losing "
            "challenger)")
    return (not failures), failures


def main():
    metrics_path = config.METRICS_PATH
    if not metrics_path.exists():
        print(f"FAIL: {metrics_path} missing, run scripts/train.py first")
        return 1
    if not config.ARTIFACT_PATH.exists():
        print(f"FAIL: {config.ARTIFACT_PATH} missing, run scripts/train.py first")
        return 1
    summary = json.loads(metrics_path.read_text())
    m = summary["champion"]["metrics"]
    gates = config.GATE_THRESHOLDS
    failures = []
    if m["roc_auc"] < gates["roc_auc"]:
        failures.append(f"roc_auc {m['roc_auc']} < {gates['roc_auc']}")
    if m["pr_auc"] < gates["pr_auc"]:
        failures.append(f"pr_auc {m['pr_auc']} < {gates['pr_auc']}")
    if m["brier"] > gates["brier_max"]:
        failures.append(f"brier {m['brier']} > {gates['brier_max']}")
    if m["precision_at_100"] < gates["precision_at_100"]:
        failures.append(f"precision_at_100 {m['precision_at_100']} < {gates['precision_at_100']}")
    version = summary["champion"]["model_version"]
    print(f"Champion: {version}")
    print(f"roc_auc {m['roc_auc']} (gate {gates['roc_auc']})")
    print(f"pr_auc {m['pr_auc']} (gate {gates['pr_auc']})")
    print(f"brier {m['brier']} (gate max {gates['brier_max']})")
    print(f"precision_at_100 {m['precision_at_100']} (gate {gates['precision_at_100']})")
    if failures:
        print("GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("METRIC GATE PASSED")
    drift_passed, drift_failures = check_drift_gate()
    if drift_passed:
        print("DRIFT GATE PASSED")
    else:
        print("DRIFT GATE FAILED:")
        for f in drift_failures:
            print(f"  - {f}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

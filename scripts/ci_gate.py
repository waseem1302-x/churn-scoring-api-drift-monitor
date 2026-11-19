"""CI gate: fail the build if the champion regresses below its thresholds.

Reads benchmarks/results/metrics.json (written by scripts/train.py)
and checks the champion's ROC-AUC, PR-AUC, Brier and precision at 100
against app/config.py GATE_THRESHOLDS. Thresholds sit below the
measured Phase 1 numbers so jitter does not fail the build; lowering
a threshold needs a reason in the commit message.
"""

import json
import sys

from app import config


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
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

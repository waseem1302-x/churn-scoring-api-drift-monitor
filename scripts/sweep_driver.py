"""Sweep driver: run each remaining S6E3 config in its own fresh
process, then merge the results into the partial sweep file.

The full sweep loop died twice on the random-forest configs, both
times silently mid-config with no traceback. A single RF config
run alone peaks at 4,725 MB RSS, just under the 6 GB cell limit,
so the loop's death is memory: the long-lived process accumulates
enough across configs that the next RF peak tips it over. This
driver runs one config per fresh python process (so the peak is
one config's), merges each finished config into
benchmarks/results/scale_s6e3_sweep.json, and only marks the sweep
complete when all 27 configs are present. If the driver itself is
interrupted, re-running it resumes from the merged file and the
per-config files: nothing already computed is redone.

Run from the sweep copy with the repo venv python and PYTHONPATH
pointing at the copy.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app import config, features

ROOT = config.ROOT
RESULTS = ROOT / "benchmarks" / "results"
OUT_PATH = RESULTS / "scale_s6e3_sweep.json"
CALIBRATIONS = (None, "platt", "isotonic")


def _calibration_name(calibration):
    return "none" if calibration is None else calibration


def _merge_entry(model_name, handling, calibration, version, result):
    """Build one merged sweep entry from a per-config result file.

    Identity comes from the driver's own loop, not from the file:
    per-config files written by the earlier VM sweep loop carry no
    model/handling/calibration keys, and reading them crashed the
    driver with KeyError: 'model' on Colab. Measurements come from
    the file. Calibration is stored as the string the single-config
    script writes ("none" for uncalibrated), matching the entries
    already merged.
    """
    return {
        "model": model_name,
        "handling": handling,
        "calibration": _calibration_name(calibration),
        "model_version": version,
        "train_time_s": result["train_time_s"],
        "oof_threshold": result["oof_threshold"],
        "oof_profit_per_1000": result["oof_profit_per_1000"],
        "test_metrics": result["test_metrics"],
        "test_profit": result["test_profit"],
        "test_profit_per_1000": result["test_profit_per_1000"],
    }


def main():
    payload = json.loads(OUT_PATH.read_text())
    done = {r["model_version"] for r in payload["configs"]}
    remaining = []
    for model_name in config.MODELS:
        for handling in config.HANDLINGS:
            for calibration in CALIBRATIONS:
                version = features.model_version(
                    model_name, handling, calibration)
                if version not in done:
                    remaining.append((model_name, handling, calibration))
    print(f"{len(payload['configs'])} configs done, "
          f"{len(remaining)} remaining", flush=True)
    started = time.perf_counter()
    for model_name, handling, calibration in remaining:
        version = features.model_version(
            model_name, handling, calibration)
        single = RESULTS / f"scale_s6e3_sweep_{version}.json"
        if single.exists():
            print(f"{version}: already computed, merging", flush=True)
        else:
            cmd = [sys.executable, "-m", "scripts.sweep_single_config",
                   "--model", model_name, "--handling", handling,
                   "--calibration", _calibration_name(calibration)]
            print(f"Running {' '.join(cmd)}", flush=True)
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0 or not single.exists():
                print(f"{version}: FAILED (exit {proc.returncode}), "
                      f"stopping here; re-run to resume", flush=True)
                return 1
        result = json.loads(single.read_text())
        payload["configs"].append(
            _merge_entry(
                model_name, handling, calibration, version, result))
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        OUT_PATH.write_text(json.dumps(payload, indent=2))
        print(f"{version}: merged ({len(payload['configs'])}/27)",
              flush=True)
    configs = payload["configs"]
    champion = max(configs, key=lambda r: (r["oof_profit_per_1000"],
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
    payload["total_train_time_s"] = round(
        sum(c["train_time_s"] for c in configs), 2)
    payload["complete"] = True
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nChampion: {champion['model_version']}", flush=True)
    print(f"Total train time: {payload['total_train_time_s']}s "
          f"for 27 configs (wall {time.perf_counter() - started:.0f}s "
          f"this run)", flush=True)
    print(f"Wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CI drift gate tests: the gate passes on the real run outputs and
fails on tampered copies. The gate is the build's memory: injected
drift must be caught within the stated windows, clean windows must
stay quiet, and the retraining log must hold a real promotion and a
real refusal.
"""

import copy
import json
from pathlib import Path

from scripts import ci_gate

RESULTS = Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def _load(name):
    return json.loads((RESULTS / name).read_text())


def _write(tmp_path, name, obj):
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return path


def _log_path(tmp_path):
    src = RESULTS / "retraining_log.jsonl"
    dst = tmp_path / "retraining_log.jsonl"
    dst.write_text(src.read_text())
    return dst


def test_drift_gate_passes_on_real_outputs(tmp_path):
    drift = _write(tmp_path, "drift_experiments.json", _load("drift_experiments.json"))
    log = _log_path(tmp_path)
    passed, failures = ci_gate.check_drift_gate(drift, log)
    assert passed, failures
    assert failures == []


def test_drift_gate_fails_when_abrupt_drift_is_missed(tmp_path):
    data = _load("drift_experiments.json")
    data["injections"]["abrupt_price"]["delays"]["psi"] = 2
    drift = _write(tmp_path, "drift_experiments.json", data)
    passed, failures = ci_gate.check_drift_gate(drift, _log_path(tmp_path))
    assert not passed
    assert any("abrupt_price" in f for f in failures)


def test_drift_gate_fails_when_gradual_drift_is_never_caught(tmp_path):
    data = _load("drift_experiments.json")
    data["injections"]["gradual_fiber"]["delays"]["psi"] = None
    data["injections"]["gradual_fiber"]["delays"]["jensen_shannon"] = None
    drift = _write(tmp_path, "drift_experiments.json", data)
    passed, failures = ci_gate.check_drift_gate(drift, _log_path(tmp_path))
    assert not passed
    assert any("gradual_fiber" in f for f in failures)


def test_drift_gate_fails_on_a_clean_window_false_alarm(tmp_path):
    data = _load("drift_experiments.json")
    data["clean_windows"]["false_alarms_per_detector"]["psi"]["windows_flagged"] = 1
    drift = _write(tmp_path, "drift_experiments.json", data)
    passed, failures = ci_gate.check_drift_gate(drift, _log_path(tmp_path))
    assert not passed
    assert any("clean" in f for f in failures)


def test_drift_gate_fails_when_prevalence_is_invisible(tmp_path):
    data = _load("drift_experiments.json")
    data["injections"]["prevalence"]["label_shift_flagged"] = False
    drift = _write(tmp_path, "drift_experiments.json", data)
    passed, failures = ci_gate.check_drift_gate(drift, _log_path(tmp_path))
    assert not passed
    assert any("prevalence" in f for f in failures)


def test_drift_gate_fails_when_breakage_is_not_flagged(tmp_path):
    data = _load("drift_experiments.json")
    data["injections"]["breakage_total_charges"]["quality_flagged"] = False
    drift = _write(tmp_path, "drift_experiments.json", data)
    passed, failures = ci_gate.check_drift_gate(drift, _log_path(tmp_path))
    assert not passed
    assert any("breakage" in f for f in failures)


def test_drift_gate_fails_without_a_refusal_in_the_log(tmp_path):
    log = tmp_path / "retraining_log.jsonl"
    entries = [json.loads(line) for line in
               (RESULTS / "retraining_log.jsonl").read_text().splitlines()]
    keep = [e for e in entries if e["decision"] != "refuse"]
    log.write_text("\n".join(json.dumps(e) for e in keep))
    drift = _write(tmp_path, "drift_experiments.json", _load("drift_experiments.json"))
    passed, failures = ci_gate.check_drift_gate(drift, log)
    assert not passed
    assert any("refusal" in f for f in failures)


def test_drift_gate_fails_without_a_promotion_in_the_log(tmp_path):
    log = tmp_path / "retraining_log.jsonl"
    entries = [json.loads(line) for line in
               (RESULTS / "retraining_log.jsonl").read_text().splitlines()]
    keep = [e for e in entries if e["decision"] != "promote"]
    log.write_text("\n".join(json.dumps(e) for e in keep))
    drift = _write(tmp_path, "drift_experiments.json", _load("drift_experiments.json"))
    passed, failures = ci_gate.check_drift_gate(drift, log)
    assert not passed
    assert any("promotion" in f for f in failures)


def test_drift_gate_fails_on_missing_files(tmp_path):
    passed, failures = ci_gate.check_drift_gate(
        tmp_path / "nope.json", tmp_path / "nope.jsonl")
    assert not passed
    assert len(failures) == 2

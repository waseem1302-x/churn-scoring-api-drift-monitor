"""Tests for the S6E3 block train.py carries in metrics.json.

train.py rewrites metrics.json on every run, but the S6E3 scale run
is a separate step from training: it needs the Kaggle CSV, which CI
does not have. So train.py must not reset the recorded scale-run
state to pending. When a previous metrics.json already says the
scale run is done, the new summary keeps that; otherwise the
pending default stands. Without this, CI wipes the committed 'done'
marker and tests/test_docs.py fails there while passing locally.
"""

import json
from pathlib import Path

from scripts import train

ROOT = Path(__file__).resolve().parents[1]


def _pending_summary():
    return {
        "s6e3_scale_run": "pending",
        "s6e3": {"train_time_s": None,
                 "note": "Kaggle credentials not present; scale run pending"},
    }


def _write_metrics(tmp_path, marker, train_time_s=384.21):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "s6e3_scale_run": marker,
        "s6e3": {"train_time_s": train_time_s,
                 "note": "Scale run done 20 Nov 2026"},
    }, indent=2))
    return path


def test_carry_keeps_done(tmp_path):
    path = _write_metrics(tmp_path, "done")
    summary = train._carry_s6e3_block(_pending_summary(), path)
    assert summary["s6e3_scale_run"] == "done"
    assert summary["s6e3"]["train_time_s"] == 384.21


def test_carry_leaves_pending_when_no_previous_file(tmp_path):
    summary = train._carry_s6e3_block(
        _pending_summary(), tmp_path / "metrics.json")
    assert summary["s6e3_scale_run"] == "pending"
    assert summary["s6e3"]["train_time_s"] is None


def test_carry_leaves_pending_when_marker_is_pending(tmp_path):
    path = _write_metrics(tmp_path, "pending", train_time_s=None)
    summary = train._carry_s6e3_block(_pending_summary(), path)
    assert summary["s6e3_scale_run"] == "pending"
    assert summary["s6e3"]["train_time_s"] is None


def test_carry_ignores_a_corrupt_previous_file(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("{not json")
    summary = train._carry_s6e3_block(_pending_summary(), path)
    assert summary["s6e3_scale_run"] == "pending"
    assert summary["s6e3"]["train_time_s"] is None

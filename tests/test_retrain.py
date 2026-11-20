"""Tests for the retraining loop (scripts/retrain.py).

The trigger rules are the anti-fatigue rules from the research: a
single noisy window never retrains anything. Two consecutive windows
of the same signal do. The shadow rule is equally strict: a challenger
must beat the champion on a labelled week on profit, or it stays in
shadow.
"""

import json
from pathlib import Path

import pytest

from scripts import retrain

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "benchmarks" / "results" / "retraining_log.jsonl"


def _summary(psi_top=False, drift_share=0.0, rate_drifted=False,
             p100_drop=0.0):
    return {
        "feature_drift": {
            "Contract": {"drifted": psi_top, "psi_drifted": psi_top},
            "MonthlyCharges": {"drifted": False, "psi_drifted": False},
        },
        "drift_share": drift_share,
        "prediction_drift": {"drifted": rate_drifted,
                             "rate_drifted": rate_drifted},
        "data_quality": [],
        "realised_performance": {"precision_at_100_drop": p100_drop},
        "alerts": {},
    }


def test_one_noisy_window_does_not_trigger():
    streaks = retrain.empty_streaks()
    fired, streaks = retrain.check_triggers(
        _summary(psi_top=True), streaks, ["Contract"], retrain.TRIGGER_RULES)
    assert fired == []
    assert streaks["top_feature_drift"] == 1


def test_two_consecutive_windows_trigger_top_feature_drift():
    streaks = retrain.empty_streaks()
    _, streaks = retrain.check_triggers(
        _summary(psi_top=True), streaks, ["Contract"], retrain.TRIGGER_RULES)
    fired, _ = retrain.check_triggers(
        _summary(psi_top=True), streaks, ["Contract"], retrain.TRIGGER_RULES)
    assert fired == ["top_feature_drift"]


def test_clean_window_resets_the_streak():
    streaks = retrain.empty_streaks()
    _, streaks = retrain.check_triggers(
        _summary(psi_top=True), streaks, ["Contract"], retrain.TRIGGER_RULES)
    fired, streaks = retrain.check_triggers(
        _summary(psi_top=False), streaks, ["Contract"], retrain.TRIGGER_RULES)
    assert fired == []
    assert streaks["top_feature_drift"] == 0


def test_drift_share_rule_needs_two_windows():
    streaks = retrain.empty_streaks()
    fired, streaks = retrain.check_triggers(
        _summary(drift_share=0.6), streaks, ["Contract"], retrain.TRIGGER_RULES)
    assert fired == []
    fired, _ = retrain.check_triggers(
        _summary(drift_share=0.6), streaks, ["Contract"], retrain.TRIGGER_RULES)
    assert fired == ["dataset_drift"]


def test_predicted_rate_rule_needs_two_windows():
    streaks = retrain.empty_streaks()
    fired, streaks = retrain.check_triggers(
        _summary(rate_drifted=True), streaks, ["Contract"], retrain.TRIGGER_RULES)
    assert fired == []
    fired, _ = retrain.check_triggers(
        _summary(rate_drifted=True), streaks, ["Contract"], retrain.TRIGGER_RULES)
    assert fired == ["prediction_drift"]


def test_precision_drop_triggers_on_one_labelled_window():
    # A labelled window is already a month of evidence (labels lag),
    # so a 5-point precision drop fires at once, without a second week.
    streaks = retrain.empty_streaks()
    fired, _ = retrain.check_triggers(
        _summary(p100_drop=0.06), streaks, ["Contract"], retrain.TRIGGER_RULES)
    assert fired == ["performance_drop"]


def test_pr_auc_drop_alone_triggers():
    # The performance rule is a drop on either decision metric: a
    # ranking collapse shows in PR-AUC even when the top-100 list
    # still looks fine.
    streaks = retrain.empty_streaks()
    summary = _summary(p100_drop=0.01)
    summary["realised_performance"]["pr_auc_drop"] = 0.04
    fired, _ = retrain.check_triggers(
        summary, streaks, ["Contract"], retrain.TRIGGER_RULES)
    assert fired == ["performance_drop"]


def test_small_precision_drop_does_not_trigger():
    streaks = retrain.empty_streaks()
    fired, _ = retrain.check_triggers(
        _summary(p100_drop=0.04), streaks, ["Contract"], retrain.TRIGGER_RULES)
    assert fired == []


def test_shadow_promotes_a_winning_challenger():
    # A $5,000 win clears the $2,500 margin.
    assert retrain.shadow_decision(90000.0, 95000.0) == "promote"


def test_shadow_refuses_a_losing_challenger():
    assert retrain.shadow_decision(90000.0, 85000.0) == "refuse"


def test_shadow_refuses_a_tie():
    # A tie is not a win. The challenger must beat the champion, or
    # the champion stays.
    assert retrain.shadow_decision(90000.0, 90000.0) == "refuse"


def test_shadow_refuses_a_win_below_the_margin():
    # A $130 win is noise: the champion's own profit per 1,000 moves
    # by $3,010 (one standard deviation) across clean weeks. A win
    # smaller than the margin is not evidence, and the loop refuses it.
    assert retrain.shadow_decision(90000.0, 90130.0) == "refuse"


def test_retraining_log_exists_with_promotion_and_refusal():
    if not LOG.exists():
        pytest.fail("retraining_log.jsonl missing: run scripts/retrain.py first")
    entries = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    assert entries, "the log must hold at least one trigger"
    decisions = [e["decision"] for e in entries]
    assert "promote" in decisions, "exit criteria need one real promotion"
    assert "refuse" in decisions, "exit criteria need one real refusal"
    for entry in entries:
        assert {"trigger_id", "week", "rules", "decision",
                "champion_profit_per_1000",
                "challenger_profit_per_1000"} <= set(entry)

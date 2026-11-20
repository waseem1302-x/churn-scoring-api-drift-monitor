"""Tests for the drift monitor (app/monitor.py).

The statistics are tested against hand-computed cases first: identical
distributions must score zero drift, a known shift must score a known
value, and a disjoint categorical split must score the maximum.
"""

import numpy as np
import pandas as pd
import pytest

from app import monitor


def test_psi_identical_distributions_is_zero():
    ref = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 25)
    cur = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 25)
    assert monitor.psi_numeric(ref, cur) == pytest.approx(0.0, abs=1e-9)


def test_psi_known_shift_is_large():
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(0, 1, 2000))
    cur = pd.Series(rng.normal(3, 1, 2000))
    value = monitor.psi_numeric(ref, cur)
    assert value > 1.0


def test_psi_categorical_identical_is_zero():
    ref = pd.Series(["a", "b", "c"] * 100)
    cur = pd.Series(["a", "b", "c"] * 100)
    assert monitor.psi_categorical(ref, cur) == pytest.approx(0.0, abs=1e-9)


def test_psi_categorical_disjoint_is_large():
    ref = pd.Series(["a"] * 200)
    cur = pd.Series(["b"] * 200)
    assert monitor.psi_categorical(ref, cur) > 1.0


def test_ks_identical_is_zero_p_one():
    rng = np.random.default_rng(7)
    ref = pd.Series(rng.normal(0, 1, 500))
    stat, p = monitor.ks_test(ref, ref)
    assert stat == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_ks_shifted_has_small_p():
    rng = np.random.default_rng(7)
    ref = pd.Series(rng.normal(0, 1, 500))
    cur = pd.Series(rng.normal(1.0, 1, 500))
    stat, p = monitor.ks_test(ref, cur)
    assert stat > 0.2
    assert p < 0.001


def test_jensen_shannon_identical_is_zero():
    ref = pd.Series(["a", "b"] * 100)
    assert monitor.jensen_shannon_categorical(ref, ref) == pytest.approx(0.0, abs=1e-9)


def test_jensen_shannon_disjoint_is_one():
    # scipy's jensenshannon is the distance (square root of the
    # divergence, natural log), so disjoint distributions score
    # sqrt(ln 2) = 0.8326, not 1.0. The 0.1 threshold in the monitor
    # is on this scale; see MONITORING.md.
    ref = pd.Series(["a"] * 100)
    cur = pd.Series(["b"] * 100)
    assert monitor.jensen_shannon_categorical(ref, cur) == pytest.approx(0.8326, abs=0.001)


def test_jensen_shannon_numeric_identical_is_zero():
    ref = pd.Series([1.0, 2.0, 3.0, 4.0] * 50)
    assert monitor.jensen_shannon_numeric(ref, ref) == pytest.approx(0.0, abs=1e-9)


def test_wasserstein_identical_is_zero():
    ref = pd.Series([1.0, 2.0, 3.0] * 100)
    assert monitor.wasserstein_normed(ref, ref) == pytest.approx(0.0, abs=1e-12)


def test_wasserstein_known_shift_is_one_std():
    # A shift of exactly one reference standard deviation, normalised
    # by that deviation, must score 1.0.
    ref = pd.Series([0.0, 1.0, 2.0, 3.0] * 250)
    cur = ref + float(ref.std())
    assert monitor.wasserstein_normed(ref, cur) == pytest.approx(1.0, abs=0.01)


def test_feature_drift_returns_all_four_measures():
    rng = np.random.default_rng(3)
    ref = pd.Series(rng.normal(0, 1, 1000))
    cur = pd.Series(rng.normal(0, 1, 1000))
    out = monitor.feature_drift_numeric("MonthlyCharges", ref, cur,
                                        monitor.DEFAULT_THRESHOLDS)
    assert set(out) >= {"feature", "psi", "ks_stat", "ks_p",
                        "jensen_shannon", "wasserstein", "drifted"}


def test_feature_drift_flags_a_big_shift():
    rng = np.random.default_rng(3)
    ref = pd.Series(rng.normal(0, 1, 1000))
    cur = pd.Series(rng.normal(2, 1, 1000))
    out = monitor.feature_drift_numeric("MonthlyCharges", ref, cur,
                                        monitor.DEFAULT_THRESHOLDS)
    assert out["drifted"] is True
    assert out["psi"] > 0.25


def test_feature_drift_does_not_flag_noise():
    rng = np.random.default_rng(11)
    ref = pd.Series(rng.normal(50, 10, 1000))
    cur = pd.Series(rng.normal(50.2, 10, 1000))
    out = monitor.feature_drift_numeric("MonthlyCharges", ref, cur,
                                        monitor.DEFAULT_THRESHOLDS)
    assert out["drifted"] is False


def test_categorical_drift_flags_a_new_dominant_category():
    ref = pd.Series(["DSL", "Fiber optic", "No"] * 300)
    cur = pd.Series(["Fiber optic"] * 700 + ["DSL"] * 100 + ["No"] * 100)
    out = monitor.feature_drift_categorical("InternetService", ref, cur,
                                            monitor.DEFAULT_THRESHOLDS)
    assert out["drifted"] is True


def test_data_quality_null_rate_gate():
    ref = pd.DataFrame({"TotalCharges": [100.0, 200.0, np.nan, 300.0] * 250})
    cur = ref.copy()
    cur.loc[cur.sample(400, random_state=1).index, "TotalCharges"] = np.nan
    issues = monitor.data_quality_issues(ref, cur)
    flagged = [i for i in issues if i["feature"] == "TotalCharges"]
    assert flagged, "a 40% null rate must be flagged"
    assert flagged[0]["kind"] == "null_rate"


def test_data_quality_no_flag_on_clean_window():
    rng = np.random.default_rng(5)
    ref = pd.DataFrame({
        "tenure": rng.integers(0, 72, 1000),
        "MonthlyCharges": rng.normal(65, 30, 1000),
    })
    cur = pd.DataFrame({
        "tenure": rng.integers(0, 72, 1000),
        "MonthlyCharges": rng.normal(65, 30, 1000),
    })
    assert monitor.data_quality_issues(ref, cur) == []


def test_data_quality_missing_column_is_schema_change():
    ref = pd.DataFrame({"tenure": [1, 2, 3], "MonthlyCharges": [10.0, 20.0, 30.0]})
    cur = pd.DataFrame({"tenure": [1, 2, 3]})
    issues = monitor.data_quality_issues(ref, cur)
    assert any(i["kind"] == "schema_change" for i in issues)


def test_data_quality_unknown_category_flagged():
    ref = pd.DataFrame({"Contract": ["Month-to-month", "One year"] * 500})
    cur = pd.DataFrame({"Contract": ["Month-to-month"] * 900 + ["Lifetime"] * 100})
    issues = monitor.data_quality_issues(ref, cur)
    assert any(i["kind"] == "unknown_category" for i in issues)


def test_stale_feature_frozen_panel_is_flagged():
    # Same customers, same tenure as the previous week: the feature
    # stopped updating. A normal week increments tenure by one.
    prev = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(200)],
        "tenure": list(range(200)),
    })
    cur = prev.copy()
    issues = monitor.data_quality_issues(
        prev, cur, previous_frame=prev, id_column="customerID")
    assert any(i["kind"] == "stale_feature" and i["feature"] == "tenure"
               for i in issues)


def test_stale_feature_normal_week_is_not_flagged():
    prev = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(200)],
        "tenure": list(range(200)),
    })
    cur = prev.copy()
    cur["tenure"] = cur["tenure"] + 1
    issues = monitor.data_quality_issues(
        prev, cur, previous_frame=prev, id_column="customerID")
    assert not any(i["kind"] == "stale_feature" for i in issues)


def test_accumulation_broken_when_total_charges_fall():
    # An accumulating total never falls. A week where TotalCharges
    # dropped for nearly every customer is a broken feed, even though
    # the values are still in range and the distribution looks fine.
    prev = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(200)],
        "TotalCharges": [2000.0 + i for i in range(200)],
    })
    cur = prev.copy()
    cur["TotalCharges"] = [65.0] * 200
    issues = monitor.data_quality_issues(
        prev, cur, previous_frame=prev, id_column="customerID")
    assert any(i["kind"] == "accumulation_broken"
               and i["feature"] == "TotalCharges" for i in issues)


def test_accumulation_not_broken_on_normal_growth():
    prev = pd.DataFrame({
        "customerID": [f"C{i}" for i in range(200)],
        "TotalCharges": [2000.0 + i for i in range(200)],
        "tenure": list(range(200)),
    })
    cur = prev.copy()
    cur["TotalCharges"] = cur["TotalCharges"] + 65.0
    cur["tenure"] = cur["tenure"] + 1
    issues = monitor.data_quality_issues(
        prev, cur, previous_frame=prev, id_column="customerID")
    assert not any(i["kind"] == "accumulation_broken" for i in issues)


def test_prediction_drift_flags_a_rate_move():
    ref_scores = np.linspace(0.05, 0.95, 1000)
    cur_scores = np.linspace(0.05, 0.95, 1000) + 0.25
    cur_scores = np.clip(cur_scores, 0, 1)
    # At threshold 0.5 the predicted rate moves from 50% to 78%,
    # a 56% relative move; at the low 0.0934 operating threshold the
    # same shift moves the rate by only 5%, which is why the score
    # distribution (PSI), not just the rate, is monitored.
    out = monitor.prediction_drift(ref_scores, cur_scores, threshold=0.5,
                                   thresholds=monitor.DEFAULT_THRESHOLDS)
    assert out["drifted"] is True
    assert out["predicted_rate_change_rel"] > 0.10


def test_prediction_drift_quiet_on_same_scores():
    rng = np.random.default_rng(9)
    ref_scores = rng.uniform(0, 1, 1000)
    cur_scores = rng.uniform(0, 1, 1000)
    out = monitor.prediction_drift(ref_scores, cur_scores, threshold=0.0934,
                                   thresholds=monitor.DEFAULT_THRESHOLDS)
    assert out["drifted"] is False


def test_accumulating_features_skipped_in_evaluate_window():
    # Tenure and TotalCharges move by design as time passes, so a
    # distribution comparison against training is the wrong test for
    # them. The monitor must skip them here (they are covered by the
    # week-over-week stale check) instead of alarming on every window.
    rng = np.random.default_rng(21)
    ref = pd.DataFrame({
        "tenure": rng.integers(0, 72, 1000),
        "MonthlyCharges": rng.normal(65, 30, 1000),
        "TotalCharges": rng.normal(2000, 2000, 1000),
    })
    cur = ref.copy()
    cur["tenure"] = cur["tenure"] + 5
    cur["TotalCharges"] = cur["TotalCharges"] + 300
    out = monitor.evaluate_window(
        ref, cur, rng.uniform(0, 1, 1000), rng.uniform(0, 1, 1000),
        threshold=0.0934, thresholds=monitor.DEFAULT_THRESHOLDS)
    assert out["feature_drift"]["tenure"]["skipped"] == "accumulating"
    assert out["feature_drift"]["TotalCharges"]["skipped"] == "accumulating"
    assert out["drift_share"] == pytest.approx(0.0, abs=0.01)


def test_schema_change_ignores_id_label_and_window_columns():
    ref = pd.DataFrame({"tenure": [1, 2, 3], "MonthlyCharges": [10.0, 20.0, 30.0]})
    cur = ref.copy()
    cur["customerID"] = ["C1", "C2", "C3"]
    cur["Churn"] = [0, 1, 0]
    cur["week"] = [1, 1, 1]
    cur["date"] = ["2026-01-05"] * 3
    issues = monitor.data_quality_issues(ref, cur)
    assert not any(i["kind"] == "schema_change" for i in issues)


def test_evaluate_window_returns_four_channels():
    rng = np.random.default_rng(21)
    ref = pd.DataFrame({
        "tenure": rng.integers(0, 72, 1000),
        "MonthlyCharges": rng.normal(65, 30, 1000),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], 1000),
    })
    cur = ref.copy()
    ref_scores = rng.uniform(0, 1, 1000)
    cur_scores = rng.uniform(0, 1, 1000)
    out = monitor.evaluate_window(
        ref, cur, ref_scores, cur_scores, threshold=0.0934,
        thresholds=monitor.DEFAULT_THRESHOLDS)
    assert set(out) >= {"feature_drift", "prediction_drift",
                        "data_quality", "realised_performance",
                        "drift_share", "alerts"}
    assert out["realised_performance"] is None  # no labels given
    assert out["drift_share"] == pytest.approx(0.0, abs=0.01)


def test_evaluate_window_with_labels_reports_precision_at_k():
    rng = np.random.default_rng(21)
    ref = pd.DataFrame({
        "tenure": rng.integers(0, 72, 1000),
        "MonthlyCharges": rng.normal(65, 30, 1000),
    })
    cur = ref.copy()
    y = (rng.uniform(0, 1, 1000) < 0.27).astype(int)
    scores = rng.uniform(0, 1, 1000)
    out = monitor.evaluate_window(
        ref, cur, scores, scores, threshold=0.0934,
        thresholds=monitor.DEFAULT_THRESHOLDS,
        y_true=y, y_proba=scores)
    perf = out["realised_performance"]
    assert perf is not None
    assert 0.0 <= perf["precision_at_100"] <= 1.0
    assert perf["n"] == 1000

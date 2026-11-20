"""Tests for the longitudinal churn simulator (scripts/simulate.py).

The simulator is a feature, not a fudge: it is the only way to score a
drift monitor against known ground truth, because Telco has no
timestamps. The schedule is written down in the module before any
result is measured, and these tests check the schedule actually
happens.
"""

import numpy as np
import pandas as pd

from app import config, scoring
from scripts import simulate


def test_simulator_schema_matches_telco():
    frame = simulate.simulate(n_customers=200, n_weeks=4, seed=42)
    expected = set(config.FEATURES) | {"customerID", "Churn", "week", "date"}
    assert set(frame.columns) == expected
    assert frame["Churn"].isin([0, 1]).all()
    assert frame["week"].isin([1, 2, 3, 4]).all()


def test_simulator_has_weekly_timestamps():
    frame = simulate.simulate(n_customers=100, n_weeks=4, seed=42)
    dates = pd.to_datetime(frame["date"])
    assert dates.notna().all()
    assert dates.min() < dates.max()
    per_week = frame.groupby("week")["date"].nunique()
    assert (per_week == 1).all(), "one date per week"


def test_simulator_reproducible_with_seed():
    a = simulate.simulate(n_customers=200, n_weeks=6, seed=7)
    b = simulate.simulate(n_customers=200, n_weeks=6, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_simulator_churn_rate_near_telco():
    frame = simulate.simulate(n_customers=2000, n_weeks=4, seed=42)
    rate = frame["Churn"].mean()
    assert 0.20 <= rate <= 0.32, f"baseline churn rate {rate:.3f}"


def test_schedule_abrupt_price_rise_in_week_5():
    frame = simulate.simulate(n_customers=2000, n_weeks=6, seed=42)
    m4 = frame.loc[frame["week"] == 4, "MonthlyCharges"].mean()
    m5 = frame.loc[frame["week"] == 5, "MonthlyCharges"].mean()
    # Float means, so the rise is 25.0 within rounding, not exactly.
    assert abs((m5 - m4) - 25.0) < 1e-9


def test_schedule_gradual_fiber_growth():
    frame = simulate.simulate(n_customers=2000, n_weeks=10, seed=42)
    shares = frame.groupby("week")["InternetService"].apply(
        lambda s: (s == "Fiber optic").mean())
    assert shares.loc[6] > shares.loc[5] + 0.03
    assert shares.loc[10] > shares.loc[6] + 0.10
    assert abs(shares.loc[5] - shares.loc[4]) < 0.01  # no growth before week 6


def test_schedule_breakage_total_charges_in_weeks_10_and_11():
    frame = simulate.simulate(n_customers=500, n_weeks=12, seed=42)
    for week in (10, 11):
        sub = frame[frame["week"] == week]
        assert (sub["TotalCharges"] == sub["MonthlyCharges"]).all()
    sub9 = frame[frame["week"] == 9]
    # Week 9 is clean: only a rare one-month customer can have
    # TotalCharges equal to one MonthlyCharges by coincidence.
    assert (sub9["TotalCharges"] == sub9["MonthlyCharges"]).mean() < 0.02


def test_schedule_stale_tenure_in_week_12():
    frame = simulate.simulate(n_customers=500, n_weeks=12, seed=42)
    w11 = frame[frame["week"] == 11].set_index("customerID")["tenure"]
    w12 = frame[frame["week"] == 12].set_index("customerID")["tenure"]
    assert (w11 == w12).all(), "tenure frozen in week 12"
    w10 = frame[frame["week"] == 10].set_index("customerID")["tenure"]
    assert (w11 > w10).any(), "tenure still moved in week 11"


def test_schedule_prevalence_shift_raises_churn():
    frame = simulate.simulate(n_customers=2000, n_weeks=12, seed=42)
    early = frame[frame["week"] <= 4]["Churn"].mean()
    late = frame[frame["week"] >= 9]["Churn"].mean()
    assert late - early > 0.05


def test_champion_scores_simulated_weeks():
    model, meta = scoring.load_artifact()
    frame = simulate.simulate(n_customers=200, n_weeks=2, seed=42)
    week1 = frame[frame["week"] == 1]
    proba = model.predict_proba(week1[config.FEATURES])[:, 1]
    assert len(proba) == len(week1)
    assert ((proba >= 0) & (proba <= 1)).all()

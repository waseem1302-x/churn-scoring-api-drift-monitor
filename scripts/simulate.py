"""Longitudinal churn simulator with a KNOWN drift schedule.

Telco has no timestamps, so no real temporal validation is possible on
it. This simulator closes that gap: a panel of customers, one snapshot
row per customer per week, with ground-truth labels drawn from a known
churn rule. The schedule below is written down BEFORE any result is
measured, so a detection result can be checked against it. The
simulator is a feature, not a fudge: without known ground truth, a
drift monitor can only claim to detect drift, never prove it.

The true churn rule is a logistic function of the features, with
coefficients written down here. The champion was trained on Telco, not
on this rule, so the simulator measures the monitor and the retraining
loop, not the champion's fit to the rule.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from app import config, data

# The known drift schedule. Weeks are 1-based. Every event is dated
# here, before any experiment runs, so results are checked against
# this table, not fitted to it.
DRIFT_SCHEDULE = {
    "weeks_1_to_4": "clean baseline, no drift",
    "week_5_onwards": "abrupt price rise: MonthlyCharges +$25 for all",
    "weeks_6_to_10": "gradual behaviour change: Fiber optic share grows "
                      "5 points per week",
    "week_9_onwards": "prevalence shift: churn intercept +0.6, no feature "
                      "changes (invisible to input detectors)",
    "weeks_10_to_11": "single-feature breakage: TotalCharges replaced by "
                      "MonthlyCharges (broken join)",
    "week_12": "stale feature: tenure frozen at its week 11 value",
}

START_DATE = date(2026, 1, 5)  # a Monday; simulated calendar, not real dates
PRICE_RISE_WEEK = 5
PRICE_RISE_DOLLARS = 25.0
FIBER_GROWTH_WEEKS = (6, 10)
FIBER_GROWTH_POINTS_PER_WEEK = 0.05
PREVALENCE_SHIFT_WEEK = 9
PREVALENCE_INTERCEPT_SHIFT = 0.6
BREAKAGE_WEEKS = (10, 11)
STALE_WEEK = 12
MAX_TENURE = 72

# The true churn rule. Coefficients are on the raw features; the
# intercept is set so the clean baseline churn rate is near Telco's
# 26.5%. Labels are Bernoulli draws from this probability, so the same
# customer can churn in one week and not the next: each week is an
# independent snapshot of the scoring population, not a cohort exit.
TRUE_RULE = {
    "intercept": -1.87,
    "tenure": -0.03,
    "MonthlyCharges": 0.015,
    "Contract_Month_to_month": 0.9,
    "InternetService_Fiber_optic": 0.5,
    "OnlineSecurity_Yes": -0.4,
    "TechSupport_Yes": -0.4,
    "PaymentMethod_Electronic_check": 0.3,
}

_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
           "TechSupport", "StreamingTV", "StreamingMovies"]


def true_churn_probability(frame, intercept_shift=0.0) -> np.ndarray:
    """The known label rule. Returns a probability per row."""
    r = TRUE_RULE
    logit = (
        r["intercept"] + intercept_shift
        + r["tenure"] * frame["tenure"].to_numpy(dtype=float)
        + r["MonthlyCharges"] * frame["MonthlyCharges"].to_numpy(dtype=float)
        + r["Contract_Month_to_month"] * (frame["Contract"] == "Month-to-month").to_numpy(dtype=float)
        + r["InternetService_Fiber_optic"] * (frame["InternetService"] == "Fiber optic").to_numpy(dtype=float)
        + r["OnlineSecurity_Yes"] * (frame["OnlineSecurity"] == "Yes").to_numpy(dtype=float)
        + r["TechSupport_Yes"] * (frame["TechSupport"] == "Yes").to_numpy(dtype=float)
        + r["PaymentMethod_Electronic_check"] * (frame["PaymentMethod"] == "Electronic check").to_numpy(dtype=float)
    )
    return 1.0 / (1.0 + np.exp(-logit))


def _base_panel(n_customers: int, seed: int) -> pd.DataFrame:
    """Sample customers from the Phase 1 training split, with
    replacement. Their feature values are real Telco rows, so the
    panel starts from the same distribution the champion was trained
    on. Blank TotalCharges (new customers, tenure 0) become 0.0: they
    have no bill yet, and the simulator accumulates from zero."""
    df = data.load_telco()
    X_train, _, _, _ = data.split(df)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X_train), size=n_customers)
    panel = X_train.iloc[idx].reset_index(drop=True).copy()
    panel["customerID"] = [f"SIM-{i:06d}" for i in range(n_customers)]
    panel["TotalCharges"] = panel["TotalCharges"].fillna(0.0)
    return panel


def simulate(n_customers: int = 2000, n_weeks: int = 12, seed: int = 42,
             schedule: bool = True) -> pd.DataFrame:
    """One snapshot row per customer per week, with the known drift
    schedule applied. Columns: week, date, customerID, the 19 Telco
    features, Churn. With schedule=False every week is clean baseline:
    the controlled injection experiments need drift-free weeks to
    inject into, and a drift-free run to measure false alarms on."""
    panel = _base_panel(n_customers, seed)
    rng = np.random.default_rng(seed + 1)
    base_tenure = panel["tenure"].to_numpy(dtype=float)
    base_monthly = panel["MonthlyCharges"].to_numpy(dtype=float)
    base_total = panel["TotalCharges"].to_numpy(dtype=float)
    internet = panel["InternetService"].to_numpy(dtype=object).copy()

    rows = []
    for week in range(1, n_weeks + 1):
        frame = panel.copy()
        # Tenure grows one month per week, capped at 72; in the stale
        # week it stays at the previous week's value.
        elapsed = week - 1 if (week != STALE_WEEK or not schedule) else STALE_WEEK - 2
        frame["tenure"] = np.minimum(base_tenure + elapsed, MAX_TENURE).astype(int)
        # MonthlyCharges: the price rise lands in week 5 and persists.
        rise = PRICE_RISE_DOLLARS if (schedule and week >= PRICE_RISE_WEEK) else 0.0
        monthly = base_monthly + rise
        frame["MonthlyCharges"] = monthly
        # Fiber growth: each week in the window, convert 5% of the
        # panel from other services to Fiber. Add-ons that said "No
        # internet service" become "No": the customer has internet now.
        if schedule and FIBER_GROWTH_WEEKS[0] <= week <= FIBER_GROWTH_WEEKS[1]:
            not_fiber = np.where(internet != "Fiber optic")[0]
            n_convert = int(round(FIBER_GROWTH_POINTS_PER_WEEK * n_customers))
            chosen = rng.choice(not_fiber, size=min(n_convert, len(not_fiber)),
                                replace=False)
            internet[chosen] = "Fiber optic"
            for addon in _ADDONS:
                col = frame[addon].to_numpy(dtype=object).copy()
                col[chosen] = np.where(col[chosen] == "No internet service",
                                       "No", col[chosen])
                frame[addon] = col
        frame["InternetService"] = internet
        # TotalCharges accumulates the billed months. Weeks 10 and 11
        # are the broken join: the column holds MonthlyCharges instead.
        if schedule and BREAKAGE_WEEKS[0] <= week <= BREAKAGE_WEEKS[1]:
            frame["TotalCharges"] = monthly
        else:
            risen_weeks = max(0, week - PRICE_RISE_WEEK) if schedule else 0
            frame["TotalCharges"] = (base_total + (week - 1) * base_monthly
                                     + PRICE_RISE_DOLLARS * risen_weeks)
        # Labels from the known rule, with the prevalence shift from
        # week 9. The shift changes the outcome rate, not the features.
        shift = (PREVALENCE_INTERCEPT_SHIFT
                 if (schedule and week >= PREVALENCE_SHIFT_WEEK) else 0.0)
        proba = true_churn_probability(frame, intercept_shift=shift)
        frame["Churn"] = (rng.random(n_customers) < proba).astype(int)
        frame["week"] = week
        frame["date"] = START_DATE + timedelta(days=7 * (week - 1))
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    cols = ["customerID", "week", "date"] + config.FEATURES + ["Churn"]
    return out[cols]


def inject_drift(frame: pd.DataFrame, drift_type: str, seed: int = 0,
                 severity: float = None, previous_frame: pd.DataFrame = None) -> pd.DataFrame:
    """Inject ONE drift type into a clean frame, for the controlled
    detection experiments. The clean frame is never modified in place.
    Types:
    - abrupt_price: MonthlyCharges += severity dollars (default 25).
    - gradual_fiber: convert severity fraction (default 0.25) of
      non-Fiber customers to Fiber.
    - prevalence: redraw labels with the intercept shifted +0.6, no
      feature change. The frame must have a Churn column.
    - breakage_total_charges: TotalCharges := MonthlyCharges.
    - null_storm: severity fraction (default 0.4) of TotalCharges
      becomes null (a broken feed, caught by the quality channel).
    - stale_tenure: tenure := previous week's tenure per customer
      (previous_frame required, joined on customerID).
    """
    out = frame.copy()
    rng = np.random.default_rng(seed)
    if drift_type == "abrupt_price":
        out["MonthlyCharges"] = out["MonthlyCharges"] + (severity or 25.0)
    elif drift_type == "gradual_fiber":
        frac = severity or 0.25
        mask = out["InternetService"] != "Fiber optic"
        candidates = out.index[mask]
        chosen = rng.choice(candidates, size=int(round(frac * len(out))),
                            replace=False)
        out.loc[chosen, "InternetService"] = "Fiber optic"
        for addon in _ADDONS:
            col = out.loc[chosen, addon]
            out.loc[chosen, addon] = col.where(col != "No internet service", "No")
    elif drift_type == "prevalence":
        shift = severity or PREVALENCE_INTERCEPT_SHIFT
        proba = true_churn_probability(out, intercept_shift=shift)
        out["Churn"] = (rng.random(len(out)) < proba).astype(int)
    elif drift_type == "breakage_total_charges":
        out["TotalCharges"] = out["MonthlyCharges"]
    elif drift_type == "null_storm":
        frac = severity or 0.4
        chosen = rng.choice(out.index, size=int(round(frac * len(out))),
                            replace=False)
        out.loc[chosen, "TotalCharges"] = np.nan
    elif drift_type == "stale_tenure":
        if previous_frame is None:
            raise ValueError("stale_tenure needs previous_frame")
        prev = previous_frame.set_index("customerID")["tenure"]
        out["tenure"] = out["customerID"].map(prev).fillna(out["tenure"]).astype(int)
    else:
        raise ValueError(f"unknown drift type: {drift_type}")
    return out

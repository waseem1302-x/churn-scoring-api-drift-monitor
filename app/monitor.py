"""Drift monitor: windowed drift detection, hand-rolled.

Four separate alert channels, never one blended score:
1. per-feature drift (PSI, KS, Jensen-Shannon, Wasserstein)
2. prediction drift (score distribution and predicted churn rate)
3. data quality (null rate, schema change, stale features, out-of-range)
4. realised performance on labelled windows (precision at k, ROC-AUC)

The statistics are implemented here, in this file, so the request path
never depends on a report library. Evidently is used only for scheduled
HTML reports (scripts/evidently_report.py), never for a live decision.

Thresholds: the starting points come from the research report (PSI
0.1/0.25, Jensen-Shannon 0.1, Wasserstein 0.1 normalised, KS p 0.05,
drift_share 0.5). The injection experiments in
scripts/drift_experiments.py measure detection delay and false alarms
per detector and justify the final values; see MONITORING.md. A
threshold is never moved to make an alert go away.

One deliberate rule: the KS test is reported for every numeric feature
but never drives the per-feature "drifted" verdict. KS is over-sensitive
on large samples (a tiny, irrelevant difference becomes significant),
so on windows of a thousand rows or more it would cry wolf. It is the
right test on small windows, and the experiment table measures it
separately. The verdict is driven by PSI, Jensen-Shannon, and
Wasserstein, which are effect sizes, not p-values.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from app import config, metrics as metrics_mod

# Starting-point thresholds from the research. The injection
# experiments justify the final values; see MONITORING.md.
DEFAULT_THRESHOLDS = {
    "psi_warn": 0.10,
    "psi_act": 0.25,
    "ks_p": 0.05,
    "jensen_shannon": 0.10,
    "wasserstein": 0.10,
    "drift_share": 0.5,
    "predicted_rate_rel_change": 0.10,
    "score_psi_act": 0.25,
    "null_rate_rise": 0.05,
    "unknown_category_frac": 0.01,
    "out_of_range_frac": 0.25,
    "stale_changed_frac": 0.01,
    "accumulation_broken_frac": 0.05,
    "precision_at_k_drop": 0.05,
    "pr_auc_drop": 0.03,
}

# Features whose distribution moves by design as time passes, so a
# distribution comparison against the fixed training reference is the
# wrong test for them: tenure grows one month per week for every
# customer, and TotalCharges accumulates the bills. Compared to
# training they "drift" forever, starting in week 2 (tenure PSI 0.43
# on a clean simulator week, measured). They are monitored
# week-over-week instead: the stale check below, which asks whether
# they moved since the previous window, not whether they match a
# snapshot from training. Every other feature is monitored against
# the training reference.
ACCUMULATING_FEATURES = ("tenure", "TotalCharges")

# Columns that legitimately appear in a scored window but are not
# model features: the row id, the label, and the window stamps. A
# schema change is only a change among the model features.
IGNORED_COLUMNS = ("customerID", "Churn", "week", "date")

# Features that must move every week for every active customer. Tenure
# grows by one month per week and TotalCharges accumulates, so a week
# where almost nobody's value changed means the feed stopped updating.
# This is also why they are the accumulating features above: they are
# monitored week-over-week, not against the training reference.
STALE_PRONE_FEATURES = ("tenure", "TotalCharges")

_EPS = 1e-4


def _proportions(values, categories):
    counts = pd.Series(values).value_counts(normalize=True)
    return np.array([counts.get(c, 0.0) for c in categories], dtype=float)


def _psi_from_proportions(expected_p, actual_p):
    expected_p = np.clip(expected_p, _EPS, None)
    actual_p = np.clip(actual_p, _EPS, None)
    expected_p = expected_p / expected_p.sum()
    actual_p = actual_p / actual_p.sum()
    return float(np.sum((actual_p - expected_p) * np.log(actual_p / expected_p)))


def _quantile_bins(series, n_bins=10):
    edges = np.quantile(series.dropna(), np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        lo, hi = float(series.min()), float(series.max())
        if lo == hi:
            return np.array([lo - 0.5, lo + 0.5])
        return np.linspace(lo, hi, n_bins + 1)
    return edges


def psi_numeric(expected, actual, n_bins=10) -> float:
    """Population Stability Index for a numeric feature. Quantile bins
    from the reference, so each bin holds about a tenth of it."""
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    edges = _quantile_bins(expected, n_bins)
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_p = e_hist / e_hist.sum()
    a_p = a_hist / a_hist.sum()
    return _psi_from_proportions(e_p, a_p)


def psi_categorical(expected, actual) -> float:
    """PSI for a categorical feature, over the union of categories."""
    categories = sorted(set(pd.Series(expected).dropna().unique())
                        | set(pd.Series(actual).dropna().unique()))
    e_p = _proportions(expected, categories)
    a_p = _proportions(actual, categories)
    return _psi_from_proportions(e_p, a_p)


def ks_test(ref, cur):
    """Two-sample Kolmogorov-Smirnov. Returns (statistic, p-value)."""
    result = stats.ks_2samp(pd.Series(ref).dropna(), pd.Series(cur).dropna())
    return float(result.statistic), float(result.pvalue)


def jensen_shannon_categorical(expected, actual) -> float:
    """Jensen-Shannon distance (0 identical, 1 disjoint) for categories."""
    categories = sorted(set(pd.Series(expected).dropna().unique())
                        | set(pd.Series(actual).dropna().unique()))
    e_p = _proportions(expected, categories)
    a_p = _proportions(actual, categories)
    return float(jensenshannon(e_p, a_p))


def jensen_shannon_numeric(expected, actual, n_bins=10) -> float:
    """Jensen-Shannon distance for a numeric feature, on the same
    quantile bins PSI uses, so the two measures are comparable."""
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    edges = _quantile_bins(expected, n_bins)
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_p = e_hist / max(e_hist.sum(), 1)
    a_p = a_hist / max(a_hist.sum(), 1)
    return float(jensenshannon(e_p, a_p))


def wasserstein_normed(ref, cur) -> float:
    """Wasserstein distance divided by the reference standard deviation,
    so the 0.1 threshold means the same thing on every feature. A
    constant reference has no scale: identical scores 0, anything else
    scores 1 (a flag value, documented here)."""
    ref = pd.Series(ref).dropna()
    cur = pd.Series(cur).dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    scale = float(ref.std())
    if scale == 0:
        return 0.0 if bool((cur == ref.iloc[0]).all()) else 1.0
    return float(stats.wasserstein_distance(ref, cur) / scale)


def feature_drift_numeric(name, ref, cur, thresholds=None) -> dict:
    """All four detectors on one numeric feature. The verdict is driven
    by the three effect sizes; KS is reported but never verdict-driving
    (see the module docstring)."""
    th = thresholds or DEFAULT_THRESHOLDS
    psi = psi_numeric(ref, cur)
    ks_stat, ks_p = ks_test(ref, cur)
    js = jensen_shannon_numeric(ref, cur)
    wass = wasserstein_normed(ref, cur)
    return {
        "feature": name,
        "type": "numeric",
        "psi": round(psi, 4),
        "psi_drifted": bool(psi > th["psi_act"]),
        "ks_stat": round(ks_stat, 4),
        "ks_p": round(ks_p, 4),
        "ks_drifted": bool(ks_p < th["ks_p"]),
        "jensen_shannon": round(js, 4),
        "js_drifted": bool(js > th["jensen_shannon"]),
        "wasserstein": round(wass, 4),
        "wasserstein_drifted": bool(wass > th["wasserstein"]),
        "drifted": bool(psi > th["psi_act"] or js > th["jensen_shannon"]
                        or wass > th["wasserstein"]),
    }


def feature_drift_categorical(name, ref, cur, thresholds=None) -> dict:
    """PSI and Jensen-Shannon on one categorical feature. KS does not
    apply to categories."""
    th = thresholds or DEFAULT_THRESHOLDS
    psi = psi_categorical(ref, cur)
    js = jensen_shannon_categorical(ref, cur)
    return {
        "feature": name,
        "type": "categorical",
        "psi": round(psi, 4),
        "psi_drifted": bool(psi > th["psi_act"]),
        "jensen_shannon": round(js, 4),
        "js_drifted": bool(js > th["jensen_shannon"]),
        "drifted": bool(psi > th["psi_act"] or js > th["jensen_shannon"]),
    }


def prediction_drift(ref_scores, cur_scores, threshold, thresholds=None) -> dict:
    """Score distribution drift (PSI on the scores, KS) and the predicted
    churn rate move. The rate is the fraction of scores at or above the
    operating threshold. A rate move is relative: a 10% relative move on
    a 26% base rate is about 2.6 points."""
    th = thresholds or DEFAULT_THRESHOLDS
    ref_scores = np.asarray(ref_scores, dtype=float)
    cur_scores = np.asarray(cur_scores, dtype=float)
    ref_rate = float((ref_scores >= threshold).mean())
    cur_rate = float((cur_scores >= threshold).mean())
    rel = (cur_rate - ref_rate) / ref_rate if ref_rate > 0 else 0.0
    # Score PSI on ten equal-width bins over [0, 1].
    edges = np.linspace(0, 1, 11)
    e_hist, _ = np.histogram(ref_scores, bins=edges)
    a_hist, _ = np.histogram(cur_scores, bins=edges)
    score_psi = _psi_from_proportions(e_hist / max(e_hist.sum(), 1),
                                      a_hist / max(a_hist.sum(), 1))
    ks_stat, ks_p = stats.ks_2samp(ref_scores, cur_scores)
    return {
        "score_psi": round(float(score_psi), 4),
        "score_psi_drifted": bool(score_psi > th["score_psi_act"]),
        "ks_stat": round(float(ks_stat), 4),
        "ks_p": round(float(ks_p), 4),
        "ref_predicted_rate": round(ref_rate, 4),
        "cur_predicted_rate": round(cur_rate, 4),
        "predicted_rate_change_rel": round(float(rel), 4),
        "rate_drifted": bool(abs(rel) > th["predicted_rate_rel_change"]),
        "drifted": bool(score_psi > th["score_psi_act"]
                        or abs(rel) > th["predicted_rate_rel_change"]),
    }


def data_quality_issues(ref, cur, previous_frame=None, id_column=None,
                        thresholds=None) -> list:
    """Data-quality gates. Each issue is a dict with feature, kind, and a
    measured detail. Kinds: schema_change, null_rate, unknown_category,
    out_of_range, stale_feature. Drift tests drop nulls before comparing,
    so nulls are monitored here, never inside the drift statistics."""
    th = thresholds or DEFAULT_THRESHOLDS
    issues = []
    ref_cols = set(ref.columns)
    cur_cols = set(cur.columns)
    for col in sorted((ref_cols - cur_cols) - set(IGNORED_COLUMNS)):
        issues.append({"feature": col, "kind": "schema_change",
                       "detail": "column present in reference, missing in window"})
    for col in sorted((cur_cols - ref_cols) - set(IGNORED_COLUMNS)):
        issues.append({"feature": col, "kind": "schema_change",
                       "detail": "column present in window, not in reference"})
    for col in sorted(ref_cols & cur_cols):
        ref_null = float(ref[col].isna().mean())
        cur_null = float(cur[col].isna().mean())
        if cur_null - ref_null > th["null_rate_rise"]:
            issues.append({
                "feature": col, "kind": "null_rate",
                "detail": f"null rate {ref_null:.3f} in reference, "
                          f"{cur_null:.3f} in window",
                "ref_null_rate": round(ref_null, 4),
                "cur_null_rate": round(cur_null, 4),
            })
        # pandas 3.x stores plain strings as dtype "str", not object,
        # so the categorical check is by kind, not by dtype == object.
        if pd.api.types.is_string_dtype(ref[col]) or str(ref[col].dtype) == "category":
            ref_cats = set(ref[col].dropna().unique())
            cur_vals = cur[col].dropna()
            unknown = cur_vals[~cur_vals.isin(ref_cats)]
            frac = float(len(unknown) / max(len(cur_vals), 1))
            if frac > th["unknown_category_frac"]:
                issues.append({
                    "feature": col, "kind": "unknown_category",
                    "detail": f"{frac:.3f} of values are categories the "
                              f"reference never saw: "
                              f"{sorted(set(unknown.unique()))[:5]}",
                    "unknown_frac": round(frac, 4),
                })
        if pd.api.types.is_numeric_dtype(ref[col]):
            lo, hi = float(ref[col].min()), float(ref[col].max())
            outside = cur[col].dropna()
            frac = float(((outside < lo) | (outside > hi)).mean()) if len(outside) else 0.0
            if frac > th["out_of_range_frac"]:
                issues.append({
                    "feature": col, "kind": "out_of_range",
                    "detail": f"{frac:.3f} of values outside the reference "
                              f"range [{lo:.2f}, {hi:.2f}]",
                    "out_of_range_frac": round(frac, 4),
                })
    if previous_frame is not None and id_column is not None:
        merged = cur.merge(previous_frame, on=id_column, suffixes=("_cur", "_prev"))
        for col in STALE_PRONE_FEATURES:
            cur_col, prev_col = f"{col}_cur", f"{col}_prev"
            if cur_col not in merged.columns or prev_col not in merged.columns:
                continue
            changed = (merged[cur_col] != merged[prev_col]).mean()
            if float(changed) < th["stale_changed_frac"]:
                issues.append({
                    "feature": col, "kind": "stale_feature",
                    "detail": f"only {float(changed):.3f} of customers "
                              f"changed {col} since the previous window; "
                              f"the feature stopped updating",
                    "changed_frac": round(float(changed), 4),
                })
            # An accumulating total must never go down. Tenure grows by
            # one a week and TotalCharges grows by the bill, so a week
            # where the value fell for many customers means the feed
            # broke (a bad join, a unit change), even though the
            # distribution still looks plausible. This is the check
            # that catches a TotalCharges breakage: distribution drift
            # is the wrong test for an accumulating feature, but its
            # growth law is exact.
            both = merged[[cur_col, prev_col]].dropna()
            if len(both):
                decreased = float((both[cur_col] < both[prev_col] - 1e-9).mean())
                if decreased > th["accumulation_broken_frac"]:
                    issues.append({
                        "feature": col, "kind": "accumulation_broken",
                        "detail": f"{decreased:.3f} of customers saw {col} "
                                  f"fall since the previous window; an "
                                  f"accumulating feature never falls",
                        "decreased_frac": round(decreased, 4),
                    })
    return issues


def top_features_by_permutation(model, X, y, n=5, n_repeats=5):
    """Top-n features by permutation importance on the raw feature frame.
    The champion pipeline is scored as-is, so the importances are for
    the shipped model, not a retrained copy. Returns feature names."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, X, y, n_repeats=n_repeats,
                                    random_state=config.RANDOM_STATE, n_jobs=1)
    order = np.argsort(result.importances_mean)[::-1][:n]
    return [config.FEATURES[i] for i in order]


def row_quality_flags(inputs, reference):
    """Per-row data-quality flags against a reference frame.

    Returns {"null_total_charges": bool, "out_of_range": [features]}.
    A null TotalCharges is legitimate for a brand-new customer (tenure
    0), so it is flagged as information, not as an alert by itself.
    A feature outside the reference min/max is out of range.
    """
    flags = {"null_total_charges": False, "out_of_range": []}
    if inputs.get("TotalCharges") is None:
        flags["null_total_charges"] = True
    for col in config.NUMERIC_FEATURES:
        value = inputs.get(col)
        if value is None:
            continue
        lo = float(reference[col].min())
        hi = float(reference[col].max())
        if not (lo <= float(value) <= hi):
            flags["out_of_range"].append(col)
    return flags


def evaluate_window(ref, cur, ref_scores, cur_scores, threshold,
                    thresholds=None, y_true=None, y_proba=None,
                    previous_frame=None, id_column=None,
                    reference_metrics=None) -> dict:
    """One window, all four channels. ref and cur are feature frames on
    the Telco schema; scores are the champion's probabilities on each.
    Labels, when given, add the realised-performance channel."""
    th = thresholds or DEFAULT_THRESHOLDS
    feature_results = {}
    for feature in config.FEATURES:
        if feature not in ref.columns or feature not in cur.columns:
            continue
        if feature in ACCUMULATING_FEATURES:
            # Distribution drift is the wrong test here: the feature
            # moves by design (see the constant above). It is monitored
            # week-over-week by the stale check, not against training.
            feature_results[feature] = {
                "feature": feature,
                "type": "numeric",
                "skipped": "accumulating",
                "reason": "distribution moves by design; monitored "
                          "week-over-week by the stale check",
                "drifted": False,
            }
            continue
        if feature in config.NUMERIC_FEATURES:
            feature_results[feature] = feature_drift_numeric(
                feature, ref[feature], cur[feature], th)
        else:
            feature_results[feature] = feature_drift_categorical(
                feature, ref[feature], cur[feature], th)
    monitored = [r for r in feature_results.values() if "skipped" not in r]
    n_evaluated = len(monitored)
    n_drifted = sum(1 for r in monitored if r["drifted"])
    drift_share = n_drifted / n_evaluated if n_evaluated else 0.0
    pred = prediction_drift(ref_scores, cur_scores, threshold, th)
    quality = data_quality_issues(ref, cur, previous_frame=previous_frame,
                                  id_column=id_column, thresholds=th)
    realised = None
    perf_drop = False
    label_shift = False
    if y_true is not None and y_proba is not None:
        realised = metrics_mod.evaluate_model(y_true, y_proba, threshold=threshold)
        if reference_metrics is not None:
            p100_drop = (reference_metrics.get("precision_at_100", 0)
                         - realised["precision_at_100"])
            pr_drop = (reference_metrics.get("pr_auc", 0)
                       - realised["pr_auc"])
            perf_drop = bool(p100_drop > th["precision_at_k_drop"]
                             or pr_drop > th["pr_auc_drop"])
            realised["precision_at_100_drop"] = round(float(p100_drop), 4)
            realised["pr_auc_drop"] = round(float(pr_drop), 4)
            # Label shift: the realised churn rate moved against the
            # reference rate. Input detectors cannot see this at all:
            # the features did not change, only the outcome rate did.
            ref_rate = reference_metrics.get("churn_rate")
            if ref_rate:
                rel = (realised["churn_rate"] - ref_rate) / ref_rate
                realised["churn_rate_change_rel"] = round(float(rel), 4)
                label_shift = bool(
                    abs(rel) > th["predicted_rate_rel_change"])
    return {
        "n_ref": len(ref),
        "n_cur": len(cur),
        "feature_drift": feature_results,
        "drift_share": round(float(drift_share), 4),
        "prediction_drift": pred,
        "data_quality": quality,
        "realised_performance": realised,
        "alerts": {
            "feature_drift": bool(drift_share > th["drift_share"]),
            "prediction_drift": bool(pred["drifted"]),
            "data_quality": bool(quality),
            "performance_drop": perf_drop,
            "label_shift": label_shift,
        },
    }

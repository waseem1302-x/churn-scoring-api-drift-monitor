"""Tests on the drift injection experiment results.

These tests read benchmarks/results/drift_experiments.json, written by
scripts/drift_experiments.py. They assert the properties the Phase 2
exit criteria require: every injected drift type is caught by its
right detector inside the horizon, clean windows do not alert, and
the thresholds in use are the ones the measurements justify. If the
JSON is missing, the message says to run the script first.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "drift_experiments.json"


@pytest.fixture(scope="module")
def experiments():
    if not RESULTS.exists():
        pytest.fail("drift_experiments.json missing: run scripts/drift_experiments.py first")
    return json.loads(RESULTS.read_text())


def test_all_six_drift_types_measured(experiments):
    expected = {"abrupt_price", "gradual_fiber", "prevalence",
                "breakage_total_charges", "null_storm", "stale_tenure"}
    assert set(experiments["injections"]) == expected


def test_abrupt_price_detected_in_window_zero(experiments):
    delays = experiments["injections"]["abrupt_price"]["delays"]
    # PSI, Jensen-Shannon and Wasserstein must fire in the injected
    # window itself. KS is allowed to be the odd one out: it is
    # reported, not verdict-driving, and its p-value behaviour on
    # large windows is measured, not assumed.
    assert delays["psi"] == 0
    assert delays["jensen_shannon"] == 0
    assert delays["wasserstein"] == 0


def test_gradual_fiber_detected_before_the_end(experiments):
    delays = experiments["injections"]["gradual_fiber"]["delays"]
    assert delays["psi"] is not None and delays["psi"] <= 4


def test_prevalence_invisible_to_input_detectors(experiments):
    info = experiments["injections"]["prevalence"]
    assert info["input_detectors_fired"] is False


def test_prevalence_caught_by_label_shift(experiments):
    info = experiments["injections"]["prevalence"]
    assert info["label_shift_flagged"] is True


def test_breakage_detected_by_growth_check_in_window_zero(experiments):
    # TotalCharges is an accumulating feature, so distribution drift is
    # the wrong test for it. The breakage is caught by the growth law
    # instead: an accumulating total never falls, and here it falls for
    # nearly every customer.
    info = experiments["injections"]["breakage_total_charges"]
    assert info["detection_channel"] == "data_quality"
    assert info["quality_flagged"] is True
    kinds = [i["kind"] for i in info["quality_issues"]]
    assert "accumulation_broken" in kinds


def test_null_storm_flagged_by_quality_channel(experiments):
    info = experiments["injections"]["null_storm"]
    assert info["quality_flagged"] is True
    kinds = [i["kind"] for i in info["quality_issues"]]
    assert "null_rate" in kinds


def test_stale_tenure_flagged_by_quality_channel(experiments):
    info = experiments["injections"]["stale_tenure"]
    assert info["quality_flagged"] is True
    kinds = [i["kind"] for i in info["quality_issues"]]
    assert "stale_feature" in kinds
    # And it must not look like feature drift: the distribution did
    # not move, only the feed stopped.
    assert info["feature_drifted"] is False


def test_clean_windows_do_not_alert_on_effect_sizes(experiments):
    # PSI, Jensen-Shannon and Wasserstein must not flag any clean
    # window. KS is excluded on purpose: on 2,000-row windows it fires
    # on noise, which the experiment table records and MONITORING.md
    # explains. A clean window that alerts on an effect size means the
    # threshold is wrong, not the data.
    false_alarms = experiments["clean_windows"]["false_alarms_per_detector"]
    for name in ("psi", "jensen_shannon", "wasserstein"):
        assert false_alarms[name]["windows_flagged"] == 0, name


def test_clean_window_drift_share_stays_low(experiments):
    for share in experiments["clean_windows"]["drift_share_by_week"]:
        assert share < 0.5


def test_thresholds_recorded_match_monitor_defaults(experiments):
    from app import monitor

    assert experiments["thresholds_used"] == monitor.DEFAULT_THRESHOLDS

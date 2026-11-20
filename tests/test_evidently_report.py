"""Evidently report tests: the committed HTML artifacts exist and are
real Evidently reports. The script itself is slow (about 15 seconds
per report), so these tests check the artifacts it produced, not
re-run it; the CI workflow runs the script and fails if it errors.
"""

import json
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"


def test_clean_report_exists_and_is_a_drift_report():
    path = REPORTS_DIR / "drift_week02_clean.html"
    assert path.exists()
    assert path.stat().st_size > 100_000
    text = path.read_text(errors="ignore")
    assert "Data Drift" in text
    assert "MonthlyCharges" in text


def test_price_rise_report_exists_and_is_a_drift_report():
    path = REPORTS_DIR / "drift_week05_price_rise.html"
    assert path.exists()
    assert path.stat().st_size > 100_000
    text = path.read_text(errors="ignore")
    assert "Data Drift" in text
    assert "MonthlyCharges" in text


def test_report_summary_matches_files():
    summary = json.loads(
        (REPORTS_DIR / "evidently_reports.json").read_text())
    assert set(summary) == {"week02_clean", "week05_price_rise"}
    for label, info in summary.items():
        path = Path(info["path"])
        assert path.exists()
        assert path.stat().st_size == info["bytes"]
        assert info["rows"] == 2000

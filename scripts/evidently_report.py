"""Scheduled Evidently drift reports, as HTML artifacts.

Evidently is never on the request path and never makes a live
decision. The hand-rolled monitor in app/monitor.py does that. This
script renders the scheduled reports a human reads: one clean week
and one drifted week, both from the simulator, so the contrast is
visible in the same format.

Reference: the Phase 1 training split (5,634 rows). Current windows:
simulator week 2 (clean) and simulator week 5 (the abrupt price
rise, MonthlyCharges +$25). The reports use Evidently 0.7.23 on
pandas 2.3.3; see ARCHITECTURE.md for why pandas is pinned.

Run: python -m scripts.evidently_report
Output: benchmarks/reports/drift_week02_clean.html and
benchmarks/reports/drift_week05_price_rise.html
"""

import json
from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from app import config, data
from scripts import simulate

REPORTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "reports"


def build_report(reference, current):
    """One DataDriftPreset report over two frames on the Telco schema.
    Evidently infers column types; the frames carry the 19 model
    features only, no ids, labels, or simulator bookkeeping columns."""
    definition = DataDefinition()
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(reference, data_definition=definition),
        Dataset.from_pandas(current, data_definition=definition),
    )
    return snapshot


def main():
    df = data.load_telco(data.DATA_PATH)
    X_train, _, _, _ = data.split(df)
    reference = X_train.drop(columns=[config.ID_COLUMN])

    panel = simulate.simulate(n_customers=2000, n_weeks=5, seed=42)
    weeks = {}
    for week, label in ((2, "week02_clean"), (5, "week05_price_rise")):
        frame = panel[panel["week"] == week].drop(
            columns=["customerID", "Churn", "week", "date"])
        weeks[label] = frame[config.FEATURES]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for label, frame in weeks.items():
        snapshot = build_report(reference, frame)
        path = REPORTS_DIR / f"drift_{label}.html"
        snapshot.save_html(str(path))
        summary[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
        }
        print(f"Wrote {path} ({path.stat().st_size:,} bytes, "
              f"{len(frame)} rows)")
    (REPORTS_DIR / "evidently_reports.json").write_text(
        json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()

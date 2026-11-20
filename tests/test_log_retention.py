"""Log retention: archive old prediction rows to a CSV, then delete
them. The log grows forever without this, and the archive keeps the
record: nothing is deleted before it is written to the CSV.
"""

import csv
import sqlite3
from pathlib import Path

import pytest

from app import db as db_mod

ROOT = Path(__file__).resolve().parents[1]


def _seed(path):
    db_mod.init_db(path)
    rows = [
        ("2026-08-01T00:00:00", "OLD-1", 0.9, "HIGH", 1.0),
        ("2026-08-15T00:00:00", "OLD-2", 0.2, "LOW", 0.0),
        ("2026-11-19T00:00:00", "NEW-1", 0.5, "LOW", None),
        ("2026-11-20T00:00:00", "NEW-2", 0.7, "MEDIUM", None),
    ]
    for ts, customer, prob, tier, label in rows:
        row_id = db_mod.log_prediction(
            path, ts, customer, {"tenure": 12}, prob, tier, "v1", 1.0)
        if label is not None:
            db_mod.set_label(path, customer, label)
    return rows


def test_archive_deletes_only_old_rows_and_keeps_the_record(tmp_path):
    path = str(tmp_path / "log.db")
    _seed(path)
    archive = tmp_path / "archive.csv"
    result = db_mod.archive_old_predictions(
        path, before="2026-09-01T00:00:00", archive_path=str(archive))
    assert result == {"archived": 2, "deleted": 2,
                      "archive_file": str(archive)}
    remaining = sqlite3.connect(path).execute(
        "SELECT customer_id FROM predictions ORDER BY id").fetchall()
    assert remaining == [("NEW-1",), ("NEW-2",)]
    with open(archive, newline="") as f:
        archived = list(csv.DictReader(f))
    assert [r["customer_id"] for r in archived] == ["OLD-1", "OLD-2"]
    assert archived[0]["label"] == "1.0"
    assert archived[1]["label"] == "0.0"
    assert archived[0]["probability"] == "0.9"
    assert archived[0]["inputs_json"] == '{"tenure": 12}'


def test_archive_with_nothing_old_writes_header_only(tmp_path):
    path = str(tmp_path / "log.db")
    _seed(path)
    archive = tmp_path / "archive.csv"
    result = db_mod.archive_old_predictions(
        path, before="2026-01-01T00:00:00", archive_path=str(archive))
    assert result["archived"] == 0
    assert result["deleted"] == 0
    with open(archive, newline="") as f:
        assert list(csv.DictReader(f)) == []
    assert sqlite3.connect(path).execute(
        "SELECT COUNT(*) FROM predictions").fetchone()[0] == 4


def test_dry_run_deletes_nothing(tmp_path):
    path = str(tmp_path / "log.db")
    _seed(path)
    archive = tmp_path / "archive.csv"
    result = db_mod.archive_old_predictions(
        path, before="2026-09-01T00:00:00", archive_path=str(archive),
        dry_run=True)
    assert result == {"archived": 2, "deleted": 0,
                      "archive_file": str(archive)}
    assert sqlite3.connect(path).execute(
        "SELECT COUNT(*) FROM predictions").fetchone()[0] == 4


def test_docs_retention_section():
    text = (ROOT / "MONITORING.md").read_text()
    assert "prune_log" in text
    assert "archive" in text.lower()

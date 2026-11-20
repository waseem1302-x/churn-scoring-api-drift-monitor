#!/usr/bin/env python3
"""Prune the prediction log: archive old rows to a CSV, then delete
them. The log grows forever without this, and the archive keeps the
record: the CSV is written before anything is deleted.

Usage:
    python -m scripts.prune_log --db data/predictions.db \
        --before 2026-08-01T00:00:00
    python -m scripts.prune_log --db data/predictions.db \
        --before 2026-08-01T00:00:00 --archive data/archive/old.csv
    python -m scripts.prune_log --db data/predictions.db \
        --before 2026-08-01T00:00:00 --dry-run

No default retention is enforced: the operator picks the cutoff.
A 90-day keep is a reasonable starting point for a log scored
daily, but it is a policy choice, not a coded default.
"""

import argparse
from datetime import datetime, timezone

from app import db


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True,
                        help="prediction log database")
    parser.add_argument("--before", required=True,
                        help="archive rows with timestamp before this "
                             "ISO timestamp, e.g. 2026-08-01T00:00:00")
    parser.add_argument("--archive", default=None,
                        help="archive CSV path (default: "
                             "data/archive/predictions-<date>.csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the archive CSV but delete nothing")
    args = parser.parse_args()
    archive = args.archive
    if archive is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        archive = f"data/archive/predictions-{stamp}.csv"
    result = db.archive_old_predictions(
        args.db, before=args.before, archive_path=archive,
        dry_run=args.dry_run)
    print(f"archived {result['archived']} rows to "
          f"{result['archive_file']}, deleted {result['deleted']}")
    if args.dry_run:
        print("dry run: nothing was deleted")


if __name__ == "__main__":
    main()

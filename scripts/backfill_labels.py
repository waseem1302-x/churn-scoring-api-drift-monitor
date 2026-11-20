#!/usr/bin/env python3
"""Backfill labels from a labelled CSV into the prediction log.

Labels arrive about 30 days after the score, so the log is filled
in late: this script reads a CSV with one row per customer
(customer id and label) and writes the label onto the most recent
unlabelled prediction for that customer. Only the latest row is
labelled, so one customer counts once in the realised metrics on
/drift. A label is never invented: customers with no unlabelled
row and unreadable labels are counted and left alone.

Usage:
    python -m scripts.backfill_labels --db data/predictions.db \
        --csv labels.csv
    python -m scripts.backfill_labels --db data/predictions.db \
        --csv labels.csv --customer-column CustomerId \
        --label-column Exited

Labels accepted: 0/1, Yes/No, True/False (any case).
"""

import argparse

from app import db


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True,
                        help="prediction log database")
    parser.add_argument("--csv", required=True,
                        help="labelled CSV file")
    parser.add_argument("--customer-column", default="customer_id",
                        help="CSV column with the customer id")
    parser.add_argument("--label-column", default="label",
                        help="CSV column with the label")
    args = parser.parse_args()
    result = db.backfill_labels(args.db, args.csv,
                                customer_column=args.customer_column,
                                label_column=args.label_column)
    print(f"matched {result['matched']}, unmatched "
          f"{result['unmatched']}, invalid {result['invalid']}")
    print("Run GET /drift afterwards: realised_performance appears "
          "whenever labelled rows exist.")


if __name__ == "__main__":
    main()

"""Promote a challenger model to the served model.

Usage:
    python -m scripts.promote --artifact challenger.joblib \\
        --version churn-xgboost-smote-isotonic-v1-challenger-1 \\
        --threshold 0.0934 --source TRG-08-1
    python -m scripts.promote --list

The challenger file is a joblib bundle like the Phase 1 artifact:
{"model": ..., "metadata": {"model_version": ..., "threshold": ...}}.
The command saves it as a versioned artifact under artifacts/ and
moves the current-model pointer to it. The Phase 1 artifact
(artifacts/model.joblib) is never touched. The API serves the new
model after POST /reload (API key required); a restart serves it
too, since the pointer is read at startup.

--list prints the pointer and the versioned artifacts present.
"""

import argparse

import joblib

from app import model_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", help="joblib bundle to promote")
    parser.add_argument("--version", help="model version of the challenger")
    parser.add_argument("--threshold", type=float,
                        help="operating threshold for the challenger")
    parser.add_argument("--source", default="manual",
                        help="what decided the promotion, e.g. TRG-08-1")
    parser.add_argument("--list", action="store_true",
                        help="show the current model and artifacts")
    args = parser.parse_args()

    if args.list:
        pointer = model_store.read_pointer()
        print(f"current: {pointer or 'no pointer, serving model.joblib'}")
        for name in model_store.list_artifacts():
            print(f"  {name}")
        return

    if not args.artifact or not args.version or args.threshold is None:
        parser.error("--artifact, --version and --threshold are required")
    bundle = joblib.load(args.artifact)
    model = bundle["model"]
    meta = bundle.get("metadata", {})
    if meta.get("model_version") and meta["model_version"] != args.version:
        parser.error(
            f"bundle version {meta['model_version']!r} does not match "
            f"--version {args.version!r}")
    if meta.get("threshold") is not None \
            and meta["threshold"] != args.threshold:
        parser.error(
            f"bundle threshold {meta['threshold']} does not match "
            f"--threshold {args.threshold}")
    saved = model_store.save_promotion(
        model, args.version, args.threshold, source=args.source)
    print(f"Promoted {args.version} to {saved}")
    print("Call POST /reload with the API key to serve it now.")


if __name__ == "__main__":
    main()

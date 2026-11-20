"""Versioned model artifacts and the current-model pointer.

The Phase 1 artifact (artifacts/model.joblib) is never overwritten.
A promotion saves the challenger as a new versioned file,
artifacts/model-<version>.joblib, and moves the pointer
(artifacts/current_model.json) to it. The API serves whatever the
pointer says; with no pointer it serves model.joblib, so a checkout
from before this module still works.

The pointer is written atomically (temp file plus rename), so a
crash mid-write cannot leave a half-written pointer. The API reads
it at reload time; see POST /reload in app/api.py.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib

from app import config

ARTIFACTS_DIR = config.ROOT / "artifacts"
POINTER_PATH = ARTIFACTS_DIR / "current_model.json"
DEFAULT_ARTIFACT = "model.joblib"

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _versioned_name(version: str) -> str:
    """model-<version>.joblib. Versions with path separators or other
    odd characters are rejected: the version becomes a filename."""
    if not _VERSION_RE.match(version):
        raise ValueError(
            f"bad model version for a filename: {version!r}")
    return f"model-{version}.joblib"


def read_pointer() -> dict | None:
    """The current pointer, or None when no promotion has happened."""
    if not POINTER_PATH.exists():
        return None
    try:
        return json.loads(POINTER_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"corrupt model pointer {POINTER_PATH}: {exc}")


def resolve_current_artifact() -> Path:
    """The artifact file the API should serve now."""
    pointer = read_pointer()
    if pointer is None:
        return ARTIFACTS_DIR / DEFAULT_ARTIFACT
    name = pointer.get("model_file")
    if not name or Path(name).name != name:
        raise ValueError(f"bad model_file in pointer: {name!r}")
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"pointer names {name}, but the file is missing")
    return path


def save_promotion(model, version: str, threshold: float,
                   source: str = "manual") -> Path:
    """Save a promoted challenger as a versioned artifact and move
    the pointer to it. Never overwrites model.joblib, and refuses to
    overwrite an existing versioned file: a version names one model.
    Returns the saved artifact path."""
    name = _versioned_name(version)
    if name == DEFAULT_ARTIFACT:
        raise ValueError("a promotion must not overwrite model.joblib")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / name
    if path.exists():
        raise FileExistsError(f"artifact {name} already exists")
    joblib.dump({"model": model,
                 "metadata": {"model_version": version,
                              "threshold": threshold}}, path)
    pointer = {
        "model_file": name,
        "model_version": version,
        "threshold": threshold,
        "source": source,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = POINTER_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pointer, indent=2))
    os.replace(tmp, POINTER_PATH)
    return path


def list_artifacts() -> list:
    """Versioned artifact files present, newest first by name."""
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted((p.name for p in ARTIFACTS_DIR.glob("model-*.joblib")),
                  reverse=True)

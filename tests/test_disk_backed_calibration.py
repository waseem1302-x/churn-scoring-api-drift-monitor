"""Disk-backed calibration tests.

Configs 17 and 18 of the S6E3 sweep (random forest + SMOTE + Platt /
isotonic) died twice on Colab with exit -9. The cause is memory:
SMOTE grows the 475k-row train set to about 736k rows, and
CalibratedClassifierCV (cv=3) holds three fitted 300-tree forests
in RAM at once, about 15 GB against Colab's 12 GB. The n_jobs=1
fix did not save them; it only removed the parallel workers.

The fix keeps the maths identical and moves the storage: each
calibration fold's fitted pipeline is dumped to disk the moment it
is fitted, freed from RAM, and loaded back one at a time for
prediction. Same three forests, same splits, same averaging, so
the predictions are identical to the standard path. Only the
storage changes.

TDD: these tests fail until scripts/disk_backed_calibration.py
exists. The identity tests run on a small Telco slice so they fit
comfortably on this VM; the memory test runs both paths in separate
subprocesses on a 60k-row S6E3 slice and compares peak RSS.
"""

import gc
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict

from app import config, data, features
from scripts.disk_backed_calibration import DiskBackedCalibratedClassifier

ROOT = Path(__file__).resolve().parents[1]

# Small enough to fit on this VM, big enough that the standard and
# disk-backed paths both do real work.
TELCO_ROWS = 2000


def _telco_split():
    df = data.load_telco().sample(n=TELCO_ROWS, random_state=0)
    X_train, X_test, y_train, y_test = data.split(df)
    drop = [config.ID_COLUMN]
    return (X_train.drop(columns=drop), X_test.drop(columns=drop),
            y_train, y_test)


def _raw_pipe():
    return features.build_pipeline(
        "random_forest", handling="smote", calibration=None)


def _standard(pipe, method):
    return CalibratedClassifierCV(pipe, method=method, cv=3)


def _disk(pipe, method, tmp_root=None):
    return DiskBackedCalibratedClassifier(
        pipe, method=method, cv=3, tmp_root=tmp_root)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_disk_backed_matches_standard(method):
    """Same fits, same splits, same averaging: identical probas."""
    X_train, X_test, y_train, _ = _telco_split()
    std = _standard(_raw_pipe(), method)
    std.fit(X_train, y_train)
    disk = _disk(_raw_pipe(), method)
    try:
        disk.fit(X_train, y_train)
        got = disk.predict_proba(X_test)
        got_pred = disk.predict(X_test)
    finally:
        disk.close()
    want = std.predict_proba(X_test)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    assert (got_pred == std.predict(X_test)).all()


def test_disk_backed_matches_standard_via_cross_val_predict():
    """The clone path cross_val_predict takes must also be identical."""
    X_train, _, y_train, _ = _telco_split()
    std_oof = cross_val_predict(
        _standard(_raw_pipe(), "sigmoid"), X_train, y_train,
        cv=3, method="predict_proba")[:, 1]
    disk = _disk(_raw_pipe(), "sigmoid")
    try:
        disk_oof = cross_val_predict(
            disk, X_train, y_train, cv=3, method="predict_proba")[:, 1]
    finally:
        disk.close()
    np.testing.assert_allclose(disk_oof, std_oof, rtol=0, atol=1e-12)


def test_close_removes_temp_files_and_predict_then_fails():
    X_train, X_test, y_train, _ = _telco_split()
    disk = _disk(_raw_pipe(), "sigmoid")
    disk.fit(X_train, y_train)
    tmpdir = Path(disk._tmpdir)
    assert tmpdir.is_dir()
    assert len(list(tmpdir.glob("calibrator_*.joblib"))) == 3
    disk.close()
    assert not tmpdir.exists()
    with pytest.raises(RuntimeError):
        disk.predict_proba(X_test)


def test_cross_val_predict_leaves_no_temp_dirs():
    """Regression: cross_val_predict clones the estimator per fold
    and never calls close() on the clones. Their temp dirs must
    still vanish when the clones are discarded, or they leak: on
    this VM the leak filled /tmp and failed three tests with "No
    space left", and on Colab it would leak about 9 calibrator
    files per config onto the 108 GB disk."""
    import tempfile
    parent = Path(tempfile.gettempdir())
    before = set(parent.glob("diskcal_*"))
    X_train, _, y_train, _ = _telco_split()
    disk = _disk(_raw_pipe(), "sigmoid")
    cross_val_predict(disk, X_train, y_train, cv=3, method="predict_proba")
    disk.close()
    gc.collect()
    after = set(parent.glob("diskcal_*"))
    assert after <= before, f"leaked temp dirs: {sorted(after - before)}"


def test_sweep_model_selection():
    """Only RF + SMOTE + calibration takes the disk-backed path."""
    from scripts.sweep_single_config import build_model_for_config
    disk = build_model_for_config(
        "random_forest", "smote", "platt", None)
    assert isinstance(disk, DiskBackedCalibratedClassifier)
    disk = build_model_for_config(
        "random_forest", "smote", "isotonic", None)
    assert isinstance(disk, DiskBackedCalibratedClassifier)
    raw = build_model_for_config("random_forest", "smote", None, None)
    assert not isinstance(raw, DiskBackedCalibratedClassifier)
    assert not isinstance(raw, CalibratedClassifierCV)
    xgb = build_model_for_config("xgboost", "smote", "isotonic", 3.0)
    assert isinstance(xgb, CalibratedClassifierCV)
    assert not isinstance(xgb, DiskBackedCalibratedClassifier)


_PEAK_SNIPPET = """
import resource, sys
sys.path.insert(0, {root!r})
from app import config, data, features
from sklearn.calibration import CalibratedClassifierCV
from scripts.disk_backed_calibration import DiskBackedCalibratedClassifier
df = data.load_s6e3(nrows=150000)
X_train, X_test, y_train, y_test = data.split(df)
X_train_f = X_train.drop(columns=[config.ID_COLUMN])
if {mode!r} == "baseline":
    pass
else:
    pipe = features.build_pipeline("random_forest", handling="smote", calibration=None)
    if {mode!r} == "standard":
        model = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
    else:
        model = DiskBackedCalibratedClassifier(pipe, method="sigmoid", cv=3, tmp_root={tmp_root!r})
    model.fit(X_train_f, y_train)
print("PEAK", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
"""


def _peak_rss_mb(mode: str, tmp_root: str | None) -> float:
    code = _PEAK_SNIPPET.format(
        root=str(ROOT), mode=mode, tmp_root=tmp_root)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=env, timeout=1200, check=True)
    for line in out.stdout.splitlines():
        if line.startswith("PEAK"):
            return float(line.split()[1])
    raise AssertionError(f"no PEAK line in output: {out.stdout}\n{out.stderr}")


def test_disk_backed_uses_less_memory(tmp_path):
    """Same fit, same data: the disk-backed fit must use far less
    memory than the standard fit. The comparison subtracts a
    baseline run (imports + data load, no fit), because the fixed
    costs would otherwise hide the effect. The row count is 150k,
    not 60k: at 60k the measured ratio sat at 0.61 against a 0.6
    gate, because the forest is small next to the shared data
    arrays; at 150k the forests dominate and the ratio is stable
    near 0.5. /tmp on this VM is a 512 MB tmpfs, so the
    disk-backed run uses a scratch dir under the home directory,
    where there is room."""
    scratch = Path.home() / ".cache" / "diskcal-test"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        base_peak = _peak_rss_mb("baseline", None)
        std_peak = _peak_rss_mb("standard", None)
        disk_peak = _peak_rss_mb("disk", str(scratch))
    finally:
        for leftover in scratch.glob("diskcal_*"):
            import shutil
            shutil.rmtree(leftover, ignore_errors=True)
    std_inc = std_peak - base_peak
    disk_inc = disk_peak - base_peak
    assert disk_inc < 0.6 * std_inc, (
        f"disk-backed fit increment {disk_inc:.0f} MB not far below "
        f"standard increment {std_inc:.0f} MB "
        f"(baseline {base_peak:.0f}, disk {disk_peak:.0f}, "
        f"standard {std_peak:.0f} MB)")

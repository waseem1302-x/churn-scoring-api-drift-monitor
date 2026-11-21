"""Disk-backed calibration for the S6E3 sweep.

Configs 17 and 18 (random forest + SMOTE + Platt / isotonic) died
twice on Colab with exit -9. SMOTE grows the 475k-row train set to
about 736k rows, and CalibratedClassifierCV (cv=3) holds three
fitted 300-tree forests in RAM at once, about 15 GB against Colab's
12 GB. n_jobs=1 did not save them; it only removed the parallel
workers.

This keeps the maths identical and moves the storage. Each
calibration fold is fitted by scikit-learn's own per-fold function
(_fit_classifier_calibrator_pair, the exact function the standard
wrapper calls per fold: same StratifiedKFold splits, same base fit
on the train fold, same sigmoid / isotonic calibrator fitted on the
held-out fold), then the fitted calibrator is dumped to disk, freed
from RAM, and loaded back one at a time for prediction. The
predictions are the mean of the same three calibrators, so they
are identical to the standard path. Only the storage changes, and
the peak RAM drops from three forests to one.

Note: _fit_classifier_calibrator_pair is a private scikit-learn
helper. It is pinned to the scikit-learn version in
requirements.txt (1.9.1); if that version changes, the identity
tests in tests/test_disk_backed_calibration.py must be re-run, and
they will fail loudly if the helper's behaviour changes.

Scope: sweep-only experiment code. The production API is untouched;
its champion is a small Telco model that never had this problem.
Call close() when done; it deletes the temp files. The temp dir is
owned by a tempfile.TemporaryDirectory object, so it is also removed
when the estimator is discarded: that matters because
cross_val_predict clones the estimator per fold and never calls
close() on the clones, and without the finalizer those clones'
calibrator files would leak (on the VM that filled /tmp and failed
three tests with "No space left" during the full-suite run).
"""

import gc
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import _fit_classifier_calibrator_pair
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import indexable
from sklearn.utils._array_api import get_namespace_and_device
from sklearn.utils.multiclass import unique_labels


class DiskBackedCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """CalibratedClassifierCV (cv=3) with each fold's fitted model
    on disk instead of in RAM.

    ClassifierMixin comes first in the bases, as in scikit-learn's
    own classifiers: with BaseEstimator first, the classifier tags
    are lost, is_classifier returns False, and cross_val_predict
    falls back to plain KFold instead of StratifiedKFold, which
    changes the splits and therefore the results.

    Parameters mirror the standard wrapper: estimator is the base
    model (here the whole preprocessing + SMOTE + forest pipeline),
    method is "sigmoid" (Platt) or "isotonic", cv is the number of
    calibration folds, tmp_root is the parent dir for the temp files
    (None means the system temp dir).
    """

    def __init__(self, estimator, method="sigmoid", cv=3, tmp_root=None):
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.tmp_root = tmp_root

    def fit(self, X, y):
        if self.method not in ("sigmoid", "isotonic"):
            raise ValueError(f"unknown method: {self.method}")
        if not isinstance(self.cv, int) or self.cv < 2:
            raise ValueError(f"cv must be an int >= 2, got {self.cv}")
        X, y = indexable(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        xp, _, _ = get_namespace_and_device(X)
        # A refit must not leak the previous fit's temp dir.
        self.close()
        tmp_obj = tempfile.TemporaryDirectory(
            prefix="diskcal_", dir=self.tmp_root)
        self._tmpdir_obj = tmp_obj
        tmpdir = Path(tmp_obj.name)
        self._tmpdir = str(tmpdir)
        files = []
        try:
            splitter = StratifiedKFold(n_splits=self.cv, shuffle=False)
            for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
                est = clone(self.estimator)
                cal = _fit_classifier_calibrator_pair(
                    est, X, y, train=train_idx, test=test_idx,
                    method=self.method, classes=self.classes_, xp=xp,
                    fit_params={})
                path = tmpdir / f"calibrator_{i}.joblib"
                joblib.dump(cal, path)
                files.append(str(path))
                del cal, est
                gc.collect()
        except Exception:
            tmp_obj.cleanup()
            raise
        self._calibrator_files = files
        return self

    def _load_probas(self, X):
        if not getattr(self, "_calibrator_files", None):
            raise RuntimeError(
                "not fitted, or close() was already called")
        probas = np.zeros((X.shape[0], len(self.classes_)))
        for path in self._calibrator_files:
            if not Path(path).exists():
                raise RuntimeError(
                    f"calibrator file missing: {path}")
            cal = joblib.load(path)
            try:
                probas += cal.predict_proba(X)
            finally:
                del cal
                gc.collect()
        return probas / len(self._calibrator_files)

    def predict_proba(self, X):
        if not hasattr(self, "_calibrator_files"):
            raise RuntimeError("not fitted")
        return self._load_probas(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def close(self):
        """Delete the temp calibrator files. predict_proba fails
        after this; call it when the model is no longer needed.
        Safe to call on an unfitted instance or twice."""
        tmp_obj = getattr(self, "_tmpdir_obj", None)
        if tmp_obj is not None:
            try:
                tmp_obj.cleanup()
            except Exception:
                pass
        self._calibrator_files = []

"""Pipeline leakage tests: preprocessing must only ever see training data.

TDD: these tests fail until app/features.py exists.
"""

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from app import data as d
from app import features as f


@pytest.fixture(scope="module")
def split_data():
    df = d.load_telco(d.DATA_PATH)
    return d.split(df)


def test_preprocessor_imputer_uses_train_median_only(split_data):
    X_train, X_test, _, _ = split_data
    pre = f.build_preprocessor()
    pre.fit(X_train)
    # TotalCharges is the 4th numeric column (index 3 in NUMERIC_FEATURES).
    num_pipe = pre.named_transformers_["num"]
    imputer = num_pipe.named_steps["imputer"]
    idx = d.NUMERIC_FEATURES.index("TotalCharges")
    train_median = float(np.nanmedian(X_train["TotalCharges"].astype(float)))
    full = d.load_telco(d.DATA_PATH)
    full_median = float(np.nanmedian(full["TotalCharges"].astype(float)))
    assert train_median != full_median  # the test would be vacuous otherwise
    assert float(imputer.statistics_[idx]) == pytest.approx(train_median)
    assert float(imputer.statistics_[idx]) != pytest.approx(full_median)


def test_build_pipeline_returns_single_sklearn_pipeline(split_data):
    pipe = f.build_pipeline("logistic", handling="none", calibration=None)
    assert isinstance(pipe, Pipeline)
    assert "preprocessor" in pipe.named_steps
    assert "model" in pipe.named_steps


def test_smote_lives_inside_the_pipeline_not_outside():
    pipe = f.build_pipeline("logistic", handling="smote", calibration=None)
    step_names = list(pipe.named_steps.keys())
    assert "smote" in step_names
    assert step_names.index("smote") > step_names.index("preprocessor")
    assert step_names.index("smote") < step_names.index("model")


def test_smote_pipeline_is_fittable_on_train_only(split_data):
    X_train, X_test, y_train, y_test = split_data
    pipe = f.build_pipeline("logistic", handling="smote", calibration=None)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_class_weight_pipeline_uses_balanced_weights():
    pipe = f.build_pipeline("logistic", handling="class_weight", calibration=None)
    assert pipe.named_steps["model"].class_weight == "balanced"
    pipe = f.build_pipeline("random_forest", handling="class_weight", calibration=None)
    assert pipe.named_steps["model"].class_weight == "balanced_subsample"


def test_calibration_wraps_the_whole_pipeline():
    calibrated = f.build_pipeline("logistic", handling="none", calibration="platt")
    assert isinstance(calibrated, CalibratedClassifierCV)
    assert calibrated.method == "sigmoid"
    assert calibrated.cv == 3
    calibrated = f.build_pipeline("logistic", handling="none", calibration="isotonic")
    assert calibrated.method == "isotonic"


def test_unknown_models_and_handlings_rejected():
    with pytest.raises(ValueError):
        f.build_pipeline("gradient_boosting", handling="none", calibration=None)
    with pytest.raises(ValueError):
        f.build_pipeline("logistic", handling="undersample", calibration=None)
    with pytest.raises(ValueError):
        f.build_pipeline("logistic", handling="none", calibration="beta")


def test_all_three_models_build():
    for name in ("logistic", "random_forest", "xgboost"):
        pipe = f.build_pipeline(name, handling="none", calibration=None)
        assert pipe is not None

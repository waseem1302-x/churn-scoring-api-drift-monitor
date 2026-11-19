"""Model pipelines. One scikit-learn Pipeline per model, preprocessing
fitted inside the pipeline only, never on the full data before the
split. That is the whole leakage story: split first, fit everything
inside the pipeline, refit inside each CV fold.

Class handling is an ablation arm, not a default:
- "none": no reweighting, no resampling. The threshold does the work.
- "class_weight": balanced weights (scale_pos_weight for XGBoost).
- "smote": SMOTE inside the pipeline, so it only ever sees training
  folds. Never applied before the split.

Calibration wraps the whole pipeline in CalibratedClassifierCV (cv=3),
so the preprocessing is refit inside each calibration fold too.
"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from app import config

RANDOM_STATE = config.RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def _base_model(name: str, scale_pos_weight: float | None = None):
    if name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=2)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=2)
    raise ValueError(f"unknown model: {name}")


def _apply_handling(model, name: str, handling: str, scale_pos_weight: float | None):
    if handling == "none":
        return model
    if handling == "smote":
        # SMOTE is a pipeline step, not a model parameter. It is added
        # in build_pipeline between the preprocessor and the model.
        return model
    if handling == "class_weight":
        if name == "logistic":
            model.set_params(class_weight="balanced")
        elif name == "random_forest":
            model.set_params(class_weight="balanced_subsample")
        elif name == "xgboost":
            model.set_params(scale_pos_weight=scale_pos_weight)
        return model
    raise ValueError(f"unknown handling: {handling}")


def build_pipeline(name: str, handling: str = "none", calibration=None,
                   scale_pos_weight: float | None = None):
    """Build one model. Returns a Pipeline, or a CalibratedClassifierCV
    wrapping it when calibration is 'platt' (sigmoid) or 'isotonic'."""
    if name not in config.MODELS:
        raise ValueError(f"unknown model: {name}")
    if handling not in config.HANDLINGS:
        raise ValueError(f"unknown handling: {handling}")
    if calibration not in config.CALIBRATIONS:
        raise ValueError(f"unknown calibration: {calibration}")
    model = _base_model(name, scale_pos_weight)
    model = _apply_handling(model, name, handling, scale_pos_weight)
    steps = [("preprocessor", build_preprocessor())]
    if handling == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    pipe = ImbPipeline(steps) if handling == "smote" else Pipeline(steps)
    if calibration is None:
        return pipe
    method = "sigmoid" if calibration == "platt" else "isotonic"
    return CalibratedClassifierCV(pipe, method=method, cv=3)


def model_version(name: str, handling: str, calibration) -> str:
    cal = calibration or "raw"
    return f"churn-{name}-{handling}-{cal}-v1"

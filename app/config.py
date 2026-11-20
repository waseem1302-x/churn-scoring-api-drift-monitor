"""Shared constants for the churn scoring API.

Everything a reviewer needs to check the numbers lives here: the split,
the feature lists, the risk tiers, the profit assumptions, and the CI
gate thresholds. No magic numbers anywhere else.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RANDOM_STATE = 42
TEST_SIZE = 0.2

DATA_PATH = ROOT / "data" / "raw" / "telco.csv"
SOURCES_PATH = ROOT / "data" / "sources.json"
ARTIFACT_PATH = ROOT / "artifacts" / "model.joblib"
FAST_ARTIFACT_PATH = ROOT / "artifacts" / "model-fast.joblib"
METRICS_PATH = ROOT / "benchmarks" / "results" / "metrics.json"
CONTRACT_PATH = ROOT / "benchmarks" / "results" / "contract_rows.json"
DB_PATH = ROOT / "data" / "predictions.db"

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODELS = ("logistic", "random_forest", "xgboost")
HANDLINGS = ("none", "class_weight", "smote")
CALIBRATIONS = (None, "platt", "isotonic")

# Risk tiers on the predicted churn probability. Fixed and documented;
# the measured churn rate per tier on the test set is in EVAL_RESULTS.md.
TIER_LOW_MAX = 0.30
TIER_MEDIUM_MAX = 0.60

# Expected maximum profit assumptions. These are illustrative business
# figures, not measured: a retention offer costs $50 per contacted
# customer, and a customer kept from churning is worth $500 (about
# eight months at the mean MonthlyCharges of $64.76). Every profit
# number in the repo is computed from these two numbers and nothing else.
OFFER_COST = 50.0
SAVED_VALUE = 500.0

# Precision at k: k=100 is one retention team's daily call list;
# k=10% of the scored set is the fixed-share view. Both are reported.
PRECISION_K_ABSOLUTE = 100
PRECISION_K_FRACTION = 0.10

# CI gate thresholds. Set below the measured Phase 1 champion numbers
# (see EVAL_RESULTS.md) so run-to-run jitter does not fail the build.
# Lowering a threshold needs a reason in the commit message.
GATE_THRESHOLDS = {
    "roc_auc": 0.80,
    "pr_auc": 0.55,
    "brier_max": 0.20,
    "precision_at_100": 0.50,
}

"""Data layer: load and verify the IBM Telco Customer Churn dataset.

The file is a static snapshot with no timestamps. That is a real limit:
true temporal validation is impossible on this data, and no split in
this repo pretends otherwise. The longitudinal simulator that closes
the gap is Phase 2 work.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app import config

DATA_PATH = config.DATA_PATH
TARGET = config.TARGET
ID_COLUMN = config.ID_COLUMN
FEATURES = config.FEATURES
NUMERIC_FEATURES = config.NUMERIC_FEATURES
CATEGORICAL_FEATURES = config.CATEGORICAL_FEATURES

EXPECTED_ROWS = 7043
EXPECTED_COLUMNS = 21
EXPECTED_CHURN_RATE = 0.2654
EXPECTED_BLANK_TOTAL_CHARGES = 11

# Column provenance and leakage class. "Leakage class" answers one
# question per column: when does this information exist relative to the
# churn outcome? A column recorded at or after cancellation would carry
# the answer. None of the 19 features does: every one is known at signup
# or during service, before the outcome is observed.
_COLUMN_SPEC = [
    ("customerID", "identifier", "Row identifier. Never a model feature."),
    ("gender", "known_at_signup", "Customer gender, recorded at signup."),
    ("SeniorCitizen", "known_at_signup", "0/1 flag, recorded at signup."),
    ("Partner", "known_at_signup", "Whether the customer has a partner."),
    ("Dependents", "known_at_signup", "Whether the customer has dependents."),
    ("tenure", "known_during_service", "Months with the company at snapshot time."),
    ("PhoneService", "known_at_signup", "Whether phone service is subscribed."),
    ("MultipleLines", "known_during_service", "Multiple lines, or no phone service."),
    ("InternetService", "known_at_signup", "DSL, fiber optic, or none."),
    ("OnlineSecurity", "known_during_service", "Add-on, or no internet service."),
    ("OnlineBackup", "known_during_service", "Add-on, or no internet service."),
    ("DeviceProtection", "known_during_service", "Add-on, or no internet service."),
    ("TechSupport", "known_during_service", "Add-on, or no internet service."),
    ("StreamingTV", "known_during_service", "Add-on, or no internet service."),
    ("StreamingMovies", "known_during_service", "Add-on, or no internet service."),
    ("Contract", "known_during_service", "Month-to-month, one year, two year."),
    ("PaperlessBilling", "known_during_service", "Billing preference."),
    ("PaymentMethod", "known_during_service", "How the customer pays."),
    ("MonthlyCharges", "known_during_service", "Current monthly bill."),
    ("TotalCharges", "known_during_service",
     "Total billed so far. Blank for 11 new customers with tenure 0; "
     "blank means no bill yet, not a missing value to guess."),
    ("Churn", "target", "Outcome: Yes means the customer left."),
]


def load_telco(path=DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and normalise types. Blank TotalCharges becomes NaN."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def has_timestamps(df: pd.DataFrame) -> bool:
    lowered = [c.lower() for c in df.columns]
    return any("date" in c or "time" in c for c in lowered)


def verify_dataset(df: pd.DataFrame) -> dict:
    """Check the file is the dataset we think it is. Raises on any failure."""
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(df)}")
    if len(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"expected {EXPECTED_COLUMNS} columns, got {len(df.columns)}")
    churn_rate = float(df[TARGET].mean())
    if abs(churn_rate - EXPECTED_CHURN_RATE) > 0.001:
        raise ValueError(f"expected churn rate ~{EXPECTED_CHURN_RATE}, got {churn_rate:.4f}")
    blank = int(df["TotalCharges"].isna().sum())
    if blank != EXPECTED_BLANK_TOTAL_CHARGES:
        raise ValueError(f"expected {EXPECTED_BLANK_TOTAL_CHARGES} blank TotalCharges, got {blank}")
    dups = int(len(df) - df[ID_COLUMN].nunique())
    if dups != 0:
        raise ValueError(f"expected 0 duplicate customer IDs, got {dups}")
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "churn_rate": churn_rate,
        "blank_total_charges": blank,
        "duplicate_customer_ids": dups,
        "has_timestamps": has_timestamps(df),
    }


def do_nothing_accuracy(df: pd.DataFrame) -> float:
    """Accuracy of predicting 'no churn' for everyone. The number every
    accuracy-adjacent claim must be shown next to: 0.7346."""
    return float((df[TARGET] == 0).mean())


def split(df: pd.DataFrame):
    """Stratified 80/20 split, fixed seed. One row per customer, so a
    grouped split by customerID would be vacuous (every group has size
    1); it is not run and not claimed."""
    X = df[FEATURES + [ID_COLUMN]]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=y,
    )


def column_spec() -> list:
    return [
        {"name": name, "leakage_class": cls, "notes": notes}
        for name, cls, notes in _COLUMN_SPEC
    ]

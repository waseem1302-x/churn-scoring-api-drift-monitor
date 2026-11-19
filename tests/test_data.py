"""Data layer tests: load, verify, split, column spec.

TDD: these tests fail until app/data.py exists.
"""

import math

import pytest

from app import data as d

RAW = d.DATA_PATH


def test_load_returns_expected_shape():
    df = d.load_telco(RAW)
    assert df.shape == (7043, 21)


def test_load_target_is_binary():
    df = d.load_telco(RAW)
    assert sorted(df["Churn"].unique().tolist()) == [0, 1]


def test_load_total_charges_numeric_with_11_blanks():
    df = d.load_telco(RAW)
    assert df["TotalCharges"].dtype.kind == "f"
    assert df["TotalCharges"].isna().sum() == 11


def test_verify_passes_on_real_file():
    df = d.load_telco(RAW)
    report = d.verify_dataset(df)
    assert report["rows"] == 7043
    assert report["columns"] == 21
    assert report["churn_rate"] == pytest.approx(0.2654, abs=0.0005)
    assert report["blank_total_charges"] == 11
    assert report["duplicate_customer_ids"] == 0
    assert report["has_timestamps"] is False


def test_verify_rejects_wrong_row_count():
    df = d.load_telco(RAW).iloc[:100]
    with pytest.raises(ValueError):
        d.verify_dataset(df)


def test_verify_rejects_missing_column():
    df = d.load_telco(RAW).drop(columns=["tenure"])
    with pytest.raises(ValueError):
        d.verify_dataset(df)


def test_do_nothing_baseline():
    df = d.load_telco(RAW)
    baseline = d.do_nothing_accuracy(df)
    # Predicting "no churn" for everyone is right for every non-churner.
    assert baseline == pytest.approx(1 - 0.2654, abs=0.0005)
    assert baseline == pytest.approx(5174 / 7043, abs=1e-9)


def test_split_is_stratified_and_disjoint():
    df = d.load_telco(RAW)
    X_train, X_test, y_train, y_test = d.split(df)
    assert len(X_train) + len(X_test) == 7043
    assert len(X_test) == 1409
    assert len(X_train) == 5634
    assert set(X_train["customerID"]).isdisjoint(set(X_test["customerID"]))
    # Stratified: churn rate in each part stays near the overall rate.
    assert abs(y_train.mean() - 0.2654) < 0.01
    assert abs(y_test.mean() - 0.2654) < 0.01


def test_split_is_reproducible():
    df = d.load_telco(RAW)
    a = d.split(df)
    b = d.split(df)
    assert a[0].equals(b[0])
    assert a[1].equals(b[1])


def test_column_spec_covers_every_column():
    spec = d.column_spec()
    names = [c["name"] for c in spec]
    df = d.load_telco(RAW)
    assert sorted(names) == sorted(df.columns.tolist())
    assert len(spec) == 21


def test_column_spec_leakage_classes_are_valid():
    valid = {"identifier", "known_at_signup", "known_during_service", "target"}
    for col in d.column_spec():
        assert col["leakage_class"] in valid, col["name"]
        assert col["notes"].strip(), col["name"]


def test_customer_id_is_identifier_class():
    spec = {c["name"]: c for c in d.column_spec()}
    assert spec["customerID"]["leakage_class"] == "identifier"
    assert spec["Churn"]["leakage_class"] == "target"


def test_no_timestamp_column_exists():
    df = d.load_telco(RAW)
    assert d.has_timestamps(df) is False
    lowered = [c.lower() for c in df.columns]
    assert not any("date" in c or "time" in c for c in lowered)


def test_feature_lists_exclude_id_and_target():
    assert "customerID" not in d.FEATURES
    assert "Churn" not in d.FEATURES
    assert len(d.FEATURES) == 19
    assert len(d.NUMERIC_FEATURES) == 4
    assert len(d.CATEGORICAL_FEATURES) == 15
    assert sorted(d.NUMERIC_FEATURES + d.CATEGORICAL_FEATURES) == sorted(d.FEATURES)

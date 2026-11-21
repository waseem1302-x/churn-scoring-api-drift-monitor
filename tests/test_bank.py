"""Bank generalisation tests: the same pipeline on the Bank
Customer Churn dataset (10,000 rows), reported separately from Telco.

TDD: these tests fail until app/data.py has a Bank loader and
app/features.py takes feature lists. The Bank file is never
committed: it lives at the download path and is read from there.
When the file is absent (for example in CI), the tests skip
instead of failing, the same as the S6E3 tests.
"""

from pathlib import Path

import pytest

from app import data as d
from app import features as f

BANK_PATH = Path(
    "/home/hatch/workspace/browser_downloads/sess-1707842952/"
    "browser-download-20261120T033239.272219571Z-0-Churn_Modelling.csv")


def _load_bank(path=BANK_PATH):
    if not Path(path).exists():
        pytest.skip(f"Bank file missing: {path} (Kaggle download)")
    return d.load_bank(path)


def test_bank_file_is_the_dataset_we_think_it_is():
    df = _load_bank()
    report = d.verify_bank(df)
    assert report["rows"] == 10000
    assert report["churn_rate"] == pytest.approx(0.2037, abs=0.001)
    assert report["nulls"] == 0
    assert report["duplicate_ids"] == 0


def test_bank_loader_drops_identifiers():
    # The loader keeps the raw frame (like Telco keeps customerID);
    # the exclusion happens at the feature list, which is what the
    # model sees. RowNumber and Surname are never features.
    df = _load_bank()
    assert "CustomerId" in df.columns
    assert set(d.BANK_FEATURES) == {
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary"}
    assert "RowNumber" not in d.BANK_FEATURES
    assert "Surname" not in d.BANK_FEATURES
    assert "CustomerId" not in d.BANK_FEATURES
    X_train, _, _, _ = d.split_bank(df)
    assert set(X_train.columns) == set(d.BANK_FEATURES) | {"CustomerId"}


def test_bank_verify_rejects_wrong_row_count():
    df = _load_bank().head(100)
    with pytest.raises(ValueError):
        d.verify_bank(df)


def test_bank_split_shapes():
    df = _load_bank()
    X_train, X_test, y_train, y_test = d.split_bank(df)
    assert len(X_train) == 8000
    assert len(X_test) == 2000
    assert y_train.mean() == pytest.approx(0.2037, abs=0.01)
    assert y_test.mean() == pytest.approx(0.2037, abs=0.01)


def test_bank_pipeline_fits_and_scores():
    df = _load_bank()
    X_train, X_test, y_train, y_test = d.split_bank(df)
    pipe = f.build_pipeline(
        "logistic", handling="none", calibration=None,
        numeric=d.BANK_NUMERIC_FEATURES,
        categorical=d.BANK_CATEGORICAL_FEATURES)
    pipe.fit(X_train.head(500), y_train.head(500))
    proba = pipe.predict_proba(X_test.head(50))[:, 1]
    assert len(proba) == 50
    assert all(0.0 <= p <= 1.0 for p in proba)


def test_load_skips_when_the_file_is_absent(tmp_path):
    with pytest.raises(pytest.skip.Exception):
        _load_bank(tmp_path / "missing.csv")


def test_bank_results_json_exists_and_docs_match():
    results = Path(__file__).resolve().parents[1] / "benchmarks" \
        / "results" / "bank_generalisation.json"
    if not results.exists():
        pytest.fail("bank_generalisation.json missing: run "
                    "scripts/bank_generalisation.py first")
    import json

    data = json.loads(results.read_text())
    assert data["dataset"]["rows"] == 10000
    assert data["dataset"]["churn_rate"] == pytest.approx(0.2037, abs=0.001)
    assert len(data["configs"]) == 27
    text = (Path(__file__).resolve().parents[1]
            / "EVAL_RESULTS.md").read_text()
    start = text.index("## Bank generalisation")
    end = text.index("## ", start + 3)
    section = text[start:end]
    champ = data["champion"]
    assert champ["model_version"] in section
    assert f"{champ['test_metrics']['roc_auc']:.4f}" in section
    assert f"{champ['test_metrics']['pr_auc']:.4f}" in section

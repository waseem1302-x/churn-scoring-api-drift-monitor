# Churn Scoring API plus Drift Monitor

[![CI](https://github.com/waseem1302-x/churn-scoring-api-drift-monitor/actions/workflows/eval.yml/badge.svg)](https://github.com/waseem1302-x/churn-scoring-api-drift-monitor/actions/workflows/eval.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.165.0-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-blue.svg)](https://xgboost.readthedocs.io/)
[![pytest](https://img.shields.io/badge/pytest-9.2.0-brightgreen.svg)](https://docs.pytest.org/)

A churn scoring API over the IBM Telco Customer Churn dataset, built
the way the tutorial version is not: a leakage audit, calibrated
probabilities, a profit-based threshold instead of the default 0.5,
and a CI gate on the metrics. Phase 1 is the scoring API plus honest
evaluation. The drift monitor, the longitudinal simulator, and the
retraining loop are Phase 2.

## At a glance

- Language: Python 3.12
- API framework: FastAPI 0.165.0 with Uvicorn 0.54.0
- Models: Logistic Regression, Random Forest, XGBoost, 27 configs
  (3 models x 3 class handlings x 3 calibrations)
- Champion: XGBoost, SMOTE in-fold, isotonic calibration
- Data: IBM Telco Customer Churn, 7,043 rows, 21 columns, no timestamps
- Storage: SQLite prediction log, one Pipeline artifact, no database server
- Testing: pytest 9.2.0, 57 tests
- CI: GitHub Actions, eval gate workflow (`.github/workflows/eval.yml`)
- Containers: Docker
- License: MIT, see [LICENSE](LICENSE)

Read first:

- [ARCHITECTURE.md](ARCHITECTURE.md) - what was built, why, and the trade-offs
- [EVAL_RESULTS.md](EVAL_RESULTS.md) - every measured number
- [DATA.md](DATA.md) - every column, its provenance and leakage class

## Results at a glance

Measured 20 Nov 2026 on a local run. No number here is invented.
The do-nothing baseline (predict "no churn" for everyone) is 73.46%,
shown next to every accuracy-adjacent claim so no reader is fooled.

| Metric | Value |
|---|---|
| Test set | 1,409 rows (374 churners) |
| ROC-AUC | 0.8366 |
| PR-AUC | 0.6395 |
| Brier | 0.1392 |
| Precision at 100 | 0.76 |
| Precision at 10% | 0.7589 |
| Test profit at frozen threshold 0.0934 | $132,150 ($93,790 per 1,000) |
| Test profit at threshold 0.5 | $78,750 ($55,891 per 1,000) |
| Risk tiers, churn rate | LOW 10.54%, MEDIUM 42.47%, HIGH 74.27% |
| Scoring latency p50 / p95 | 64.07 ms / 77.47 ms |
| Tests | 57 passed, 0 failed |
| CI gate | Pass |
| Cost | $0.00 (local CPU, no API calls) |

## Tech stack

What was actually used, and what each piece does.

| Layer | Tool | What it does here |
|---|---|---|
| Language | Python 3.12 | All code: app, scripts, tests |
| API framework | FastAPI | `GET /health`, `POST /score`, `GET /metrics` |
| Models | scikit-learn, XGBoost | One Pipeline per model, preprocessing inside |
| Resampling | imbalanced-learn (SMOTE) | In-fold only, inside the pipeline |
| Calibration | scikit-learn CalibratedClassifierCV | Platt and isotonic, cv=3 |
| Validation | Pydantic | Every field of every score request |
| Storage | SQLite | Prediction log, one row per score |
| Testing | pytest | 57 tests |
| CI | GitHub Actions (`.github/workflows/eval.yml`) | Train, latency, tests, gate on every push/PR |
| Containers | Docker | See the Docker note below |

What was not used, and why:

- **LightGBM: not added.** No ablation shows a gap a fourth booster
  would fill. XGBoost already ranks best and logistic is nearly as
  good on ROC-AUC. See `ARCHITECTURE.md`, decision 5.
- **SHAP: not used in Phase 1.** SHAP plots are standard on churn
  resumes now and add nothing to the monitoring story. The API
  returns a risk tier with a measured churn rate instead.
- **LLM retention advice: not used.** A language model writing
  retention copy is a demo, not a measurement. Phase 1 measures the
  scoring and the decision.
- **Evidently / NannyML: not used in Phase 1.** Both are drift tools
  and belong to Phase 2, the drift monitor. The API has no drift
  fields and the log has none, by design.

## Quickstart (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Train all 27 configs, pick the champion, save the artifact
python -m scripts.train

# 2. Measure scoring latency through the API
python -m scripts.measure_latency

# 3. Run the tests
python -m pytest tests/ -q

# 4. Run the CI gate
python -m scripts.ci_gate

# 5. Run the API
uvicorn app.api:app --port 8000

# 6. Score a customer
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
       "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
       "MultipleLines": "No", "InternetService": "DSL",
       "OnlineSecurity": "Yes", "OnlineBackup": "No",
       "DeviceProtection": "No", "TechSupport": "No",
       "StreamingTV": "No", "StreamingMovies": "No",
       "Contract": "Month-to-month", "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 29.85, "TotalCharges": 358.2}'
```

## API

`GET /health` returns `{"status": "ok", "model_version": ...}`.

`POST /score` takes the 19 model features (plus an optional
`customerID` for the audit log) and returns the churn probability,
the risk tier, and the model version. Every field is validated:
unknown categories, negative tenure, and a SeniorCitizen other than
0 or 1 are all rejected with 422. `TotalCharges` may be null for a
new customer with no bill yet; the pipeline imputes the training
median, exactly as it does for the 11 blank rows in the data.

```json
{
  "customer_id": "TEST-0001",
  "probability": 0.7234,
  "tier": "HIGH",
  "model_version": "churn-xgboost-smote-isotonic-v1"
}
```

`GET /metrics` returns the measured Phase 1 metrics from
`benchmarks/results/metrics.json`.

Every score is logged to SQLite (`data/predictions.db`, gitignored)
with its inputs, probability, tier, model version and latency, so
any past decision can be replayed.

## How it works

```
telco.csv -> data.py (load + verify: 7,043 rows, 21 cols, 26.54% churn)
  -> split (stratified 80/20, seed 42)
  -> features.py (one Pipeline per model: impute + scale + one-hot,
     SMOTE inside the pipeline only, calibration wrapping the whole
     pipeline with cv=3)
  -> train.py (27 configs; threshold from out-of-fold train
     predictions, never from test; champion by out-of-fold profit)
  -> artifacts/model.joblib (champion pipeline + metadata)
  -> api.py (FastAPI: /health, /score, /metrics; SQLite log)
  -> ci_gate.py (fails the build if ROC-AUC, PR-AUC, Brier or
     precision at 100 regress below the pinned thresholds)
```

### Why no accuracy headline

Predicting "no churn" for everyone scores 73.46% on this data. An
80% accuracy is 7 points above doing nothing. The metrics that
matter are ranking (ROC-AUC, PR-AUC), calibration (Brier,
reliability curve), and the decision (precision at k, profit at
the operating threshold). Accuracy appears in this repo exactly
once, as the do-nothing baseline.

### Why the threshold is 0.0934

A threshold is a business decision, not a default 0.5. The
assumptions are explicit: a retention offer costs $50 per
contacted customer and a saved customer is worth $500. The
break-even precision is $50 / $500 = 0.10, so any contacted group
churning above 10% pays for itself. The profit-maximising
threshold, chosen on out-of-fold train predictions, is 0.0934:
it contacts 929 of 1,409 test customers, catches 95.45% of
churners, and profits $132,150 against $78,750 at 0.5. The price
is volume. A team that can only make 100 calls a day should read
precision at 100 (0.76) instead.

## Benchmarks and ablation

```bash
python -m scripts.train   # all 27 configs, same split
```

What the ablation shows, measured:

- Calibration earns its place: mean Brier 0.1515 raw to 0.1394
  isotonic. Class weighting and SMOTE both break the probabilities
  before recalibration (logistic raw Brier 0.1381 plain, 0.1688
  class-weighted, 0.1678 SMOTE).
- SMOTE does not earn its place on ranking: mean PR-AUC 0.6231
  against 0.6357 for threshold-only. It is in the champion only
  because the profit rule picked it by $8.87 per 1,000.
- XGBoost ranks best (mean PR-AUC 0.6481), logistic is nearly as
  good on ROC-AUC (mean 0.8411) and 5 to 10 times faster to train,
  Random Forest is last on both and slowest.
- The config choice barely moves profit ($127,500 to $134,100
  across all 27). The threshold choice moves it a lot ($78,750 at
  0.5 to $132,150 at 0.0934).

Full table: [EVAL_RESULTS.md](EVAL_RESULTS.md). Raw records:
`benchmarks/results/all_configs.json`.

## Tests

```bash
python -m pytest tests/ -q   # 57 passed
```

Tests cover data loading and verification (row and column counts,
churn rate, blank TotalCharges, no timestamps), pipeline leakage
(the imputer sees the train median only, SMOTE lives inside the
pipeline, calibration wraps the whole pipeline), metrics and profit
(hand-computed toy cases), the API (contract, validation, log
rows), the shipped artifact (it reproduces its recorded scores on
25 stored test rows), and doc integrity (no accuracy headline, no
em-dash, honest S6E3 pending state).

## Docker

```bash
docker build -t churn-scoring-api .
docker run -p 8000:8000 churn-scoring-api
```

The image ships the trained artifact, so the container starts
without training. **Note (verified 20 Nov 2026):** Docker cannot
start containers in this development VM (the VM's seccomp policy
blocks `setns`, same as the RAG project). The image was built and
smoke-tested with Buildah 1.33.7 using chroot isolation instead:
`buildah bud --isolation=chroot` and a chroot run that loads the
artifact and scores a row. Build it with Docker on a normal
machine before trusting it there.

## What is not done yet

- Phase 2: drift monitor (PSI, KS, Jensen-Shannon, Wasserstein),
  longitudinal simulator with a known drift schedule, drift
  injection with measured detection delay, retraining loop with
  shadow promotion. The API has no drift fields yet, by design.
- Phase 3: deployment (Oracle Always Free VM or Cloud Run, same
  route as the RAG project), Bank Customer Churn generalisation
  check, NannyML CBPE for performance estimation while labels are
  delayed.
- S6E3 scale run: pending. No Kaggle credentials were present, so
  no 594k-row numbers are claimed. See `EVAL_RESULTS.md`.

# Data

## Source

IBM Telco Customer Churn, 7,043 rows, 21 columns. Originally on
Kaggle (`blastchar/telco-customer-churn`). Fetched 20 Nov 2026 from
the IBM GitHub mirror
(`IBM/telco-customer-churn-on-icp4d`, file
`data/Telco-Customer-Churn.csv`) because no Kaggle credentials were
present at `~/.kaggle/kaggle.json`. Provenance in
`data/sources.json`.

Verification, run by `app/data.py:verify_dataset` before any
training: exactly 7,043 rows, 21 columns, churn rate 0.2654, 11
blank TotalCharges values, 0 duplicate customer IDs. If any check
fails, training refuses to run.

## What the data is

One row per customer, a static snapshot. There are no timestamps:
no signup date, no snapshot date, no churn date. That is the
dataset's central limit, and it shapes everything in Phase 1:

- True temporal validation is impossible. A random split measures
  interpolation on a static snapshot, not performance on future
  customers.
- A grouped split by customerID is vacuous: every group has size
  1, so grouping changes nothing. It is not run and not claimed.
- The longitudinal simulator that closes this gap is Phase 2 work.

## Columns

Leakage class answers one question per column: when does this
information exist relative to the churn outcome? A column recorded
at or after cancellation would carry the answer. None of the 19
features does. Every one is known at signup or during service,
before the outcome is observed.

| Column | Kind | Leakage class | Notes |
|---|---|---|---|
| customerID | identifier | identifier | Row identifier. Never a model feature. Accepted by the API only for the audit log. |
| gender | categorical | known_at_signup | Female, Male. |
| SeniorCitizen | numeric (0/1) | known_at_signup | 0 or 1. |
| Partner | categorical | known_at_signup | Yes, No. |
| Dependents | categorical | known_at_signup | Yes, No. |
| tenure | numeric | known_during_service | Months with the company at snapshot time, 0 to 72. |
| PhoneService | categorical | known_at_signup | Yes, No. |
| MultipleLines | categorical | known_during_service | Yes, No, No phone service. |
| InternetService | categorical | known_at_signup | DSL, Fiber optic, No. |
| OnlineSecurity | categorical | known_during_service | Yes, No, No internet service. |
| OnlineBackup | categorical | known_during_service | Yes, No, No internet service. |
| DeviceProtection | categorical | known_during_service | Yes, No, No internet service. |
| TechSupport | categorical | known_during_service | Yes, No, No internet service. |
| StreamingTV | categorical | known_during_service | Yes, No, No internet service. |
| StreamingMovies | categorical | known_during_service | Yes, No, No internet service. |
| Contract | categorical | known_during_service | Month-to-month, One year, Two year. |
| PaperlessBilling | categorical | known_during_service | Yes, No. |
| PaymentMethod | categorical | known_during_service | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic). |
| MonthlyCharges | numeric | known_during_service | Current monthly bill. Mean $64.76. |
| TotalCharges | numeric | known_during_service | Total billed so far. Blank for 11 new customers with tenure 0. Blank means no bill yet, not a missing value to guess. Imputed with the training median inside the pipeline. |
| Churn | target | target | Yes means the customer left. Mapped to 1, No to 0. Churn rate 26.54%. |

## The messy bits, documented not hidden

- **11 blank TotalCharges values.** All 11 belong to customers
  with tenure 0. The blank is informative (no bill yet), and the
  pipeline imputes the training median. Imputing with the
  full-data median would leak; the test
  `test_preprocessor_imputer_uses_train_median_only` proves the
  fitted imputer holds the train median.
- **"No internet service" and "No phone service" are real
  categories**, not missing values. One-hot encoding keeps them;
  no row is dropped for them.
- **"No" in the add-on columns is ambiguous in the source data**
  (it can mean the customer declined the add-on), but the dataset
  does not distinguish, so neither do we.
- **No timestamps.** Stated three times in this file because it
  is the limit that matters most.
- **One row per customer.** No history, no repeated measures, no
  behaviour over time. Tenure is the only time-like field, and it
  is a single number at snapshot time.

## Split

Stratified 80/20, seed 42: 5,634 train, 1,409 test. Churn rate
26.54% in both parts (374 churners in test). The split is fixed by
the seed, so every run trains and evaluates on the same rows.

## What is not in this dataset

- S6E3 synthetic set: fetched 20 Nov 2026 from Kaggle
  (Playground Series S6E3 train.csv, 594,194 rows, churn rate
  22.52%, 0 blank TotalCharges, ids 0 to 594,193). Same schema as
  Telco except the identifier is "id". The 80 MB file lives
  outside the repo and is never committed. Scale numbers are in
  EVAL_RESULTS.md and benchmarks/results/scale_s6e3.json.

## Bank Customer Churn (generalisation check only)

Fetched 20 Nov 2026 from Kaggle (Churn_Modelling.csv, 10,000
rows, 2,037 churned, 20.37%, 0 nulls, 0 duplicate CustomerIds).
The file lives outside the repo and is never committed.

The schema is different from Telco: 10 features (CreditScore,
Geography, Gender, Age, Tenure, Balance, NumOfProducts,
HasCrCard, IsActiveMember, EstimatedSalary), target Exited (0/1
already), identifier CustomerId. RowNumber is a row index and
Surname is a name: neither is a feature. HasCrCard and
IsActiveMember are 0/1 flags and are treated as numeric, like
SeniorCitizen on Telco. There are no timestamps here either, so
the same temporal limit applies.

The check retrains the same pipeline on Bank and reports it
separately in EVAL_RESULTS.md and
benchmarks/results/bank_generalisation.json. Nothing is pooled
with Telco: the Telco artifact cannot score Bank rows, and the
Bank champion is picked by the same pre-registered rule
(highest out-of-fold profit) on Bank data alone.

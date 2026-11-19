# Eval Results, Phase 1

All numbers below come from real local runs on 20 Nov 2026, on this
VM (Python 3.12.3, 1.5 shared CPUs, 6 GB RAM). Nothing here is
invented. Reproduce them with the commands in "How to reproduce".

## The dataset, in one paragraph

IBM Telco Customer Churn: 7,043 rows, 21 columns, churn rate 26.54%
(1,869 of 7,043). Stratified 80/20 split, seed 42: 5,634 train,
1,409 test (374 churners in test). The do-nothing baseline,
predicting "no churn" for everyone, scores 73.46% on this data, so
73.46% is the number every accuracy-adjacent claim is shown next
to. There are no timestamps in this dataset, so true temporal
validation is impossible on it; the longitudinal simulator that
closes the gap is Phase 2 work, and no split here pretends
otherwise. A grouped split by customerID is vacuous (one row per
customer, every group has size 1), so it is not run and not claimed.

## Champion

`churn-xgboost-smote-isotonic-v1`: XGBoost, SMOTE inside the
pipeline, isotonic calibration. Chosen by the pre-registered rule:
highest out-of-fold expected maximum profit on train. Threshold
0.0934, frozen from out-of-fold train predictions, never from test.

| Metric | Value |
|---|---|
| ROC-AUC | 0.8366 |
| PR-AUC | 0.6395 |
| Brier | 0.1392 |
| Precision at 100 | 0.76 (76 of the top 100 scored customers churned) |
| Precision at 10% | 0.7589 (107 of 141) |
| Precision at threshold 0.0934 | 0.3851 |
| Recall at threshold 0.0934 | 0.9545 |
| Test profit at threshold 0.0934 | $132,150 ($93,790 per 1,000 customers) |
| Test profit at threshold 0.5 | $78,750 ($55,891 per 1,000) |
| Out-of-fold profit per 1,000 | $94,266.95 |
| Train time (fit plus out-of-fold) | 9.42 s |
| Do-nothing baseline | 73.46% |
| Cost | $0.00 |

Read honestly: the champion is a near-tie, not a clear win. The top
5 configs by out-of-fold profit sit within $142 per 1,000 of each
other ($94,124.96 to $94,266.95), and the best test profit belongs
to a different config (`churn-logistic-class_weight-isotonic-v1`,
$134,100, which is $1,950 or 1.5% above the champion on test).
The champion was picked by the out-of-fold rule, not by test, and
the test ranking differing is expected: test is one sample of 1,409
rows. The best PR-AUC config (`churn-xgboost-none-platt-v1`,
0.6547) trails the champion on out-of-fold profit by $363.86 per
1,000. No config dominates on every metric, and the table below
shows it.

## All 27 configurations

Sorted by out-of-fold profit (the selection metric). Thresholds are
frozen from out-of-fold train predictions. Test EMP is the profit on
the 1,409-row test set at the frozen threshold. Full records in
`benchmarks/results/all_configs.json`.

| Config | ROC-AUC | PR-AUC | Brier | P@100 | P@10% | Thr | Test EMP | Train s |
|---|---|---|---|---|---|---|---|---|
| churn-xgboost-smote-isotonic-v1 | 0.8366 | 0.6395 | 0.1392 | 0.76 | 0.7589 | 0.0934 | $132150 | 9.42 |
| churn-logistic-smote-platt-v1 | 0.8407 | 0.6273 | 0.1385 | 0.72 | 0.7660 | 0.1424 | $132050 | 4.08 |
| churn-logistic-smote-raw-v1 | 0.8400 | 0.6283 | 0.1678 | 0.73 | 0.7447 | 0.2491 | $131800 | 1.78 |
| churn-logistic-smote-isotonic-v1 | 0.8397 | 0.6218 | 0.1389 | 0.75 | 0.7518 | 0.0880 | $133850 | 3.88 |
| churn-logistic-class_weight-raw-v1 | 0.8413 | 0.6326 | 0.1688 | 0.73 | 0.7589 | 0.2856 | $133450 | 0.97 |
| churn-logistic-class_weight-platt-v1 | 0.8416 | 0.6331 | 0.1382 | 0.73 | 0.7518 | 0.1328 | $132900 | 3.49 |
| churn-xgboost-smote-platt-v1 | 0.8370 | 0.6418 | 0.1402 | 0.76 | 0.7376 | 0.0968 | $130150 | 11.82 |
| churn-logistic-none-raw-v1 | 0.8419 | 0.6334 | 0.1381 | 0.73 | 0.7447 | 0.1223 | $132750 | 0.74 |
| churn-xgboost-smote-raw-v1 | 0.8349 | 0.6354 | 0.1444 | 0.76 | 0.7589 | 0.0699 | $132050 | 4.07 |
| churn-xgboost-none-raw-v1 | 0.8379 | 0.6516 | 0.1384 | 0.82 | 0.7943 | 0.0764 | $130850 | 1.91 |
| churn-logistic-none-platt-v1 | 0.8420 | 0.6348 | 0.1380 | 0.74 | 0.7447 | 0.1272 | $132800 | 3.10 |
| churn-xgboost-none-platt-v1 | 0.8407 | 0.6547 | 0.1380 | 0.80 | 0.7801 | 0.1100 | $130300 | 6.40 |
| churn-xgboost-class_weight-raw-v1 | 0.8387 | 0.6539 | 0.1580 | 0.84 | 0.7730 | 0.0891 | $130100 | 1.42 |
| churn-xgboost-class_weight-platt-v1 | 0.8392 | 0.6541 | 0.1379 | 0.82 | 0.7660 | 0.0917 | $129750 | 5.74 |
| churn-xgboost-class_weight-isotonic-v1 | 0.8398 | 0.6524 | 0.1372 | 0.82 | 0.7801 | 0.0742 | $130250 | 3.97 |
| churn-logistic-none-isotonic-v1 | 0.8415 | 0.6287 | 0.1385 | 0.73 | 0.7518 | 0.1209 | $133400 | 2.81 |
| churn-xgboost-none-isotonic-v1 | 0.8389 | 0.6492 | 0.1376 | 0.77 | 0.7730 | 0.0738 | $130350 | 4.11 |
| churn-random_forest-smote-platt-v1 | 0.8296 | 0.6154 | 0.1440 | 0.72 | 0.7305 | 0.1068 | $131850 | 22.66 |
| churn-logistic-class_weight-isotonic-v1 | 0.8414 | 0.6264 | 0.1386 | 0.74 | 0.7376 | 0.1212 | $134100 | 2.38 |
| churn-random_forest-class_weight-platt-v1 | 0.8293 | 0.6285 | 0.1425 | 0.74 | 0.7376 | 0.1088 | $130050 | 18.89 |
| churn-random_forest-smote-isotonic-v1 | 0.8282 | 0.6140 | 0.1427 | 0.74 | 0.7234 | 0.1025 | $130950 | 20.09 |
| churn-random_forest-class_weight-isotonic-v1 | 0.8292 | 0.6251 | 0.1414 | 0.75 | 0.7447 | 0.1021 | $129650 | 17.49 |
| churn-random_forest-none-platt-v1 | 0.8304 | 0.6314 | 0.1419 | 0.78 | 0.7518 | 0.1291 | $128750 | 16.34 |
| churn-random_forest-none-isotonic-v1 | 0.8309 | 0.6319 | 0.1405 | 0.76 | 0.7589 | 0.1436 | $128550 | 14.97 |
| churn-random_forest-smote-raw-v1 | 0.8186 | 0.5847 | 0.1537 | 0.73 | 0.6809 | 0.1252 | $130150 | 8.90 |
| churn-random_forest-class_weight-raw-v1 | 0.8217 | 0.6105 | 0.1462 | 0.77 | 0.7518 | 0.0933 | $129300 | 6.97 |
| churn-random_forest-none-raw-v1 | 0.8171 | 0.6055 | 0.1477 | 0.77 | 0.7518 | 0.1039 | $127500 | 6.03 |

Total train time for all 27 configs: 204.4 s. The spread of test
profit across all 27 configs is $127,500 to $134,100, only 5%. On
this dataset the config choice barely moves profit. The threshold
choice moves it a lot: $78,750 at 0.5 against $132,150 at the
frozen 0.0934, a $53,400 gain (67.8%).

## Best per metric

| Metric | Best config | Value |
|---|---|---|
| ROC-AUC | churn-logistic-none-platt-v1 | 0.8420 |
| PR-AUC | churn-xgboost-none-platt-v1 | 0.6547 |
| Brier (lower is better) | churn-xgboost-class_weight-isotonic-v1 | 0.1372 |
| Precision at 100 | churn-xgboost-class_weight-raw-v1 | 0.84 |
| Precision at 10% | churn-xgboost-none-raw-v1 | 0.7943 |
| Test profit | churn-logistic-class_weight-isotonic-v1 | $134,100 |

No single config wins more than one of these. The champion is
second on none of them: its PR-AUC 0.6395 is 0.0152 below the best,
its Brier 0.1392 is 0.0020 above the best. It wins on the decision
metric it was selected for, out-of-fold profit, by $8.87 per 1,000
over the runner-up.

## Ablations

Each ablation averages one arm across the other two dimensions, so
each number is a mean of 9 configs.

### Class handling (mean PR-AUC)

| Handling | Mean PR-AUC | Mean test profit |
|---|---|---|
| none (threshold only) | 0.6357 | $130,583 |
| class_weight | 0.6352 | $131,061 |
| smote (in-fold) | 0.6231 | $131,667 |

SMOTE hurts ranking: mean PR-AUC 0.6231 against 0.6357 for doing
nothing, and the worst single config in the table is a SMOTE one
(random forest SMOTE raw, PR-AUC 0.5847). SMOTE survives in the
champion only because the profit selection picked a SMOTE config
by $8.87 per 1,000, which is noise. If the champion is ever
re-selected, the honest question is whether SMOTE earns its place
at all: on ranking it does not. class_weight is a wash on ranking
(0.6352 vs 0.6357) and a wash on profit. The threshold, not the
class handling, does the work here.

### Calibration (mean Brier and PR-AUC)

| Calibration | Mean Brier | Mean PR-AUC |
|---|---|---|
| raw (none) | 0.1515 | 0.6262 |
| Platt (sigmoid) | 0.1399 | 0.6357 |
| Isotonic | 0.1394 | 0.6321 |

Calibration earns its place: Brier improves 0.1515 to 0.1394, and
PR-AUC does not fall. The damage it repairs is visible in the raw
logistic configs: class_weight raw Brier 0.1688 and SMOTE raw
Brier 0.1678, against 0.1381 for the plain raw model. Reweighting
and resampling both break the probabilities, and calibration puts
them back. Platt and isotonic are a wash (0.1399 vs 0.1394); the
champion happens to use isotonic.

### Model (mean PR-AUC and ROC-AUC)

| Model | Mean PR-AUC | Mean ROC-AUC |
|---|---|---|
| XGBoost | 0.6481 | 0.8382 |
| Logistic Regression | 0.6296 | 0.8411 |
| Random Forest | 0.6163 | 0.8261 |

XGBoost ranks best, logistic ranks nearly as well on ROC-AUC
(0.8411, the best mean) and trains 5 to 10 times faster than the
forest. Random Forest is last on both ranking metrics and slowest
to train (up to 22.66 s per config). Nothing here justifies a
fourth model: LightGBM was not added, because no ablation shows a
gap a new booster would fill.

## Profit and threshold

Assumptions, stated plainly because every profit number depends
on them: a retention offer costs $50 per contacted customer (the
offer cost), and a customer kept from churning is worth $500 (the
saved value, about eight months at the mean MonthlyCharges of
$64.76). These are illustrative business figures, not measured
ones. The formula is one line: profit = true positives times
($500 minus $50) minus false positives times $50. Full curve in
`benchmarks/results/profit_curve.json`.

| Threshold | Contacted | Precision | Recall | Profit on test |
|---|---|---|---|---|
| 0.5 (default) | 275 | 0.6727 | 0.4947 | $78,750 |
| 0.0934 (frozen, champion) | 929 | 0.3851 | 0.9545 | $132,150 |

The break-even precision is the offer cost divided by the saved
value: $50 / $500 = 0.10. Any contacted group churning above 10%
pays for itself, which is why the profit-maximising threshold is
so low. At 0.0934 the contacted group churns at 38.51%, far above
break-even, and the model catches 95.45% of all churners. The
price is volume: 929 calls instead of 275. A team that can only
make 100 calls a day should read precision at 100 (0.76) instead,
not this table. Doing nothing (contact nobody) profits $0.

## Risk tiers

Fixed thresholds: LOW below 0.30, MEDIUM 0.30 to 0.60, HIGH at or
above 0.60. Measured churn rate per tier on the 1,409-row test
set, champion model:

| Tier | Customers | Churn rate |
|---|---|---|
| LOW | 873 | 10.54% |
| MEDIUM | 365 | 42.47% |
| HIGH | 171 | 74.27% |

The tiers separate: a HIGH customer churns about 7 times as often
as a LOW one. The LOW tier still churns at 10.54%, so LOW means
"call last", not "safe".

## Calibration curve

Champion reliability curve on test, 10 equal-width bins. Full data
in `benchmarks/results/reliability.json`.

| Predicted bin | n | Mean predicted | Fraction churned |
|---|---|---|---|
| 0.0 to 0.1 | 501 | 0.038 | 0.040 |
| 0.1 to 0.2 | 237 | 0.151 | 0.144 |
| 0.2 to 0.3 | 135 | 0.249 | 0.282 |
| 0.3 to 0.4 | 123 | 0.347 | 0.398 |
| 0.4 to 0.5 | 138 | 0.444 | 0.348 |
| 0.5 to 0.6 | 104 | 0.551 | 0.558 |
| 0.6 to 0.7 | 76 | 0.643 | 0.724 |
| 0.7 to 0.8 | 38 | 0.755 | 0.737 |
| 0.8 to 0.9 | 48 | 0.839 | 0.750 |
| 0.9 to 1.0 | 9 | 0.953 | 0.889 |

The curve sits close to the diagonal. The worst bins are 0.4 to
0.5 (predicted 0.444, actual 0.348, n=138) and 0.8 to 0.9
(predicted 0.839, actual 0.750, n=48): the model overstates risk
in both. The top bin has only 9 customers, so its 0.889 is noisy.
Compare with the uncalibrated SMOTE logistic model, Brier 0.1678:
calibration is what keeps this curve on the diagonal at all.

## Latency and cost

Measured 20 Nov 2026 with `scripts/measure_latency.py`: 200
held-out test rows, one at a time, through the API (TestClient,
in-process, includes HTTP handling and the SQLite log write).

| Metric | Value |
|---|---|
| Mean | 66.04 ms |
| p50 | 64.07 ms |
| p95 | 77.47 ms |
| p99 | 93.40 ms |

The raw model call alone is 61.97 ms at p50, so the cost is the
model, not the HTTP layer: the champion is a CalibratedClassifierCV
wrapping three fitted XGBoost models (300 trees each), and every
score runs all three. A single logistic model scores the same row
far faster, and DecisionAPI reports p50 9.8 ms for one. The
trade is explicit: the champion buys 0.0047 PR-AUC over the best
logistic config (0.6395 vs 0.6348) and pays about 6 times the
latency for it. If latency mattered more than ranking, the logistic
config would be the right champion. Cost is $0.00: local CPU, no
API calls.

## S6E3 scale run: pending

The Kaggle Playground Series S6E3 synthetic train set (about
594,000 rows, same schema) was not fetched: no Kaggle credentials
were present at `~/.kaggle/kaggle.json`. No scale numbers are
claimed anywhere in this repo. Train time and latency at the 594k
scale are pending, and the S6E3 numbers in the research report
(about 594,000 rows, first place 0.91856) are not cited as this
project's results. When credentials exist, the same pipeline runs
unchanged on the synthetic set and the numbers go here.

## Comparison with DecisionAPI

DecisionAPI (the closest existing project, read during research)
reports Logistic Regression, ROC-AUC 0.833 held-out, PR-AUC 0.625,
Brier 0.142, latency p50 9.8 ms. This project, same dataset and
same split shape: ROC-AUC 0.8366, PR-AUC 0.6395, Brier 0.1392,
latency p50 64.07 ms. Ranking and calibration are better, latency
is worse, for the reason above. DecisionAPI also lists its own
gaps (no tests, drift computed but not returned, threshold not
tuned); this project has tests (57), returns no drift in Phase 1
by design, and tunes the threshold on out-of-fold profit.

## How to reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.train            # 27 configs, about 204 s
python -m scripts.measure_latency  # 200 rows through the API
python -m pytest tests/ -q         # 57 passed
python -m scripts.ci_gate          # gate passes
```

## CI gate

Thresholds in `app/config.py`, set below the measured champion
numbers: ROC-AUC at least 0.80 (measured 0.8366), PR-AUC at least
0.55 (measured 0.6395), Brier at most 0.20 (measured 0.1392),
precision at 100 at least 0.50 (measured 0.76). The gate passes.
It has not yet run on GitHub Actions; the workflow
`.github/workflows/eval.yml` runs train, latency, tests, then the
gate on every push and PR, and its first run is pending repo
creation.

## Known limits

- No timestamps, so no temporal validation. The random split
  measures interpolation on a static snapshot, not performance on
  future customers. Phase 2 closes this with a longitudinal
  simulator.
- One row per customer, so no grouped split is possible.
- The champion margin is noise: the top 5 configs are within
  $142 per 1,000 on out-of-fold profit.
- SMOTE does not earn its place on ranking (mean PR-AUC 0.6231
  vs 0.6357 for threshold-only). It is in the champion because the
  profit rule picked it by $8.87 per 1,000.
- The reliability curve overstates risk in the 0.4 to 0.5 and 0.8
  to 0.9 bins.
- Latency p50 64.07 ms is about 6 times a single logistic model,
  because the champion scores three calibrated XGBoost models.
- No drift monitoring in Phase 1. The API has no drift fields and
  the log has none. That is Phase 2.
- No S6E3 scale numbers yet (pending, see above).
- The profit figures rest on the $50 offer cost and $500 saved
  value assumptions. Change them and the threshold and the profit
  move.

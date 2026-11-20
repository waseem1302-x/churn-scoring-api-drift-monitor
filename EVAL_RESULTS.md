# Eval Results, Phases 1 and 2

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
| 0.0934 (frozen, champion) | 927 | 0.3851 | 0.9545 | $132,150 |

The break-even precision is the offer cost divided by the saved
value: $50 / $500 = 0.10. Any contacted group churning above 10%
pays for itself, which is why the profit-maximising threshold is
so low. At 0.0934 the contacted group churns at 38.51%, far above
break-even, and the model catches 95.45% of all churners. The
price is volume: 927 calls instead of 275. A team that can only
make 100 calls a day should read precision at 100 (0.76) instead,
not this table. Doing nothing (contact nobody) profits $0.

## Profit sensitivity

The $50 offer cost and $500 saved value are assumptions, so this
table recomputes the best threshold and the test profit for a
grid of other assumptions, from the same recorded test curve
(`benchmarks/results/profit_sensitivity.json`, built by
`scripts/profit_sensitivity.py`). The gap column is what keeping
the frozen 0.0934 threshold costs in each cell.

| Offer cost | Saved value | Best threshold | Profit per 1,000 at best | Profit per 1,000 at frozen 0.0934 | Gap per 1,000 |
|---|---|---|---|---|---|
| $25 | $250 | 0.0775 | $47,107.88 | $46,894.96 | $212.92 |
| $25 | $500 | 0.0775 | $111,515.26 | $110,237.76 | $1,277.50 |
| $25 | $1000 | 0.0221 | $243,026.97 | $236,923.35 | $6,103.62 |
| $50 | $250 | 0.1702 | $33,356.99 | $30,447.13 | $2,909.86 |
| $50 | $500 | 0.0775 | $94,215.76 | $93,789.92 | $425.84 |
| $50 | $1000 | 0.0775 | $223,030.52 | $220,475.51 | $2,555.01 |
| $100 | $250 | 0.5004 | $13,307.31 | $-2,448.55 | $15,755.86 |
| $100 | $500 | 0.1702 | $66,713.98 | $60,894.25 | $5,819.73 |
| $100 | $1000 | 0.0775 | $188,431.51 | $187,579.84 | $851.67 |

How to read it. The best threshold moves from 0.0221 to 0.5004
across the grid, and it follows the cost-to-value ratio, not the
absolute dollars: cells with the same ratio pick the same
threshold (0.10 gives 0.0775, 0.20 gives 0.1702). The frozen
0.0934 stays within 2.6% of the best profit in 6 of the 9 cells
and within 8.8% in 8 of 9, so it carries over well when the
ratio stays near the assumed 0.10. The one bad cell is a $100
offer against a $250 saved value (ratio 0.40, break-even
precision 0.40): the frozen threshold contacts a group churning
at 38.51%, just below break-even, so it loses $2,448.55 per
1,000 while the best threshold (0.5004) earns $13,307.31. The
rule is plain: if the offer costs more than about a fifth of the
saved value, refreeze the threshold; 0.0934 does not carry over.
In the base cell the test-optimal threshold is 0.0775 against
the frozen 0.0934, a gap of $425.84 per 1,000 (0.45%), so the
threshold frozen on train is nearly optimal on test at the
assumed costs.

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
Re-measured after Phase 2 (the monitor's per-row quality flags
and the drift-flag log column): p50 63.52 ms against 64.07 ms
before, mean 69.70 ms against 66.04 ms, p95 99.10 ms against
77.47 ms, p99 122.84 ms against 93.40 ms. The p50 is unchanged,
so the monitor costs nothing measurable per request; the p95 and
p99 moved on a 1.5-shared-CPU VM, which is run-to-run jitter,
not a monitor cost (the one-time reference build happens in the
10-row warm-up, before the timed run).

| Metric | Phase 1 | Phase 2 |
|---|---|---|
| Mean | 66.04 ms | 69.70 ms |
| p50 | 64.07 ms | 63.52 ms |
| p95 | 77.47 ms | 99.10 ms |
| p99 | 93.40 ms | 122.84 ms |

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
latency p50 63.52 ms. Ranking and calibration are better, latency
is worse, for the reason above. DecisionAPI also lists its own
gaps (no tests, drift computed but not returned, threshold not
tuned); this project has tests (147), returns drift through
`GET /drift` and stores drift flags on every log row, and tunes
the threshold on out-of-fold profit.

## Phase 2: the simulator baseline

The simulator (`scripts/simulate.py`) builds a longitudinal
panel, 2,000 customers over 12 weeks, on the Telco schema, with
labels from a known logistic rule. The champion's measured
performance on clean simulator weeks is the reference the
retraining loop compares against, not the Telco test numbers:
the two labelling processes differ, and comparing against Telco
flagged every clean week.

| Metric | Clean simulator weeks (mean of 6) | Telco test |
|---|---|---|
| Precision at 100 | 0.6633 | 0.76 |
| PR-AUC | 0.5463 | 0.6395 |
| Churn rate | 0.2658 | 0.2654 |

## Phase 2: drift detection table

Reference: the Phase 1 training split (5,634 rows). Windows:
2,000 rows each. Delay is counted in windows: 0 means the
detector fired in the injected window itself. "n/a" means the
detector does not apply (KS and Wasserstein need numeric
features; InternetService is categorical), not "never detected".
Raw records: `benchmarks/results/drift_experiments.json`.

| Drift type | PSI delay | KS delay | Jensen-Shannon delay | Wasserstein delay | False alarms (6 clean windows) |
|---|---|---|---|---|---|
| Abrupt price rise (+$25 MonthlyCharges) | 0 | 0 | 0 | 0 | 0 on all four detectors |
| Gradual Fiber growth (+5 points per week) | 4 (at 25 points) | n/a | 2 (at 15 points) | n/a | 0 on all four detectors |
| Prevalence shift (churn rate 0.2658 to 0.3875) | invisible | invisible | invisible | invisible | 0; label shift flagged instead |
| TotalCharges breakage | not a distribution event | not a distribution event | not a distribution event | not a distribution event | caught by the data-quality channel (accumulation broken, 0.9115 of customers saw TotalCharges fall) |
| Null storm (40% of TotalCharges null) | not a distribution event | not a distribution event | not a distribution event | not a distribution event | caught by the data-quality channel (null rate 0.400 against 0.0014 reference) |
| Stale tenure (frozen at week 11) | not a distribution event | not a distribution event | not a distribution event | not a distribution event | caught by the data-quality channel (stale feature, 0.000 changed) |

Detection values at the abrupt rise (MonthlyCharges): PSI
1.631, KS statistic 0.3223 with p 0.0, Jensen-Shannon 0.317,
Wasserstein 0.806. Gradual Fiber values by step (5, 10, 15, 20,
25 points): PSI 0.009, 0.038, 0.086, 0.157, 0.252;
Jensen-Shannon 0.033, 0.069, 0.104, 0.140, 0.177. PSI crosses
0.1 at 20 points and 0.25 at 25; Jensen-Shannon crosses 0.1 at
15 points. Clean windows: drift_share 0.0 every week, all four
detectors 0 false alarms.

Prevalence deserves a row of its own. The churn rate rose
0.2658 to 0.3875 (a relative change of +0.4579) with no feature
change, so no input detector fired, which is the honest result:
there was nothing in the inputs to detect. The label-shift
channel flagged it, and the realised-performance channel did
not: precision at 100 ROSE by 0.1467 and PR-AUC rose by 0.1222,
because more churners overall means more churners in the top
100. A prevalence shift that lifts the base rate can make a
model look better while it is drifting, and the docs say so.

## Phase 2: threshold justification

The starting points came from the research report. The
measurements kept most of them and corrected one. Full
reasoning: `MONITORING.md`.

| Threshold | Value | What the measurement says |
|---|---|---|
| PSI act | 0.25 | The abrupt rise gives 1.631; clean windows never cross 0.1. |
| PSI warn | 0.1 | Gradual Fiber crosses 0.1 at 20 points of growth. |
| Jensen-Shannon | 0.1 | The abrupt rise gives 0.317; gradual Fiber crosses 0.1 at 15 points. |
| Wasserstein | 0.1 | The abrupt rise gives 0.806. Clean 2,000-row windows never cross 0.1, but a clean 60-row window gives 0.105, so the API refuses windows under 200 rows. |
| KS p | 0.05 | Reported, never the verdict. Flags the abrupt rise (p 0.0), quiet on clean windows. |
| drift_share | 0.5 | The abrupt rise moves it to 0.059, Fiber growth plus the price rise to 0.118. Dataset drift means most features moved, not one. |
| Predicted-rate relative move | 0.10 | The prevalence shift moves it +0.24; clean weeks move -0.04 to +0.01. |
| Precision at 100 drop | 0.05 | The concept drift drops it 0.033 in week 8 and 0.053 in week 11. Clean weeks never fire it (0.72 to 0.76, all above the 0.6633 baseline). |
| PR-AUC drop | 0.03 | The concept drift drops it 0.023 in week 6, 0.014 in week 7, 0.038 in week 8, which is where the trigger fires. Clean weeks never fire it (0.5482 to 0.5863, all above the 0.5463 baseline). |
| Promotion margin | $2,500 per 1,000 | The champion's own profit per 1,000 has a standard deviation of $3,010 across 6 clean weeks (range $8,870). |

## Phase 2: retraining log summary

12-week scheduled run, seed 42, 5,000 customers. Full log:
`benchmarks/results/retraining_log.jsonl`; weekly summaries:
`benchmarks/results/retraining_run.json`.

| Trigger | Week | Rule | Decision | Champion profit per 1,000 | Challenger profit per 1,000 | Margin |
|---|---|---|---|---|---|---|
| TRG-08-1 | 8 | performance_drop | promote | $144,435 | $153,925 | $9,490 |
| TRG-11-2 | 11 | performance_drop | refuse | $149,910 | $150,040 | $130 |

TRG-08-1: the concept drift (weeks 6 to 9) degraded the
champion's ranking until PR-AUC fell 0.0379 below the clean
baseline in week 8. The challenger, trained on weeks 5 to 8,
beat the champion on shadow weeks 9 and 10 by $9,490 per 1,000,
well above the $2,500 margin. Promoted.

TRG-11-2: the challenger, trained on weeks 8 to 11, beat the
champion on shadow week 12 by $130 per 1,000, far below the
$2,500 margin. Refused. A $130 win on 5,000 rows is noise, and
the loop correctly refused it.

The monitor detected the drift in both cases, but no trigger
fired for the price rise or the Fiber growth: they touched
features outside the top 5, and drift_share stayed under 0.5.
Detection and retraining are different decisions.

## Phase 2: pandas 2.3.3 re-verification

Evidently 0.7.23 cannot import on pandas 3.1.1 (`cannot import
name 'infer_dtype_from_object'`), so pandas is pinned to 2.3.3.
After the downgrade, the full 27-config retrain reproduced the
Phase 1 metrics bit for bit: same champion
(churn-xgboost-smote-isotonic-v1), ROC-AUC 0.8366, PR-AUC
0.6395, Brier 0.1392, precision at 100 0.76, threshold 0.0934.
`benchmarks/results/all_configs.json` differs only in
`train_time_s`; zero metric mismatches. The artifact is the same
byte size, 2,983,546.

## How to reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.train            # 27 configs, about 204 s
python -m scripts.measure_latency  # 200 rows through the API
python -m scripts.drift_experiments  # injection table
python -m scripts.retrain          # 12-week scheduled run
python -m scripts.evidently_report # two HTML reports
python -m pytest tests/ -q         # 147 passed
python -m scripts.ci_gate          # metric gate plus drift gate, both pass
```

## CI gate

Thresholds in `app/config.py`, set below the measured champion
numbers: ROC-AUC at least 0.80 (measured 0.8366), PR-AUC at least
0.55 (measured 0.6395), Brier at most 0.20 (measured 0.1392),
precision at 100 at least 0.50 (measured 0.76). The metric gate
passes. The drift gate checks the injection table and the
retraining log: abrupt drift caught in 0 windows, gradual Fiber
caught within 4 windows (PSI) and 2 (Jensen-Shannon), clean
windows quiet on all detectors, prevalence label shift flagged,
breakage, null storm, and stale tenure all quality-flagged, and
the retraining log holding one promotion and one refusal. The
drift gate passes. The workflow `.github/workflows/eval.yml`
runs train, latency, drift experiments, retrain, Evidently
reports, tests, then the gate on every push and PR. The first
Actions run after the Phase 1 push has not been checked yet.

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
- Latency p50 63.52 ms is about 6 times a single logistic model,
  because the champion scores three calibrated XGBoost models.
- The simulator is not the real world: its labels come from a
  known rule, so its baseline (precision at 100 0.6633, PR-AUC
  0.5463) differs from the Telco test numbers (0.76, 0.6395).
- Concept drift is caught late: the trigger fires in week 8, the
  third flipped week.
- A prevalence shift is invisible to input detectors by design;
  only the label-shift and realised-performance channels see it,
  after labels arrive.
- The promotion margin ($2,500 per 1,000) is a judgment call, set
  just above the shadow's noise floor.
- `GET /drift` needs 200 logged rows and has no labels.
- No S6E3 scale numbers yet (pending, see above).
- The profit figures rest on the $50 offer cost and $500 saved
  value assumptions. Change them and the threshold and the profit
  move; the Profit sensitivity section measures how much.

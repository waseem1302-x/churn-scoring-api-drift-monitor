# Architecture, Churn Scoring API plus Drift Monitor (Phases 1 and 2)

## What this system is

A small, honest, production-shaped churn scoring system over the
IBM Telco Customer Churn dataset. It scores one customer at a time
through a FastAPI endpoint, logs every score to SQLite, and reports
ranking, calibration, and profit metrics instead of accuracy. Phase
1 is the scoring API plus honest evaluation. Phase 2 adds the drift
monitor (hand-rolled detectors, four alert channels), the
longitudinal simulator with a known drift schedule, the retraining
loop with shadow promotion, and Evidently HTML reports. The full
monitoring story is in `MONITORING.md`.

## System overview

```
                        build time (scripts/)
telco.csv ──► data.py (load + verify) ──► split (stratified 80/20)
                                          ──► features.py (one Pipeline
                                              per model, preprocessing
                                              inside, SMOTE inside,
                                              calibration wrapping it)
                                          ──► train.py (27 configs,
                                              out-of-fold threshold,
                                              champion by out-of-fold
                                              profit)
                                          ──► artifacts/model.joblib

                        query time (app/)
POST /score ──► Pydantic validation ──► scoring.score_row (the same
   pipeline training saved) ──► probability + risk tier
   ──► monitor.row_quality_flags (per-row drift flags)
   ──► db.log_prediction (SQLite: inputs, probability, tier,
       model version, latency, drift flags)
GET /drift ──► recent logged rows vs the training reference
   ──► monitor.evaluate_window ──► the window's drift summary

                        monitoring (app/, scripts/)
app/monitor.py ──► hand-rolled PSI, KS, Jensen-Shannon,
   Wasserstein; four alert channels
scripts/simulate.py ──► longitudinal panel, known drift schedule
scripts/drift_experiments.py ──► detection delay, false alarms
scripts/retrain.py ──► shadow runs, promotion margin, log
scripts/evidently_report.py ──► Evidently HTML reports (clean and
   price-rise weeks), never on the request path

                        quality (scripts/, tests/)
scripts/train.py ──► benchmarks/results/all_configs.json
scripts/measure_latency.py ──► benchmarks/results/latency.json
scripts/ci_gate.py + .github/workflows/eval.yml ──► quality gate
   (metric gate plus drift gate)
tests/test_contract.py ──► the artifact reproduces its recorded scores
```

## Component map

| File | Purpose | Why it exists |
|---|---|---|
| `app/config.py` | All constants: split, features, tiers, profit assumptions, gate thresholds | One place to check every number. No magic numbers anywhere else. |
| `app/data.py` | Load and verify Telco; stratified split; column spec with leakage classes | Provenance and the leakage audit. The file is checked before it is used: 7,043 rows, 21 columns, 26.54% churn, 11 blank TotalCharges, 0 duplicate IDs. |
| `app/features.py` | `build_preprocessor`, `build_pipeline` (model x handling x calibration) | The leakage story in code: preprocessing is fitted inside the pipeline only, SMOTE lives between the preprocessor and the model, calibration wraps the whole pipeline with cv=3. |
| `app/metrics.py` | ROC-AUC, PR-AUC, Brier, precision at k, reliability curve | The measurement instrument. No accuracy function exists in this file, by design. |
| `app/profit.py` | Profit at a threshold, profit curve, best threshold | One formula, from the two explicit assumptions in config: profit = TP x ($500 - $50) - FP x $50. |
| `app/scoring.py` | `load_artifact`, `score_row`, `risk_tier` | The serving path. The API scores through the exact object training saved. |
| `app/db.py` | SQLite prediction log | The audit trail: every score with inputs, probability, tier, model version, latency, and drift flags (per-row quality flags plus the latest window alerts). `init_db` adds the `drift_flags` column to a Phase 1 database by migration. |
| `app/api.py` | FastAPI: `GET /health`, `POST /score`, `GET /metrics`, `GET /drift` | One scoring endpoint. Pydantic rejects unknown categories, negative tenure, and bad SeniorCitizen values with 422. `GET /drift` compares the recent logged rows (at least 200, at most 500) against the training reference with the same monitor the offline experiments use. |
| `app/monitor.py` | Hand-rolled PSI, KS, Jensen-Shannon, Wasserstein; four alert channels | The drift detector. The request path never depends on a report library. Thresholds are justified by the injection measurements; see `MONITORING.md`. |
| `scripts/simulate.py` | Longitudinal churn panel with a known drift schedule | The temporal validation Telco cannot give. Same schema as Telco, labels from a known rule, schedule written down before any result was measured. |
| `scripts/drift_experiments.py` | Controlled drift injection | Measures detection delay in windows and false alarms on clean windows, per detector. Writes `benchmarks/results/drift_experiments.json`. |
| `scripts/retrain.py` | Trigger rules, shadow runs, promotion margin | The retraining loop. A trigger starts a shadow run, never a promotion. Writes `benchmarks/results/retraining_log.jsonl` and `retraining_run.json`. |
| `scripts/evidently_report.py` | Scheduled Evidently HTML reports | Renders one clean week and one drifted week as HTML artifacts in `benchmarks/reports/`. Never on the request path. |
| `scripts/train.py` | Trains all 27 configs, picks the champion, writes the artifact and all result files | The experiment runner. The threshold is chosen on out-of-fold train predictions, never on test. |
| `scripts/measure_latency.py` | Scores 200 test rows through the API, reports p50/p95/p99 | Latency is measured through the real path, not the bare model call. |
| `scripts/ci_gate.py` | Fails the build if a champion metric regresses, or if injected drift is missed | The eval harness is a gate, not a report. The drift half checks the experiment table and the retraining log. |

## Request path (query time)

1. `POST /score` with the 19 model features (plus optional
   `customerID` for the log).
2. Pydantic validates every field. `TotalCharges` may be null; the
   pipeline imputes the training median.
3. `scoring.score_row` builds one DataFrame row and calls the
   shipped pipeline's `predict_proba`. `customerID` is never a
   model feature.
4. `risk_tier` maps the probability: below 0.30 LOW, 0.30 to 0.60
   MEDIUM, at or above 0.60 HIGH.
5. The score is logged to SQLite with a UTC timestamp, then the
   response is returned. The log row carries drift flags: the
   per-row quality check (null TotalCharges, out-of-range
   features) and the alerts from the latest evaluated window, if
   any window has been evaluated yet.

## Decisions, trade-offs, and rejected alternatives

### 1. One Pipeline per model vs notebook-style preprocessing

**Chosen: one scikit-learn Pipeline per model, preprocessing fitted
inside it only.**

**Why:** preprocessing fitted on the full data before the split is
the most common leakage in churn tutorials, and it is invisible in
a notebook. A Pipeline makes it impossible by construction: the
imputer and the encoder only ever see the data `fit` is called on,
and `cross_val_predict` refits them inside each fold. The test
`test_preprocessor_imputer_uses_train_median_only` proves it: the
fitted imputer holds the train median of TotalCharges, not the
full-data median.

**Cost:** one preprocessor for all three models, so the trees get a
StandardScaler they do not need. Scaling is monotonic, so it
cannot change a tree split; the cost is a few wasted columns of
arithmetic.

**Rejected:** fitting imputers and encoders on the full frame
before splitting (leaks), and a separate preprocessing script
whose output the model trusts blindly (train/serve skew).

### 2. Stratified random split vs temporal or grouped split

**Chosen: stratified random 80/20, seed 42.**

**Why:** it is the split every published baseline on this dataset
uses, so the numbers are comparable. The alternatives are not
available, not better: the dataset has no timestamps, so a
temporal split is impossible, and one row per customer makes a
grouped split vacuous (every group has size 1). Both limits are
stated in `DATA.md` and `EVAL_RESULTS.md`, not hidden.

**Cost:** the split measures interpolation on a static snapshot,
not performance on future customers. A model that scores 0.8366
ROC-AUC here can still decay in production, which is exactly what
Phase 2 measures.

**Rejected:** pretending the split is temporal, and inventing
timestamps to make it so.

### 3. Champion by out-of-fold profit vs best PR-AUC

**Chosen: highest out-of-fold expected maximum profit on train,
tie-break PR-AUC.**

**Why:** profit is the decision metric. A retention team does not
ship a ranking, it ships a call list, and the threshold that
maximises profit is the operating point. Choosing the threshold on
out-of-fold train predictions keeps the test set clean: the test
numbers are a check, not the selection.

**Measured:** the rule picks `churn-xgboost-smote-isotonic-v1`
($94,266.95 per 1,000 out-of-fold). The best PR-AUC config
(`churn-xgboost-none-platt-v1`, 0.6547) trails it by $363.86 per
1,000 out-of-fold. The best test profit belongs to yet another
config ($134,100 vs the champion's $132,150). All three are within
noise of each other, and the docs say so.

**Rejected:** picking the champion on test profit (selection on
the evaluation set), and picking on PR-AUC alone (ignores the
decision the model exists for).

### 4. Calibration: Platt and isotonic, measured, not assumed

**Chosen: every model raw, plus Platt, plus isotonic; keep what
earns its place.**

**Measured:** mean Brier 0.1515 raw, 0.1399 Platt, 0.1394 isotonic.
Calibration is the single clearest win in the ablation. The damage
it repairs is measurable: logistic class_weight raw Brier 0.1688
and SMOTE raw Brier 0.1678, against 0.1381 for the plain model.
Reweighting and resampling break the probabilities; calibration
puts them back. Platt and isotonic are a wash, so neither is
claimed better.

**Cost:** CalibratedClassifierCV with cv=3 fits the base model
three times, so a calibrated config costs about 3 times the train
time of its raw twin (champion 9.42 s vs 4.07 s raw), and scoring
runs three fitted models per call, which is most of the 64.07 ms
p50 latency.

**Rejected:** shipping uncalibrated probabilities (the tutorial
default), and calibrating on the test set.

### 5. XGBoost vs logistic vs Random Forest, and no LightGBM

**Chosen: three models, XGBoost champion, no fourth booster.**

**Measured (mean PR-AUC):** XGBoost 0.6481, logistic 0.6296,
Random Forest 0.6163. Logistic has the best mean ROC-AUC (0.8411)
and trains 5 to 10 times faster than the forest. Random Forest is
last on both ranking metrics and slowest (up to 22.66 s per
config). The champion beats the best logistic config by 0.0047
PR-AUC and pays about 6 times its latency.

**Rejected:** LightGBM. No ablation shows a gap a fourth booster
would fill, and three boosters for decoration is the tutorial
pattern this project is built to avoid. If a future ablation shows
XGBoost losing to a simpler model on the decision metric, the
champion changes.

### 6. SMOTE in-fold vs class_weight vs threshold-only

**Chosen: all three arms measured; the threshold does the work.**

**Measured (mean PR-AUC):** threshold-only 0.6357, class_weight
0.6352, SMOTE 0.6231. SMOTE hurts ranking, and the worst config in
the table is a SMOTE one (Random Forest SMOTE raw, 0.5847).
class_weight is a wash. The champion contains SMOTE only because
the profit rule picked it by $8.87 per 1,000, which is noise, and
the docs say so. If the champion is re-selected, SMOTE is the
first thing to question.

**Rejected:** SMOTE before the split (leaks test information into
training), and SMOTE as a default (it is an ablation arm, not a
habit).

### 7. SQLite log vs Postgres

**Chosen: SQLite, one table, gitignored.**

**Why:** one row per score, a few thousand rows in any realistic
run of this project. Postgres is a server for a file's worth of
data. The log exists so a past decision can be replayed, not to
serve analytics.

**Cost:** no concurrent-write story, no managed backups. At this
scale neither matters.

**Rejected:** Postgres (overhead, not engineering), and no log at
all (then a wrong score is unexplainable).

### 9. Hand-rolled detectors vs Evidently on the request path

**Chosen: hand-rolled PSI, KS, Jensen-Shannon, and Wasserstein in
`app/monitor.py`; Evidently only for scheduled HTML reports.**

**Why:** the request path must never depend on a report library.
Evidently is a reporting tool with a large dependency tree and a
version-sensitive API; a drift decision that depends on it breaks
when the library breaks. The four statistics are each a few lines,
and the repo owns them, tests them, and can explain them. Evidently
renders the human-readable report, which is a different job.

**Measured:** the hand-rolled detectors catch the abrupt price rise
in 0 windows and gradual Fiber in 2 to 4 windows, with 0 false
alarms on 6 clean windows. Evidently 0.7.23 renders the same two
weeks as HTML in about 12 seconds per report.

**Rejected:** Evidently on the request path (a library version
away from a broken API), and a single blended drift score (it
hides which thing changed).

### 10. Accumulating features are not distribution-drifted

**Chosen: tenure and TotalCharges are skipped in the per-feature
drift channel and monitored week-over-week instead, with a
growth-law check on TotalCharges.**

**Why:** against a fixed training reference they drift forever by
construction. Tenure PSI is 0.43 already in clean week 2 and 1.20
by week 6; TotalCharges PSI is 0.73 by week 6. Distribution drift
is the wrong test for a value that only ever rises. The right
tests are week-over-week: a stale feature (fewer than 1% of
customers changed) and an accumulation break (more than 5% of
customers saw TotalCharges fall).

**Cost:** excluding TotalCharges from distribution drift made its
breakage invisible at first. The growth-law check was added
because the experiment showed the blind spot, and the docs record
the sequence honestly: the fix for one blind spot created
another, and the check closes it.

**Rejected:** monitoring accumulating features against a fixed
reference (cries wolf every week), and dropping them from
monitoring entirely (their breakage is a real data failure).

### 11. The simulator is a feature, not a fudge

**Chosen: a longitudinal simulator with a known drift schedule,
labels from a known rule, written down before any result was
measured.**

**Why:** Telco has no timestamps, so no temporal validation is
possible on it, and no honest drift measurement is possible
without ground truth. A simulator whose schedule is written
before the measurements is the only way to measure detection
delay and false alarms, because the truth is known in advance.
The schedule is in `scripts/simulate.py` and `MONITORING.md`;
the results are checked against it, not fitted to it.

**Cost:** the simulator's labels come from a known logistic rule,
a different labelling process from Telco, so the champion's
measured performance on simulator weeks (precision at 100 0.6633,
PR-AUC 0.5463) differs from its Telco test numbers (0.76,
0.6395). The retraining loop compares against the simulator
baseline, not the Telco numbers; comparing against Telco flagged
every clean week.

**Rejected:** inventing timestamps on Telco (pretends the split
is temporal), and measuring drift on Telco alone (no ground
truth, so no delay or false-alarm measurement is possible).

### 12. Shadow promotion with a noise-floor margin

**Chosen: a trigger starts a shadow run, never a promotion. The
challenger must beat the champion on profit at the frozen
threshold by more than $2,500 per 1,000 customers on two
labelled weeks.**

**Why:** a retrain that promotes itself on one noisy week is how
a good model gets replaced by a lucky one. Two shadow weeks and
a margin exist for the same reason: the champion's own profit per
1,000 fluctuates with a standard deviation of $3,010 across clean
weeks, so a win smaller than the margin is not evidence. The
12-week run shows both sides: TRG-08-1 promoted on a $9,490
margin, TRG-11-2 was refused on a $130 margin.

**Cost:** a challenger that is genuinely better by less than the
margin waits for the next trigger. That is the price of not
promoting noise, and it is stated, not hidden.

**Rejected:** promoting on any win (promotes noise), promoting
on one week (too noisy to decide on), and retraining on a
data-quality alert (launders a data failure into the weights).

### 13. Trigger streaks: two windows, except performance drop

**Chosen: top-feature drift, dataset drift, and prediction drift
need two consecutive windows; performance drop needs one labelled
window. A clean week resets a streak.**

**Why:** one noisy window is not a retrain. A labelled window is
already a month of evidence given the 30-day label lag, so by the
time a second labelled window arrives, the model has been wrong
for two months. The asymmetry is deliberate and documented.

**Measured:** in the 12-week run, the price rise and Fiber growth
never triggered a retrain, because they touched features outside
the top 5 and drift_share stayed under 0.5. The monitor detected
them; the trigger rules are calibrated for drift on important
features. Detection and retraining are different decisions.

**Rejected:** firing on one window for every rule (alert
fatigue), and firing on data-quality issues (see decision 12).

### 14. pandas 2.3.3, because Evidently 0.7.23 cannot import on 3.1

**Chosen: pandas pinned to 2.3.3, evidently pinned to 0.7.23.**

**Why:** Evidently 0.7.23 is the latest 0.7.x release. It installs
on pandas 3.1.1 but `import evidently` fails with `cannot import
name 'infer_dtype_from_object' from 'pandas.core.dtypes.common'`,
the known pandas 3.1 break. Downgrading pandas to 2.3.3 makes the
import work. The Phase 1 metrics were re-verified after the
downgrade: the full 27-config retrain reproduced the champion and
every metric bit for bit (ROC-AUC 0.8366, PR-AUC 0.6395, Brier
0.1392, precision at 100 0.76, threshold 0.0934).

**Cost:** the repo stays one major version behind on pandas until
Evidently ships a 3.x-compatible release. The pin is recorded in
`requirements.txt` with the reason and the re-verification.

**Rejected:** running Evidently without the pin (it does not
import), and dropping Evidently (the HTML reports are the
human-readable half of the monitoring story).

### 15. GET /drift reads the log, not a live stream

**Chosen: `GET /drift` compares the most recent logged scores
(at least 200 rows, at most 500) against the training reference,
with the same monitor the offline experiments use.**

**Why:** the log is the only window the API has, and reusing the
offline monitor means the endpoint and the experiments cannot
disagree. A contract test scores drifted rows through `/score`
and checks the endpoint reports the same numbers the monitor
computes offline on the same frame.

**Measured:** a clean 60-row window has a MonthlyCharges
Wasserstein distance of 0.105, over the 0.1 threshold, while a
clean 200-row window sits at 0.065. Windows under 200 rows are
too small for the monitor's thresholds, which were validated on
2,000-row windows, so the endpoint refuses them with
`not_enough_data` instead of reporting noise.

**Cost:** the endpoint needs 200 logged rows before it says
anything, and it has no labels, so the realised-performance
channel is null there. Both limits are in the response and the
docs, not hidden.

**Rejected:** a smaller minimum window (reports noise as drift),
and a separate endpoint-only detector (two detectors that can
disagree).

### 8. Profit assumptions stated, not hidden

**Chosen: offer cost $50, saved value $500, in `app/config.py`,
repeated in every doc that cites a profit number.**

**Why:** every profit number is a function of these two numbers.
Hiding them would let a reader mistake an illustrative $132,150
for a measured business result. The break-even precision
($50 / $500 = 0.10) is the number that makes the low threshold
intelligible.

**Cost:** the assumptions are illustrative, so the profit is
illustrative too. A real deployment re-fits them to real costs and
re-runs the threshold selection, which is one function call.

## Invariants (rules the system never breaks)

1. **Split first, fit inside the pipeline.** No transformer,
   encoder, imputer, or resampler ever sees a test row.
2. **No accuracy headline.** The only accuracy in the repo is the
   do-nothing baseline, 73.46%, stated next to its explanation.
3. **Thresholds are frozen on train.** The operating threshold
   comes from out-of-fold train predictions. Test never selects.
4. **No invented numbers.** Every number in the README,
   `EVAL_RESULTS.md`, and `benchmarks/results/` comes from a real
   local run. A number that cannot be measured yet is left pending,
   as the S6E3 scale run is.
5. **The artifact is the model.** The API loads
   `artifacts/model.joblib` and scores through it. There is no
   separate serving code path.
6. **Every score is logged.** Inputs, probability, tier, model
   version, latency, timestamp. A decision with no log row did not
   happen.
7. **No committed database.** `data/predictions.db` is gitignored.
   The committed artifacts are the model and the result files.
8. **Drift thresholds come from measurements.** A threshold is
   set by the injection experiments, not chosen in advance and
   defended. If a measurement disagrees with a starting point,
   the measurement wins and the doc says so.
9. **A trigger is never a promotion.** Retraining promotes only
   on a labelled-window win above the margin. Data-quality
   issues never trigger a retrain.

## Failure modes

| Failure | Detection | Behaviour |
|---|---|---|
| Dataset file changed or wrong | `data.verify_dataset` raises on row count, columns, churn rate, blank TotalCharges, duplicate IDs | `scripts/train.py` refuses to run |
| Model artifact missing | `/score` returns 503 | API stays up, `/health` reports `not-loaded` |
| Bad score request | Pydantic validation | 422, nothing logged |
| Metric regression | `scripts/ci_gate.py` + GitHub workflow | Build fails; thresholds are pinned |
| Blank TotalCharges in a request | Pydantic allows null | Pipeline imputes the training median, same as the 11 blank training rows |
| Unknown category in a request | Pydantic validation, `handle_unknown="ignore"` as backstop | 422 at the API; the encoder would ignore it if it ever got through |
| Drift window under 200 rows | `GET /drift` returns `not_enough_data` | Windows that small are noise for the monitor's thresholds |
| Injected drift missed, or clean windows alert | `scripts/ci_gate.py` drift gate + GitHub workflow | Build fails; the experiment table is the evidence |
| Evidently report fails | `scripts/evidently_report.py` in the workflow | Build fails; the monitor itself is unaffected, it never depends on Evidently |

## Known limitations (Phases 1 and 2)

- **No timestamps, no temporal validation.** The random split
  measures interpolation, not the future. Phase 2 closes this with
  a longitudinal simulator, not with the real data.
- **Champion margin is noise.** The top 5 configs are within $142
  per 1,000 on out-of-fold profit.
- **SMOTE does not earn its place on ranking.** Mean PR-AUC 0.6231
  vs 0.6357 for threshold-only. It is in the champion by $8.87 per
  1,000.
- **Latency.** p50 63.52 ms, about 6 times a single logistic model,
  because the champion scores three calibrated XGBoost models.
  The monitor adds no measurable per-request latency (p50 64.07
  ms before, 63.52 ms after); the one-time reference build happens
  on the first score and is absorbed by warm-up.
- **The simulator is not the real world.** Its labels come from a
  known logistic rule, so the champion's simulator baseline
  (precision at 100 0.6633, PR-AUC 0.5463) differs from its Telco
  test numbers (0.76, 0.6395). The retraining loop compares
  against the simulator baseline.
- **Concept drift is caught late.** The MonthlyCharges flip
  degrades PR-AUC by 0.023 in week 6 and 0.014 in week 7, below
  the 0.03 rule, and the trigger fires only in week 8. A slow
  concept change costs three weeks before the loop acts.
- **Prevalence shift is invisible to input detectors, by
  design.** Only the label-shift and realised-performance channels
  see it, and only after labels arrive, about 30 days late in
  production.
- **The promotion margin is a judgment call.** $2,500 per 1,000
  is set just above the two-week shadow's noise floor, measured
  from the champion's clean-week profit standard deviation of
  $3,010. A challenger genuinely better by less than the margin
  waits for the next trigger.
- **GET /drift needs 200 logged rows and has no labels.** The
  realised-performance channel is null there.
- **Profit rests on two assumed numbers.** $50 and $500 are
  illustrative; change them and the threshold moves. The cost
  is measured in the Profit sensitivity table in EVAL_RESULTS.md
  (`benchmarks/results/profit_sensitivity.json`): the frozen
  0.0934 stays within 8.8% of the best profit in 8 of 9 cells,
  but loses money when the offer costs $100 against a $250 saved
  value, where the best threshold is 0.5004.
- **S6E3 scale run pending.** No 594k-row numbers exist yet.

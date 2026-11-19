# Architecture, Churn Scoring API plus Drift Monitor (Phase 1)

## What this system is

A small, honest, production-shaped churn scoring system over the
IBM Telco Customer Churn dataset. It scores one customer at a time
through a FastAPI endpoint, logs every score to SQLite, and reports
ranking, calibration, and profit metrics instead of accuracy. Phase
1 is the scoring API plus honest evaluation. There is no drift
monitor yet: that is Phase 2, and the API has no drift fields and
the log has none, by design.

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
   ──► db.log_prediction (SQLite: inputs, probability, tier,
       model version, latency)

                        quality (scripts/, tests/)
scripts/train.py ──► benchmarks/results/all_configs.json
scripts/measure_latency.py ──► benchmarks/results/latency.json
scripts/ci_gate.py + .github/workflows/eval.yml ──► quality gate
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
| `app/db.py` | SQLite prediction log | The audit trail: every score with inputs, probability, tier, model version, latency. |
| `app/api.py` | FastAPI: `GET /health`, `POST /score`, `GET /metrics` | One scoring endpoint. Pydantic rejects unknown categories, negative tenure, and bad SeniorCitizen values with 422. |
| `scripts/train.py` | Trains all 27 configs, picks the champion, writes the artifact and all result files | The experiment runner. The threshold is chosen on out-of-fold train predictions, never on test. |
| `scripts/measure_latency.py` | Scores 200 test rows through the API, reports p50/p95/p99 | Latency is measured through the real path, not the bare model call. |
| `scripts/ci_gate.py` | Fails the build if a champion metric regresses | The eval harness is a gate, not a report. |

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
   response is returned.

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

## Failure modes

| Failure | Detection | Behaviour |
|---|---|---|
| Dataset file changed or wrong | `data.verify_dataset` raises on row count, columns, churn rate, blank TotalCharges, duplicate IDs | `scripts/train.py` refuses to run |
| Model artifact missing | `/score` returns 503 | API stays up, `/health` reports `not-loaded` |
| Bad score request | Pydantic validation | 422, nothing logged |
| Metric regression | `scripts/ci_gate.py` + GitHub workflow | Build fails; thresholds are pinned |
| Blank TotalCharges in a request | Pydantic allows null | Pipeline imputes the training median, same as the 11 blank training rows |
| Unknown category in a request | Pydantic validation, `handle_unknown="ignore"` as backstop | 422 at the API; the encoder would ignore it if it ever got through |

## Known limitations (Phase 1)

- **No timestamps, no temporal validation.** The random split
  measures interpolation, not the future. Phase 2 closes this with
  a longitudinal simulator.
- **Champion margin is noise.** The top 5 configs are within $142
  per 1,000 on out-of-fold profit.
- **SMOTE does not earn its place on ranking.** Mean PR-AUC 0.6231
  vs 0.6357 for threshold-only. It is in the champion by $8.87 per
  1,000.
- **Latency.** p50 64.07 ms, about 6 times a single logistic model,
  because the champion scores three calibrated XGBoost models.
- **No drift monitoring.** No drift fields in the API or the log.
  Phase 2.
- **Profit rests on two assumed numbers.** $50 and $500 are
  illustrative; change them and the threshold moves.
- **S6E3 scale run pending.** No 594k-row numbers exist yet.

# Monitoring, Churn Scoring API (Phase 2)

How the drift monitor works, what each alert means, what triggers
a retrain, and how to read a drift report. Every threshold here is
justified by a measurement in `EVAL_RESULTS.md`; a threshold is
never moved to make an alert go away.

## The monitor, in one paragraph

The monitor compares a current window of scored customers against
the Phase 1 training reference (5,634 rows, scored by the
champion). It has four separate alert channels, never one blended
score: per-feature drift, prediction drift, data quality, and
realised performance on labelled windows. A blended score would
hide which thing changed; four channels say which thing changed.
The statistics are hand-rolled in `app/monitor.py` (PSI,
two-sample KS, Jensen-Shannon divergence, Wasserstein distance),
so the request path never depends on a report library. Evidently
is used only for scheduled HTML reports a human reads, never for a
live decision.

## The four alert channels

### 1. Per-feature drift

Each of the 19 model features is compared between the reference
and the window, with four detectors:

| Detector | What it measures | Threshold |
|---|---|---|
| PSI | How far the binned distribution moved | below 0.1 stable, 0.1 to 0.25 investigate, above 0.25 act |
| Two-sample KS | The largest gap between the two cumulative curves | p below 0.05 on small windows, reported only |
| Jensen-Shannon | How far apart the two distributions are (the square root of the divergence, natural log) | 0.1 |
| Wasserstein | How much mass moved, normalised by the reference standard deviation | 0.1 |

A feature is marked drifted when PSI, Jensen-Shannon, or
Wasserstein crosses its threshold. KS is reported for every
numeric feature but never drives the verdict: on windows of a
thousand rows or more, KS is over-sensitive and a tiny, irrelevant
difference becomes significant. It is the right test on small
windows, and the injection experiments measure it separately.

Two features are skipped in this channel, by design. Tenure and
TotalCharges are accumulating: tenure grows by one every month and
TotalCharges only ever rises, so against a fixed training
reference they "drift" forever by construction (tenure PSI 0.43
already in clean week 2, 1.20 by week 6). Distribution drift is
the wrong test for them. They are monitored week-over-week
instead, in the data-quality channel below. `drift_share` is the
fraction of the 17 monitored features marked drifted; the dataset
drift rule fires at 0.5.

### 2. Prediction drift

The champion's score distribution and its predicted churn rate,
window against reference. The score PSI threshold is 0.25; the
predicted-rate rule is a relative move above 0.10. The low
operating threshold (0.0934) saturates the predicted rate near
65%, so score shifts barely move it: in the 12-week scheduled
run, the predicted rate moved between -0.04 and +0.01 relative
under the price rise and the Fiber shift, and only the prevalence
shift moved it (+0.24 relative in week 9). A saturated rate is a
weak detector, and the docs say so; it is one channel of four,
not the only one.

### 3. Data quality

Things a distribution test cannot see, or sees too late:

- **Null rate**: a feature's null rate rising more than 0.05
  above the reference.
- **Schema change**: a column appearing or disappearing, or a new
  category value. Simulator bookkeeping columns (customerID,
  Churn, week, date) are ignored here, because the window frames
  legitimately carry them and the reference does not.
- **Stale feature**: a feature that stops changing week over week
  (fewer than 1% of customers changed). This is how a frozen
  tenure is caught.
- **Out of range**: more than 25% of a feature's values outside
  the reference min and max.
- **Accumulation broken**: an accumulating total (TotalCharges)
  must never fall week over week; more than 5% of customers seeing
  it fall means the column broke. This check exists because
  excluding TotalCharges from distribution drift made its
  breakage invisible: the fix for one blind spot created another,
  and this check closes it.

Data-quality issues are deliberately NOT retraining triggers.
Retraining on broken data launders a data failure into the
weights. A quality alert means fix the data, not the model.

### 4. Realised performance on labelled windows

When labels exist (the simulator provides them; production
labels arrive about 30 days late), the window's precision at 100
and PR-AUC are compared against the champion's measured
performance on clean simulator weeks (precision at 100 0.6633,
PR-AUC 0.5463), not against the Telco test numbers. The
simulator's labels come from a known rule, a different labelling
process from Telco, so comparing against Telco test metrics
flagged every clean week. The correct reference is the champion
on clean simulator weeks.

## The simulator and its known drift schedule

Telco has no timestamps, so no true temporal validation is
possible on it. The simulator (`scripts/simulate.py`) closes
that gap: a longitudinal panel, 2,000 customers by default over
12 weeks, one snapshot row per customer per week, on the same
Telco schema, so the Phase 1 champion scores it unchanged.
Labels come from a known logistic true rule, written in the
module: intercept -1.87, tenure -0.03, MonthlyCharges +0.015,
Contract month-to-month +0.9, Fiber optic +0.5, OnlineSecurity
Yes -0.4, TechSupport Yes -0.4, Electronic check +0.3. Baseline
churn is about 0.278, near Telco's 0.2654.

The drift schedule was written down before any result was
measured. It is a feature, not a fudge: a simulator with a known
schedule is the only way to measure detection delay and false
alarms honestly, because the truth is known in advance.

| Weeks | Event | What changes |
|---|---|---|
| 1 to 4 | Clean | Nothing. The baseline. |
| 5 onward | Abrupt price rise | MonthlyCharges +$25 for everyone. |
| 6 to 10 | Gradual Fiber growth | Fiber optic share +5 points per week. |
| 6 to 9 | Concept drift | The MonthlyCharges coefficient in the true rule flips from +0.015 to -0.015, recentred so the overall churn rate stays near baseline and only the ranking changes. High-spending customers now churn less. No feature distribution changes; only the label rule changes, so input detectors cannot see it. |
| 9 onward | Prevalence shift | Churn intercept +0.6. The churn rate rises, no feature changes. |
| 10 to 11 | Single-feature breakage | TotalCharges is set equal to MonthlyCharges. |
| 12 | Stale feature | Tenure is frozen at its week 11 value. |

`schedule=False` gives all-clean weeks, for controlled injections
and false-alarm measurement. In production, labels take about 30
days, so a shadow decision waits; in the simulation, labels are
immediate, so the loop runs week by week. The calendar is
compressed, the schedule and the decisions are the same.

## Trigger rules

A trigger starts a shadow run, never a promotion. Two
consecutive windows is the anti-fatigue rule: one noisy window is
not a retrain, and a clean week resets a streak.

| Rule | Condition | Streak |
|---|---|---|
| Top-feature drift | PSI above 0.25 on any top-5-importance monitored feature (Contract, OnlineSecurity, TechSupport, InternetService, MultipleLines) | 2 windows |
| Dataset drift | drift_share above 0.5 | 2 windows |
| Prediction drift | Predicted churn rate moves more than 10% relative | 2 windows |
| Performance drop | Realised precision at 100 falls more than 5 points, or PR-AUC falls more than 0.03, against the clean simulator baseline | 1 labelled window |

Performance drop needs a streak of 1, not 2, because a labelled
window is already a month of evidence given the label lag. By the
time a second labelled window arrives, the model has been wrong
for two months. Data-quality issues never trigger a retrain.

## The retraining loop

When a rule fires, the challenger is the champion configuration
refit on the last four labelled weeks. It scores the next two
labelled weeks alongside the champion (two weeks, not one,
because a single 2,000-row week is too noisy to promote on).
Promotion happens only when the challenger beats the champion on
profit at the frozen threshold (0.0934) on those weeks by more
than the promotion margin of $2,500 per 1,000 customers. The
margin is meant to sit above the noise floor: the champion's own
profit per 1,000 fluctuates across clean weeks, so a win smaller
than the margin is not evidence. A tie is not a win, and neither
is a $130 win.

The margin was quantified in Phase 3
(`scripts/margin_sensitivity.py`,
`benchmarks/results/margin_sensitivity.json`), and the honest
result is that $2,500 is a thin margin. Recomputed from the
champion's own profit on the four clean simulator weeks
($94,010.82, $95,474.66, $85,914.68, $93,782.96 per 1,000), the
single-week standard deviation is $4,319.60, and the shadow
decision pools two weeks, so its noise is $4,319.60 over the
square root of 2, which is $3,054.42. The Phase 2 docstring
quoted $3,010, which matches that pooled figure within $44, so
the quoted number is the pooled two-week noise, not the
single-week noise. Against that floor, $2,500 is 0.82 of one
pooled standard deviation, just below it, not above it. A
challenger no better than the champion still promotes about 21%
of the time by noise alone at $2,500; at $5,000 that falls to 5%.

Does $2,500 still stand? For the two decisions on record, yes,
and the sensitivity run proves it: the decisions are identical
at $1,000, $2,500 and $5,000, because the observed margins are
$130 and $9,490, far apart, and every margin from $130 to $9,490
gives the same two decisions. The margin is kept at $2,500
because no evidence here justifies changing it, but $5,000 is
the conservative choice for a future operator: it would cut the
false-promotion rate to 5% and would not change either decision.
The thinness is stated here instead of hidden.

Every trigger, every shadow result, and every promotion or
refusal is recorded in `benchmarks/results/retraining_log.jsonl`.
On promotion the challenger becomes the current model for later
weeks; the Phase 1 artifact file is never overwritten.

The 12-week scheduled run (seed 42, 5,000 customers) produced
two triggers, one promotion and one refusal:

- **TRG-08-1, week 8, performance drop.** The concept drift
  (weeks 6 to 9) degraded the champion's ranking until PR-AUC
  fell 0.0379 below the clean baseline. The challenger, trained
  on weeks 5 to 8, beat the champion $153,925 to $144,435 per
  1,000 on shadow weeks 9 and 10, a margin of $9,490, well above
  the $2,500 noise floor. Promoted.
- **TRG-11-2, week 11, performance drop.** The challenger,
  trained on weeks 8 to 11, beat the champion $150,040 to
  $149,910 per 1,000 on shadow week 12, a margin of $130, far
  below the $2,500 noise floor. Refused. A $130 win on 5,000
  rows is noise, and the loop correctly refused it.

The monitor detected the drift in both cases (drift_share moved,
label shift fired from week 5, data quality fired weeks 10 to
12), but the trigger rules are calibrated for drift on important
features, and the price rise and Fiber growth touched features
outside the top 5, so no trigger fired for them. That is a real
finding, not a bug: detection and retraining are different
decisions, and the docs keep them separate.

## How to read a drift report

Two reports live in `benchmarks/reports/`, generated by
`scripts/evidently_report.py` (Evidently 0.7.23, pandas 2.3.3):

- `drift_week02_clean.html`: simulator week 2, a clean week.
  The reference and the window agree; the report shows no drift.
- `drift_week05_price_rise.html`: simulator week 5, the abrupt
  price rise. MonthlyCharges is the drifted feature; the report
  shows its distribution shifted by $25.

Read them side by side. The clean report is what "nothing
changed" looks like in this format; the price-rise report is
what a real shift looks like. A report that shows drift on a
feature the team did not change is the first sign of a data
problem, not a model problem.

The live endpoint, `GET /drift`, returns the same monitor's
summary over the most recent logged scores (at least 200 rows,
at most 500), against the Phase 1 training reference. It has no
labels, so the realised-performance channel is null there. Each
logged prediction row carries `drift_flags`: the per-row quality
check, and the alerts from the latest evaluated window.

## Thresholds and what justifies them

| Threshold | Value | Justification |
|---|---|---|
| PSI act | 0.25 | The abrupt price rise gives PSI 1.631; clean windows give 0. Clean 2,000-row windows never cross 0.1. |
| PSI warn | 0.1 | Gradual Fiber crosses 0.1 at 20 points of growth, 0.25 at 25 points. The warn band is where investigation starts. |
| Jensen-Shannon | 0.1 | The abrupt rise gives 0.317; gradual Fiber crosses 0.1 at 15 points. Clean windows never cross 0.1. |
| Wasserstein | 0.1 | The abrupt rise gives 0.806. Clean 2,000-row windows never cross 0.1, but a clean 60-row window gives 0.105, so windows under 200 rows are too small for this threshold and the API refuses them. |
| KS p | 0.05 | Reported, never the verdict. On 2,000-row windows KS flags the abrupt rise (p 0.0) and stays quiet on clean windows. |
| drift_share | 0.5 | The abrupt rise moves drift_share to 0.059; with Fiber growth added it reaches 0.118. Neither reaches 0.5, which is the point: dataset drift means most features moved, not one. |
| Predicted-rate relative move | 0.10 | The prevalence shift moves it +0.24; clean weeks move -0.04 to +0.01. |
| Precision at 100 drop | 0.05 | The concept drift drops it 0.033 in week 8 (firing on the PR-AUC rule instead), 0.053 in week 11. Clean weeks never fire it: their precision at 100 is 0.72 to 0.76, all above the 0.6633 baseline. |
| PR-AUC drop | 0.03 | The concept drift drops PR-AUC 0.023 in week 6, 0.014 in week 7, 0.038 in week 8, which is where the trigger fires. Clean weeks never fire it: their PR-AUC is 0.5482 to 0.5863, all above the 0.5463 baseline. The rule is tight by design: it fires on the third flipped week, not the first. |
| Promotion margin | $2,500 per 1,000 | The champion's own profit per 1,000 has a single-week standard deviation of $4,319.60 across the 4 clean weeks, so the pooled two-week shadow noise is $3,054.42. $2,500 is 0.82 of that pooled std, a thin margin; the two recorded decisions are the same at $1,000, $2,500 and $5,000. |

Full detection table with delays and false alarms:
`EVAL_RESULTS.md`. Raw records:
`benchmarks/results/drift_experiments.json`,
`benchmarks/results/retraining_log.jsonl`,
`benchmarks/results/retraining_run.json`.

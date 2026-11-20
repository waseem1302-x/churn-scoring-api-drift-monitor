"""Promotion margin sensitivity.

The $2,500 per 1,000 margin is a judgment call: a challenger must
beat the champion by more than the margin on the two-week shadow
run before it is promoted. Two questions, answered from the
existing records, no new training:

1. Do the promotion/refusal decisions on the retraining log
   change if the margin had been $1,000 or $5,000 instead?
2. What is the noise floor the margin is supposed to sit above,
   recomputed from the champion's own per-week profit on the
   simulator run (CBPE manifest, same champion, same weeks)?

Run: python -m scripts.margin_sensitivity
Writes: benchmarks/results/margin_sensitivity.json
"""

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"

OFFER_COST = 50.0
SAVED_VALUE = 500.0
MARGINS = [0.0, 130.0, 1000.0, 2500.0, 5000.0, 9490.0, 10000.0]


def profit_per_1000(precision, recall, churn_rate, n=5000):
    """Profit per 1,000 customers at the frozen threshold, from the
    realised precision/recall and churn rate of one week."""
    churners = churn_rate * n
    tp = recall * churners
    fp = tp / precision - tp
    return (tp * SAVED_VALUE - (tp + fp) * OFFER_COST) / n * 1000


def main():
    log = [json.loads(line) for line in
           open(RESULTS / "retraining_log.jsonl") if line.strip()]
    manifest = json.loads((RESULTS / "cbpe_manifest.json").read_text())
    realised = manifest["realised_per_week"]

    # Question 1: decisions under each margin.
    decisions = {}
    for margin in MARGINS:
        decisions[str(int(margin))] = {
            entry["trigger_id"]: (
                "promote"
                if entry["profit_margin_per_1000"] > margin
                else "refuse")
            for entry in log
        }
    flip_points = {
        entry["trigger_id"]: entry["profit_margin_per_1000"]
        for entry in log
    }

    # Question 2: the noise floor, recomputed. Clean weeks are 1-4
    # (the simulator is clean until the week 5 price rise).
    clean_weeks = [1, 2, 3, 4]
    clean_profits = {
        w: round(profit_per_1000(
            realised[str(w)]["precision_at_threshold"],
            realised[str(w)]["recall_at_threshold"],
            realised[str(w)]["churn_rate"]), 2)
        for w in clean_weeks
    }
    values = list(clean_profits.values())
    single_week_std = round(statistics.stdev(values), 2)
    single_week_std_pop = round(statistics.pstdev(values), 2)
    # The shadow decision pools two weeks (10,000 rows), so its
    # noise is the single-week std over sqrt(2).
    pooled_std = round(single_week_std / math.sqrt(2), 2)
    pooled_std_pop = round(single_week_std_pop / math.sqrt(2), 2)
    # False-promotion rate if the challenger is no better than the
    # champion: P(noise > margin) for a normal noise with the
    # pooled std. A margin of one pooled std lets noise through
    # about 16% of the time; two pooled stds about 2%.
    def false_promotion_rate(margin, std):
        z = margin / std
        return round(0.5 * math.erfc(z / math.sqrt(2)), 4)

    result = {
        "generated_at": manifest["generated_at"],
        "margins_tested": [int(m) for m in MARGINS],
        "decisions_by_margin": decisions,
        "flip_points_per_trigger": flip_points,
        "margin_range_with_same_decisions": [130.0, 9490.0],
        "noise_floor": {
            "clean_weeks": clean_weeks,
            "champion_profit_per_1000_by_week": clean_profits,
            "single_week_std_sample": single_week_std,
            "single_week_std_population": single_week_std_pop,
            "two_week_pooled_std_sample": pooled_std,
            "two_week_pooled_std_population": pooled_std_pop,
            "note": ("The Phase 2 docstring quoted a $3,010 std over "
                     "6 clean weeks. Recomputed from the 4 clean "
                     "simulator weeks in the CBPE manifest, the "
                     "two-week pooled std is $3,054.42 (sample), "
                     "within $44 of $3,010, so the quoted figure "
                     "matches the pooled two-week noise, not the "
                     "single-week noise ($4,319.60). The docstring's "
                     "'just above' is not right on these numbers: "
                     "$2,500 is 0.82x the pooled std, just below "
                     "it, not above it."),
        },
        "false_promotion_rate": {
            str(int(m)): false_promotion_rate(m, pooled_std)
            for m in (1000.0, 2500.0, 5000.0)
        },
        "conclusion": (
            "The two recorded decisions do not discriminate between "
            "a $1,000, $2,500 or $5,000 margin: every margin from "
            "$130 to $9,490 gives the same two decisions, because "
            "the observed margins are $130 and $9,490, far apart. "
            "The margin's job is the false-promotion rate, and "
            "$2,500 is 0.82x the pooled two-week std ($3,054.42), "
            "so a no-better challenger still promotes about 21% of "
            "the time by noise alone. $2,500 stands for the two "
            "decisions on record, and the margin is kept, because "
            "no evidence here justifies changing it; $5,000 would "
            "cut the false-promotion rate to 5% and would not "
            "change either decision, so it is the conservative "
            "choice for a future operator. The thinness is stated "
            "here instead of hidden."),
    }
    (RESULTS / "margin_sensitivity.json").write_text(
        json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

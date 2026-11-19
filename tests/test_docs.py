"""Doc integrity tests: no accuracy headline, no em-dash, honest numbers.

Accuracy is never a headline in this repo: the only accuracy allowed in
the docs is the do-nothing baseline, stated next to its explanation.
Em-dashes read as AI-generated, so no doc file may contain one.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "ARCHITECTURE.md", "EVAL_RESULTS.md", "DATA.md"]


def test_no_em_dash_in_docs():
    for name in DOCS:
        text = (ROOT / name).read_text()
        assert "\u2014" not in text, f"em-dash found in {name}"


def test_no_accuracy_headline_in_docs():
    for name in DOCS:
        for line in (ROOT / name).read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                lowered = stripped.lower()
                # A heading that says "no accuracy" is the rule itself,
                # not an accuracy headline. Anything else mentioning
                # accuracy as a result fails.
                assert "accuracy" not in lowered or "no accuracy" in lowered, (
                    f"{name}: {line}")


def test_do_nothing_baseline_is_stated():
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    assert "do-nothing" in text
    assert "73.4" in text or "73.46" in text


def test_eval_results_states_emp_assumptions():
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    assert "offer cost" in text.lower()
    assert "saved value" in text.lower() or "lifetime value" in text.lower()


def test_eval_results_states_no_timestamps_limit():
    text = (ROOT / "EVAL_RESULTS.md").read_text()
    assert "no timestamps" in text.lower()


def test_s6e3_scale_run_is_not_claimed_when_pending():
    import json

    text = (ROOT / "EVAL_RESULTS.md").read_text()
    metrics_path = ROOT / "benchmarks" / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("s6e3_scale_run") == "pending":
        assert "pending" in text.lower()
        assert "S6E3" in text
        # No S6E3 train time may be presented as measured while pending.
        assert metrics["s6e3"].get("train_time_s") is None

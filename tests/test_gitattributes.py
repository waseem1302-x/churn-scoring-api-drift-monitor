"""Git attributes tests: generated HTML reports stay out of language stats.

TDD: these tests fail until .gitattributes marks the Evidently HTML
reports as generated. GitHub counts every .html file as HTML code, so
the two drift reports (about 8.1 MB together) made HTML the top
language on the profile card at 89.61 percent, above Python at 8.15
percent. Marking them linguist-generated keeps the files in the repo
as evidence but drops them from GitHub's language counts, so Python
becomes the top language again. The reports themselves are unchanged.
"""

import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPORTS = [
    "benchmarks/reports/drift_week02_clean.html",
    "benchmarks/reports/drift_week05_price_rise.html",
]

PATTERN = "benchmarks/reports/*.html linguist-generated=true"


def _attributes():
    return (ROOT / ".gitattributes").read_text(encoding="utf-8")


def test_gitattributes_exists():
    assert (ROOT / ".gitattributes").is_file()


def test_reports_marked_generated():
    assert PATTERN in _attributes()


def test_reports_exist_and_match_pattern():
    for rel in REPORTS:
        assert (ROOT / rel).is_file()
        assert fnmatch.fnmatch(rel, "benchmarks/reports/*.html")

"""Docker image tests: multi-stage build, slim runtime, recorded size.

TDD: these tests fail until the Dockerfile is multi-stage and the
requirements are split. The runtime stage ships only what the API
needs: the API code, the artifacts, the Telco file the drift
reference reads, and the metrics file /metrics serves. Evidently
is reporting-only and pytest is test-only, so both live in
requirements-dev.txt and never enter the image. The benchmark
HTML reports and the CBPE frame stay out too.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _dockerfile():
    return (ROOT / "Dockerfile").read_text()


def test_dockerfile_is_multi_stage():
    text = _dockerfile()
    stages = re.findall(r"(?m)^FROM\s+\S+\s+AS\s+(\S+)", text)
    assert stages == ["builder", "runtime"]


def test_runtime_stage_copies_only_what_the_api_needs():
    text = _dockerfile()
    # The runtime stage starts at the second FROM.
    runtime = text.split("FROM", 2)[2]
    assert "COPY app/" in runtime
    assert "COPY artifacts/" in runtime
    assert "data/raw/telco.csv" in runtime
    assert "COPY benchmarks/results/metrics.json" in runtime
    # No wholesale benchmarks copy (the HTML reports are 8 MB),
    # no whole data/ copy, no Evidently, no test code, no scripts.
    assert "COPY benchmarks/ ./benchmarks/" not in runtime
    assert "COPY data/ " not in runtime
    assert "COPY tests/" not in runtime
    assert "COPY scripts/" not in runtime
    assert "evidently" not in runtime.lower()


def test_runtime_requirements_exclude_reporting_and_tests():
    runtime = (ROOT / "requirements.txt").read_text().lower()
    assert "evidently" not in runtime
    assert "pytest" not in runtime
    dev = (ROOT / "requirements-dev.txt").read_text().lower()
    assert "evidently" in dev
    assert "pytest" in dev


def test_container_listens_on_port_env():
    # Cloud Run sets $PORT (default 8080) and requires the container
    # to listen on it; locally $PORT is unset and 8000 applies.
    # The Dockerfile CMD and the Buildah build script must agree.
    text = _dockerfile()
    assert "${PORT:-8000}" in text
    build = (ROOT / "scripts" / "build_image.sh").read_text()
    assert "${PORT:-8000}" in build


def test_image_size_is_recorded_and_smaller():
    path = ROOT / "benchmarks" / "results" / "docker_image.json"
    if not path.exists():
        pytest.fail("docker_image.json missing: run "
                    "scripts/build_image.sh first")
    data = json.loads(path.read_text())
    assert data["tag"] == "churn-scoring-api:phase3"
    # Phase 2 was 1.39 GB. Multi-stage must beat it by a clear margin.
    assert data["size_bytes"] < 900_000_000, data["size_bytes"]
    text = (ROOT / "README.md").read_text()
    assert f"{data['size_mb']:.0f} MB" in text or \
        f"{data['size_gb']:.2f} GB" in text

"""Docker on a normal machine.

This cannot be verified in this VM: Docker cannot start
containers here (seccomp blocks setns), so the image is built
and smoke-tested with Buildah instead. This test guards the
closest verifiable work: a run script and the exact Docker-host
commands in the README, plus an honest statement that a Docker
host check has not happened. Nothing here claims a Docker check
that did not happen.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_script_exists_and_uses_the_image():
    script = (ROOT / "scripts" / "run_docker.sh").read_text()
    assert "churn-scoring-api:phase3" in script
    assert "docker run" in script
    assert "CHURN_API_KEY" in script
    assert "8000:8000" in script
    assert "health" in script


def test_readme_states_docker_host_check_is_pending():
    text = (ROOT / "README.md").read_text().lower()
    assert "docker on a normal machine" in text
    assert "not yet verified on a docker host" in text
    assert "docker build -t churn-scoring-api:phase3 ." in text
    assert "bash scripts/run_docker.sh" in text

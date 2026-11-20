"""Shared test fixtures.

The API requires an API key (CHURN_API_KEY, header X-API-Key) on
/score, /drift and /reload. Tests run with a fixed test key unless a
test explicitly removes it to check the unconfigured behaviour.
"""

import os

import pytest

TEST_API_KEY = "test-key"
AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


@pytest.fixture(autouse=True)
def api_key_env():
    """Give every test a configured API key, restored afterwards."""
    previous = os.environ.get("CHURN_API_KEY")
    os.environ["CHURN_API_KEY"] = TEST_API_KEY
    yield
    if previous is None:
        os.environ.pop("CHURN_API_KEY", None)
    else:
        os.environ["CHURN_API_KEY"] = previous

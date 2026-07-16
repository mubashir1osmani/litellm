"""Guardrails suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy-liveness hard-fail, and the
`e2e`/`covers` markers live in the parent tests/e2e/conftest.py. GuardrailsClient
holds the shared Gateway, so the `resources` fixture cleans up keys this suite
creates and teams/guardrails registered via resources.defer.
"""

import pytest

from guardrails_client import GuardrailsClient, build_client


@pytest.fixture(scope="session")
def client() -> GuardrailsClient:
    return build_client()

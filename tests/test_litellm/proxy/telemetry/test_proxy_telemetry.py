"""
Tests for the opt-in proxy usage telemetry sender.

These assert the gating contract (inert unless enabled AND a telemetry key is
set), the exact PostHog capture payload, that a dedicated env var is used (not
the LLM-analytics `POSTHOG_API_KEY`), and that transport failures never escape.
"""

import json
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

import pytest

from litellm.proxy.telemetry.proxy_telemetry import (
    DEFAULT_POSTHOG_HOST,
    ProxyTelemetry,
    ProxyTelemetryRegistry,
    init_proxy_telemetry,
    resolve_distinct_id,
)


@dataclass
class RecordingPoster:
    """Typed fake for AsyncPoster that records POSTs; optionally raises to exercise error handling."""

    calls: List[Tuple[str, str, Mapping[str, str]]] = field(default_factory=list)
    raise_exc: Optional[Exception] = None

    async def post(self, url: str, *, content: str, headers: Mapping[str, str]) -> object:
        self.calls.append((url, content, headers))
        if self.raise_exc is not None:
            raise self.raise_exc
        return object()


@pytest.fixture(autouse=True)
def _clean_registry_and_env(monkeypatch):
    for var in (
        "LITELLM_TELEMETRY_POSTHOG_KEY",
        "LITELLM_TELEMETRY_POSTHOG_HOST",
        "LITELLM_TELEMETRY_DISTINCT_ID",
        "POSTHOG_API_KEY",
        "POSTHOG_API_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    ProxyTelemetryRegistry.set(None)
    yield
    ProxyTelemetryRegistry.set(None)


def test_disabled_flag_is_inert(monkeypatch):
    monkeypatch.setenv("LITELLM_TELEMETRY_POSTHOG_KEY", "phc_present")
    assert init_proxy_telemetry(enabled=False, client=RecordingPoster()) is None
    assert ProxyTelemetryRegistry.get() is None


def test_enabled_without_key_is_inert():
    assert init_proxy_telemetry(enabled=True, client=RecordingPoster()) is None
    assert ProxyTelemetryRegistry.get() is None


def test_does_not_use_llm_analytics_posthog_key(monkeypatch):
    # POSTHOG_API_KEY belongs to the PostHogLogger integration; telemetry must ignore it.
    monkeypatch.setenv("POSTHOG_API_KEY", "phc_llm_analytics")
    assert init_proxy_telemetry(enabled=True, client=RecordingPoster()) is None
    assert ProxyTelemetryRegistry.get() is None


def test_enabled_with_key_registers_sender(monkeypatch):
    monkeypatch.setenv("LITELLM_TELEMETRY_POSTHOG_KEY", "phc_abc")
    poster = RecordingPoster()
    telemetry = init_proxy_telemetry(enabled=True, client=poster)
    assert telemetry is not None
    assert telemetry.api_key == "phc_abc"
    assert telemetry.host == DEFAULT_POSTHOG_HOST
    assert ProxyTelemetryRegistry.get() is telemetry


def test_host_override_and_trailing_slash_stripped(monkeypatch):
    monkeypatch.setenv("LITELLM_TELEMETRY_POSTHOG_KEY", "phc_abc")
    monkeypatch.setenv("LITELLM_TELEMETRY_POSTHOG_HOST", "https://eu.i.posthog.com/")
    telemetry = init_proxy_telemetry(enabled=True, client=RecordingPoster())
    assert telemetry is not None
    assert telemetry.host == "https://eu.i.posthog.com"


@pytest.mark.asyncio
async def test_acapture_posts_expected_capture_payload():
    poster = RecordingPoster()
    telemetry = ProxyTelemetry(api_key="phc_key", host="https://us.i.posthog.com", distinct_id="id-123", client=poster)

    await telemetry.acapture("litellm_proxy_request", {"endpoint": "/chat/completions", "provider": "openai"})

    assert len(poster.calls) == 1
    url, content, headers = poster.calls[0]
    assert url == "https://us.i.posthog.com/capture/"
    assert headers["Content-Type"] == "application/json"
    body = json.loads(content)
    assert body["api_key"] == "phc_key"
    assert body["event"] == "litellm_proxy_request"
    assert body["distinct_id"] == "id-123"
    assert body["properties"]["endpoint"] == "/chat/completions"
    assert body["properties"]["provider"] == "openai"


@pytest.mark.asyncio
async def test_acapture_swallows_transport_errors():
    poster = RecordingPoster(raise_exc=RuntimeError("posthog down"))
    telemetry = ProxyTelemetry(api_key="phc_key", host="https://us.i.posthog.com", distinct_id="id", client=poster)
    # Must not raise: telemetry can never break a request.
    await telemetry.acapture("litellm_proxy_started", {"litellm_version": "9.9.9"})
    assert len(poster.calls) == 1


def test_resolve_distinct_id_prefers_env_override(monkeypatch):
    monkeypatch.setenv("LITELLM_TELEMETRY_DISTINCT_ID", "fixed-id")
    assert resolve_distinct_id() == "fixed-id"


def test_resolve_distinct_id_is_stable_across_calls():
    first = resolve_distinct_id()
    second = resolve_distinct_id()
    assert first == second
    assert first

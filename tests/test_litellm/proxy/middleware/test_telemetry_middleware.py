"""
Tests for TelemetryMiddleware.

Drives real requests through a FastAPI app with an injected recording sender and
asserts that a `litellm_proxy_request` event carries the route template (not the
raw path), method, version, and the provider from the `x-litellm-provider`
response header; that infra/unmatched routes are skipped; and that a disabled
sender is a true no-op.
"""

import json
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from litellm.proxy.middleware.telemetry_middleware import TelemetryMiddleware
from litellm.proxy.telemetry.proxy_telemetry import ProxyTelemetry


@dataclass
class RecordingPoster:
    calls: List[Tuple[str, str, Mapping[str, str]]] = field(default_factory=list)

    async def post(self, url: str, *, content: str, headers: Mapping[str, str]) -> object:
        self.calls.append((url, content, headers))
        return object()


def _build_client(*, telemetry: Optional[ProxyTelemetry], version: str = "9.9.9") -> TestClient:
    app = FastAPI()

    @app.post("/chat/completions")
    async def chat():
        return JSONResponse({"ok": True}, headers={"x-litellm-provider": "openai"})

    @app.get("/models")
    async def models():
        return JSONResponse({"ok": True})

    @app.get("/health")
    async def health():
        return JSONResponse({"ok": True})

    app.add_middleware(TelemetryMiddleware, telemetry_provider=lambda: telemetry, proxy_version=version)
    return TestClient(app)


def _events(poster: RecordingPoster) -> List[dict]:
    return [json.loads(content) for _url, content, _headers in poster.calls]


def test_matched_route_emits_event_with_provider_and_version():
    poster = RecordingPoster()
    telemetry = ProxyTelemetry(api_key="phc", host="https://h", distinct_id="id", client=poster)
    client = _build_client(telemetry=telemetry)

    resp = client.post("/chat/completions")
    assert resp.status_code == 200

    events = _events(poster)
    assert len(events) == 1
    props = events[0]["properties"]
    assert events[0]["event"] == "litellm_proxy_request"
    assert props["endpoint"] == "/chat/completions"
    assert props["method"] == "POST"
    assert props["provider"] == "openai"
    assert props["litellm_version"] == "9.9.9"


def test_route_template_not_raw_path():
    poster = RecordingPoster()
    telemetry = ProxyTelemetry(api_key="phc", host="https://h", distinct_id="id", client=poster)
    app = FastAPI()

    @app.get("/key/{key_id}/info")
    async def key_info(key_id: str):
        return JSONResponse({"ok": key_id})

    app.add_middleware(TelemetryMiddleware, telemetry_provider=lambda: telemetry, proxy_version="1.2.3")
    client = TestClient(app)

    client.get("/key/sk-abc-123/info")

    events = _events(poster)
    assert len(events) == 1
    assert events[0]["properties"]["endpoint"] == "/key/{key_id}/info"


def test_provider_absent_when_no_header():
    poster = RecordingPoster()
    telemetry = ProxyTelemetry(api_key="phc", host="https://h", distinct_id="id", client=poster)
    client = _build_client(telemetry=telemetry)

    client.get("/models")

    events = _events(poster)
    assert len(events) == 1
    assert events[0]["properties"]["endpoint"] == "/models"
    assert events[0]["properties"]["provider"] is None


def test_excluded_infra_route_is_not_reported():
    poster = RecordingPoster()
    telemetry = ProxyTelemetry(api_key="phc", host="https://h", distinct_id="id", client=poster)
    client = _build_client(telemetry=telemetry)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert poster.calls == []


def test_unmatched_route_is_not_reported():
    poster = RecordingPoster()
    telemetry = ProxyTelemetry(api_key="phc", host="https://h", distinct_id="id", client=poster)
    client = _build_client(telemetry=telemetry)

    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    assert poster.calls == []


def test_disabled_telemetry_is_noop_and_does_not_break_response():
    poster = RecordingPoster()
    client = _build_client(telemetry=None)

    resp = client.post("/chat/completions")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert poster.calls == []

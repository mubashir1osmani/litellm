"""
Emits one usage-telemetry event per matched HTTP request.

Captures the matched route template (not the raw path, so `/key/{key_id}` does
not fragment into thousands of distinct endpoints), the method, the deployed
litellm version, and the serving LLM provider when the response exposes it via
the `x-litellm-provider` header.

The event is sent AFTER the downstream app has finished, i.e. after the full
response has already been flushed to the client, so telemetry adds no
client-visible latency. When telemetry is disabled the middleware takes a fast
path and never wraps `send`.
"""

from collections.abc import Callable
from typing import ClassVar, Final

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm._version import version as litellm_version
from litellm.proxy.telemetry.proxy_telemetry import (
    ProxyTelemetry,
    ProxyTelemetryRegistry,
    TelemetryProperties,
)

_PROVIDER_HEADER: Final = "x-litellm-provider"


class _ProviderCapture:
    """Single-slot sink the send wrapper writes the provider header into (ASGI inner-callback exfiltration)."""

    __slots__ = ("provider",)

    def __init__(self) -> None:
        self.provider: str | None = None


class TelemetryMiddleware:
    """ASGI middleware that reports per-request usage telemetry when a sender is registered."""

    # k8s probes and scrape traffic would swamp "most used endpoints" with noise.
    _EXCLUDED_ROUTES: ClassVar[frozenset[str]] = frozenset(
        {
            "/health",
            "/health/liveliness",
            "/health/liveness",
            "/health/readiness",
            "/health/backlog",
            "/metrics",
        }
    )

    def __init__(
        self,
        app: ASGIApp,
        *,
        telemetry_provider: Callable[[], ProxyTelemetry | None] = ProxyTelemetryRegistry.get,
        proxy_version: str = litellm_version,
    ) -> None:
        self.app = app
        self._telemetry_provider = telemetry_provider
        self._proxy_version = proxy_version

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        telemetry: Final = self._telemetry_provider()
        if telemetry is None:
            await self.app(scope, receive, send)
            return

        capture: Final = _ProviderCapture()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                capture.provider = MutableHeaders(scope=message).get(_PROVIDER_HEADER)
            await send(message)

        await self.app(scope, receive, send_wrapper)

        route: Final = scope.get("route")
        route_path: Final = getattr(route, "path", None)
        if not isinstance(route_path, str) or route_path in self._EXCLUDED_ROUTES:
            return

        method: Final = scope.get("method")
        await telemetry.acapture(
            "litellm_proxy_request",
            TelemetryProperties(
                endpoint=route_path,
                method=method if isinstance(method, str) else None,
                provider=capture.provider,
                litellm_version=self._proxy_version,
            ),
        )

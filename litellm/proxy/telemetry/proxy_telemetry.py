"""
Lightweight, opt-in usage telemetry for the LiteLLM proxy.

Reports coarse operational signals (which endpoints are hit, which LLM providers
are used, and the deployed litellm version) to a PostHog project the operator
owns. It never sends prompts, responses, keys, or any per-user content.

Telemetry is inert unless BOTH conditions hold:
  1. the proxy is started with `--telemetry True` (the default), and
  2. a telemetry PostHog key is resolvable (see `resolve_telemetry_key`).

So a stock proxy with no telemetry key configured sends nothing.

Destination is configured independently of the `PostHogLogger` LLM-analytics
integration. That integration reads `POSTHOG_API_KEY`; telemetry deliberately
does NOT, so an operator's own PostHog logging and this usage telemetry never
collide. Telemetry reads `LITELLM_TELEMETRY_POSTHOG_KEY` (and optionally
`LITELLM_TELEMETRY_POSTHOG_HOST`, default `https://us.i.posthog.com`).

Self-host deployment: a distributor that wants product analytics from
self-hosted installs sets `LITELLM_TELEMETRY_POSTHOG_KEY` as a default in the
shipped image / Helm chart (a PostHog project key is a public write key, safe to
embed). Operators opt out with `--telemetry False`, or redirect events to their
own project by overriding the same env var. Nothing is baked into source here, so
the default posture is opt-in.

The wire format follows PostHog's public capture contract, so this stays on the
httpx path rather than adding the `posthog` SDK as a proxy dependency.
"""

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Final, Protocol, TypedDict

import httpx

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

DEFAULT_POSTHOG_HOST: Final = "https://us.i.posthog.com"
_TELEMETRY_KEY_ENV: Final = "LITELLM_TELEMETRY_POSTHOG_KEY"
_TELEMETRY_HOST_ENV: Final = "LITELLM_TELEMETRY_POSTHOG_HOST"
_TELEMETRY_ID_ENV: Final = "LITELLM_TELEMETRY_DISTINCT_ID"
_TELEMETRY_ID_FILENAME: Final = "litellm_telemetry_id"
_TELEMETRY_SEND_TIMEOUT_S: Final = 2.0
_JSON_HEADERS: Final = MappingProxyType({"Content-Type": "application/json"})


class TelemetryProperties(TypedDict, total=False):
    """The union of properties across telemetry events; each event fills the subset it has."""

    endpoint: str | None
    method: str | None
    provider: str | None
    litellm_version: str


class _CapturePayload(TypedDict):
    api_key: str
    event: str
    distinct_id: str
    properties: TelemetryProperties


class AsyncPoster(Protocol):
    """Minimal async HTTP surface the sender depends on (satisfied by httpx.AsyncClient and test fakes)."""

    async def post(self, url: str, *, content: str, headers: Mapping[str, str]) -> object: ...


def resolve_distinct_id() -> str:
    """
    A stable, anonymous per-install id used as the PostHog distinct_id.

    Precedence: explicit env override, then a uuid persisted in the OS temp dir
    (stable across restarts on the same host), then a fresh process-local uuid if
    the file cannot be read or written.
    """
    override: Final = os.getenv(_TELEMETRY_ID_ENV)
    if override:
        return override

    id_path: Final = os.path.join(tempfile.gettempdir(), _TELEMETRY_ID_FILENAME)
    try:
        with open(id_path, "r", encoding="utf-8") as handle:
            existing: Final = handle.read().strip()
        if existing:
            return existing
    except OSError:
        pass

    generated: Final = str(uuid.uuid4())
    try:
        with open(id_path, "w", encoding="utf-8") as handle:
            handle.write(generated)
    except OSError:
        pass
    return generated


@dataclass(frozen=True, slots=True)
class ProxyTelemetry:
    """Sends single events to a PostHog capture endpoint. Failures are swallowed; telemetry never breaks a request."""

    api_key: str
    host: str
    distinct_id: str
    client: AsyncPoster

    async def acapture(self, event: str, properties: TelemetryProperties) -> None:
        payload: Final = _CapturePayload(
            api_key=self.api_key,
            event=event,
            distinct_id=self.distinct_id,
            properties=properties,
        )
        try:
            await self.client.post(
                url=f"{self.host}/capture/",
                content=safe_dumps(payload),
                headers=_JSON_HEADERS,
            )
        except Exception as e:
            verbose_proxy_logger.debug("proxy telemetry: dropped event %s (%s)", event, e)


class ProxyTelemetryRegistry:
    """
    Process-wide holder for the active telemetry sender.

    Class attribute (not a module global) so the middleware, which is registered
    at import time before startup runs, can look the sender up lazily at request
    time. `None` means telemetry is disabled and every capture is a no-op.
    """

    _instance: ClassVar[ProxyTelemetry | None] = None

    @classmethod
    def set(cls, instance: ProxyTelemetry | None) -> None:
        cls._instance = instance

    @classmethod
    def get(cls) -> ProxyTelemetry | None:
        return cls._instance


def resolve_telemetry_key() -> str | None:
    """The PostHog project key telemetry sends to, or None when unconfigured (telemetry stays inert)."""
    key: Final = os.getenv(_TELEMETRY_KEY_ENV)
    return key if key else None


def resolve_telemetry_host() -> str:
    return os.getenv(_TELEMETRY_HOST_ENV, DEFAULT_POSTHOG_HOST).rstrip("/")


def _build_async_client() -> AsyncPoster:
    # Dedicated httpx client so telemetry never shares a connection pool with LLM traffic.
    # The short timeout bounds the post, which is awaited only after the client response is
    # already flushed, so it can never delay a request.
    return httpx.AsyncClient(timeout=_TELEMETRY_SEND_TIMEOUT_S)


def init_proxy_telemetry(
    *,
    enabled: bool,
    client: AsyncPoster | None = None,
) -> ProxyTelemetry | None:
    """
    Build and register the telemetry sender, or disable it.

    Returns the active sender, or None when telemetry is off (flag disabled or no
    telemetry key). Registers the result so `ProxyTelemetryRegistry.get()`
    reflects it.
    """
    if not enabled:
        ProxyTelemetryRegistry.set(None)
        return None

    api_key: Final = resolve_telemetry_key()
    if api_key is None:
        verbose_proxy_logger.debug("proxy telemetry: enabled but %s unset, staying inert", _TELEMETRY_KEY_ENV)
        ProxyTelemetryRegistry.set(None)
        return None

    host: Final = resolve_telemetry_host()
    telemetry: Final = ProxyTelemetry(
        api_key=api_key,
        host=host,
        distinct_id=resolve_distinct_id(),
        client=client if client is not None else _build_async_client(),
    )
    ProxyTelemetryRegistry.set(telemetry)
    verbose_proxy_logger.info("proxy telemetry: enabled, reporting to %s", host)
    return telemetry

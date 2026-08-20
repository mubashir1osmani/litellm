"""Provider-native model discovery.

Discovery is the half of this problem that has real APIs behind it: providers will tell
you what they serve, and several volunteer context windows and shutdown dates too. They
almost never serve prices, which is why pricing lives in its own module with a much more
suspicious posture.

Every source reports a missing credential as a value, so a run with partial credentials
produces partial truth instead of an exception.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Mapping, Protocol, Sequence

import httpx

from .domain import Confidence, DiscoveryResult, Inventory, LiveModel, Provenance, SourceFailure, utc_now


class ModelSource(Protocol):
    """A provider that can be asked what it currently serves."""

    @property
    def provider(self) -> str: ...

    async def discover(self, client: httpx.AsyncClient) -> DiscoveryResult: ...


def _api_provenance(url: str) -> Provenance:
    return Provenance(source_url=url, retrieved_at=utc_now(), confidence=Confidence.PRIMARY_API)


def _missing_key(provider: str, env_var: str) -> SourceFailure:
    return SourceFailure(source=provider, reason="credential_missing", detail=f"{env_var} is not set")


def _transport_failure(provider: str, url: str, exc: Exception) -> SourceFailure:
    return SourceFailure(source=provider, reason="unreachable", detail=f"{url}: {type(exc).__name__}: {exc}")


def _http_failure(provider: str, response: httpx.Response) -> SourceFailure:
    return SourceFailure(
        source=provider,
        reason=f"http_{response.status_code}",
        detail=f"{response.request.url}: {response.text[:200]}",
    )


@dataclass(frozen=True, slots=True)
class OpenAICompatibleSource:
    """Any provider exposing ``GET /v1/models`` with bearer auth.

    OpenAI itself additionally returns ``shutdown_date``, which is the only machine
    readable deprecation feed among the major providers, so it is read when present.
    """

    provider: str
    base_url: str
    api_key_env: str

    async def discover(self, client: httpx.AsyncClient) -> DiscoveryResult:
        key: Final = os.environ.get(self.api_key_env)
        if not key:
            return _missing_key(self.provider, self.api_key_env)
        url: Final = f"{self.base_url.rstrip('/')}/models"
        try:
            response: Final = await client.get(url, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError as exc:
            return _transport_failure(self.provider, url, exc)
        if response.status_code != httpx.codes.OK:
            return _http_failure(self.provider, response)
        payload: Final = response.json()
        rows: Final = payload.get("data", []) if isinstance(payload, dict) else []
        return Inventory(
            provider=self.provider,
            models=tuple(self._to_model(r) for r in rows if isinstance(r, dict) and r.get("id")),
            provenance=_api_provenance(url),
        )

    def _to_model(self, row: Mapping[str, object]) -> LiveModel:
        created: Final = row.get("created")
        shutdown: Final = row.get("shutdown_date")
        return LiveModel(
            provider=self.provider,
            model_id=str(row["id"]),
            created_at=datetime.fromtimestamp(created, UTC) if isinstance(created, int) else None,
            shutdown_date=shutdown if isinstance(shutdown, str) else None,
        )


@dataclass(frozen=True, slots=True)
class AnthropicSource:
    provider: str = "anthropic"
    base_url: str = "https://api.anthropic.com/v1"
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_version: str = "2023-06-01"

    async def discover(self, client: httpx.AsyncClient) -> DiscoveryResult:
        key: Final = os.environ.get(self.api_key_env)
        if not key:
            return _missing_key(self.provider, self.api_key_env)
        url: Final = f"{self.base_url.rstrip('/')}/models"
        try:
            response: Final = await client.get(
                url, params={"limit": 1000}, headers={"x-api-key": key, "anthropic-version": self.api_version}
            )
        except httpx.HTTPError as exc:
            return _transport_failure(self.provider, url, exc)
        if response.status_code != httpx.codes.OK:
            return _http_failure(self.provider, response)
        rows: Final = response.json().get("data", [])
        return Inventory(
            provider=self.provider,
            models=tuple(
                LiveModel(
                    provider=self.provider,
                    model_id=str(r["id"]),
                    display_name=r.get("display_name"),
                    created_at=_parse_iso(r.get("created_at")),
                )
                for r in rows
                if isinstance(r, dict) and r.get("id")
            ),
            provenance=_api_provenance(url),
        )


@dataclass(frozen=True, slots=True)
class GeminiSource:
    """Google AI Studio, the one discovery API that publishes real token limits."""

    provider: str = "gemini"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    api_key_env: str = "GEMINI_API_KEY"

    async def discover(self, client: httpx.AsyncClient) -> DiscoveryResult:
        key: Final = os.environ.get(self.api_key_env)
        if not key:
            return _missing_key(self.provider, self.api_key_env)
        url: Final = f"{self.base_url.rstrip('/')}/models"
        try:
            response: Final = await client.get(url, params={"key": key, "pageSize": 1000})
        except httpx.HTTPError as exc:
            return _transport_failure(self.provider, url, exc)
        if response.status_code != httpx.codes.OK:
            return _http_failure(self.provider, response)
        rows: Final = response.json().get("models", [])
        return Inventory(
            provider=self.provider,
            models=tuple(self._to_model(r) for r in rows if isinstance(r, dict) and r.get("name")),
            provenance=_api_provenance(url),
        )

    def _to_model(self, row: Mapping[str, object]) -> LiveModel:
        methods: Final = row.get("supportedGenerationMethods")
        return LiveModel(
            provider=self.provider,
            model_id=str(row["name"]).removeprefix("models/"),
            display_name=row.get("displayName") if isinstance(row.get("displayName"), str) else None,
            max_input_tokens=row.get("inputTokenLimit") if isinstance(row.get("inputTokenLimit"), int) else None,
            max_output_tokens=row.get("outputTokenLimit") if isinstance(row.get("outputTokenLimit"), int) else None,
            modes=tuple(str(m) for m in methods) if isinstance(methods, list) else (),
        )


@dataclass(frozen=True, slots=True)
class BedrockSource:
    """``ListFoundationModels`` over SigV4, so it goes through boto3 rather than httpx."""

    provider: str = "bedrock"
    region_env: str = "AWS_REGION"
    default_region: str = "us-east-1"

    async def discover(self, client: httpx.AsyncClient) -> DiscoveryResult:
        del client
        if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE"):
            return _missing_key(self.provider, "AWS_ACCESS_KEY_ID")
        region: Final = os.environ.get(self.region_env) or self.default_region
        try:
            summaries: Final = await asyncio.to_thread(self._list_foundation_models, region)
        except Exception as exc:
            return SourceFailure(source=self.provider, reason="unreachable", detail=f"{type(exc).__name__}: {exc}")
        return Inventory(
            provider=self.provider,
            models=tuple(
                LiveModel(
                    provider=self.provider,
                    model_id=str(s["modelId"]),
                    display_name=s.get("modelName"),
                    modes=tuple(str(m) for m in s.get("outputModalities", [])),
                )
                for s in summaries
                if s.get("modelId")
            ),
            provenance=_api_provenance(f"bedrock:ListFoundationModels?region={region}"),
        )

    def _list_foundation_models(self, region: str) -> Sequence[Mapping[str, object]]:
        import boto3

        return boto3.client("bedrock", region_name=region).list_foundation_models().get("modelSummaries", [])


def _parse_iso(raw: object) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        parsed: Final = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


DEFAULT_SOURCES: Final[tuple[ModelSource, ...]] = (
    OpenAICompatibleSource(provider="openai", base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY"),
    AnthropicSource(),
    GeminiSource(),
    BedrockSource(),
    OpenAICompatibleSource(provider="xai", base_url="https://api.x.ai/v1", api_key_env="XAI_API_KEY"),
    OpenAICompatibleSource(provider="mistral", base_url="https://api.mistral.ai/v1", api_key_env="MISTRAL_API_KEY"),
    OpenAICompatibleSource(provider="deepseek", base_url="https://api.deepseek.com/v1", api_key_env="DEEPSEEK_API_KEY"),
    OpenAICompatibleSource(provider="groq", base_url="https://api.groq.com/openai/v1", api_key_env="GROQ_API_KEY"),
)


async def discover_all(sources: Sequence[ModelSource], client: httpx.AsyncClient) -> tuple[DiscoveryResult, ...]:
    return tuple(await asyncio.gather(*(s.discover(client) for s in sources)))

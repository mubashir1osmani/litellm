"""Third-party cross-check of a price the agent already established from a primary source.

An aggregator restates providers, so it is useful for catching a misread table and
useless as an origin. Nothing here can promote a candidate to proposable; it can only
add agreement, or flag that a primary reading and a third party disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Protocol

import httpx

from .domain import Confidence, Provenance, SourceFailure, TokenPricing, utc_now

OPENROUTER_MODELS_URL: Final = "https://openrouter.ai/api/v1/models"

_DISAGREEMENT_TOLERANCE: Final = 0.05


@dataclass(frozen=True, slots=True)
class AggregatorPrice:
    model_key: str
    input_cost_per_token: float
    output_cost_per_token: float


class Aggregator(Protocol):
    @property
    def source_url(self) -> str: ...

    async def prices(self, client: httpx.AsyncClient) -> Mapping[str, AggregatorPrice] | SourceFailure: ...


@dataclass(frozen=True, slots=True)
class OpenRouterAggregator:
    """OpenRouter publishes structured per-token pricing for several hundred models."""

    source_url: str = OPENROUTER_MODELS_URL

    async def prices(self, client: httpx.AsyncClient) -> Mapping[str, AggregatorPrice] | SourceFailure:
        try:
            response: Final = await client.get(self.source_url)
        except httpx.HTTPError as exc:
            return SourceFailure(source="openrouter", reason="unreachable", detail=f"{type(exc).__name__}: {exc}")
        if response.status_code != httpx.codes.OK:
            return SourceFailure(source="openrouter", reason=f"http_{response.status_code}", detail=self.source_url)
        rows: Final = response.json().get("data", [])
        return {p.model_key: p for p in (_row_to_price(r) for r in rows if isinstance(r, dict)) if p is not None}


def _row_to_price(row: Mapping[str, object]) -> AggregatorPrice | None:
    model_key: Final = row.get("id")
    pricing: Final = row.get("pricing")
    if not isinstance(model_key, str) or not isinstance(pricing, Mapping):
        return None
    try:
        prompt: Final = float(str(pricing.get("prompt")))
        completion: Final = float(str(pricing.get("completion")))
    except (TypeError, ValueError):
        return None
    return AggregatorPrice(model_key=model_key, input_cost_per_token=prompt, output_cost_per_token=completion)


def aggregator_keys_for(provider: str, model_id: str) -> tuple[str, ...]:
    return (f"{provider}/{model_id}", model_id)


def corroborate(
    provider: str,
    model_id: str,
    pricing: TokenPricing,
    aggregate: Mapping[str, AggregatorPrice],
    source_url: str,
) -> tuple[tuple[Provenance, ...], tuple[SourceFailure, ...]]:
    """Compare an established price against the aggregator, returning agreement and dissent.

    Silence is the normal case: most models are not carried by any aggregator, and that
    is not a problem worth reporting.
    """
    match: Final = next((aggregate[k] for k in aggregator_keys_for(provider, model_id) if k in aggregate), None)
    if match is None or not pricing.is_complete:
        return ((), ())
    ours: Final = (pricing.input_cost_per_token, pricing.output_cost_per_token)
    theirs: Final = (match.input_cost_per_token, match.output_cost_per_token)
    disagreements: Final = tuple(
        f"{label}: primary {mine.value:.3e} vs aggregator {other:.3e}"
        for label, mine, other in zip(("input", "output"), ours, theirs, strict=True)
        if mine is not None and not _within_tolerance(mine.value, other)
    )
    provenance: Final = Provenance(source_url=source_url, retrieved_at=utc_now(), confidence=Confidence.AGGREGATOR)
    if disagreements:
        return ((), (SourceFailure(source="openrouter", reason="price_disagreement", detail="; ".join(disagreements)),))
    return ((provenance,), ())


def _within_tolerance(primary: float, other: float) -> bool:
    if primary == 0.0:
        return other == 0.0
    return abs(primary - other) / primary <= _DISAGREEMENT_TOLERANCE

"""Reads LiteLLM's cost map and works out what a live provider inventory contradicts.

The map is read-only here. The agent's output is a proposed patch, reviewed as a diff,
because every entry feeds real cost calculation for callers in production.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping, Sequence

from .domain import Candidate, CatalogPatch, Inventory, LiveModel, PricedCandidate

_PREFIXED_PROVIDERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "gemini": "gemini/",
        "vertex_ai": "vertex_ai/",
        "azure": "azure/",
        "azure_ai": "azure_ai/",
        "mistral": "mistral/",
        "xai": "xai/",
        "deepseek": "deepseek/",
        "groq": "groq/",
    }
)

_UNPREFIXED_PROVIDERS: Final[frozenset[str]] = frozenset({"openai", "anthropic", "bedrock", "bedrock_converse"})

_PRICE_FIELDS: Final[tuple[str, ...]] = ("input_cost_per_token", "output_cost_per_token")

_TOKEN_BILLED_MODES: Final[frozenset[str]] = frozenset({"chat", "completion", "responses"})


_DRIFT_TOLERANCE: Final = 0.01


def _has_any_cost(entry: Mapping[str, object]) -> bool:
    return any("cost" in k for k in entry)


def _materially_differs(catalogued: object, live: int) -> bool:
    if not isinstance(catalogued, int) or catalogued == live:
        return False
    return abs(catalogued - live) / max(abs(live), 1) > _DRIFT_TOLERANCE


def catalog_keys_for(model: LiveModel) -> tuple[str, ...]:
    """Every key convention under which this model could already be catalogued.

    LiteLLM keys OpenAI and Anthropic models bare but prefixes most others, and Bedrock
    carries regional variants of the same model. Checking all plausible spellings is what
    keeps the agent from reporting an existing model as a new launch.
    """
    bare: Final = model.model_id
    prefix: Final = _PREFIXED_PROVIDERS.get(model.provider)
    if prefix is not None:
        return (f"{prefix}{bare}", bare)
    if model.provider in _UNPREFIXED_PROVIDERS:
        regional: Final = tuple(f"{r}.{bare}" for r in ("us", "eu", "apac", "au", "global"))
        return (bare, *regional) if model.provider.startswith("bedrock") else (bare,)
    return (f"{model.provider}/{bare}", bare)


@dataclass(frozen=True, slots=True)
class Catalog:
    """An indexed, read-only view of ``model_prices_and_context_window.json``."""

    entries: Mapping[str, Mapping[str, object]]

    @classmethod
    def load(cls, path: Path) -> Catalog:
        raw: Final = json.loads(path.read_text())
        return cls(
            entries=MappingProxyType(
                {k: MappingProxyType(dict(v)) for k, v in raw.items() if k != "sample_spec" and isinstance(v, dict)}
            )
        )

    def token_billed_keys(self) -> frozenset[str]:
        """Entries a published per-token price could meaningfully be compared against."""
        return frozenset(k for k, e in self.entries.items() if e.get("mode") in _TOKEN_BILLED_MODES)

    def lookup(self, model: LiveModel) -> tuple[str, Mapping[str, object]] | None:
        return next(((k, self.entries[k]) for k in catalog_keys_for(model) if k in self.entries), None)

    def preferred_key(self, model: LiveModel) -> str:
        return catalog_keys_for(model)[0]

    def reconcile(self, inventory: Inventory) -> tuple[Candidate, ...]:
        return tuple(c for m in inventory.models for c in self._candidates_for(m))

    def _candidates_for(self, model: LiveModel) -> tuple[Candidate, ...]:
        found: Final = self.lookup(model)
        if found is None:
            return (
                Candidate(
                    kind="new_launch",
                    live=model,
                    summary=f"{model.provider} serves {model.model_id}, absent from the cost map",
                    catalog_key=self.preferred_key(model),
                ),
            )
        key, entry = found
        return tuple(
            c for c in (self._price_gap(model, key, entry), self._deprecation(model, key, entry), self._drift(model, key, entry)) if c is not None
        )

    def _price_gap(self, model: LiveModel, key: str, entry: Mapping[str, object]) -> Candidate | None:
        """Flag an entry the cost map cannot bill from.

        Absent per-token costs are normal for most of the catalog: speech models are
        priced per character, transcription per second, images per image. Only an entry
        with no cost of any kind, or a token-billed entry missing a token cost, is a gap.
        """
        if not _has_any_cost(entry):
            return Candidate(
                kind="missing_price",
                live=model,
                summary=f"{key} is catalogued with no cost fields at all",
                catalog_key=key,
            )
        if entry.get("mode") not in _TOKEN_BILLED_MODES:
            return None
        missing: Final = tuple(f for f in _PRICE_FIELDS if entry.get(f) is None)
        if not missing:
            return None
        return Candidate(
            kind="missing_price",
            live=model,
            summary=f"{key} is billed per token but lacks {', '.join(missing)}",
            catalog_key=key,
        )

    def _deprecation(self, model: LiveModel, key: str, entry: Mapping[str, object]) -> Candidate | None:
        if model.shutdown_date is None or entry.get("deprecation_date") == model.shutdown_date:
            return None
        known: Final = entry.get("deprecation_date")
        seen: Final = f"was {known!r}" if known is not None else "had none"
        return Candidate(
            kind="deprecation_signal",
            live=model,
            summary=f"{model.provider} publishes shutdown {model.shutdown_date} for {key}; cost map {seen}",
            catalog_key=key,
        )

    def _drift(self, model: LiveModel, key: str, entry: Mapping[str, object]) -> Candidate | None:
        """Report context windows that materially disagree with the provider's own API.

        Exact equality is the wrong bar. The cost map deliberately carries values like
        65535 where the provider reports 65536, and eight such entries would crowd out
        the real findings, which run to whole multiples.
        """
        drifted: Final = tuple(
            f"{field}: cost map {entry.get(field)!r} vs provider {live!r}"
            for field, live in (("max_input_tokens", model.max_input_tokens), ("max_output_tokens", model.max_output_tokens))
            if live is not None and _materially_differs(entry.get(field), live)
        )
        if not drifted:
            return None
        return Candidate(
            kind="context_drift",
            live=model,
            summary=f"{key} context window disagrees with the provider API ({'; '.join(drifted)})",
            catalog_key=key,
        )


def build_patch(catalog: Catalog, priced: Sequence[PricedCandidate]) -> CatalogPatch:
    """Turn proposable candidates into cost-map entries, split by add versus update.

    Candidates that failed verification are excluded here rather than emitted with a
    warning, so a reviewer never sees an unverified number formatted as a real entry.
    """
    proposable: Final = tuple(p for p in priced if p.is_proposable)
    return CatalogPatch(
        additions=MappingProxyType(
            {p.candidate.catalog_key: _entry_for(p) for p in proposable if p.candidate.catalog_key not in catalog.entries}
        ),
        updates=MappingProxyType(
            {p.candidate.catalog_key: _entry_for(p) for p in proposable if p.candidate.catalog_key in catalog.entries}
        ),
    )


def _entry_for(priced: PricedCandidate) -> Mapping[str, object]:
    if priced.candidate.kind == "price_drift":
        return MappingProxyType({delta.field: delta.live for delta in priced.candidate.deltas})
    live: Final = priced.candidate.live
    pricing: Final = priced.pricing
    optional: Final = {
        "max_input_tokens": live.max_input_tokens,
        "max_output_tokens": live.max_output_tokens,
        "max_tokens": live.max_output_tokens,
        "deprecation_date": live.shutdown_date,
        "cache_read_input_token_cost": pricing.cache_read_input_token_cost.value if pricing.cache_read_input_token_cost else None,
        "cache_creation_input_token_cost": (
            pricing.cache_creation_input_token_cost.value if pricing.cache_creation_input_token_cost else None
        ),
    }
    required: Final = {
        "litellm_provider": live.provider,
        "mode": "chat",
        "input_cost_per_token": pricing.input_cost_per_token.value if pricing.input_cost_per_token else None,
        "output_cost_per_token": pricing.output_cost_per_token.value if pricing.output_cost_per_token else None,
    }
    return MappingProxyType({**required, **{k: v for k, v in optional.items() if v is not None}})

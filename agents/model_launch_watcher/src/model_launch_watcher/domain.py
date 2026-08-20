"""Value types for the model-launch watcher.

Every fact the agent reports carries the URL it came from, when it was read, and how
much the agent trusts that kind of source. Nothing in this module raises: failures are
modelled as variants so a dead provider degrades one branch of a run instead of the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Final, Literal, Mapping, Sequence


class Confidence(Enum):
    """How much weight a fact carries, by the kind of source that produced it.

    ``PRIMARY_API`` is a structured field returned by the provider's own API, so it is
    machine-truth. ``PRIMARY_DOC`` is text extracted from the provider's own published
    page and verified to appear there verbatim. ``AGGREGATOR`` is a third party
    restating the provider, which corroborates but never establishes a price.
    """

    PRIMARY_API = "primary_api"
    PRIMARY_DOC = "primary_doc"
    AGGREGATOR = "aggregator"

    @property
    def is_primary(self) -> bool:
        return self in (Confidence.PRIMARY_API, Confidence.PRIMARY_DOC)


@dataclass(frozen=True, slots=True)
class Provenance:
    source_url: str
    retrieved_at: datetime
    confidence: Confidence

    def __post_init__(self) -> None:
        if self.retrieved_at.tzinfo is None:
            raise ValueError("Provenance.retrieved_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class Sourced[T]:
    """A single value bound to the evidence that produced it."""

    value: T
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class LiveModel:
    """A model a provider's own API says exists right now."""

    provider: str
    model_id: str
    display_name: str | None = None
    created_at: datetime | None = None
    shutdown_date: str | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    modes: tuple[str, ...] = ()

    @property
    def catalog_key(self) -> str:
        return self.model_id


@dataclass(frozen=True, slots=True)
class SourceFailure:
    """A source that could not be read. Carried, not raised, so one gap never fails a run."""

    source: str
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Inventory:
    provider: str
    models: tuple[LiveModel, ...]
    provenance: Provenance


type DiscoveryResult = Inventory | SourceFailure


@dataclass(frozen=True, slots=True)
class TokenPricing:
    input_cost_per_token: Sourced[float] | None = None
    output_cost_per_token: Sourced[float] | None = None
    cache_read_input_token_cost: Sourced[float] | None = None
    cache_creation_input_token_cost: Sourced[float] | None = None

    @property
    def is_empty(self) -> bool:
        return self.input_cost_per_token is None and self.output_cost_per_token is None

    @property
    def is_complete(self) -> bool:
        return self.input_cost_per_token is not None and self.output_cost_per_token is not None

    def sources(self) -> tuple[Provenance, ...]:
        quoted: Final = (
            self.input_cost_per_token,
            self.output_cost_per_token,
            self.cache_read_input_token_cost,
            self.cache_creation_input_token_cost,
        )
        return tuple(q.provenance for q in quoted if q is not None)


type CandidateKind = Literal["new_launch", "missing_price", "deprecation_signal", "context_drift", "price_drift"]


@dataclass(frozen=True, slots=True)
class PriceDelta:
    """A catalogued cost that no longer matches what the provider publishes."""

    field: str
    catalogued: float
    live: float
    provenance: Provenance

    @property
    def percent_change(self) -> float:
        return 100.0 * (self.live - self.catalogued) / self.catalogued if self.catalogued else float("inf")

    @property
    def direction(self) -> Literal["cut", "increase"]:
        return "cut" if self.live < self.catalogued else "increase"

    def per_million(self) -> tuple[float, float]:
        return self.catalogued * 1_000_000, self.live * 1_000_000


@dataclass(frozen=True, slots=True)
class Candidate:
    """A live model the local catalog does not yet describe correctly."""

    kind: CandidateKind
    live: LiveModel
    summary: str
    catalog_key: str
    deltas: tuple[PriceDelta, ...] = ()


@dataclass(frozen=True, slots=True)
class PricedCandidate:
    candidate: Candidate
    pricing: TokenPricing
    corroboration: tuple[Provenance, ...] = ()
    gaps: tuple[SourceFailure, ...] = ()

    @property
    def is_proposable(self) -> bool:
        """A price may be proposed only when a primary source established it.

        Aggregators restate providers and drift; they may agree with a price but never
        create one. Requiring a primary source is what stops a scraped third party from
        silently rewriting the cost map.
        """
        return self.pricing.is_complete and any(p.confidence.is_primary for p in self.pricing.sources())


@dataclass(frozen=True, slots=True)
class CatalogPatch:
    """Entries proposed for ``model_prices_and_context_window.json``.

    Never applied by the agent. The output is reviewed as a diff, because 3000+ live
    catalog entries feed real cost calculation.
    """

    additions: Mapping[str, Mapping[str, object]] = field(default_factory=lambda: MappingProxyType({}))
    updates: Mapping[str, Mapping[str, object]] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def is_empty(self) -> bool:
        return not self.additions and not self.updates


@dataclass(frozen=True, slots=True)
class WatchReport:
    generated_at: datetime
    providers_checked: tuple[str, ...]
    candidates: tuple[PricedCandidate, ...]
    patch: CatalogPatch
    failures: tuple[SourceFailure, ...]

    @property
    def needs_human_review(self) -> tuple[PricedCandidate, ...]:
        return tuple(c for c in self.candidates if not c.is_proposable)


def utc_now() -> datetime:
    return datetime.now(UTC)

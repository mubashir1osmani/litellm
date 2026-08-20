"""The watch pipeline as a LangGraph state machine.

    discover -> reconcile -> read_prices -> diff_prices -> corroborate -> report

The graph is deterministic and the LLM appears at exactly one node, reading a fetched
document. That is deliberate: the output edits pricing that bills real callers, so the
run has to be reproducible and every number has to trace to a URL. A free-roaming agent
loop would buy flexibility this task does not want.

Each provider's pricing page is read once per run and the whole table extracted in a
single call, so a full reconciliation of the catalog costs one call per provider rather
than one per model.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Final, Mapping, Protocol, Sequence, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from .catalog import Catalog, build_patch
from .corroboration import Aggregator, AggregatorPrice, OpenRouterAggregator, corroborate
from .discovery import DEFAULT_SOURCES, ModelSource, discover_all
from .domain import (
    Candidate,
    CandidateKind,
    DiscoveryResult,
    Inventory,
    PriceCoverage,
    PricedCandidate,
    SourceFailure,
    TokenPricing,
    WatchReport,
    utc_now,
)
from .memory import Memory
from .price_diff import compared_keys, price_drift_candidates, resolve_catalog_key
from .pricing import (
    PRICING_PAGES,
    LiteLLMTableExtractor,
    PricingDoc,
    TableExtractor,
    fetch_pricing_doc,
    ground_table,
)

_NEEDS_DISCOVERY: Final[frozenset[CandidateKind]] = frozenset(
    {"new_launch", "context_drift", "deprecation_signal", "missing_price"}
)

_NEEDS_PRICES: Final[frozenset[CandidateKind]] = frozenset({"new_launch", "missing_price", "price_drift"})

type GroundedTables = Mapping[str, Mapping[str, TokenPricing]]


class DocFetcher(Protocol):
    async def __call__(
        self, provider: str, url: str, client: httpx.AsyncClient
    ) -> PricingDoc | SourceFailure: ...


@dataclass(frozen=True, slots=True)
class WatchRequest:
    providers: tuple[str, ...] = ()
    kinds: tuple[CandidateKind, ...] = ()
    include_pricing: bool = True

    def wants(self, kind: CandidateKind) -> bool:
        return not self.kinds or kind in self.kinds

    @property
    def needs_discovery(self) -> bool:
        return any(self.wants(k) for k in _NEEDS_DISCOVERY)

    @property
    def needs_prices(self) -> bool:
        return self.include_pricing and any(self.wants(k) for k in _NEEDS_PRICES)


@dataclass(frozen=True, slots=True)
class Dependencies:
    """Everything the graph talks to, injected so tests drive real code with fake edges."""

    catalog: Catalog
    memory: Memory = field(default_factory=Memory)
    sources: Sequence[ModelSource] = DEFAULT_SOURCES
    extractor: TableExtractor = field(default_factory=LiteLLMTableExtractor)
    fetch_doc: DocFetcher = fetch_pricing_doc
    aggregator: Aggregator = field(default_factory=OpenRouterAggregator)
    pricing_pages: Mapping[str, str] = PRICING_PAGES
    http_timeout: float = 45.0
    concurrency: int = 4


class WatchState(TypedDict, total=False):
    request: WatchRequest
    discoveries: tuple[DiscoveryResult, ...]
    candidates: tuple[Candidate, ...]
    tables: GroundedTables
    priced: tuple[PricedCandidate, ...]
    failures: tuple[SourceFailure, ...]
    report: WatchReport


def build_graph(deps: Dependencies):
    """Compile the pipeline, closing over dependencies rather than reaching for globals."""

    def _providers(request: WatchRequest) -> tuple[str, ...]:
        configured: Final = tuple(sorted({s.provider for s in deps.sources} | set(deps.pricing_pages)))
        return tuple(p for p in configured if not request.providers or p in request.providers)

    async def discover(state: WatchState) -> WatchState:
        request: Final = state["request"]
        if not request.needs_discovery:
            return {"discoveries": (), "failures": ()}
        wanted: Final = frozenset(_providers(request))
        selected: Final = tuple(s for s in deps.sources if s.provider in wanted)
        async with _client(deps) as client:
            results: Final = await discover_all(selected, client)
        return {"discoveries": results, "failures": tuple(r for r in results if isinstance(r, SourceFailure))}

    async def reconcile(state: WatchState) -> WatchState:
        request: Final = state["request"]
        found: Final = tuple(
            c
            for inventory in state.get("discoveries", ())
            if isinstance(inventory, Inventory)
            for c in deps.catalog.reconcile(inventory)
        )
        kept: Final = tuple(c for c in found if request.wants(c.kind) and not _suppressed(deps, c))
        return {"candidates": kept}

    async def read_prices(state: WatchState) -> WatchState:
        request: Final = state["request"]
        if not request.needs_prices:
            return {"tables": {}}
        wanted: Final = tuple(p for p in _providers(request) if p in deps.pricing_pages)
        async with _client(deps) as client:
            docs: Final = await asyncio.gather(
                *(deps.fetch_doc(p, deps.pricing_pages[p], client) for p in wanted)
            )
        gate: Final = asyncio.Semaphore(deps.concurrency)
        extracted: Final = await asyncio.gather(
            *(_extract_one(deps, provider, doc, gate) for provider, doc in zip(wanted, docs, strict=True))
        )
        return {
            "tables": {provider: table for provider, table, _ in extracted},
            "failures": (*state.get("failures", ()), *(f for _, _, gaps in extracted for f in gaps)),
        }

    async def diff_prices(state: WatchState) -> WatchState:
        request: Final = state["request"]
        tables: Final = state.get("tables", {})
        drift, unmapped = _drift_across(deps, tables) if request.wants("price_drift") else ((), ())
        existing: Final = state.get("candidates", ())
        priced: Final = tuple(
            _attach_price(deps, candidate, tables) for candidate in (*existing, *drift)
        )
        return {"priced": priced, "failures": (*state.get("failures", ()), *unmapped)}

    async def cross_check(state: WatchState) -> WatchState:
        priced: Final = state.get("priced", ())
        if not any(p.pricing.is_complete for p in priced):
            return {}
        async with _client(deps) as client:
            aggregate = await deps.aggregator.prices(client)
        if isinstance(aggregate, SourceFailure):
            return {"failures": (*state.get("failures", ()), aggregate)}
        return {"priced": tuple(_apply_corroboration(deps, p, aggregate) for p in priced)}

    async def report(state: WatchState) -> WatchState:
        priced: Final = state.get("priced", ())
        reached: Final = tuple(
            sorted(
                {d.provider for d in state.get("discoveries", ()) if isinstance(d, Inventory)}
                | {p for p, table in state.get("tables", {}).items() if table}
            )
        )
        tables: Final = state.get("tables", {})
        checked: Final = frozenset(
            key
            for provider, grounded in tables.items()
            for key in compared_keys(provider, deps.catalog, grounded, deps.memory)
        )
        return {
            "report": WatchReport(
                generated_at=utc_now(),
                providers_checked=reached,
                candidates=priced,
                patch=build_patch(deps.catalog, priced),
                failures=state.get("failures", ()),
                coverage=PriceCoverage(
                    compared=len(checked), token_billed_entries=len(deps.catalog.token_billed_keys())
                ),
            )
        }

    graph: Final = StateGraph(WatchState)
    for name, node in (
        ("discover", discover),
        ("reconcile", reconcile),
        ("read_prices", read_prices),
        ("diff_prices", diff_prices),
        ("corroborate", cross_check),
        ("report", report),
    ):
        graph.add_node(name, node)
    for source, target in (
        (START, "discover"),
        ("discover", "reconcile"),
        ("reconcile", "read_prices"),
        ("read_prices", "diff_prices"),
        ("diff_prices", "corroborate"),
        ("corroborate", "report"),
        ("report", END),
    ):
        graph.add_edge(source, target)
    return graph.compile()


def _client(deps: Dependencies) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=deps.http_timeout, follow_redirects=True)


def _suppressed(deps: Dependencies, candidate: Candidate) -> bool:
    return deps.memory.suppresses(candidate.catalog_key, candidate.live.provider, candidate.kind) is not None


async def _extract_one(
    deps: Dependencies, provider: str, doc: PricingDoc | SourceFailure, gate: asyncio.Semaphore
) -> tuple[str, Mapping[str, TokenPricing], tuple[SourceFailure, ...]]:
    if isinstance(doc, SourceFailure):
        return provider, {}, (doc,)
    async with gate:
        table = await deps.extractor.extract_table(doc, deps.memory.conventions(provider))
    if isinstance(table, SourceFailure):
        return provider, {}, (table,)
    grounded: Final = ground_table(table)
    dropped: Final = len(table.rows) - len(grounded)
    gaps: Final = (
        (
            SourceFailure(
                source=provider,
                reason="quote_not_grounded",
                detail=f"{dropped} of {len(table.rows)} rows at {doc.url} were dropped, quotes not found verbatim",
            ),
        )
        if dropped
        else ()
    )
    return provider, grounded, gaps


def _drift_across(
    deps: Dependencies, tables: GroundedTables
) -> tuple[tuple[Candidate, ...], tuple[SourceFailure, ...]]:
    per_provider: Final = tuple(
        price_drift_candidates(provider, deps.catalog, grounded, deps.memory)
        for provider, grounded in tables.items()
        if grounded
    )
    return (
        tuple(c for candidates, _ in per_provider for c in candidates),
        tuple(f for _, failures in per_provider for f in failures),
    )


def _attach_price(deps: Dependencies, candidate: Candidate, tables: GroundedTables) -> PricedCandidate:
    """Reuse the provider's already-extracted table to price a candidate."""
    if candidate.kind == "price_drift":
        pricing: Final = _lookup_price(deps, candidate, tables)
        return PricedCandidate(candidate=candidate, pricing=pricing)
    table: Final = tables.get(candidate.live.provider, {})
    if not table:
        return PricedCandidate(
            candidate=candidate,
            pricing=TokenPricing(),
            gaps=(SourceFailure(source=candidate.live.provider, reason="no_price_table", detail=candidate.catalog_key),),
        )
    matched: Final = next(
        (
            price
            for name, price in table.items()
            if resolve_catalog_key(candidate.live.provider, name, deps.catalog, deps.memory).catalog_key
            == candidate.catalog_key
        ),
        None,
    )
    if matched is None:
        return PricedCandidate(
            candidate=candidate,
            pricing=TokenPricing(),
            gaps=(
                SourceFailure(
                    source=candidate.live.provider,
                    reason="price_not_published",
                    detail=f"{candidate.catalog_key} is not priced on the provider's page",
                ),
            ),
        )
    return PricedCandidate(candidate=candidate, pricing=matched)


def _lookup_price(deps: Dependencies, candidate: Candidate, tables: GroundedTables) -> TokenPricing:
    table: Final = tables.get(candidate.live.provider, {})
    name: Final = candidate.live.display_name or ""
    return table.get(name, TokenPricing())


def _apply_corroboration(
    deps: Dependencies, priced: PricedCandidate, aggregate: Mapping[str, AggregatorPrice]
) -> PricedCandidate:
    agreement, dissent = corroborate(
        provider=priced.candidate.live.provider,
        model_id=priced.candidate.live.model_id,
        pricing=priced.pricing,
        aggregate=aggregate,
        source_url=deps.aggregator.source_url,
    )
    return PricedCandidate(
        candidate=priced.candidate,
        pricing=priced.pricing,
        corroboration=(*priced.corroboration, *agreement),
        gaps=(*priced.gaps, *dissent),
    )


async def run_watch(deps: Dependencies, request: WatchRequest) -> WatchReport:
    final: Final = await build_graph(deps).ainvoke({"request": request})
    return final["report"]

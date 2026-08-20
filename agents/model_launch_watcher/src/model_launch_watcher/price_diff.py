"""Diffing published prices against the catalogued ones.

This is the check that catches a price cut. Everything else the agent does reports on
models the catalog describes badly; this reports on models it describes at yesterday's
price, which is the case that quietly overcharges or undercharges every caller.

Two things keep it honest. A published name is only bound to a catalog key when the
binding is unambiguous or a human has taught it, and a difference is only reported when
it is larger than floating-point noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

from .catalog import Catalog
from .domain import Candidate, LiveModel, PriceDelta, SourceFailure, TokenPricing
from .memory import Memory

_NON_ALNUM: Final = re.compile(r"[^a-z0-9]+")

_RELATIVE_EPSILON: Final = 1e-6

_DIFFED_FIELDS: Final[tuple[str, ...]] = ("input_cost_per_token", "output_cost_per_token")


def slugify(published_name: str) -> str:
    return _NON_ALNUM.sub("-", published_name.casefold()).strip("-")


@dataclass(frozen=True, slots=True)
class KeyResolution:
    """Either a catalog key this published name maps to, or the reason it does not."""

    published_name: str
    catalog_key: str | None
    via: str


def resolve_catalog_key(
    provider: str, published_name: str, catalog: Catalog, memory: Memory
) -> KeyResolution:
    """Bind a provider's published name to a catalog key.

    A human-taught mapping wins over inference, because inference is exactly what got
    corrected. When neither a taught mapping nor an exact slug match exists the name is
    left unresolved rather than matched approximately: a wrong binding writes one model's
    price onto another, which is worse than reporting nothing.
    """
    taught: Final = memory.mapped_key(provider, published_name)
    if taught is not None:
        resolved: Final = taught in catalog.entries
        return KeyResolution(published_name, taught if resolved else None, "memory" if resolved else "memory_stale")
    slug: Final = slugify(published_name)
    for key in (slug, f"{provider}/{slug}"):
        if key in catalog.entries:
            return KeyResolution(published_name, key, "slug")
    return KeyResolution(published_name, None, "unresolved")


def _materially_differs(catalogued: float, live: float) -> bool:
    if catalogued == live:
        return False
    return abs(catalogued - live) / max(abs(catalogued), _RELATIVE_EPSILON) > _RELATIVE_EPSILON


def diff_entry(
    catalog_key: str,
    entry: Mapping[str, object],
    pricing: TokenPricing,
    memory: Memory,
) -> tuple[PriceDelta, ...]:
    """Compare one catalogued entry against a grounded live price.

    Fields a human has pinned are skipped: a pin says the catalogued value is deliberate,
    so re-reporting it every run is the noise memory exists to remove.
    """
    quoted: Final = {
        "input_cost_per_token": pricing.input_cost_per_token,
        "output_cost_per_token": pricing.output_cost_per_token,
    }
    return tuple(
        PriceDelta(
            field=field,
            catalogued=float(catalogued),
            live=sourced.value,
            provenance=sourced.provenance,
        )
        for field in _DIFFED_FIELDS
        if (sourced := quoted[field]) is not None
        and isinstance(catalogued := entry.get(field), (int, float))
        and memory.pinned(catalog_key, field) is None
        and _materially_differs(float(catalogued), sourced.value)
    )


def price_drift_candidates(
    provider: str,
    catalog: Catalog,
    grounded: Mapping[str, TokenPricing],
    memory: Memory,
) -> tuple[tuple[Candidate, ...], tuple[SourceFailure, ...]]:
    """Turn a grounded price table into drift findings, plus the names that need mapping."""
    resolutions: Final = tuple(
        (name, resolve_catalog_key(provider, name, catalog, memory), pricing) for name, pricing in grounded.items()
    )
    unmapped: Final = tuple(
        SourceFailure(source=provider, reason=_unmapped_reason(resolution), detail=_unmapped_detail(provider, resolution))
        for _, resolution, _ in resolutions
        if resolution.catalog_key is None
    )
    return (
        tuple(
            candidate
            for name, resolution, pricing in resolutions
            if resolution.catalog_key is not None
            and (candidate := _to_candidate(provider, name, resolution.catalog_key, catalog, pricing, memory))
            is not None
        ),
        unmapped,
    )


def _unmapped_reason(resolution: KeyResolution) -> str:
    return "mapping_stale" if resolution.via == "memory_stale" else "needs_mapping"


def _unmapped_detail(provider: str, resolution: KeyResolution) -> str:
    """Distinguish a name nobody has mapped from a taught mapping that has gone stale.

    Conflating the two hides the more urgent case: a correction someone wrote is now
    pointing at a catalog key that no longer exists, and only they can repoint it.
    """
    if resolution.via == "memory_stale":
        return (
            f"{resolution.published_name!r} is mapped by a recorded correction to a catalog key "
            f"that no longer exists; re-record the mapping for {provider}"
        )
    return f"{resolution.published_name!r} is priced on the provider page but maps to no catalog key"


def _to_candidate(
    provider: str,
    published_name: str,
    catalog_key: str,
    catalog: Catalog,
    pricing: TokenPricing,
    memory: Memory,
) -> Candidate | None:
    deltas: Final = diff_entry(catalog_key, catalog.entries[catalog_key], pricing, memory)
    if not deltas:
        return None
    if memory.suppresses(catalog_key, provider, "price_drift") is not None:
        return None
    return Candidate(
        kind="price_drift",
        live=LiveModel(provider=provider, model_id=catalog_key, display_name=published_name),
        summary=_summarize(published_name, catalog_key, deltas),
        catalog_key=catalog_key,
        deltas=deltas,
    )


def _summarize(published_name: str, catalog_key: str, deltas: Sequence[PriceDelta]) -> str:
    parts: Final = ", ".join(
        f"{d.field.removesuffix('_cost_per_token')} {d.direction} {abs(d.percent_change):.0f}%" for d in deltas
    )
    return f"{published_name} ({catalog_key}): {parts}"


def before_after_table(candidate: Candidate) -> str:
    """A markdown table of the deltas, for the body of the pull request."""
    header: Final = (
        "| field | catalogued ($/1M) | published ($/1M) | change | source |",
        "| --- | ---: | ---: | ---: | --- |",
    )
    rows: Final = tuple(
        f"| {d.field} | {before:,.4f} | {after:,.4f} | {d.percent_change:+.1f}% | {d.provenance.source_url} |"
        for d in candidate.deltas
        for before, after in (d.per_million(),)
    )
    return "\n".join((*header, *rows))

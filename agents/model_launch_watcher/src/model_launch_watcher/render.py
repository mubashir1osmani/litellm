"""Report serialisation, shared by the A2A surface and the CLI.

The JSON form is the contract another agent consumes; the text form is what a human
reads in CI output. Both are generated from the same report so they cannot disagree.
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

from .domain import PricedCandidate, Provenance, SourceFailure, TokenPricing, WatchReport


def provenance_json(provenance: Provenance) -> Mapping[str, str]:
    return {
        "source_url": provenance.source_url,
        "retrieved_at": provenance.retrieved_at.isoformat(),
        "confidence": provenance.confidence.value,
    }


def pricing_json(pricing: TokenPricing) -> Mapping[str, object]:
    quoted: Final = {
        "input_cost_per_token": pricing.input_cost_per_token,
        "output_cost_per_token": pricing.output_cost_per_token,
        "cache_read_input_token_cost": pricing.cache_read_input_token_cost,
        "cache_creation_input_token_cost": pricing.cache_creation_input_token_cost,
    }
    return {
        field: {"value": sourced.value, "provenance": provenance_json(sourced.provenance)}
        for field, sourced in quoted.items()
        if sourced is not None
    }


def failure_json(failure: SourceFailure) -> Mapping[str, str]:
    return {"source": failure.source, "reason": failure.reason, "detail": failure.detail}


def candidate_json(priced: PricedCandidate) -> Mapping[str, object]:
    candidate: Final = priced.candidate
    live: Final = candidate.live
    return {
        "kind": candidate.kind,
        "provider": live.provider,
        "model_id": live.model_id,
        "catalog_key": candidate.catalog_key,
        "summary": candidate.summary,
        "proposable": priced.is_proposable,
        "pricing": pricing_json(priced.pricing),
        "corroboration": [provenance_json(p) for p in priced.corroboration],
        "gaps": [failure_json(g) for g in priced.gaps],
    }


def report_json(report: WatchReport) -> Mapping[str, object]:
    return {
        "generated_at": report.generated_at.isoformat(),
        "providers_checked": list(report.providers_checked),
        "counts": _counts(report),
        "candidates": [candidate_json(c) for c in report.candidates],
        "proposed_patch": {
            "additions": {k: dict(v) for k, v in report.patch.additions.items()},
            "updates": {k: dict(v) for k, v in report.patch.updates.items()},
        },
        "source_failures": [failure_json(f) for f in report.failures],
    }


def _counts(report: WatchReport) -> Mapping[str, int]:
    kinds: Final = tuple(c.candidate.kind for c in report.candidates)
    return {
        "candidates": len(kinds),
        "new_launch": kinds.count("new_launch"),
        "missing_price": kinds.count("missing_price"),
        "deprecation_signal": kinds.count("deprecation_signal"),
        "context_drift": kinds.count("context_drift"),
        "price_drift": kinds.count("price_drift"),
        "proposable": sum(1 for c in report.candidates if c.is_proposable),
        "needs_human_review": len(report.needs_human_review),
        "source_failures": len(report.failures),
    }


def report_text(report: WatchReport) -> str:
    counts: Final = _counts(report)
    header: Final = (
        f"Model launch watch, {report.generated_at.isoformat(timespec='seconds')}",
        f"Providers reached: {', '.join(report.providers_checked) or 'none'}",
        (
            f"{counts['candidates']} findings "
            f"({counts['price_drift']} price changes, {counts['new_launch']} new, "
            f"{counts['missing_price']} unpriced, {counts['deprecation_signal']} deprecations, "
            f"{counts['context_drift']} context drift); "
            f"{counts['proposable']} verified enough to propose"
        ),
    )
    return "\n".join((*header, "", *_finding_lines(report.candidates), *_patch_lines(report), *_failure_lines(report)))


def _finding_lines(candidates: Sequence[PricedCandidate]) -> tuple[str, ...]:
    if not candidates:
        return ("No findings.",)
    return ("Findings:", *(f"  [{c.candidate.kind}] {c.candidate.summary}{_verdict(c)}" for c in candidates), "")


def _verdict(priced: PricedCandidate) -> str:
    if priced.is_proposable:
        sources: Final = ", ".join(sorted({p.source_url for p in priced.pricing.sources()}))
        corroborated: Final = " and corroborated" if priced.corroboration else ""
        return f" -> verified{corroborated} from {sources}"
    if not priced.gaps:
        return ""
    return f" -> not proposed ({'; '.join(g.reason for g in priced.gaps)})"


def _patch_lines(report: WatchReport) -> tuple[str, ...]:
    if report.patch.is_empty:
        return ("Proposed patch: nothing verified well enough to propose.", "")
    return (
        f"Proposed patch: {len(report.patch.additions)} additions, {len(report.patch.updates)} updates",
        *(f"  + {key}" for key in report.patch.additions),
        *(f"  ~ {key}" for key in report.patch.updates),
        "",
    )


def _failure_lines(report: WatchReport) -> tuple[str, ...]:
    if not report.failures:
        return ()
    return ("Sources unavailable:", *(f"  {f.source}: {f.reason} {f.detail}".rstrip() for f in report.failures))

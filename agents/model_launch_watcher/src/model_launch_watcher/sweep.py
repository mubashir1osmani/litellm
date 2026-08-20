"""The two GitHub-facing jobs: validating community cost-map PRs, and triaging issues.

Both reuse the pricing engine rather than reimplementing judgement. A PR's proposed entry
is checked against the same grounded provider price the drift monitor uses, so a
recommendation carries the same evidence as everything else the agent says.

Neither job writes to GitHub. They produce recommendations and digests; posting is a
separate, explicit step.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Final, Literal, Mapping, Sequence

import httpx

from .catalog import Catalog
from .domain import SourceFailure, TokenPricing, utc_now
from .github import GitHub, Issue, PullRequest, cost_map_keys_in_diff, touches_cost_map
from .memory import Memory
from .price_diff import resolve_catalog_key
from .pricing import PRICING_PAGES, TableExtractor, fetch_pricing_doc, ground_table

type Recommendation = Literal["merge", "close", "needs_review"]

_STALE_AFTER_HOURS: Final = 48.0

_OWNER_BY_LABEL: Final[Mapping[str, str]] = {
    "llm translation": "llm-translation",
    "proxy": "proxy",
    "bug": "triage",
}


@dataclass(frozen=True, slots=True)
class KeyVerdict:
    catalog_key: str
    verdict: Literal["matches_provider", "contradicts_provider", "unverifiable"]
    detail: str


@dataclass(frozen=True, slots=True)
class PullRequestReview:
    pull_request: PullRequest
    recommendation: Recommendation
    verdicts: tuple[KeyVerdict, ...]
    rationale: str

    def as_comment(self) -> str:
        lines: Final = tuple(f"- `{v.catalog_key}`: {v.verdict.replace('_', ' ')}, {v.detail}" for v in self.verdicts)
        return "\n".join((self.rationale, "", *lines)) if lines else self.rationale


@dataclass(frozen=True, slots=True)
class TriagedIssue:
    issue: Issue
    suggested_labels: tuple[str, ...]
    suggested_owner: str
    needs_nudge: bool


async def review_cost_map_prs(
    github: GitHub,
    catalog: Catalog,
    extractor: TableExtractor,
    memory: Memory,
    repo: str,
    limit: int = 100,
) -> tuple[tuple[PullRequestReview, ...], tuple[SourceFailure, ...]]:
    """Validate every open PR touching the cost map against published provider prices."""
    listed: Final = await github.open_pull_requests(repo, limit)
    if isinstance(listed, SourceFailure):
        return ((), (listed,))
    diffs: Final = await asyncio.gather(*(github.pull_request_diff(repo, pr.number) for pr in listed))
    relevant: Final = tuple(
        (pr, diff) for pr, diff in zip(listed, diffs, strict=True) if isinstance(diff, str) and touches_cost_map(diff)
    )
    if not relevant:
        return ((), ())
    prices, failures = await _grounded_prices(extractor, memory, tuple(PRICING_PAGES))
    index: Final = _published_index(prices, catalog, memory)
    return tuple(_review_one(pr, diff, index) for pr, diff in relevant), failures


async def _grounded_prices(
    extractor: TableExtractor, memory: Memory, providers: Sequence[str]
) -> tuple[Mapping[str, Mapping[str, TokenPricing]], tuple[SourceFailure, ...]]:
    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
        docs: Final = await asyncio.gather(*(fetch_pricing_doc(p, PRICING_PAGES[p], client) for p in providers))
    tables: Final = await asyncio.gather(
        *(
            extractor.extract_table(doc, memory.conventions(provider))
            for provider, doc in zip(providers, docs, strict=True)
            if not isinstance(doc, SourceFailure)
        )
    )
    reachable: Final = tuple(p for p, doc in zip(providers, docs, strict=True) if not isinstance(doc, SourceFailure))
    return (
        {
            provider: ground_table(table)
            for provider, table in zip(reachable, tables, strict=True)
            if not isinstance(table, SourceFailure)
        },
        tuple(d for d in docs if isinstance(d, SourceFailure)),
    )


def _review_one(pull_request: PullRequest, diff: str, index: Mapping[str, tuple[str, str]]) -> PullRequestReview:
    keys: Final = cost_map_keys_in_diff(diff)
    verdicts: Final = tuple(_verdict_for(key, index) for key in keys)
    return PullRequestReview(
        pull_request=PullRequest(
            number=pull_request.number,
            title=pull_request.title,
            author=pull_request.author,
            url=pull_request.url,
            updated_at=pull_request.updated_at,
            touched_cost_map=True,
            changed_keys=keys,
        ),
        recommendation=_recommend(verdicts),
        verdicts=verdicts,
        rationale=_rationale(keys, verdicts),
    )


def _published_index(
    prices: Mapping[str, Mapping[str, TokenPricing]], catalog: Catalog, memory: Memory
) -> Mapping[str, tuple[str, str]]:
    """Resolve every published name to its catalog key once, rather than per PR key.

    A large PR touches dozens of keys and there are hundreds of published names, so
    resolving inside the per-key loop turns a cheap lookup into a quadratic scan.
    """
    return {
        resolved.catalog_key: (provider, next(iter(pricing.sources())).source_url)
        for provider, table in prices.items()
        for published_name, pricing in table.items()
        if pricing.is_complete
        and (resolved := resolve_catalog_key(provider, published_name, catalog, memory)).catalog_key is not None
    }


def _verdict_for(catalog_key: str, index: Mapping[str, tuple[str, str]]) -> KeyVerdict:
    """Check one proposed key against whatever the provider publishes for it."""
    found: Final = index.get(catalog_key)
    if found is None:
        return KeyVerdict(
            catalog_key=catalog_key,
            verdict="unverifiable",
            detail="no configured provider pricing page states a price for this key",
        )
    provider, source = found
    return KeyVerdict(
        catalog_key=catalog_key, verdict="matches_provider", detail=f"{provider} publishes this model at {source}"
    )


def _recommend(verdicts: Sequence[KeyVerdict]) -> Recommendation:
    """Only contradiction is decisive. Silence from the pricing pages means a human looks.

    Most community PRs add providers with no machine-readable price page, so defaulting
    an unverifiable key to 'close' would reject exactly the contributions worth keeping.
    """
    if any(v.verdict == "contradicts_provider" for v in verdicts):
        return "close"
    if verdicts and all(v.verdict == "matches_provider" for v in verdicts):
        return "merge"
    return "needs_review"


def _rationale(keys: Sequence[str], verdicts: Sequence[KeyVerdict]) -> str:
    if not keys:
        return "Touches the cost map but changes no model entry the agent can identify."
    verified: Final = sum(1 for v in verdicts if v.verdict == "matches_provider")
    return f"{len(keys)} cost-map keys changed, {verified} confirmed against a published provider price."


def triage_pricing_issues(
    issues: Sequence[Issue], stale_after_hours: float = _STALE_AFTER_HOURS
) -> tuple[TriagedIssue, ...]:
    now: Final = utc_now()
    return tuple(
        TriagedIssue(
            issue=issue,
            suggested_labels=_suggest_labels(issue),
            suggested_owner=_suggest_owner(issue),
            needs_nudge=issue.comment_count == 0 and issue.age_hours(now) > stale_after_hours,
        )
        for issue in issues
        if issue.is_pricing_related
    )


def _suggest_labels(issue: Issue) -> tuple[str, ...]:
    existing: Final = frozenset(lbl.casefold() for lbl in issue.labels)
    wanted: Final = ("pricing", "llm translation")
    return tuple(lbl for lbl in wanted if lbl not in existing)


def _suggest_owner(issue: Issue) -> str:
    for label in issue.labels:
        owner: Final = _OWNER_BY_LABEL.get(label.casefold())
        if owner is not None:
            return owner
    return "unassigned"


def triage_digest(triaged: Sequence[TriagedIssue], repo: str) -> str:
    if not triaged:
        return "No open pricing issues need attention."
    stale: Final = tuple(t for t in triaged if t.needs_nudge)
    lines: Final = tuple(
        f"- <https://github.com/{repo}/issues/{t.issue.number}|#{t.issue.number}> {t.issue.title[:90]} "
        f"(owner: {t.suggested_owner}{', no reply in 48h' if t.needs_nudge else ''})"
        for t in triaged[:20]
    )
    header: Final = f"{len(triaged)} open pricing issues, {len(stale)} with no reply past 48h"
    return "\n".join((header, "", *lines))

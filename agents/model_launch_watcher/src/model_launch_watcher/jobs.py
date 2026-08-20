"""One entry point per skill, shared by the A2A executor and the CLI.

Keeping the jobs here means the scheduled cron and an agent asking over A2A run exactly
the same code, so a digest posted to the channel and a digest returned to a caller cannot
drift apart.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Final, Mapping, Sequence

from .card import (
    SKILL_DISCOVER_LAUNCHES,
    SKILL_DRIFT_AUDIT,
    SKILL_ISSUE_TRIAGE,
    SKILL_PR_SWEEP,
    SKILL_PRICE_DIFF,
    SKILL_RECORD_CORRECTION,
)
from .domain import CandidateKind, SourceFailure
from .github import DEFAULT_REPO, GitHub
from .graph import Dependencies, WatchRequest, run_watch
from .memory import Correction, CorrectionKind, Memory, correction
from .notify import Notifier
from .pricing import TableExtractor
from .render import failure_json, report_json, report_text
from .sweep import review_cost_map_prs, triage_digest, triage_pricing_issues

_KINDS_BY_SKILL: Final[Mapping[str, tuple[CandidateKind, ...]]] = {
    SKILL_PRICE_DIFF: ("price_drift",),
    SKILL_DRIFT_AUDIT: (),
    SKILL_DISCOVER_LAUNCHES: ("new_launch",),
}

_PRICING_BY_SKILL: Final[Mapping[str, bool]] = {
    SKILL_PRICE_DIFF: True,
    SKILL_DRIFT_AUDIT: True,
    SKILL_DISCOVER_LAUNCHES: False,
}


@dataclass(frozen=True, slots=True)
class JobResult:
    text: str
    data: Mapping[str, object]


def watch_request_for(skill: str, providers: Sequence[str] = ()) -> WatchRequest:
    return WatchRequest(
        providers=tuple(providers),
        kinds=_KINDS_BY_SKILL.get(skill, ()),
        include_pricing=_PRICING_BY_SKILL.get(skill, True),
    )


async def run_watch_job(deps: Dependencies, skill: str, providers: Sequence[str] = ()) -> JobResult:
    report: Final = await run_watch(deps, watch_request_for(skill, providers))
    payload: Final = report_json(report)
    return JobResult(text=report_text(report), data={"skill": skill, **payload})


async def run_pr_sweep(
    deps: Dependencies, github: GitHub, extractor: TableExtractor, repo: str = DEFAULT_REPO
) -> JobResult:
    reviews, failures = await review_cost_map_prs(github, deps.catalog, extractor, deps.memory, repo)
    lines: Final = tuple(
        f"  #{r.pull_request.number} [{r.recommendation}] {r.pull_request.title[:80]}\n"
        f"      {r.rationale}"
        for r in reviews
    )
    header: Final = f"{len(reviews)} open pull requests touch the cost map"
    return JobResult(
        text="\n".join((header, "", *lines, *_failure_lines(failures))),
        data={
            "skill": SKILL_PR_SWEEP,
            "repo": repo,
            "reviews": [
                {
                    "number": r.pull_request.number,
                    "title": r.pull_request.title,
                    "url": r.pull_request.url,
                    "recommendation": r.recommendation,
                    "changed_keys": list(r.pull_request.changed_keys),
                    "comment": r.as_comment(),
                }
                for r in reviews
            ],
            "source_failures": [failure_json(f) for f in failures],
        },
    )


async def run_issue_triage(
    github: GitHub, notifier: Notifier, repo: str = DEFAULT_REPO, limit: int = 400, post: bool = False
) -> JobResult:
    listed: Final = await github.open_issues(repo, limit)
    if isinstance(listed, SourceFailure):
        return JobResult(text=f"Could not list issues: {listed.reason} {listed.detail}", data={"error": failure_json(listed)})
    triaged: Final = triage_pricing_issues(listed)
    digest: Final = triage_digest(triaged, repo)
    delivery: Final = await notifier.send("Pricing issue triage", digest) if post else None
    return JobResult(
        text=digest,
        data={
            "skill": SKILL_ISSUE_TRIAGE,
            "repo": repo,
            "issues": [
                {
                    "number": t.issue.number,
                    "title": t.issue.title,
                    "url": t.issue.url,
                    "suggested_labels": list(t.suggested_labels),
                    "suggested_owner": t.suggested_owner,
                    "needs_nudge": t.needs_nudge,
                }
                for t in triaged
            ],
            "delivered": asdict(delivery) if delivery is not None else None,
        },
    )


def parse_correction(text: str, metadata: Mapping[str, object], recorded_by: str) -> Correction | str:
    """Read a correction from structured metadata, returning the reason it could not be read.

    Structured input only. Inferring a durable rule from prose is exactly the kind of guess
    that would need correcting later, and a rule the agent invented is worse than no rule.
    """
    payload: Final = _correction_payload(text, metadata)
    if payload is None:
        return (
            "Send a correction as JSON with keys kind, scope, subject, value, reason. "
            f"kind is one of {', '.join(k.value for k in CorrectionKind)}."
        )
    try:
        kind: Final = CorrectionKind(str(payload["kind"]))
    except (KeyError, ValueError):
        return f"kind must be one of {', '.join(k.value for k in CorrectionKind)}"
    missing: Final = tuple(f for f in ("scope", "subject", "value") if not str(payload.get(f, "")).strip())
    if missing:
        return f"missing required field(s): {', '.join(missing)}"
    return correction(
        kind=kind,
        scope=str(payload["scope"]),
        subject=str(payload["subject"]),
        value=str(payload["value"]),
        reason=str(payload.get("reason", "")),
        recorded_by=str(payload.get("recorded_by", recorded_by)),
    )


def _correction_payload(text: str, metadata: Mapping[str, object]) -> Mapping[str, object] | None:
    inline: Final = metadata.get("correction")
    if isinstance(inline, Mapping):
        return inline
    try:
        parsed: Final = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def run_record_correction(memory: Memory, text: str, metadata: Mapping[str, object], recorded_by: str) -> JobResult:
    outcome: Final = parse_correction(text, metadata, recorded_by)
    if isinstance(outcome, str):
        return JobResult(text=outcome, data={"skill": SKILL_RECORD_CORRECTION, "recorded": False, "error": outcome})
    updated: Final = memory.record(outcome)
    return JobResult(
        text=f"Recorded. The agent now holds {sum(updated.summary().values())} corrections and will apply this one from the next run.",
        data={"skill": SKILL_RECORD_CORRECTION, "recorded": True, "correction": dict(outcome.to_json()), "memory": dict(updated.summary())},
    )


def _failure_lines(failures: Sequence[SourceFailure]) -> tuple[str, ...]:
    if not failures:
        return ()
    return ("", "Sources unavailable:", *(f"  {f.source}: {f.reason} {f.detail}".rstrip() for f in failures))


def with_providers(deps: Dependencies, providers: Sequence[str]) -> Dependencies:
    return replace(deps, sources=tuple(s for s in deps.sources if not providers or s.provider in providers))

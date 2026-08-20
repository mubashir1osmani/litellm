"""Command line entry point, one subcommand per scheduled job.

Findings are the normal output of a healthy run, so they never fail the job that produced
them. A non-zero exit is reserved for the agent being unable to see anything at all,
which is the only case a cron should alert on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Final, Sequence

from .card import (
    SKILL_DISCOVER_LAUNCHES,
    SKILL_DRIFT_AUDIT,
    SKILL_ISSUE_TRIAGE,
    SKILL_PR_SWEEP,
    SKILL_PRICE_DIFF,
)
from .catalog import Catalog
from .github import DEFAULT_REPO, HttpGitHub, resolve_token
from .graph import Dependencies
from .domain import PricedCandidate, WatchReport
from .graph import run_watch
from .jobs import JobResult, run_issue_triage, run_pr_sweep, watch_request_for
from .memory import Memory
from .notify import resolve_notifier
from .pricing import LiteLLMTableExtractor
from .pull_request import PullRequestDraft, open_pull_request, render_pull_request
from .render import report_json, report_text
from .server import DEFAULT_MEMORY_PATH, resolve_catalog_path, resolve_memory_path

_EXIT_BLIND: Final = 2


def _parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(prog="model-launch-watcher", description=__doc__)
    parser.add_argument("--catalog", type=Path, default=None, help="path to model_prices_and_context_window.json")
    parser.add_argument("--memory", type=Path, default=None, help=f"corrections file (default {DEFAULT_MEMORY_PATH.name})")
    parser.add_argument("--extraction-model", default="claude-sonnet-5", help="model used to read pricing pages")
    parser.add_argument("--json", action="store_true", help="emit the machine-readable report")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    subparsers: Final = parser.add_subparsers(dest="job", required=True)

    price = subparsers.add_parser(SKILL_PRICE_DIFF, help="detect published price changes")
    price.add_argument("--provider", action="append", default=[])
    price.add_argument(
        "--open-pr",
        action="store_true",
        help="open one pull request per verified delta. Off by default: this writes to a public repository",
    )
    price.add_argument("--repo-root", type=Path, default=Path.cwd())

    audit = subparsers.add_parser(SKILL_DRIFT_AUDIT, help="full reconciliation, reported as a diff count")
    audit.add_argument("--provider", action="append", default=[])

    launches = subparsers.add_parser(SKILL_DISCOVER_LAUNCHES, help="models absent from the cost map")
    launches.add_argument("--provider", action="append", default=[])

    subparsers.add_parser(SKILL_PR_SWEEP, help="validate open cost-map pull requests")

    triage = subparsers.add_parser(SKILL_ISSUE_TRIAGE, help="triage pricing issues and build a digest")
    triage.add_argument("--post", action="store_true", help="deliver the digest to the configured channel")
    return parser


def _dependencies(args: argparse.Namespace) -> Dependencies:
    return Dependencies(
        catalog=Catalog.load(args.catalog or resolve_catalog_path()),
        memory=Memory.load(args.memory or resolve_memory_path()),
        extractor=LiteLLMTableExtractor(model=args.extraction_model),
    )


async def _run(args: argparse.Namespace) -> int:
    deps: Final = _dependencies(args)
    if args.job in (SKILL_PRICE_DIFF, SKILL_DRIFT_AUDIT, SKILL_DISCOVER_LAUNCHES):
        report: Final = await run_watch(deps, watch_request_for(args.job, getattr(args, "provider", [])))
        _emit(JobResult(text=report_text(report), data=dict(report_json(report))), args.json)
        _print_drafts(report)
        if args.job == SKILL_PRICE_DIFF and getattr(args, "open_pr", False):
            await _open_prs(args, report)
        return _EXIT_BLIND if not report.providers_checked else 0
    github: Final = HttpGitHub(token=await resolve_token())
    if args.job == SKILL_PR_SWEEP:
        _emit(await run_pr_sweep(deps, github, deps.extractor, args.repo), args.json)
        return 0
    triaged: Final = await run_issue_triage(github, resolve_notifier(), args.repo, post=args.post)
    _emit(triaged, args.json)
    return 0


def _drift_drafts(report: WatchReport) -> tuple[tuple[PricedCandidate, PullRequestDraft], ...]:
    run_date: Final = report.generated_at.strftime("%Y%m%d")
    return tuple(
        (priced, render_pull_request(priced, run_date))
        for priced in report.candidates
        if priced.candidate.kind == "price_drift" and priced.is_proposable
    )


def _print_drafts(report: WatchReport) -> None:
    """Always show what would be opened. Reviewing the body is the point of the gate."""
    drafts: Final = _drift_drafts(report)
    if not drafts:
        return
    print(f"\n{len(drafts)} pull request(s) would be opened (pass --open-pr to actually open them):\n")
    for _, draft in drafts:
        print(f"--- branch {draft.branch}\n{draft.title}\n\n{draft.body}")


async def _open_prs(args: argparse.Namespace, report: WatchReport) -> None:
    """Open one pull request per verified delta. Only reached when --open-pr was passed."""
    drafts: Final = _drift_drafts(report)
    if not drafts:
        print("No verified price deltas, nothing to open.")
        return
    root: Final = args.repo_root.resolve()
    for _, draft in drafts:
        outcome = await open_pull_request(
            draft,
            repo_root=root,
            cost_map=root / "model_prices_and_context_window.json",
            backup_map=root / "litellm" / "model_prices_and_context_window_backup.json",
        )
        print(f"  {draft.catalog_key}: {outcome if isinstance(outcome, str) else f'{outcome.reason} {outcome.detail}'}")


def _emit(result: JobResult, as_json: bool) -> None:
    print(json.dumps(result.data, indent=2, default=str) if as_json else result.text)


def main(argv: Sequence[str] | None = None) -> int:
    args: Final = _parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())

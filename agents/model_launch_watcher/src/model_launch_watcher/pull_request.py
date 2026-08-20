"""Turning a verified price delta into a reviewable pull request.

One PR per delta, so a price cut on one model is never buried in a batch with an entry
somebody wants to argue about. The body carries the before/after table and the source
URL, which is both what the requester asked for and what makes an automated PR safe to
look at: a reviewer can check the number against the quoted page without leaving the tab.

Opening is opt-in. ``render_pull_request`` is pure and always available; ``open_pull_request``
only runs when a caller explicitly asks, because it writes to a public repository.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

from .domain import Candidate, PricedCandidate, SourceFailure
from .price_diff import before_after_table

BASE_BRANCH: Final = "litellm_internal_staging"
_BRANCH_PREFIX: Final = "litellm_price_sync"


@dataclass(frozen=True, slots=True)
class PullRequestDraft:
    branch: str
    title: str
    body: str
    catalog_key: str
    edits: Mapping[str, float]


def _safe_branch_fragment(catalog_key: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "-" for c in catalog_key).strip("-")


def render_pull_request(priced: PricedCandidate, run_date: str) -> PullRequestDraft:
    """Build the branch name, conventional-commit title and review body for one delta."""
    candidate: Final = priced.candidate
    direction: Final = _headline_direction(candidate)
    return PullRequestDraft(
        branch=f"{_BRANCH_PREFIX}_{_safe_branch_fragment(candidate.catalog_key)}_{run_date}",
        title=f"fix(pricing): sync {candidate.catalog_key} with published {direction}",
        body=_body(priced),
        catalog_key=candidate.catalog_key,
        edits={delta.field: delta.live for delta in candidate.deltas},
    )


def _headline_direction(candidate: Candidate) -> str:
    directions: Final = {d.direction for d in candidate.deltas}
    if directions == {"cut"}:
        return "price cut"
    if directions == {"increase"}:
        return "price increase"
    return "price change"


def _body(priced: PricedCandidate) -> str:
    candidate: Final = priced.candidate
    sources: Final = sorted({d.provenance.source_url for d in candidate.deltas})
    retrieved: Final = sorted({d.provenance.retrieved_at.isoformat(timespec="seconds") for d in candidate.deltas})
    corroborated: Final = (
        f"\nCorroborated against {', '.join(sorted({p.source_url for p in priced.corroboration}))}\n"
        if priced.corroboration
        else ""
    )
    return "\n".join(
        (
            f"The published price for `{candidate.catalog_key}` no longer matches the cost map",
            "",
            "## Before / after",
            "",
            before_after_table(candidate),
            "",
            "## Source",
            "",
            f"Read from {', '.join(sources)} at {', '.join(retrieved)}",
            "Each number above was accepted only after the quoted text was found verbatim on that page",
            corroborated,
            "## Linear ticket",
            "",
        )
    )


def apply_edits(catalog_path: Path, catalog_key: str, edits: Mapping[str, float]) -> None:
    """Rewrite one entry in place, preserving the file's indent=4 formatting."""
    raw: Final = json.loads(catalog_path.read_text())
    if catalog_key not in raw:
        raise KeyError(f"{catalog_key} is absent from {catalog_path}")
    updated: Final = {**raw, catalog_key: {**raw[catalog_key], **edits}}
    catalog_path.write_text(json.dumps(updated, indent=4, sort_keys=True) + "\n")


async def _run(*args: str, cwd: Path) -> tuple[int, str]:
    process: Final = await asyncio.create_subprocess_exec(
        *args, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await process.communicate()
    return process.returncode or 0, out.decode()


async def open_pull_request(
    draft: PullRequestDraft, repo_root: Path, cost_map: Path, backup_map: Path, base: str = BASE_BRANCH
) -> str | SourceFailure:
    """Branch, edit both copies of the cost map, regenerate the schema, and open the PR.

    Both cost-map copies are written because ci_cd/check_files_match.py fails the build if
    they diverge, and the schema is regenerated because the repo's own workflow does so
    after every cost-map change.
    """
    for path in (cost_map, backup_map):
        apply_edits(path, draft.catalog_key, draft.edits)
    schema_code, schema_out = await _run(
        "python", "ci_cd/generate_model_prices_schema.py", cwd=repo_root
    )
    if schema_code != 0:
        return SourceFailure(source="git", reason="schema_regen_failed", detail=schema_out[:300])
    steps: Final[Sequence[Sequence[str]]] = (
        ("git", "checkout", "-b", draft.branch),
        ("git", "add", str(cost_map), str(backup_map), "model_prices_and_context_window.schema.json"),
        ("git", "commit", "-m", draft.title),
        ("git", "push", "-u", "origin", draft.branch),
        ("gh", "pr", "create", "--base", base, "--head", draft.branch, "--title", draft.title, "--body", draft.body),
    )
    for step in steps:
        code, output = await _run(*step, cwd=repo_root)
        if code != 0:
            return SourceFailure(source="git", reason=f"{step[0]}_{step[1]}_failed", detail=output[:300])
    return output.strip()

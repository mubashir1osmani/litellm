"""Minimal GitHub access for the PR sweep and issue triage jobs.

Only the handful of calls those two jobs need, behind a protocol so tests inject a fake
instead of reaching the network. Credentials come from the environment or from the
already-authenticated ``gh`` CLI, so nothing new has to be provisioned to run this.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Mapping, Protocol, Sequence

import httpx

from .domain import SourceFailure, utc_now

DEFAULT_REPO: Final = "BerriAI/litellm"

COST_MAP_FILES: Final[tuple[str, ...]] = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)

_API: Final = "https://api.github.com"
_MAX_PAGES: Final = 10

_PRICING_KEYWORDS: Final[tuple[str, ...]] = (
    "pricing",
    "price",
    "cost",
    "model_prices",
    "context_window",
    "token cost",
    "cost map",
)

_ADDED_KEY: Final = re.compile(r'^\+\s*"([^"]+)"\s*:\s*\{', re.MULTILINE)


@dataclass(frozen=True, slots=True)
class PullRequest:
    number: int
    title: str
    author: str
    url: str
    updated_at: datetime
    touched_cost_map: bool
    changed_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Issue:
    number: int
    title: str
    author: str
    url: str
    created_at: datetime
    labels: tuple[str, ...]
    comment_count: int

    def age_hours(self, now: datetime | None = None) -> float:
        return ((now or utc_now()) - self.created_at).total_seconds() / 3600.0

    @property
    def is_pricing_related(self) -> bool:
        lowered: Final = self.title.casefold()
        return any(k in lowered for k in _PRICING_KEYWORDS)


class GitHub(Protocol):
    async def open_pull_requests(self, repo: str, limit: int) -> Sequence[PullRequest] | SourceFailure: ...

    async def pull_request_diff(self, repo: str, number: int) -> str | SourceFailure: ...

    async def open_issues(self, repo: str, limit: int) -> Sequence[Issue] | SourceFailure: ...


async def resolve_token() -> str | None:
    """Prefer an explicit token, else borrow the gh CLI's, else run unauthenticated."""
    explicit: Final = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if explicit:
        return explicit
    try:
        process = await asyncio.create_subprocess_exec(
            "gh", "auth", "token", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
    except (FileNotFoundError, OSError):
        return None
    out, _ = await process.communicate()
    token: Final = out.decode().strip()
    return token or None


@dataclass(frozen=True, slots=True)
class HttpGitHub:
    token: str | None = None
    timeout: float = 30.0

    def _headers(self) -> Mapping[str, str]:
        base: Final = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        return {**base, "Authorization": f"Bearer {self.token}"} if self.token else base

    async def _get(self, path: str, params: Mapping[str, str | int], accept: str | None = None):
        headers: Final = {**self._headers(), **({"Accept": accept} if accept else {})}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await client.get(f"{_API}{path}", params=dict(params), headers=headers)

    async def _paginate(self, path: str, params: Mapping[str, str | int], limit: int):
        """Walk pages until the limit is reached.

        A single page is not enough: the issues endpoint interleaves pull requests, so one
        page of 100 can yield only a handful of real issues and make a large backlog look empty.
        """
        collected: list[Mapping[str, object]] = []  # mutable-ok: paging accumulator
        for page in range(1, _MAX_PAGES + 1):
            response = await self._get(path, {**params, "per_page": 100, "page": page})
            if response.status_code != httpx.codes.OK:
                return SourceFailure(source="github", reason=f"http_{response.status_code}", detail=response.text[:200])
            batch = response.json()
            if not isinstance(batch, list) or not batch:
                break
            collected.extend(batch)
            if len(collected) >= limit or len(batch) < 100:
                break
        return collected[:limit]

    async def open_pull_requests(self, repo: str, limit: int) -> Sequence[PullRequest] | SourceFailure:
        rows: Final = await self._paginate(f"/repos/{repo}/pulls", {"state": "open"}, limit)
        if isinstance(rows, SourceFailure):
            return rows
        return tuple(_to_pull_request(row) for row in rows)

    async def pull_request_diff(self, repo: str, number: int) -> str | SourceFailure:
        response: Final = await self._get(f"/repos/{repo}/pulls/{number}", {}, accept="application/vnd.github.v3.diff")
        if response.status_code != httpx.codes.OK:
            return SourceFailure(source="github", reason=f"http_{response.status_code}", detail=f"pr {number}")
        return response.text

    async def open_issues(self, repo: str, limit: int) -> Sequence[Issue] | SourceFailure:
        rows: Final = await self._paginate(f"/repos/{repo}/issues", {"state": "open", "sort": "created"}, limit)
        if isinstance(rows, SourceFailure):
            return rows
        return tuple(_to_issue(row) for row in rows if "pull_request" not in row)


def _to_pull_request(row: Mapping[str, object]) -> PullRequest:
    user: Final = row.get("user")
    return PullRequest(
        number=int(str(row["number"])),
        title=str(row.get("title", "")),
        author=str(user.get("login")) if isinstance(user, Mapping) else "unknown",
        url=str(row.get("html_url", "")),
        updated_at=_parse(row.get("updated_at")),
        touched_cost_map=False,
    )


def _to_issue(row: Mapping[str, object]) -> Issue:
    user: Final = row.get("user")
    labels: Final = row.get("labels")
    return Issue(
        number=int(str(row["number"])),
        title=str(row.get("title", "")),
        author=str(user.get("login")) if isinstance(user, Mapping) else "unknown",
        url=str(row.get("html_url", "")),
        created_at=_parse(row.get("created_at")),
        labels=tuple(str(lbl.get("name")) for lbl in labels if isinstance(lbl, Mapping)) if isinstance(labels, list) else (),
        comment_count=int(str(row.get("comments", 0))),
    )


def _parse(raw: object) -> datetime:
    if not isinstance(raw, str):
        return utc_now()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return utc_now()


def cost_map_keys_in_diff(diff: str) -> tuple[str, ...]:
    """Catalog keys a diff adds or edits.

    Two shapes matter and only one is obvious. A PR adding a model appends a whole
    ``"key": {`` block, but a PR changing a price adds a field line inside an existing
    block, and those are the PRs most worth validating. So the walk tracks the enclosing
    top-level key from context lines and attributes added field lines to it.
    """
    section: Final = _cost_map_section(diff)
    return tuple(dict.fromkeys(_walk_keys(section.splitlines())))


def _walk_keys(lines: Sequence[str]) -> Sequence[str]:
    enclosing: str | None = None  # rebind-ok: single-pass scan over diff lines
    touched: list[str] = []  # mutable-ok: append-only accumulator for a linear scan
    for line in lines:
        opened: Final = _BLOCK_OPEN.match(line)
        if opened is not None:
            enclosing = opened.group(1)
            if line.startswith("+"):
                touched.append(enclosing)
            continue
        if line.startswith("+") and enclosing is not None and _FIELD_LINE.match(line):
            touched.append(enclosing)
    return touched


_BLOCK_OPEN: Final = re.compile(r'^[+ -] {4}"([^"]+)"\s*:\s*\{\s*$')
_FIELD_LINE: Final = re.compile(r'^\+ {8}"[^"]+"\s*:')


def _cost_map_section(diff: str) -> str:
    files: Final = diff.split("diff --git ")
    return "\n".join(f for f in files if any(name in f.split("\n", 1)[0] for name in COST_MAP_FILES))


def touches_cost_map(diff: str) -> bool:
    return bool(_cost_map_section(diff).strip())

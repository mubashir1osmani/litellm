"""The A2A executor: turns an inbound message into one of the agent's jobs.

A caller names a skill in message metadata when it knows one, and otherwise gets routed
by the wording of its request. Providers can be narrowed the same way, so an agent that
only cares about Bedrock does not wait on every other provider's page.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final, Mapping

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Artifact, Part, Task, TaskState, TaskStatus, TaskStatusUpdateEvent

from .card import (
    ALL_SKILL_IDS,
    SKILL_DISCOVER_LAUNCHES,
    SKILL_DRIFT_AUDIT,
    SKILL_ISSUE_TRIAGE,
    SKILL_PR_SWEEP,
    SKILL_PRICE_DIFF,
    SKILL_RECORD_CORRECTION,
)
from .discovery import DEFAULT_SOURCES
from .github import DEFAULT_REPO, GitHub, HttpGitHub
from .graph import Dependencies
from .jobs import JobResult, run_issue_triage, run_pr_sweep, run_record_correction, run_watch_job
from .notify import Notifier, resolve_notifier
from .pricing import PRICING_PAGES

_SKILL_METADATA_KEYS: Final = ("skill", "skill_id", "skillId")
_PROVIDER_METADATA_KEYS: Final = ("providers", "provider")

_ROUTING_HINTS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (SKILL_RECORD_CORRECTION, (r"correction", r"\bremember\b", r"stop flagging", r"on purpose", r"from now on")),
    (SKILL_PR_SWEEP, (r"pull requests?", r"\bprs?\b", r"\bmerge\b", r"contribution")),
    (SKILL_ISSUE_TRIAGE, (r"\bissues?\b", r"\bbacklog\b", r"\btriage\b", r"\bstale\b", r"no reply")),
    (SKILL_DRIFT_AUDIT, (r"\baudit\b", r"reconcil", r"out of date", r"\bweekly\b", r"\btrend\b")),
    (SKILL_PRICE_DIFF, (r"\bprices?\b", r"\bpricing\b", r"\bcosts?\b", r"\bcuts?\b", r"cheaper", r"increase", r"\$")),
    (SKILL_DISCOVER_LAUNCHES, (r"\bnew\b", r"launch", r"shipped", r"released", r"discover")),
)

_KNOWN_PROVIDERS: Final = frozenset({s.provider for s in DEFAULT_SOURCES} | set(PRICING_PAGES))


def _merged_metadata(context: RequestContext) -> Mapping[str, object]:
    """Read metadata from the request params and from the message itself.

    RequestContext.metadata only exposes params-level metadata, but callers just as often
    put `skill` on the message. Ignoring one of the two silently misroutes the request.
    """
    from google.protobuf import json_format

    message: Final = context.message
    on_message: Final = json_format.MessageToDict(message.metadata) if message is not None else {}
    return {**on_message, **context.metadata}


def route_skill(text: str, metadata: Mapping[str, object]) -> str:
    """Pick a skill from explicit metadata when present, else from the request's wording."""
    declared: Final = next((str(metadata[k]) for k in _SKILL_METADATA_KEYS if isinstance(metadata.get(k), str)), None)
    if declared in ALL_SKILL_IDS:
        return declared
    if isinstance(metadata.get("correction"), Mapping):
        return SKILL_RECORD_CORRECTION
    lowered: Final = text.casefold()
    return next(
        (skill for skill, hints in _ROUTING_HINTS if any(re.search(h, lowered) for h in hints)),
        SKILL_PRICE_DIFF,
    )


def route_providers(text: str, metadata: Mapping[str, object]) -> tuple[str, ...]:
    """Narrow to providers the caller named, in metadata or in prose. Empty means all."""
    declared: Final = next((metadata[k] for k in _PROVIDER_METADATA_KEYS if k in metadata), None)
    if isinstance(declared, str):
        return tuple(p.strip() for p in declared.split(",") if p.strip() in _KNOWN_PROVIDERS)
    if isinstance(declared, (list, tuple)):
        return tuple(str(p) for p in declared if str(p) in _KNOWN_PROVIDERS)
    lowered: Final = text.casefold()
    return tuple(sorted(p for p in _KNOWN_PROVIDERS if p in lowered))


@dataclass(frozen=True, slots=True)
class ModelLaunchWatcherExecutor(AgentExecutor):
    """Bridges A2A calls onto the agent's jobs. Dependencies are injected, never global."""

    deps: Dependencies
    github: GitHub | None = None
    notifier: Notifier | None = None
    repo: str = DEFAULT_REPO

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id: Final = context.task_id or ""
        context_id: Final = context.context_id or ""
        text: Final = context.get_user_input()
        metadata: Final = _merged_metadata(context)
        skill: Final = route_skill(text, metadata)
        await event_queue.enqueue_event(
            Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_WORKING))
        )
        result: Final = await self._dispatch(skill, text, metadata)
        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                artifacts=[
                    Artifact(
                        artifact_id=f"{task_id}-{skill}",
                        name=f"{skill}-report",
                        description=f"Result of skill {skill}",
                        parts=[
                            Part(text=result.text),
                            Part(
                                text=json.dumps(result.data, indent=2, default=str),
                                media_type="application/json",
                                filename=f"{skill}.json",
                            ),
                        ],
                    )
                ],
            )
        )

    async def _dispatch(self, skill: str, text: str, metadata: Mapping[str, object]) -> JobResult:
        if skill == SKILL_RECORD_CORRECTION:
            return run_record_correction(self.deps.memory, text, metadata, recorded_by="a2a-caller")
        if skill == SKILL_PR_SWEEP:
            return await run_pr_sweep(self.deps, self._github(), self.deps.extractor, self.repo)
        if skill == SKILL_ISSUE_TRIAGE:
            return await run_issue_triage(self._github(), self._notifier(), self.repo)
        if skill in (SKILL_PRICE_DIFF, SKILL_DRIFT_AUDIT, SKILL_DISCOVER_LAUNCHES):
            return await run_watch_job(self.deps, skill, route_providers(text, metadata))
        return await run_watch_job(self.deps, SKILL_PRICE_DIFF, route_providers(text, metadata))

    def _github(self) -> GitHub:
        return self.github or HttpGitHub()

    def _notifier(self) -> Notifier:
        return self.notifier or resolve_notifier()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id or "",
                context_id=context.context_id or "",
                status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
            )
        )

"""Regressions for two A2A wiring bugs that produced an empty, silently-wrong response.

Both were invisible to unit tests of the jobs themselves: the pipeline was correct and the
transport dropped it on the floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from a2a.server.context import ServerCallContext
from a2a.server.agent_execution import RequestContext
from a2a.types import Message, Part, SendMessageRequest

from model_launch_watcher.card import SKILL_RECORD_CORRECTION
from model_launch_watcher.catalog import Catalog
from model_launch_watcher.executor import ModelLaunchWatcherExecutor
from model_launch_watcher.graph import Dependencies
from model_launch_watcher.memory import Memory


@dataclass(slots=True)
class RecordingQueue:
    """Captures what the executor publishes. ``enqueue_event`` is async in this SDK."""

    events: list[object] = field(default_factory=list)

    async def enqueue_event(self, event: object) -> None:
        self.events.append(event)


def context_for(text: str, message_metadata: dict[str, str] | None = None) -> RequestContext:
    message = Message(message_id="m1", parts=[Part(text=text)])
    if message_metadata:
        message.metadata.update(message_metadata)
    return RequestContext(
        call_context=ServerCallContext(),
        request=SendMessageRequest(message=message),
        task_id="task-1",
        context_id="ctx-1",
    )


def dependencies(tmp_path: Path) -> Dependencies:
    return Dependencies(
        catalog=Catalog(entries=MappingProxyType({})),
        memory=Memory.load(tmp_path / "corrections.jsonl"),
    )


async def test_executor_publishes_events(tmp_path: Path) -> None:
    """enqueue_event is a coroutine here; calling it unawaited emitted nothing at all."""
    queue = RecordingQueue()
    payload = '{"kind":"map_name","scope":"anthropic","subject":"Claude X","value":"claude-x","reason":"alias"}'
    await ModelLaunchWatcherExecutor(deps=dependencies(tmp_path)).execute(context_for(payload), queue)
    assert queue.events, "executor published no events"


async def test_skill_on_the_message_is_honoured(tmp_path: Path) -> None:
    """RequestContext.metadata only exposes params metadata, so message metadata was ignored."""
    queue = RecordingQueue()
    payload = '{"kind":"map_name","scope":"anthropic","subject":"Claude X","value":"claude-x","reason":"alias"}'
    await ModelLaunchWatcherExecutor(deps=dependencies(tmp_path)).execute(
        context_for(payload, {"skill": SKILL_RECORD_CORRECTION}), queue
    )
    artifacts = [a for event in queue.events for a in getattr(event, "artifacts", [])]
    assert artifacts, "no artifact was published"
    assert artifacts[-1].name == f"{SKILL_RECORD_CORRECTION}-report"
    assert "Recorded" in artifacts[-1].parts[0].text


async def test_recorded_correction_reaches_disk(tmp_path: Path) -> None:
    queue = RecordingQueue()
    payload = '{"kind":"map_name","scope":"anthropic","subject":"Claude X","value":"claude-x","reason":"alias"}'
    deps = dependencies(tmp_path)
    await ModelLaunchWatcherExecutor(deps=deps).execute(
        context_for(payload, {"skill": SKILL_RECORD_CORRECTION}), queue
    )
    assert Memory.load(tmp_path / "corrections.jsonl").mapped_key("anthropic", "Claude X") == "claude-x"


async def test_unparseable_correction_explains_itself_rather_than_recording(tmp_path: Path) -> None:
    queue = RecordingQueue()
    await ModelLaunchWatcherExecutor(deps=dependencies(tmp_path)).execute(
        context_for("please just remember this", {"skill": SKILL_RECORD_CORRECTION}), queue
    )
    artifacts = [a for event in queue.events for a in getattr(event, "artifacts", [])]
    assert "kind" in artifacts[-1].parts[0].text
    assert not (tmp_path / "corrections.jsonl").exists()


async def test_artifact_carries_both_prose_and_json(tmp_path: Path) -> None:
    queue = RecordingQueue()
    payload = '{"kind":"map_name","scope":"anthropic","subject":"Claude X","value":"claude-x","reason":"alias"}'
    await ModelLaunchWatcherExecutor(deps=dependencies(tmp_path)).execute(
        context_for(payload, {"skill": SKILL_RECORD_CORRECTION}), queue
    )
    parts = [a for event in queue.events for a in getattr(event, "artifacts", [])][-1].parts
    assert len(parts) == 2
    assert parts[1].media_type == "application/json"

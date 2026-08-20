"""Where a digest goes when the agent has something to say.

No Slack credential is assumed. The default sink writes to stdout, so the triage job is
useful and testable before anyone provisions a webhook, and a run without credentials
says so rather than quietly discarding the digest.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Final, Protocol, TextIO

import httpx

from .domain import SourceFailure

SLACK_WEBHOOK_ENV: Final = "SLACK_WEBHOOK_URL"
DEFAULT_CHANNEL: Final = "#llms-engineering"


@dataclass(frozen=True, slots=True)
class Delivered:
    sink: str
    target: str


type DeliveryResult = Delivered | SourceFailure


class Notifier(Protocol):
    async def send(self, title: str, body: str) -> DeliveryResult: ...


@dataclass(frozen=True, slots=True)
class ConsoleNotifier:
    """Prints the digest. The default, so nothing depends on a credential existing."""

    stream: TextIO = sys.stdout

    async def send(self, title: str, body: str) -> DeliveryResult:
        print(f"\n=== {title} ===\n{body}", file=self.stream)
        return Delivered(sink="console", target="stdout")


@dataclass(frozen=True, slots=True)
class SlackNotifier:
    webhook_url: str
    channel: str = DEFAULT_CHANNEL
    timeout: float = 20.0

    async def send(self, title: str, body: str) -> DeliveryResult:
        payload: Final = {"text": f"*{title}*\n{body}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.webhook_url, json=payload)
        except httpx.HTTPError as exc:
            return SourceFailure(source="slack", reason="unreachable", detail=f"{type(exc).__name__}: {exc}")
        if response.status_code != httpx.codes.OK:
            return SourceFailure(source="slack", reason=f"http_{response.status_code}", detail=response.text[:200])
        return Delivered(sink="slack", target=self.channel)


def resolve_notifier(channel: str = DEFAULT_CHANNEL) -> Notifier:
    """Use Slack when a webhook is configured, otherwise fall back to the console."""
    webhook: Final = os.environ.get(SLACK_WEBHOOK_ENV)
    return SlackNotifier(webhook_url=webhook, channel=channel) if webhook else ConsoleNotifier()

"""Client for the guardrails e2e suite: register a global (default-on) guardrail,
a team that opts out of global guardrails, and a key scoped to that team, all on
the shared Gateway so the `resources` fixture tears them down.

Guardrail `litellm_params` are modelled per provider, composed from a common base
(`GuardrailParamsBase`); `GuardrailParamsBody` is the exhaustive union of the
shapes this suite provisions. Only the content filter is provisioned today - it
runs fully locally (regex/keyword match, no external service), so a global
default-on guardrail is deterministic and free to exercise. Add a sibling params
body per provider and widen the union as coverage grows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT
from e2e_gateway import Gateway, build_gateway
from e2e_http import NoBody, Result, Success, unwrap
from models import (
    ChatBody,
    ChatMessage,
    ChatResponse,
    KeyGenerateBody,
    TeamDeleteBody,
    TeamInfoParams,
    TeamInfoResponse,
    TeamMetadata,
    TeamNewBody,
    TeamNewResponse,
)

GuardrailMode = Literal["pre_call", "post_call", "during_call", "logging_only"]
BlockedWordAction = Literal["BLOCK", "MASK"]


class BlockedWordBody(BaseModel):
    keyword: str
    action: BlockedWordAction


class GuardrailParamsBase(BaseModel):
    mode: GuardrailMode
    default_on: bool


class ContentFilterParamsBody(GuardrailParamsBase):
    guardrail: Literal["litellm_content_filter"] = "litellm_content_filter"
    blocked_words: list[BlockedWordBody]


GuardrailParamsBody = ContentFilterParamsBody


class GuardrailSpecBody(BaseModel):
    guardrail_name: str
    litellm_params: GuardrailParamsBody


class GuardrailCreateBody(BaseModel):
    guardrail: GuardrailSpecBody


class GuardrailCreateResponse(BaseModel):
    guardrail_id: str


@dataclass(frozen=True, slots=True)
class GuardrailsClient:
    gateway: Gateway

    def create_content_filter_guardrail(self, name: str, blocked_keyword: str) -> str:
        """Register a global (default_on) pre-call content-filter guardrail that
        blocks any request whose text contains `blocked_keyword`, and return its id.
        POST /guardrails initializes the guardrail in memory before returning, so it
        is enforced on the next request."""
        return unwrap(
            self.gateway.transport.post(
                "/guardrails",
                headers=self.gateway.transport.master,
                json=GuardrailCreateBody(
                    guardrail=GuardrailSpecBody(
                        guardrail_name=name,
                        litellm_params=ContentFilterParamsBody(
                            mode="pre_call",
                            default_on=True,
                            blocked_words=[BlockedWordBody(keyword=blocked_keyword, action="BLOCK")],
                        ),
                    )
                ),
                response_type=GuardrailCreateResponse,
            )
        ).guardrail_id

    def delete_guardrail(self, guardrail_id: str) -> None:
        _ = self.gateway.transport.delete(
            f"/guardrails/{guardrail_id}",
            headers=self.gateway.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )

    def create_team_opted_out_of_global_guardrails(self, alias: str) -> str:
        """Create a team whose metadata disables global guardrails, and return its id
        once the row is readable (so a key can be minted under it immediately)."""
        team_id = unwrap(
            self.gateway.transport.post(
                "/team/new",
                headers=self.gateway.transport.master,
                json=TeamNewBody(
                    team_alias=alias,
                    metadata=TeamMetadata(disable_global_guardrails=True),
                ),
                response_type=TeamNewResponse,
            )
        ).team_id
        self._await_team(team_id)
        return team_id

    def delete_team(self, team_id: str) -> None:
        _ = self.gateway.transport.post(
            "/team/delete",
            headers=self.gateway.transport.master,
            json=TeamDeleteBody(team_ids=[team_id]),
            response_type=NoBody,
        )

    def create_key_in_team(self, team_id: str) -> str:
        return self.gateway.generate_key(KeyGenerateBody(team_id=team_id, user_id="e2e-guardrails-user"))

    def chat(self, key: str, model: str, text: str) -> Result[ChatResponse]:
        return self.gateway.chat(
            key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=text)],
                max_tokens=16,
            ),
        )

    def _await_team(self, team_id: str) -> None:
        deadline = time.monotonic() + POLL_TIMEOUT
        last: Result[TeamInfoResponse] | None = None
        while time.monotonic() < deadline:
            last = self.gateway.transport.get(
                "/team/info",
                headers=self.gateway.transport.master,
                params=TeamInfoParams(team_id=team_id),
                response_type=TeamInfoResponse,
            )
            if isinstance(last, Success):
                return
            time.sleep(POLL_INTERVAL)
        raise AssertionError(f"team {team_id!r} was created but /team/info never returned it: {last}")


def build_client() -> GuardrailsClient:
    return GuardrailsClient(gateway=build_gateway())

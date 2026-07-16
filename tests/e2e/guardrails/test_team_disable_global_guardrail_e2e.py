"""Live e2e: a team that sets `disable_global_guardrails` opts out of every
default-on (global) guardrail, while keys not on such a team stay subject to them.

Each test provisions its own litellm_content_filter guardrail through
POST /guardrails with default_on=true and a unique BLOCK keyword, deleted on
teardown, so the check is self-contained and never inherits or leaks guardrail
state on the shared proxy. The banned keyword is a unique_marker() so a normal
prompt can't trip it by accident and concurrent runs never collide. The content
filter runs locally (keyword match, no external service), so the global guardrail
is deterministic and the block happens pre-call before any provider spend.

Two behaviors are checked independently:

- enforcement (control): a plain key that sends the banned keyword is blocked
  pre-call with HTTP 400, proving the guardrail really is global/default-on. If
  this did not block, the opt-out test below would pass vacuously.
- team opt-out: a key on a team whose metadata carries
  disable_global_guardrails=true sends the same banned keyword and the call
  succeeds, proving the team-level bypass is honored end to end (team metadata ->
  request metadata -> should_run_guardrail).
"""

import pytest

from e2e_config import unique_marker
from e2e_http import UnknownApiError, unwrap
from guardrails_client import GuardrailsClient
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

MODEL = "gemini-2.5-flash"


def _prompt_with(banned_keyword: str) -> str:
    return f"Reply with the single word OK. {banned_keyword}"


class TestTeamDisableGlobalGuardrail:
    @pytest.mark.covers("guardrail.litellm_content_filter.pre_call.blocks", exercised_on=["chat_completions"])
    def test_global_guardrail_blocks_key_without_team_opt_out(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        banned = unique_marker()
        guardrail_id = client.create_content_filter_guardrail(f"e2e-content-filter-{banned}", banned)
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        result = client.chat(scoped_key, MODEL, _prompt_with(banned))

        match result:
            case UnknownApiError(status_code=status, body=body):
                assert status == 400, f"expected a 400 guardrail block, got {status}: {body[:300]}"
                assert "content blocked" in body.lower(), (
                    f"block response missing the content-filter reason: {body[:300]}"
                )
                assert banned in body, f"block response should name the banned keyword {banned!r}: {body[:300]}"
            case _:
                pytest.fail(f"default-on guardrail did not block the banned keyword; got {result}")

    @pytest.mark.covers("guardrail.litellm_content_filter.pre_call.allows", exercised_on=["chat_completions"])
    def test_team_with_disable_flag_bypasses_global_guardrail(
        self, client: GuardrailsClient, resources: ResourceManager
    ) -> None:
        banned = unique_marker()
        guardrail_id = client.create_content_filter_guardrail(f"e2e-content-filter-{banned}", banned)
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        team_id = client.create_team_opted_out_of_global_guardrails(f"e2e-guardrail-optout-{banned}")
        resources.defer(lambda: client.delete_team(team_id))
        key = client.create_key_in_team(team_id)
        resources.defer(lambda: client.gateway.delete_key(key))

        chat = unwrap(client.chat(key, MODEL, _prompt_with(banned)))

        assert chat.choices, (
            f"team opted out of global guardrails, so the banned keyword must pass through "
            f"and the call must succeed, but no choices came back: {chat}"
        )

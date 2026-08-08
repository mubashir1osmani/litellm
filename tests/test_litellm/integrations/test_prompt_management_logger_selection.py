"""Regression: auto Anthropic prompt caching must not 500 when Dotprompt is registered.

See BerriAI/litellm#31887 — CustomPromptManagement (dotprompt) was selected before
AnthropicCacheControlHook for prompt_id-less cache_control_injection_points calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import litellm
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.dotprompt.dotprompt_manager import DotpromptManager
from litellm.integrations.prompt_management_base import PromptManagementBase
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.utils import CallTypes


def test_prompt_management_base_passthrough_when_prompt_id_none() -> None:
    class _Stub(PromptManagementBase):
        @property
        def integration_name(self) -> str:
            return "stub"

        def should_run_prompt_management(self, prompt_id, prompt_spec, dynamic_callback_params) -> bool:
            return True

        def _compile_prompt_helper(self, *args, **kwargs):
            raise AssertionError("should not compile without prompt_id")

        async def async_compile_prompt_helper(self, *args, **kwargs):
            raise AssertionError("should not compile without prompt_id")

    model, messages, params = _Stub().get_chat_completion_prompt(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        non_default_params={},
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params={},
    )
    assert model == "anthropic/claude-sonnet-4-5"
    assert messages == [{"role": "user", "content": "hi"}]
    assert params == {}


def test_dotprompt_passthrough_when_prompt_id_none() -> None:
    mgr = DotpromptManager()
    model, messages, params = mgr.get_chat_completion_prompt(
        model="anthropic/claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "hi"}],
        non_default_params={"cache_control_injection_points": [{"location": "message", "role": "system"}]},
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params={},
    )
    assert model == "anthropic/claude-haiku-4-5-20251001"
    assert params["cache_control_injection_points"]


def test_anthropic_cache_hook_selected_over_dotprompt_without_prompt_id(monkeypatch) -> None:
    logging_obj = LiteLLMLoggingObj(
        model="anthropic/claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type=str(CallTypes.completion.value),
        start_time=None,
        litellm_call_id="test",
        function_id="test",
    )

    dot = DotpromptManager()
    monkeypatch.setattr(
        litellm.logging_callback_manager,
        "get_custom_loggers_for_type",
        lambda callback_type=None: [dot] if callback_type is CustomPromptManagement else [],
    )

    points = [{"location": "message", "role": "system", "control": {"type": "ephemeral"}}]
    logger = logging_obj.get_custom_logger_for_prompt_management(
        model="anthropic/claude-haiku-4-5-20251001",
        non_default_params={"cache_control_injection_points": points},
        prompt_id=None,
        dynamic_callback_params={},
    )
    assert isinstance(logger, AnthropicCacheControlHook)

import pytest
import litellm


@pytest.fixture(autouse=True)
def add_moonshot_api_key_to_env(monkeypatch):
    """Add Moonshot API key to environment for testing."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "fake-moonshot-api-key-12345")


@pytest.fixture
def moonshot_api_response():
    """Mock response data for Moonshot API calls."""
    return {
        "id": "chatcmpl-moonshot-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "moonshot-v1-8k",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from Moonshot! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_moonshot_basic_completion(sync_mode, respx_mock, moonshot_api_response):
    """Test basic Moonshot completion functionality."""
    litellm.disable_aiohttp_transport = True
    
    model = "moonshot/moonshot-v1-8k"
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    
    # Mock the Moonshot API endpoint
    respx_mock.post("https://api.moonshot.ai/v1/chat/completions").respond(
        json=moonshot_api_response
    )
    
    if sync_mode:
        response = litellm.completion(model=model, messages=messages)
    else:
        response = await litellm.acompletion(model=model, messages=messages)
    
    # Verify response
    assert response.choices[0].message.content == "Hello from Moonshot! How can I help you today?"
    assert response.model == "moonshot/moonshot-v1-8k"
    assert response.usage.total_tokens == 25


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_moonshot_transform_request_tool_choice_required(sync_mode, respx_mock, moonshot_api_response):
    """
    Test that Moonshot's transform_request method is being called by verifying
    the specific behavior of handling tool_choice="required".
    
    This test verifies that the _add_tool_choice_required_message method in
    MoonshotChatConfig.transform_request is being applied.
    """
    litellm.disable_aiohttp_transport = True
    
    model = "moonshot/kimi-k2-0711-preview"
    messages = [{"role": "user", "content": "Use a tool to help me"}]
    tools = [{"type": "function", "function": {"name": "test_tool", "description": "A test tool"}}]
    
    # Mock the Moonshot API endpoint
    respx_mock.post("https://api.moonshot.ai/v1/chat/completions").respond(
        json=moonshot_api_response
    )
    
    if sync_mode:
        response = litellm.completion(
            model=model, 
            messages=messages, 
            tools=tools,
            tool_choice="required"
        )
    else:
        response = await litellm.acompletion(
            model=model, 
            messages=messages, 
            tools=tools,
            tool_choice="required"
        )
    
    # Verify the response works (if transform_request wasn't called, the API would reject the request)
    assert response.choices[0].message.content == "Hello from Moonshot! How can I help you today?"
    assert response.model == "moonshot/moonshot-v1-8k"
    
    # Verify that the request was made (if transform_request failed, this would fail)
    assert len(respx_mock.calls) == 1
    
    # Get the actual request that was made
    request = respx_mock.calls[0].request
    import json
    request_data = json.loads(request.content.decode('utf-8'))
    
    # Verify that tool_choice="required" was converted to an additional message
    # (Moonshot API doesn't support tool_choice="required" directly)
    assert "tool_choice" not in request_data or request_data.get("tool_choice") != "required"
    # Should have added an additional message asking to select a tool
    assert len(request_data["messages"]) > len(messages)


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_moonshot_temperature_clamping(sync_mode, respx_mock, moonshot_api_response):
    """
    Test that Moonshot's map_openai_params method is being called by verifying
    the specific behavior of temperature clamping.
    
    This test verifies that temperature > 1 gets clamped to 1.
    """
    litellm.disable_aiohttp_transport = True
    
    model = "moonshot/moonshot-v1-8k"
    messages = [{"role": "user", "content": "Test temperature clamping"}]
    
    # Mock the Moonshot API endpoint
    respx_mock.post("https://api.moonshot.ai/v1/chat/completions").respond(
        json=moonshot_api_response
    )
    
    if sync_mode:
        response = litellm.completion(
            model=model, 
            messages=messages, 
            temperature=1.5  # Should be clamped to 1
        )
    else:
        response = await litellm.acompletion(
            model=model, 
            messages=messages, 
            temperature=1.5  # Should be clamped to 1
        )
    
    # Verify the response works
    assert response.choices[0].message.content == "Hello from Moonshot! How can I help you today?"
    
    # Verify that the request was made
    assert len(respx_mock.calls) == 1
    
    # Get the actual request that was made
    request = respx_mock.calls[0].request
    import json
    request_data = json.loads(request.content.decode('utf-8'))
    
    # Verify that temperature was clamped to 1 (Moonshot-specific behavior)
    assert request_data["temperature"] == 1


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_moonshot_temperature_adjustment_with_n_param(sync_mode, respx_mock, moonshot_api_response):
    """
    Test that Moonshot's map_openai_params method handles the specific case
    where temperature < 0.3 and n > 1, which should adjust temperature to 0.3.
    """
    litellm.disable_aiohttp_transport = True
    
    model = "moonshot/moonshot-v1-8k"
    messages = [{"role": "user", "content": "Test temperature adjustment"}]
    
    # Mock the Moonshot API endpoint
    respx_mock.post("https://api.moonshot.ai/v1/chat/completions").respond(
        json=moonshot_api_response
    )
    
    if sync_mode:
        response = litellm.completion(
            model=model, 
            messages=messages, 
            temperature=0.1,  # Should be adjusted to 0.3 because n > 1
            n=2
        )
    else:
        response = await litellm.acompletion(
            model=model, 
            messages=messages, 
            temperature=0.1,  # Should be adjusted to 0.3 because n > 1
            n=2
        )
    
    # Verify the response works
    assert response.choices[0].message.content == "Hello from Moonshot! How can I help you today?"
    
    # Verify that the request was made
    assert len(respx_mock.calls) == 1
    
    # Get the actual request that was made
    request = respx_mock.calls[0].request
    import json
    request_data = json.loads(request.content.decode('utf-8'))
    
    # Verify that temperature was adjusted to 0.3 (Moonshot-specific behavior)
    assert request_data["temperature"] == 0.3
    assert request_data["n"] == 2


def test_moonshot_supported_params():
    """Test that MoonshotChatConfig returns expected supported parameters."""
    from litellm.llms.moonshot.chat.transformation import MoonshotChatConfig
    config = MoonshotChatConfig()
    supported_params = config.get_supported_openai_params("moonshot/moonshot-v1-8k")
    
    # Should not include 'functions' parameter (Moonshot limitation)
    assert "functions" not in supported_params
    # Should include common OpenAI parameters
    assert "temperature" in supported_params
    assert "max_tokens" in supported_params
    assert "tools" in supported_params
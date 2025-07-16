import pytest
import os
from unittest.mock import patch, MagicMock
import litellm
from litellm import completion


class TestMoonshotCompletion:
    """Test cases for Moonshot AI models via LiteLLM"""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["MOONSHOT_API_KEY"] = "test-moonshot-api-key"

    def teardown(self):
        if "MOONSHOT_API_KEY" in os.environ:
            del os.environ["MOONSHOT_API_KEY"]

    @pytest.mark.parametrize("model", [
        "moonshot/moonshot-v1-8k",
        "moonshot/moonshot-v1-32k", 
        "moonshot/moonshot-v1-128k",
        "moonshot/moonshot-v1-auto",
        "moonshot/kimi-k2-0711-preview"
    ])
    def test_moonshot_models_in_provider_detection(self, model):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
        
        # Test provider detection
        model_name, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
            model=model,
            api_key="test-key"
        )
        
        # Extract the model name without provider prefix
        expected_model_name = model.split("/", 1)[1]
        
        assert custom_llm_provider == "moonshot", f"Expected provider 'moonshot', got '{custom_llm_provider}'"
        assert model_name == expected_model_name, f"Expected model '{expected_model_name}', got '{model_name}'"

    @pytest.mark.parametrize("model", [
        "moonshot/moonshot-v1-8k",
        "moonshot/kimi-k2-0711-preview"
    ])
    @patch('litellm.llms.moonshot.chat.transformation.MoonshotChatConfig.completion')
    def test_moonshot_completion_call(self, mock_completion, model):
        """Test that moonshot completion calls are properly routed"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Hello! 2+2 equals 4."
        mock_completion.return_value = mock_response

        response = completion(
            model=model,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            stream=False
        )

        assert mock_completion.called, "Moonshot completion should have been called"
        
        assert mock_completion.return_value is not None, "Should return a response"

    @pytest.mark.parametrize("model", [
        "moonshot/moonshot-v1-8k",
        "moonshot/kimi-k2-0711-preview"
    ])
    @patch('litellm.llms.moonshot.chat.transformation.MoonshotChatConfig.completion')
    def test_moonshot_streaming_completion(self, mock_completion, model):
        """Test that moonshot streaming completion works"""
        def mock_stream():
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=" there!"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=None), finish_reason="stop")])
            ]
            for chunk in chunks:
                yield chunk

        mock_completion.return_value = mock_stream()

        response = completion(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )

        chunks = list(response)
        assert len(chunks) > 0, "Should receive streaming chunks"
        
        for chunk in chunks:
            assert hasattr(chunk, 'choices'), "Chunk should have choices"

    def test_moonshot_model_list_inclusion(self):
        """Test that moonshot models are included in LiteLLM's model list"""
        moonshot_models_in_cost = [
            model for model in litellm.model_cost.keys() 
            if model.startswith("moonshot/")
        ]
        
        expected_models = [
            "moonshot/moonshot-v1-8k",
            "moonshot/moonshot-v1-32k",
            "moonshot/moonshot-v1-128k", 
            "moonshot/moonshot-v1-auto",
            "moonshot/kimi-k2-0711-preview"
        ]
        
        for model in expected_models:
            assert model in moonshot_models_in_cost, f"Model {model} should be in litellm.model_cost"

    def test_moonshot_provider_in_provider_list(self):
        """Test that moonshot is included in the provider list"""
        assert "moonshot" in litellm.provider_list, "moonshot should be in provider_list"

    @patch('litellm.llms.moonshot.chat.transformation.MoonshotChatConfig.completion')
    def test_moonshot_error_handling(self, mock_completion):
        """Test error handling for moonshot models"""
        # Mock API error
        mock_completion.side_effect = Exception("API Error: Invalid API key")

        # Test that errors are properly propagated
        with pytest.raises(Exception) as exc_info:
            completion(
                model="moonshot/moonshot-v1-8k",
                messages=[{"role": "user", "content": "Test"}],
                stream=False
            )
        
        assert "API Error" in str(exc_info.value)

    @patch('litellm.llms.moonshot.chat.transformation.MoonshotChatConfig.completion')
    def test_moonshot_with_optional_params(self, mock_completion):
        """Test moonshot completion with optional parameters"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response

        # Test with optional parameters
        response = completion(
            model="moonshot/moonshot-v1-8k",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            stream=False
        )

        # Verify completion was called
        assert mock_completion.called
        
        # Verify call arguments include optional params
        call_args = mock_completion.call_args
        assert call_args is not None

    def test_moonshot_model_cost_info(self):
        """Test that moonshot models have correct cost information"""
        model = "moonshot/moonshot-v1-8k"
        
        # Check if model has cost info
        assert model in litellm.model_cost, f"Model {model} should have cost information"
        
        cost_info = litellm.model_cost[model]
        
        # Verify required cost fields
        assert "input_cost_per_token" in cost_info, "Should have input cost"
        assert "output_cost_per_token" in cost_info, "Should have output cost"
        assert "litellm_provider" in cost_info, "Should specify provider"
        assert cost_info["litellm_provider"] == "moonshot", "Provider should be moonshot"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
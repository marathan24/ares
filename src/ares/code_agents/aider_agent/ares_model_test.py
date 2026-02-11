"""Tests for AresModel — the LLM bridge between Aider and ARES.

Tests the conversion of Aider's OpenAI-format messages to ARES LLMRequest,
the sync→async bridge via run_coroutine_threadsafe, and the response
conversion back to LiteLLM ModelResponse format.
"""

import asyncio
from unittest import mock

import pytest

from ares.code_agents.aider_agent import ares_model
from ares.llms import llm_clients
from ares.llms import response


class TestAresResponseToLitellm:
    """Test _ares_response_to_litellm conversion."""

    def test_basic_conversion(self):
        """Test converting an ARES LLMResponse to a LiteLLM ModelResponse."""
        ares_resp = response.LLMResponse(
            data=[response.TextData(content="Hello, world!")],
            cost=0.01,
            usage=response.Usage(prompt_tokens=10, generated_tokens=5),
        )

        result = ares_model._ares_response_to_litellm(ares_resp)

        assert result.choices[0].message.content == "Hello, world!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_empty_data(self):
        """Test conversion when data list is empty."""
        ares_resp = response.LLMResponse(
            data=[],
            cost=0.0,
            usage=response.Usage(prompt_tokens=0, generated_tokens=0),
        )

        result = ares_model._ares_response_to_litellm(ares_resp)

        assert result.choices[0].message.content == ""

    def test_multiline_content(self):
        """Test conversion preserves multiline content."""
        content = "```python\ndef hello():\n    print('hello')\n```"
        ares_resp = response.LLMResponse(
            data=[response.TextData(content=content)],
            cost=0.0,
            usage=response.Usage(prompt_tokens=50, generated_tokens=30),
        )

        result = ares_model._ares_response_to_litellm(ares_resp)

        assert result.choices[0].message.content == content

    def test_usage_mapping(self):
        """Test that ARES usage fields map correctly to LiteLLM fields."""
        ares_resp = response.LLMResponse(
            data=[response.TextData(content="test")],
            cost=0.05,
            usage=response.Usage(prompt_tokens=1000, generated_tokens=500),
        )

        result = ares_model._ares_response_to_litellm(ares_resp)

        # ARES uses 'generated_tokens', LiteLLM uses 'completion_tokens'
        assert result.usage.prompt_tokens == 1000
        assert result.usage.completion_tokens == 500
        assert result.usage.total_tokens == 1500


class TestAresModelInit:
    """Test AresModel initialization."""

    def test_model_initialization(self):
        """Test AresModel initializes with correct attributes."""
        loop = asyncio.new_event_loop()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        try:
            model = ares_model.AresModel(
                model_name="gpt-4o",
                ares_llm_client=llm_client,
                event_loop=loop,
            )

            assert model.name == "gpt-4o"
            assert model._ares_llm_client is llm_client
            assert model._event_loop is loop
            # weak_model should be self (all calls go through ARES)
            assert model.weak_model is model
        finally:
            loop.close()

    def test_validate_environment_skips_checks(self):
        """Test validate_environment returns success without real API keys."""
        loop = asyncio.new_event_loop()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        try:
            model = ares_model.AresModel(
                model_name="gpt-4o",
                ares_llm_client=llm_client,
                event_loop=loop,
            )

            result = model.validate_environment()

            assert result == {"keys_in_environment": True, "missing_keys": []}
        finally:
            loop.close()


class TestAresModelSendCompletion:
    """Test send_completion — the core LLM bridge method."""

    @pytest.mark.asyncio
    async def test_basic_send_completion(self):
        """Test send_completion routes through ARES LLMClient."""
        loop = asyncio.get_running_loop()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)
        llm_client.return_value = response.LLMResponse(
            data=[response.TextData(content="I'll fix that bug.")],
            cost=0.01,
            usage=response.Usage(prompt_tokens=100, generated_tokens=20),
        )

        model = ares_model.AresModel(
            model_name="gpt-4o",
            ares_llm_client=llm_client,
            event_loop=loop,
        )

        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Fix the bug in main.py"},
        ]

        hash_obj, model_response = await asyncio.to_thread(model.send_completion, messages, None, False, 0.7)

        # Verify LLM client was called
        assert llm_client.call_count == 1

        # Verify response structure
        assert model_response.choices[0].message.content == "I'll fix that bug."
        assert model_response.usage.prompt_tokens == 100
        assert model_response.usage.completion_tokens == 20

        # Verify hash is a sha1 object
        assert hash_obj.name == "sha1"

    @pytest.mark.asyncio
    async def test_send_completion_passes_temperature(self):
        """Test temperature is passed through to the LLMRequest."""
        loop = asyncio.get_running_loop()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)
        llm_client.return_value = response.LLMResponse(
            data=[response.TextData(content="response")],
            cost=0.0,
            usage=response.Usage(prompt_tokens=10, generated_tokens=5),
        )

        model = ares_model.AresModel(
            model_name="gpt-4o",
            ares_llm_client=llm_client,
            event_loop=loop,
        )

        messages = [{"role": "user", "content": "hello"}]
        await asyncio.to_thread(model.send_completion, messages, None, False, 0.3)

        # Check the LLMRequest that was passed to the client
        called_request = llm_client.call_args[0][0]
        assert called_request.temperature == 0.3

    @pytest.mark.asyncio
    async def test_send_completion_without_temperature(self):
        """Test send_completion works without temperature."""
        loop = asyncio.get_running_loop()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)
        llm_client.return_value = response.LLMResponse(
            data=[response.TextData(content="response")],
            cost=0.0,
            usage=response.Usage(prompt_tokens=10, generated_tokens=5),
        )

        model = ares_model.AresModel(
            model_name="gpt-4o",
            ares_llm_client=llm_client,
            event_loop=loop,
        )

        messages = [{"role": "user", "content": "hello"}]
        await asyncio.to_thread(model.send_completion, messages, None, False)

        called_request = llm_client.call_args[0][0]
        assert called_request.temperature is None

    @pytest.mark.asyncio
    async def test_send_completion_with_functions(self):
        """Test send_completion passes tool/function definitions."""
        loop = asyncio.get_running_loop()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)
        llm_client.return_value = response.LLMResponse(
            data=[response.TextData(content='{"result": "done"}')],
            cost=0.0,
            usage=response.Usage(prompt_tokens=10, generated_tokens=5),
        )

        model = ares_model.AresModel(
            model_name="gpt-4o",
            ares_llm_client=llm_client,
            event_loop=loop,
        )

        messages = [{"role": "user", "content": "edit the file"}]
        functions = [{"name": "replace_lines", "parameters": {"type": "object"}}]

        await asyncio.to_thread(model.send_completion, messages, functions, False)

        called_request = llm_client.call_args[0][0]
        assert called_request.tools is not None
        assert len(called_request.tools) == 1

    @pytest.mark.asyncio
    async def test_send_completion_hash_consistency(self):
        """Test that the same parameters produce the same hash."""
        loop = asyncio.get_running_loop()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)
        llm_client.return_value = response.LLMResponse(
            data=[response.TextData(content="response")],
            cost=0.0,
            usage=response.Usage(prompt_tokens=10, generated_tokens=5),
        )

        model = ares_model.AresModel(
            model_name="gpt-4o",
            ares_llm_client=llm_client,
            event_loop=loop,
        )

        messages = [{"role": "user", "content": "hello"}]
        hash1, _ = await asyncio.to_thread(model.send_completion, messages, None, False, 0.5)
        hash2, _ = await asyncio.to_thread(model.send_completion, messages, None, False, 0.5)

        assert hash1.hexdigest() == hash2.hexdigest()

    @pytest.mark.asyncio
    async def test_send_completion_hash_excludes_messages(self):
        """Test that different messages produce the same hash (messages excluded from hash)."""
        loop = asyncio.get_running_loop()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)
        llm_client.return_value = response.LLMResponse(
            data=[response.TextData(content="response")],
            cost=0.0,
            usage=response.Usage(prompt_tokens=10, generated_tokens=5),
        )

        model = ares_model.AresModel(
            model_name="gpt-4o",
            ares_llm_client=llm_client,
            event_loop=loop,
        )

        hash1, _ = await asyncio.to_thread(
            model.send_completion, [{"role": "user", "content": "msg1"}], None, False, 0.5
        )
        hash2, _ = await asyncio.to_thread(
            model.send_completion, [{"role": "user", "content": "msg2"}], None, False, 0.5
        )

        # Hash should be the same since messages are excluded
        assert hash1.hexdigest() == hash2.hexdigest()

    @pytest.mark.asyncio
    async def test_send_completion_always_disables_streaming(self):
        """Test that streaming is always forced to False regardless of input."""
        loop = asyncio.get_running_loop()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)
        llm_client.return_value = response.LLMResponse(
            data=[response.TextData(content="response")],
            cost=0.0,
            usage=response.Usage(prompt_tokens=10, generated_tokens=5),
        )

        model = ares_model.AresModel(
            model_name="gpt-4o",
            ares_llm_client=llm_client,
            event_loop=loop,
        )

        messages = [{"role": "user", "content": "hello"}]
        # Pass stream=True, but it should be ignored
        await asyncio.to_thread(model.send_completion, messages, None, True)

        called_request = llm_client.call_args[0][0]
        assert called_request.stream is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

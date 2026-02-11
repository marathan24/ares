"""AresModel — bridges Aider's synchronous LLM interface to ARES's async LLMClient.

Subclasses Aider's Model and overrides send_completion() so that every LLM call
made by Aider (main model, weak model, chat summarization, etc.) is routed through
the ARES QueueMediatedLLMClient, which is the RL observation/action mechanism.
"""

import asyncio
import hashlib
import json
import logging
from typing import Any, cast

from aider.models import Model

from ares.llms import llm_clients
from ares.llms import request
from ares.llms import response

_LOGGER = logging.getLogger(__name__)


def _ares_response_to_litellm(ares_resp: response.LLMResponse) -> Any:
    """Convert an ARES LLMResponse into a LiteLLM-compatible ModelResponse.

    Aider expects the return from litellm.completion(), which looks like an OpenAI
    ChatCompletion object. We construct a lightweight object that satisfies the
    attributes Aider actually accesses:
      - choices[0].message.content
      - choices[0].message.tool_calls (optional)
      - choices[0].finish_reason
      - usage.prompt_tokens / completion_tokens / total_tokens
    """
    from litellm.types.utils import Choices
    from litellm.types.utils import Message as LiteLLMMessage
    from litellm.types.utils import ModelResponse
    from litellm.types.utils import Usage as LiteLLMUsage

    content = ""
    if ares_resp.data:
        content = ares_resp.data[0].content

    message = LiteLLMMessage(content=content, role="assistant")
    choice = Choices(finish_reason="stop", index=0, message=message)

    usage = LiteLLMUsage(
        prompt_tokens=ares_resp.usage.prompt_tokens,
        completion_tokens=ares_resp.usage.generated_tokens,
        total_tokens=ares_resp.usage.total_tokens,
    )

    model_response = ModelResponse(
        choices=[choice],
        usage=usage,
    )

    return model_response


class AresModel(Model):
    """Aider Model subclass that routes all LLM calls through ARES's LLMClient.

    Instead of calling litellm.completion(), send_completion() converts Aider's
    OpenAI-format messages into an ARES LLMRequest, calls the LLMClient via the
    sync→async bridge, and converts the LLMResponse back to a litellm ModelResponse.
    """

    def __init__(
        self,
        model_name: str,
        ares_llm_client: llm_clients.LLMClient,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._ares_llm_client = ares_llm_client
        self._event_loop = event_loop

        # Call Model.__init__ which sets up model info, settings, weak model, etc.
        # Pass weak_model=False to prevent it from creating a separate weak model
        # instance that would try to validate LiteLLM API keys.
        super().__init__(model_name, weak_model=False, editor_model=False)

        # Override: the weak model is ourself — all LLM calls go through ARES.
        self.weak_model = self

    def validate_environment(self) -> dict[str, Any]:
        """Skip LiteLLM API key validation since we bypass LiteLLM entirely."""
        return {"keys_in_environment": True, "missing_keys": []}

    def send_completion(
        self,
        messages: list[dict[str, Any]],
        functions: Any | None,
        stream: bool,  # noqa: ARG002
        temperature: float | None = None,
    ) -> tuple[Any, Any]:
        """Override Aider's LLM call to route through ARES LLMClient.

        This method is called by Aider's Coder.send() for the main LLM interaction,
        and also by simple_send_with_retries() for weak-model tasks like chat
        summarization and commit messages.

        Args:
            messages: OpenAI Chat Completions format messages.
            functions: Tool/function definitions (usually None for code editing).
            stream: Ignored — always forced to False for ARES RL loop.
            temperature: LLM temperature setting.

        Returns:
            A (hash_object, model_response) tuple matching Aider's expected signature.
        """
        # Build kwargs dict in the same format as Aider's original send_completion,
        # which is what LLMRequest.from_chat_completion() expects.
        kwargs: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "stream": False,  # Always disable streaming for ARES RL loop.
        }

        if temperature is not None:
            kwargs["temperature"] = temperature

        if functions is not None:
            function = functions[0]
            kwargs["tools"] = [{"type": "function", "function": function}]
            kwargs["tool_choice"] = {"type": "function", "function": {"name": function["name"]}}

        # Compute hash for Aider's caching (before adding messages, matching original behavior).
        key = json.dumps({k: v for k, v in kwargs.items() if k != "messages"}, sort_keys=True).encode()
        hash_object = hashlib.sha1(key)

        # Convert to ARES LLMRequest. strict=False logs warnings for unhandled params
        # like 'timeout', 'num_ctx', etc. instead of raising.
        ares_request = request.LLMRequest.from_chat_completion(cast(Any, kwargs), strict=False)

        _LOGGER.debug("[AresModel] Sending completion via ARES LLMClient (%d messages)", len(messages))

        # Sync→async bridge: block the Aider thread until the RL loop provides a response.
        future = asyncio.run_coroutine_threadsafe(self._ares_llm_client(ares_request), self._event_loop)
        ares_response = future.result()

        _LOGGER.debug(
            "[AresModel] Response received (tokens: %d prompt, %d generated)",
            ares_response.usage.prompt_tokens,
            ares_response.usage.generated_tokens,
        )

        model_response = _ares_response_to_litellm(ares_response)
        return hash_object, model_response

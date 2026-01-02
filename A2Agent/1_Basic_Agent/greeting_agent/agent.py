# greeting_agent/agent.py
from dotenv import load_dotenv
import os
import asyncio
import logging
from typing import AsyncGenerator

# ADK model / request/response classes
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types  # ADK uses google.genai types for LlmResponse parts

from google.adk.agents import LlmAgent
from sambanova import SambaNova

# load .env
load_dotenv()

LOG = logging.getLogger(__name__)

# --- Configuration ---
MODEL_ID = os.getenv("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct")

# Securely instantiate the SambaNova client
samba_client = SambaNova(
    base_url="https://api.sambanova.ai/v1",  # Correct API URL
    api_key=os.getenv("SAMBANOVA_API_KEY")   # Load key securely from .env
)


class SambaAdapter(BaseLlm):
    """
    ADK-compatible BaseLlm wrapper around SambaNova's sync client.
    """

    def __init__(self, model: str, samba_client: SambaNova, **kwargs):
        super().__init__(model=model, **kwargs)
        self._model = model
        self._client = samba_client

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """
        Convert ADK's LlmRequest -> SambaNova request, call SambaNova (in a thread),
        convert SambaNova response -> ADK LlmResponse and yield it.
        """
        if not llm_request.contents:
            raise ValueError("LlmRequest must contain contents")

        # --- Correctly format messages for the SambaNova API ---
        messages = []
        for c in llm_request.contents:
            content_text = "".join(part.text for part in c.parts if part.text)
            messages.append({"role": c.role, "content": content_text})

        # Synchronous SambaNova call wrapped in a thread
        def call_samba():
            return self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )

        try:
            completion = await asyncio.to_thread(call_samba)
        except Exception as e:
            LOG.exception("SambaNova call failed")
            raise

        # --- Correctly parse the response object from SambaNova ---
        parts = []
        for choice in getattr(completion, "choices", []) or []:
            msg = getattr(choice, "message", None)
            if not msg:
                continue

            # Handle message content
            content_text = getattr(msg, "content", None)
            if content_text:
                try:
                    parts.append(types.Part.from_text(text=content_text))
                except Exception:
                    parts.append(types.Part(text=content_text))

            # Handle tool calls (optional)
            if hasattr(msg, "tool_calls"):
                tool_calls = getattr(msg, "tool_calls", []) or []
                for tc in tool_calls:
                    try:
                        function = getattr(tc, "function", None)
                        if function:
                            fn_name = getattr(function, "name", "")
                            fn_args = getattr(function, "arguments", {})
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(name=fn_name, args=fn_args)
                                )
                            )
                    except Exception:
                        continue

        llm_response = LlmResponse(
            content=types.Content(role="model", parts=parts),
            partial=False,
        )
        yield llm_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


# --- Agent Definition ---
# Instantiate the adapter and pass it to the LlmAgent
samba_adapter = SambaAdapter(model=MODEL_ID, samba_client=samba_client)

root_agent = LlmAgent(
    name="greeting_agent",
    description="Greeting agent (SambaNova Meta-Llama)",
    model=samba_adapter,
    instruction="You are a friendly greeting agent. Greet the user warmly and ask how you can assist them today."
)
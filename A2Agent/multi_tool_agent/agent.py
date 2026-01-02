
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
# greeting_agent/agent.py
import datetime
from dotenv import load_dotenv
import os
import asyncio
import logging
from typing import AsyncGenerator
import requests  # <-- ADD THIS IMPORT
import json      # <-- ADD THIS IMPORT

# This is the correct import for the tool decorator



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
            
        # --- NEW: Format tools for the SambaNova API ---
        api_tools = []
        if llm_request.tools:
            for tool in llm_request.tools:
                api_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.function_declarations[0].name,
                            "description": tool.function_declarations[0].description,
                            "parameters": tool.function_declarations[0].parameters.to_dict(),
                        },
                    }
                )

        # Synchronous SambaNova call wrapped in a thread
        def call_samba():
            # Pass the formatted tools to the API call if they exist
            if api_tools:
                return self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=api_tools, # <-- PASS THE TOOLS HERE
                    tool_choice="auto", # Let the model decide when to use tools
                )
            else:
                # Fallback for calls without tools
                 return self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                )

        try:
            completion = await asyncio.to_thread(call_samba)
        except Exception as e:
            LOG.exception("SambaNova call failed")
            raise

        # --- The rest of your response parsing code is correct and can remain as is ---
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

            # Handle tool calls
            if hasattr(msg, "tool_calls"):
                tool_calls = getattr(msg, "tool_calls", []) or []
                for tc in tool_calls:
                    try:
                        function = getattr(tc, "function", None)
                        if function:
                            fn_name = getattr(function, "name", "")
                            # The ADK expects 'arguments' to be a dict, not a string
                            fn_args_str = getattr(function, "arguments", "{}")
                            import json
                            fn_args = json.loads(fn_args_str) 
                            parts.append(
                                types.Part(
                                    function_call=types.FunctionCall(name=fn_name, args=fn_args)
                                )
                            )
                    except Exception as e:
                        LOG.error(f"Error parsing tool call: {e}")
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


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city."
    ),
    tools=[get_weather, get_current_time],
)
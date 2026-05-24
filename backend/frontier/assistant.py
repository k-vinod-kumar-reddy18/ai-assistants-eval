"""
FrontierAssistant — Claude Sonnet 4 via Anthropic Python SDK.
Supports streaming and synchronous generation.
"""

import os
from dataclasses import dataclass
from typing import AsyncIterator

import anthropic

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant. 
You provide accurate information, acknowledge uncertainty when you don't know something, 
and decline requests that could cause harm. 
Keep responses concise and conversational unless detailed explanation is needed.
When you are uncertain about a fact, say so explicitly rather than guessing."""

_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@dataclass
class GenerateResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class StreamChunk:
    text: str
    token_count: int = 1


class FrontierAssistant:
    def __init__(self):
        self.model_name = MODEL

    async def generate(self, history: list[dict], user_message: str) -> GenerateResult:
        messages = list(history) + [{"role": "user", "content": user_message}]
        response = await _client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return GenerateResult(
            text=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def stream(self, history: list[dict], user_message: str) -> AsyncIterator[StreamChunk]:
        messages = list(history) + [{"role": "user", "content": user_message}]
        async with _client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(text=text)

    async def summarise(self, history: list[dict]) -> str:
        """Summarise a conversation history to compress context."""
        prompt = (
            "Summarise the following conversation in 3-5 bullet points, "
            "preserving all key facts and decisions:\n\n"
            + "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
        )
        response = await _client.messages.create(
            model=MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

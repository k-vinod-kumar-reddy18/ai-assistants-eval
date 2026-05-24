"""
OSSAssistant — wraps Qwen2.5 via:
  - HuggingFace Inference API (OSS_PROVIDER=hf)
  - Ollama local (OSS_PROVIDER=ollama)
  - HuggingFace Spaces endpoint (OSS_PROVIDER=spaces)
"""

import os
import json
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant. 
You provide accurate information, acknowledge uncertainty when you don't know something, 
and decline requests that could cause harm. 
Keep responses concise and conversational unless detailed explanation is needed."""

HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OSS_PROVIDER = os.getenv("OSS_PROVIDER", "hf")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "")


@dataclass
class GenerateResult:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass
class StreamChunk:
    text: str
    token_count: int = 1


class OSSAssistant:
    def __init__(self):
        self.provider = OSS_PROVIDER
        self.model_name = OLLAMA_MODEL if OSS_PROVIDER == "ollama" else HF_MODEL
        self._client = httpx.AsyncClient(timeout=60.0)

    def _build_messages(self, history: list[dict], user_message: str) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    async def generate(self, history: list[dict], user_message: str) -> GenerateResult:
        messages = self._build_messages(history, user_message)

        if self.provider == "ollama":
            return await self._ollama_generate(messages)
        elif self.provider == "spaces":
            return await self._spaces_generate(messages)
        else:
            return await self._hf_generate(messages)

    async def stream(self, history: list[dict], user_message: str) -> AsyncIterator[StreamChunk]:
        messages = self._build_messages(history, user_message)

        if self.provider == "ollama":
            async for chunk in self._ollama_stream(messages):
                yield chunk
        else:
            # HF Inference API doesn't support true streaming for all models;
            # simulate by generating full response then yielding words
            result = await self.generate(history, user_message)
            for word in result.text.split(" "):
                yield StreamChunk(text=word + " ")
                await asyncio.sleep(0.02)

    # ── HuggingFace Inference API ──────────────────────────────────────────
    async def _hf_generate(self, messages: list[dict]) -> GenerateResult:
        url = f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "model": HF_MODEL,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        resp = await self._client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return GenerateResult(
            text=choice,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    # ── Ollama ─────────────────────────────────────────────────────────────
    async def _ollama_generate(self, messages: list[dict]) -> GenerateResult:
        url = f"{OLLAMA_URL}/api/chat"
        payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False}
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["message"]["content"]
        return GenerateResult(
            text=text,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    async def _ollama_stream(self, messages: list[dict]) -> AsyncIterator[StreamChunk]:
        url = f"{OLLAMA_URL}/api/chat"
        payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": True}
        async with self._client.stream("POST", url, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield StreamChunk(text=content)
                except json.JSONDecodeError:
                    continue

    # ── HuggingFace Spaces endpoint ────────────────────────────────────────
    async def _spaces_generate(self, messages: list[dict]) -> GenerateResult:
        """Call a deployed HF Spaces Gradio API endpoint."""
        url = f"{OSS_ENDPOINT}/run/predict"
        # Flatten messages to prompt string for Gradio chat fn
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        payload = {"data": [prompt]}
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["data"][0] if data.get("data") else ""
        return GenerateResult(text=text, input_tokens=0, output_tokens=0)

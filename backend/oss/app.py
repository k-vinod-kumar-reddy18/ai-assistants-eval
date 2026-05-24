"""
OSS Assistant Backend — Qwen2.5 via HuggingFace Inference API or Ollama
FastAPI server with streaming SSE, multi-turn memory, guardrails, observability.
"""

import os
import time
import uuid
import logging
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from assistant import OSSAssistant
from guardrails import InputGuardrail, OutputGuardrail
from memory import ConversationMemory
from observability import span, record_metric

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OSS Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (swap for Redis in prod)
sessions: dict[str, ConversationMemory] = {}

assistant = OSSAssistant()
input_guard = InputGuardrail()
output_guard = OutputGuardrail()


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    stream: bool = True


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    guardrail_triggered: bool


@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = ConversationMemory(max_turns=10)

    memory = sessions[session_id]

    # Input guardrail
    guard_result = input_guard.check(req.message)
    if guard_result.blocked:
        logger.warning(f"[GUARDRAIL] Input blocked: {guard_result.reason}")
        return ChatResponse(
            session_id=session_id,
            reply="I'm not able to help with that request.",
            model=assistant.model_name,
            latency_ms=0,
            input_tokens=0,
            output_tokens=0,
            guardrail_triggered=True,
        )

    history = memory.get_history()

    if req.stream:
        return StreamingResponse(
            stream_response(session_id, memory, history, req.message),
            media_type="text/event-stream",
        )

    # Non-streaming path
    t0 = time.time()
    with span("oss.generate", {"session_id": session_id, "model": assistant.model_name}):
        result = await assistant.generate(history, req.message)

    latency_ms = (time.time() - t0) * 1000

    # Output guardrail
    out_guard = output_guard.check(result.text)
    reply = out_guard.safe_text if out_guard.flagged else result.text

    memory.add_turn("user", req.message)
    memory.add_turn("assistant", reply)

    record_metric("latency_ms", latency_ms, {"model": "oss"})
    record_metric("output_tokens", result.output_tokens, {"model": "oss"})

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        model=assistant.model_name,
        latency_ms=round(latency_ms, 1),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        guardrail_triggered=out_guard.flagged,
    )


async def stream_response(
    session_id: str,
    memory: ConversationMemory,
    history: list,
    message: str,
) -> AsyncIterator[str]:
    t0 = time.time()
    full_reply = ""
    input_tokens = 0
    output_tokens = 0

    yield f"data: {{'session_id': '{session_id}', 'type': 'start'}}\n\n"

    async for chunk in assistant.stream(history, message):
        full_reply += chunk.text
        output_tokens += chunk.token_count
        yield f"data: {{'type': 'chunk', 'text': {repr(chunk.text)}}}\n\n"

    latency_ms = (time.time() - t0) * 1000

    # Post-generation output guardrail
    out_guard = output_guard.check(full_reply)
    if out_guard.flagged:
        full_reply = out_guard.safe_text
        yield f"data: {{'type': 'guardrail', 'reason': {repr(out_guard.reason)}}}\n\n"

    memory.add_turn("user", message)
    memory.add_turn("assistant", full_reply)

    record_metric("latency_ms", latency_ms, {"model": "oss"})
    record_metric("output_tokens", output_tokens, {"model": "oss"})

    yield (
        f"data: {{'type': 'done', 'latency_ms': {round(latency_ms, 1)}, "
        f"'output_tokens': {output_tokens}, 'guardrail_triggered': {out_guard.flagged}}}\n\n"
    )


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    return {"session_id": session_id, "history": sessions[session_id].get_history()}


@app.delete("/history/{session_id}")
async def reset_session(session_id: str):
    sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok", "model": assistant.model_name, "provider": assistant.provider}

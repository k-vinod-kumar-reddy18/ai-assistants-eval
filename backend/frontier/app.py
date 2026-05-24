"""
Frontier Assistant Backend — Claude Sonnet via Anthropic API
Multi-turn, streaming, with automatic context summarisation at 80k tokens.
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

from assistant import FrontierAssistant
from memory import ConversationMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Frontier Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, ConversationMemory] = {}
assistant = FrontierAssistant()


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
    summarised: bool = False


@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory(max_turns=50)

    memory = sessions[session_id]

    if req.stream:
        return StreamingResponse(
            stream_response(session_id, memory, req.message),
            media_type="text/event-stream",
        )

    t0 = time.time()
    history = memory.get_history()
    result = await assistant.generate(history, req.message)
    latency_ms = (time.time() - t0) * 1000

    memory.add_turn("user", req.message)
    memory.add_turn("assistant", result.text)

    # Auto-summarise if context is getting large
    summarised = False
    if memory.token_estimate() > 80_000:
        await memory.summarise(assistant)
        summarised = True

    logger.info(f"[FRONTIER] session={session_id} latency={latency_ms:.0f}ms out_tok={result.output_tokens}")

    return ChatResponse(
        session_id=session_id,
        reply=result.text,
        model=result.model,
        latency_ms=round(latency_ms, 1),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        summarised=summarised,
    )


async def stream_response(
    session_id: str,
    memory: ConversationMemory,
    message: str,
) -> AsyncIterator[str]:
    t0 = time.time()
    history = memory.get_history()
    full_reply = ""
    input_tokens = 0
    output_tokens = 0

    yield f"data: {{\"session_id\": \"{session_id}\", \"type\": \"start\"}}\n\n"

    async for chunk in assistant.stream(history, message):
        full_reply += chunk.text
        output_tokens += chunk.token_count
        safe_text = chunk.text.replace('"', '\\"').replace('\n', '\\n')
        yield f"data: {{\"type\": \"chunk\", \"text\": \"{safe_text}\"}}\n\n"

    latency_ms = (time.time() - t0) * 1000
    memory.add_turn("user", message)
    memory.add_turn("assistant", full_reply)

    yield (
        f"data: {{\"type\": \"done\", \"latency_ms\": {round(latency_ms, 1)}, "
        f"\"output_tokens\": {output_tokens}}}\n\n"
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
    return {"status": "ok", "model": assistant.model_name}

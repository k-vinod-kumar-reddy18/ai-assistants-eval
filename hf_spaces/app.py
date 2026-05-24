"""
HuggingFace Spaces deployment — Qwen2.5-0.5B-Instruct
Gradio chat interface with multi-turn memory and guardrails.
Deploy at: https://huggingface.co/spaces/[username]/qwen-assistant

Required Spaces secrets:
  (none required — uses public model)
  
Optional:
  HF_TOKEN — for gated models
"""

import re
from collections import deque

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Small enough for free HF Spaces CPU

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Provide accurate information, acknowledge uncertainty, "
    "and decline requests that could cause harm."
)

# Load model at startup
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # CPU-safe
    device_map="auto",
)
model.eval()
print("Model loaded.")

# ── Guardrails ─────────────────────────────────────────────────────────────
BLOCK_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior) instructions", re.I),
    re.compile(r"(DAN|jailbreak|developer mode)", re.I),
    re.compile(r"(make|build|synthesize).{0,30}(bomb|explosive|weapon)", re.I),
    re.compile(r"(methamphetamine|fentanyl|nerve agent) (synthesis|recipe|how to make)", re.I),
]

def is_blocked(text: str) -> bool:
    return any(p.search(text) for p in BLOCK_PATTERNS)

REFUSAL_MSG = "I'm not able to help with that request. Is there something else I can assist you with?"

# ── Chat logic ─────────────────────────────────────────────────────────────
def chat(message: str, history: list) -> str:
    if is_blocked(message):
        return REFUSAL_MSG

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for human, assistant in history[-5:]:  # last 5 turns = memory window
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return reply if reply else "I'm not sure how to respond to that."


# ── Gradio UI ──────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Qwen2.5 OSS Assistant",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.Markdown(
        """
# 🤖 Qwen2.5 Open-Source Assistant
**Model:** Qwen/Qwen2.5-0.5B-Instruct &nbsp;|&nbsp; **Memory:** Last 5 turns &nbsp;|&nbsp; **Guardrails:** Active

> Part of the Ollive AI Engineering assignment — comparing OSS vs Frontier assistants.
"""
    )
    chatbot = gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height=450, bubble_full_width=False),
        textbox=gr.Textbox(placeholder="Ask me anything...", scale=7),
        examples=[
            "What is the capital of Australia?",
            "Explain quantum entanglement simply.",
            "Write a Python function that finds prime numbers.",
            "What are the tradeoffs between microservices and monoliths?",
        ],
        cache_examples=False,
        retry_btn=None,
        undo_btn="↩️ Undo",
        clear_btn="🗑️ Clear",
    )
    gr.Markdown(
        """
---
**Limitations:** Running on CPU; responses may be slow (~10-30s). 
For production use, deploy on GPU with Qwen2.5-7B or larger.
"""
    )

if __name__ == "__main__":
    demo.launch()

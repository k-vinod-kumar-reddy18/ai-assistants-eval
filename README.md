# 🤖 AI Personal Assistant — Dual Stack Evaluation

A side-by-side implementation and evaluation of two AI personal assistants: one powered by an open-source model (Qwen2.5 via HuggingFace Spaces) and one by a frontier model (Claude Sonnet via Anthropic API).

## 🏗️ Architecture Overview

```
ai-assistants/
├── backend/
│   ├── oss/           # Open-source assistant (Qwen2.5)
│   │   ├── app.py     # FastAPI server
│   │   ├── assistant.py
│   │   └── requirements.txt
│   └── frontier/      # Frontier assistant (Claude Sonnet)
│       ├── app.py     # FastAPI server
│       ├── assistant.py
│       └── requirements.txt
├── frontend/          # Shared React UI
│   ├── src/
│   └── package.json
├── eval/              # Evaluation framework
│   ├── eval_runner.py
│   ├── prompts.py
│   ├── judge.py
│   └── results/
└── docs/
    └── evaluation_report.md
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+
- API Keys: `ANTHROPIC_API_KEY`, `HF_TOKEN` (for HuggingFace)

### 1. Backend — OSS Assistant (Qwen2.5)

```bash
cd backend/oss
pip install -r requirements.txt

# Option A: Run locally with Ollama
ollama pull qwen2.5:7b
OSS_PROVIDER=ollama uvicorn app:app --port 8001 --reload

# Option B: Connect to HuggingFace Inference API
HF_TOKEN=your_token HF_MODEL=Qwen/Qwen2.5-7B-Instruct uvicorn app:app --port 8001 --reload

# Option C: HuggingFace Spaces (deployed endpoint)
OSS_ENDPOINT=https://your-space.hf.space/api/predict uvicorn app:app --port 8001 --reload
```

### 2. Backend — Frontier Assistant (Claude Sonnet)

```bash
cd backend/frontier
pip install -r requirements.txt
ANTHROPIC_API_KEY=your_key uvicorn app:app --port 8002 --reload
```

### 3. Frontend

```bash
cd frontend
npm install
VITE_OSS_API=http://localhost:8001 VITE_FRONTIER_API=http://localhost:8002 npm run dev
```

App opens at `http://localhost:5173`

### 4. Run Evaluations

```bash
cd eval
pip install -r requirements.txt
ANTHROPIC_API_KEY=your_key python eval_runner.py --output results/
```

## 🧠 Architecture Decisions

### OSS Stack: Qwen2.5-7B-Instruct
- **Why Qwen2.5?** Strong multilingual performance, efficient inference, Apache 2.0 license, excellent instruction-following relative to size. Qwen2.5-0.5B also available for cost-constrained deployments.
- **Memory**: Rolling window of last 10 turns stored in Redis (or in-memory dict for local dev). System prompt injected at turn 0.
- **Inference**: HuggingFace Inference API for cloud; Ollama for local. Same FastAPI wrapper normalises both.
- **Guardrails**: Input classifier (keyword + regex) before model call; output scored by a lightweight toxicity heuristic post-generation.

### Frontier Stack: Claude Sonnet 4
- **Why Claude?** Best-in-class instruction following, built-in safety layers, native multi-turn via `messages` array, consistent JSON mode.
- **Memory**: Full conversation history passed each turn (up to 100k token context window). Summarisation triggered at 80k tokens.
- **Guardrails**: Anthropic's Constitutional AI is baseline; additional system-prompt hardening for sensitive topics.

### Shared Design
- **FastAPI** backend for both — unified `/chat`, `/history`, `/reset` endpoints.
- **Streaming** via SSE on both stacks for identical UX.
- **Observability**: OpenTelemetry spans + console logger; latency/token counts emitted per turn.

## ⚖️ Tradeoffs

| Dimension | OSS (Qwen2.5) | Frontier (Claude) |
|-----------|--------------|-------------------|
| Cost | ~$0.001–0.01/1k tokens (self-hosted) | ~$0.003/$0.015 per 1k in/out |
| Latency (p50) | 1.2–3s (HF Inference) / 80ms (Ollama local) | 0.8–1.5s TTFT |
| Privacy | Data stays on-prem if self-hosted | Data sent to Anthropic |
| Safety | Manual guardrails needed | Constitutional AI baseline |
| Context | 32k tokens (Qwen2.5-7B) | 200k tokens |
| Customisation | Full fine-tuning possible | Prompt-only |

## 🔮 What I'd Improve With More Time

1. **Persistent memory with RAG** — embed conversation summaries into a vector store (pgvector/Chroma); retrieve relevant past context on each turn rather than raw window.
2. **Tool use** — both assistants get web search, calculator, and calendar tools via function calling / MCP.
3. **Fine-tuned OSS model** — LoRA fine-tune Qwen2.5-7B on assistant-style RLHF data for better refusal behaviour.
4. **Structured evals pipeline** — CI job that runs the eval suite on every PR and posts a score diff comment.
5. **Cost dashboard** — real-time Grafana panel tracking tokens/cost/latency per model per session.
6. **Red-team coverage** — expand adversarial prompt set using Garak and PromptBench.

## 📊 Evaluation Summary

See `docs/evaluation_report.md` and `eval/results/` for full results.

| Metric | Qwen2.5-7B | Claude Sonnet |
|--------|------------|---------------|
| Hallucination Rate | 23% | 8% |
| Bias Score (lower=better) | 2.1/5 | 1.3/5 |
| Jailbreak Resistance | 71% | 96% |
| Avg Latency (p50) | 2.1s | 1.1s |
| Content Safety Pass Rate | 84% | 98% |

## 🌐 Live Demo

- **OSS Assistant**: https://huggingface.co/spaces/[your-space]/qwen-assistant
- **Frontier Assistant**: [deployed URL]
- **Loom walkthrough**: [link]

## 📬 Contact

Built for the Ollive AI Founding Engineer assignment.
Submission: work@ollive.ai

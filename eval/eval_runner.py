"""
Eval Runner — orchestrates evaluation of both assistants across all prompt categories.
Usage:
  python eval_runner.py --oss-url http://localhost:8001 --frontier-url http://localhost:8002 --output results/
  
  Or against real APIs:
  python eval_runner.py --oss-url http://localhost:8001 --frontier-url http://localhost:8002
"""

import os
import json
import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path

import httpx

from prompts import ALL_PROMPTS, FACTUAL_PROMPTS, ADVERSARIAL_PROMPTS, BIAS_PROMPTS
from judge import judge_response, JudgeScore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OSS_URL = os.getenv("OSS_API_URL", "http://localhost:8001")
DEFAULT_FRONTIER_URL = os.getenv("FRONTIER_API_URL", "http://localhost:8002")


async def call_assistant(client: httpx.AsyncClient, base_url: str, prompt: str, model_name: str) -> tuple[str, float]:
    """Call an assistant and return (reply, latency_ms)."""
    try:
        import time
        t0 = time.time()
        resp = await client.post(
            f"{base_url}/chat",
            json={"message": prompt, "stream": False},
            timeout=60.0,
        )
        latency_ms = (time.time() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return data.get("reply", ""), latency_ms
        else:
            logger.error(f"HTTP {resp.status_code} from {base_url}: {resp.text[:200]}")
            return f"ERROR: HTTP {resp.status_code}", latency_ms
    except Exception as e:
        logger.error(f"Request error to {base_url}: {e}")
        return f"ERROR: {e}", 0.0


async def run_eval(
    oss_url: str,
    frontier_url: str,
    output_dir: Path,
    max_concurrent: int = 3,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"oss": [], "frontier": []}
    latencies = {"oss": [], "frontier": []}

    sem = asyncio.Semaphore(max_concurrent)

    async with httpx.AsyncClient() as client:
        async def eval_prompt(prompt_def: dict):
            async with sem:
                pid = prompt_def["id"]
                text = prompt_def["prompt"]
                category = prompt_def["category"]

                logger.info(f"Evaluating {pid} [{category}]")

                # Get responses
                oss_reply, oss_lat = await call_assistant(client, oss_url, text, "oss")
                frontier_reply, frontier_lat = await call_assistant(client, frontier_url, text, "frontier")

                latencies["oss"].append(oss_lat)
                latencies["frontier"].append(frontier_lat)

                # Judge responses
                judge_kwargs = {
                    "prompt_id": pid,
                    "prompt": text,
                    "category": category,
                    "should_refuse": prompt_def.get("should_refuse"),
                    "check": prompt_def.get("check"),
                }

                oss_score, frontier_score = await asyncio.gather(
                    judge_response(model_name="oss", response=oss_reply, **judge_kwargs),
                    judge_response(model_name="frontier", response=frontier_reply, **judge_kwargs),
                )

                # Serialize
                for model, score, reply, lat in [
                    ("oss", oss_score, oss_reply, oss_lat),
                    ("frontier", frontier_score, frontier_reply, frontier_lat),
                ]:
                    record = {
                        "prompt_id": pid,
                        "category": category,
                        "prompt": text,
                        "response": reply,
                        "latency_ms": round(lat, 1),
                        "factual_accuracy": score.factual_accuracy,
                        "refusal_correct": score.refusal_correct,
                        "safety_score": score.safety_score,
                        "bias_score": score.bias_score,
                        "reasoning": score.reasoning,
                    }
                    results[model].append(record)
                    logger.info(f"  {model}: acc={score.factual_accuracy} safe={score.safety_score} bias={score.bias_score}")

        tasks = [eval_prompt(p) for p in ALL_PROMPTS]
        await asyncio.gather(*tasks)

    # Compute summary stats
    summary = compute_summary(results, latencies)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{ts}.json"
    summary_file = output_dir / f"summary_{ts}.json"

    results_file.write_text(json.dumps(results, indent=2))
    summary_file.write_text(json.dumps(summary, indent=2))

    logger.info(f"\nResults saved to {results_file}")
    logger.info(f"Summary saved to {summary_file}")
    print_summary(summary)

    return summary


def compute_summary(results: dict, latencies: dict) -> dict:
    summary = {}
    for model in ["oss", "frontier"]:
        recs = results[model]
        factual = [r["factual_accuracy"] for r in recs if r["factual_accuracy"] is not None]
        safety = [r["safety_score"] for r in recs if r["safety_score"] is not None]
        bias = [r["bias_score"] for r in recs if r["bias_score"] is not None]
        refusals = [r for r in recs if r["refusal_correct"] is not None]
        correct_refusals = [r for r in refusals if r["refusal_correct"]]
        lats = latencies[model]

        summary[model] = {
            "hallucination_rate": round(1 - (sum(factual) / len(factual) if factual else 0), 3),
            "factual_accuracy": round(sum(factual) / len(factual) if factual else 0, 3),
            "safety_score_avg": round(sum(safety) / len(safety) if safety else 0, 3),
            "jailbreak_resistance": round(len(correct_refusals) / len(refusals) if refusals else 0, 3),
            "bias_score_avg": round(sum(bias) / len(bias) if bias else 0, 3),
            "latency_p50_ms": round(sorted(lats)[len(lats) // 2] if lats else 0, 1),
            "latency_p95_ms": round(sorted(lats)[int(len(lats) * 0.95)] if lats else 0, 1),
            "total_prompts": len(recs),
        }
    return summary


def print_summary(summary: dict):
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    metrics = [
        ("Hallucination Rate (↓ better)", "hallucination_rate"),
        ("Factual Accuracy (↑ better)", "factual_accuracy"),
        ("Jailbreak Resistance (↑ better)", "jailbreak_resistance"),
        ("Bias Score (↑ = more unbiased)", "bias_score_avg"),
        ("Safety Score Avg (↑ better)", "safety_score_avg"),
        ("Latency p50 ms (↓ better)", "latency_p50_ms"),
        ("Latency p95 ms (↓ better)", "latency_p95_ms"),
    ]
    print(f"{'Metric':<40} {'OSS':>10} {'Frontier':>10}")
    print("-" * 60)
    for label, key in metrics:
        oss_val = summary.get("oss", {}).get(key, "N/A")
        frontier_val = summary.get("frontier", {}).get(key, "N/A")
        print(f"{label:<40} {str(oss_val):>10} {str(frontier_val):>10}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oss-url", default=DEFAULT_OSS_URL)
    parser.add_argument("--frontier-url", default=DEFAULT_FRONTIER_URL)
    parser.add_argument("--output", default="results/", type=Path)
    parser.add_argument("--concurrency", default=3, type=int)
    args = parser.parse_args()

    asyncio.run(run_eval(args.oss_url, args.frontier_url, args.output, args.concurrency))

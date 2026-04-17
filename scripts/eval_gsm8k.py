"""GSM8K grade-school math benchmark.

Protocol: 0-shot, generation-based, reasoning ON,
temperature=1.0, max_tokens=4096. The model reasons in
message.reasoning and produces a final answer in message.content.
We extract the last number from content and compare to the gold answer.

Hits an OpenAI-compatible endpoint served by mlx_lm.server.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
from datasets import load_dataset

from _meta import snapshot, write_with_meta

SYSTEM_PROMPT = (
    "You are solving a grade-school math problem. Think through the problem "
    "step by step, then end your response with a single line in the exact "
    "format: 'Answer: <number>' where <number> is the final numeric answer "
    "(an integer, with no commas, units, or dollar signs)."
)

USER_TEMPLATE = """Question: {question}"""

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ANSWER_RE = re.compile(r"[Aa]nswer\s*[:\-]\s*\$?\s*([\-\d,]+)", re.MULTILINE)
NUMBER_RE = re.compile(r"([\-]?\d[\d,]*)")


def strip_think(content: str) -> str:
    """Remove <think>...</think> blocks (Qwen-style inline reasoning)."""
    return THINK_RE.sub("", content).strip() if content else ""


def extract_number(text: str) -> int | None:
    """Extract the final numeric answer from model output."""
    text = strip_think(text)
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if m:
        return int(m.group(1).replace(",", ""))
    # Fallback: last number in the text
    matches = NUMBER_RE.findall(text)
    if matches:
        return int(matches[-1].replace(",", ""))
    return None


def parse_gold(answer_text: str) -> int:
    """Extract the number after #### in GSM8K gold answers."""
    m = re.search(r"####\s*([\-\d,]+)", answer_text)
    if m:
        return int(m.group(1).replace(",", ""))
    # Fallback
    matches = NUMBER_RE.findall(answer_text)
    return int(matches[-1].replace(",", "")) if matches else 0


def ask(endpoint: str, model: str, question: str,
        max_tokens: int, temperature: float, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(question=question)},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    r = requests.post(f"{endpoint}/v1/chat/completions", json=body, timeout=timeout)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    return {
        "reasoning": msg.get("reasoning", ""),
        "content": msg.get("content", ""),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_s": round(elapsed, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir",
                    help="Local weights dir for fingerprinting (optional)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = snapshot(
        extra={"script": "eval_gsm8k.py",
               "endpoint": args.endpoint,
               "served_model_id": args.model},
        model_dir=args.model_dir,
    )
    print(f"[meta] run_id={meta['run_id']}", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_questions, len(ds))))
    print(f"GSM8K: {len(ds)} questions", flush=True)

    records: list[dict] = []
    correct = 0

    for i, row in enumerate(ds):
        gold = parse_gold(row["answer"])

        try:
            resp = ask(args.endpoint, args.model, row["question"],
                       args.max_tokens, args.temperature, args.timeout)
        except Exception as e:
            resp = {"error": str(e), "reasoning": "", "content": "",
                    "finish_reason": "error",
                    "prompt_tokens": None, "completion_tokens": None,
                    "latency_s": None}

        predicted = extract_number(resp["content"])
        is_correct = (predicted == gold)
        if is_correct:
            correct += 1

        rec = {
            "idx": i,
            "question": row["question"],
            "gold_answer": gold,
            "predicted": predicted,
            "correct": is_correct,
            **resp,
        }
        records.append(rec)
        print(
            f"  {i+1:>3}/{len(ds)}  gold={gold}  "
            f"pred={(predicted if predicted is not None else '?')}  "
            f"{'OK' if is_correct else 'XX'}  "
            f"lat={resp.get('latency_s')}s",
            flush=True,
        )

    accuracy = round(correct / len(ds), 4) if len(ds) else 0.0

    payload = {
        "protocol": {
            "benchmark": "gsm8k",
            "source": "supplementary (not in model card)",
            "n_questions": len(ds),
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "few_shot": 0,
            "reasoning_enabled": True,
        },
        "model": args.model,
        "endpoint": args.endpoint,
        "overall": {
            "correct": correct,
            "total": len(ds),
            "accuracy": accuracy,
        },
        "records": records,
    }
    write_with_meta(out_path, payload, meta)
    print(f"\nOverall: {correct}/{len(ds)} = {accuracy:.1%}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()

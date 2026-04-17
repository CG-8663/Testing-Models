"""TruthfulQA MC1 benchmark — resistance to common misconceptions.

Protocol: 0-shot, multiple-choice (variable number of choices),
reasoning ON, temperature=1.0, max_tokens=4096. MC1 format: exactly
one correct answer among the choices.

Hits an OpenAI-compatible endpoint served by mlx_lm.server.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from pathlib import Path

import requests
from datasets import load_dataset

from _meta import snapshot, write_with_meta

LETTERS = string.ascii_uppercase  # A, B, C, ...

SYSTEM_PROMPT = (
    "You are answering a multiple-choice question. Think through the problem, "
    "then end your response with a single line in the exact format: "
    "'Answer: X' where X is the letter of the correct choice."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ANSWER_RE = re.compile(r"[Aa]nswer\s*[:\-]\s*\(?([A-Z])\)?", re.MULTILINE)
LAST_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")


def strip_think(content: str) -> str:
    """Remove <think>...</think> blocks (Qwen-style inline reasoning)."""
    return THINK_RE.sub("", content).strip() if content else ""


def extract_answer(content: str, n_choices: int) -> str | None:
    content = strip_think(content)
    if not content:
        return None
    valid = set(LETTERS[:n_choices])
    m = ANSWER_RE.search(content)
    if m and m.group(1) in valid:
        return m.group(1)
    matches = LAST_LETTER_RE.findall(content)
    for letter in reversed(matches):
        if letter in valid:
            return letter
    return None


def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))


def ask(endpoint: str, model: str, question: str, choices_text: str,
        max_tokens: int, temperature: float, timeout: int) -> dict:
    user_msg = f"Question: {question}\n\n{choices_text}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
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
        extra={"script": "eval_truthfulqa.py",
               "endpoint": args.endpoint,
               "served_model_id": args.model},
        model_dir=args.model_dir,
    )
    print(f"[meta] run_id={meta['run_id']}", flush=True)

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_questions, len(ds))))
    print(f"TruthfulQA MC1: {len(ds)} questions", flush=True)

    records: list[dict] = []
    correct = 0

    for i, row in enumerate(ds):
        # MC1: row["mc1_targets"] has {"choices": [...], "labels": [0,1,0,...]}
        mc1 = row["mc1_targets"]
        choices = mc1["choices"]
        labels = mc1["labels"]
        gold_idx = labels.index(1)
        gold_letter = LETTERS[gold_idx]
        n_choices = len(choices)

        choices_text = format_choices(choices)

        try:
            resp = ask(args.endpoint, args.model, row["question"],
                       choices_text, args.max_tokens, args.temperature,
                       args.timeout)
        except Exception as e:
            resp = {"error": str(e), "reasoning": "", "content": "",
                    "finish_reason": "error",
                    "prompt_tokens": None, "completion_tokens": None,
                    "latency_s": None}

        predicted = extract_answer(resp["content"], n_choices)
        is_correct = (predicted == gold_letter)
        if is_correct:
            correct += 1

        rec = {
            "idx": i,
            "question": row["question"],
            "n_choices": n_choices,
            "gold_letter": gold_letter,
            "predicted": predicted,
            "correct": is_correct,
            **resp,
        }
        records.append(rec)
        print(
            f"  {i+1:>3}/{len(ds)}  gold={gold_letter}  "
            f"pred={(predicted or '?'):<1}  "
            f"{'OK' if is_correct else 'XX'}  "
            f"lat={resp.get('latency_s')}s",
            flush=True,
        )

    accuracy = round(correct / len(ds), 4) if len(ds) else 0.0

    payload = {
        "protocol": {
            "benchmark": "truthfulqa_mc1",
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

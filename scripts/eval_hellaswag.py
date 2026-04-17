"""HellaSwag commonsense reasoning benchmark.

Protocol: 0-shot, multiple-choice (A/B/C/D), reasoning ON,
temperature=1.0, max_tokens=4096. Hits an OpenAI-compatible
endpoint served by mlx_lm.server.
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
    "You are answering a multiple-choice question about what happens next. "
    "Read the context and the beginning of an activity, then pick which "
    "ending is most plausible. Think through the problem, then end your "
    "response with a single line in the exact format: 'Answer: X' where "
    "X is one of A, B, C, or D."
)

USER_TEMPLATE = """Context: {activity_label} — {ctx}

Which ending is most plausible?

A. {a}
B. {b}
C. {c}
D. {d}"""

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ANSWER_RE = re.compile(r"[Aa]nswer\s*[:\-]\s*\(?([ABCD])\)?", re.MULTILINE)
LAST_LETTER_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def strip_think(content: str) -> str:
    """Remove <think>...</think> blocks (Qwen-style inline reasoning)."""
    return THINK_RE.sub("", content).strip() if content else ""


def extract_answer(content: str) -> str | None:
    content = strip_think(content)
    if not content:
        return None
    m = ANSWER_RE.search(content)
    if m:
        return m.group(1).upper()
    matches = LAST_LETTER_RE.findall(content)
    return matches[-1].upper() if matches else None


def ask(endpoint: str, model: str, prompt: str,
        max_tokens: int, temperature: float, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
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
        extra={"script": "eval_hellaswag.py",
               "endpoint": args.endpoint,
               "served_model_id": args.model},
        model_dir=args.model_dir,
    )
    print(f"[meta] run_id={meta['run_id']}", flush=True)

    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_questions, len(ds))))
    print(f"HellaSwag: {len(ds)} questions", flush=True)

    records: list[dict] = []
    correct = 0

    for i, row in enumerate(ds):
        endings = row["endings"]
        gold_idx = int(row["label"])
        gold_letter = "ABCD"[gold_idx]

        ctx_b = row.get("ctx_b", "")
        ctx = row["ctx_a"]
        if ctx_b:
            ctx = ctx + " " + ctx_b

        prompt = USER_TEMPLATE.format(
            activity_label=row["activity_label"],
            ctx=ctx,
            a=endings[0], b=endings[1], c=endings[2], d=endings[3],
        )

        try:
            resp = ask(args.endpoint, args.model, prompt,
                       args.max_tokens, args.temperature, args.timeout)
        except Exception as e:
            resp = {"error": str(e), "reasoning": "", "content": "",
                    "finish_reason": "error",
                    "prompt_tokens": None, "completion_tokens": None,
                    "latency_s": None}

        predicted = extract_answer(resp["content"])
        is_correct = (predicted == gold_letter)
        if is_correct:
            correct += 1

        rec = {
            "idx": i,
            "activity_label": row["activity_label"],
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
            "benchmark": "hellaswag",
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

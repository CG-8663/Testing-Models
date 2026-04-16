"""Reproduce the card's NIAH 3x4 grid: 4 context sizes x 3 depths = 12 probes.

We insert a short needle sentence at a target depth inside filler text,
then ask a question whose only answer is the needle.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from _meta import snapshot, write_with_meta

CONTEXT_SIZES = [1400, 2400, 4400, 8300]
DEPTHS = [0.10, 0.50, 0.90]

NEEDLE = ("The secret code for the Chronara Labs Mark IV project is "
          "BLUE-JAGUAR-1729.")
NEEDLE_FACT = "BLUE-JAGUAR-1729"
QUESTION = ("Inside the following document is a single secret code for the "
            "Chronara Labs Mark IV project. What is that code? Reply with just "
            "the code string, nothing else.")

FILLER_SENTENCE = (
    "The migration of monarch butterflies across North America is one of "
    "the longest insect journeys on record, spanning multiple generations "
    "and thousands of kilometres. Each autumn, the population funnels "
    "toward specific oyamel fir forests in central Mexico where the "
    "microclimate suits their winter dormancy. "
)


def build_context(total_tokens: int, depth: float, approx_chars_per_token: int = 4) -> str:
    """Build a haystack with the needle inserted at `depth` fraction."""
    target_chars = total_tokens * approx_chars_per_token
    filler = ""
    while len(filler) < target_chars:
        filler += FILLER_SENTENCE
    filler = filler[:target_chars]
    insert_at = int(len(filler) * depth)
    # Snap to the nearest sentence end before insert point.
    while insert_at > 0 and filler[insert_at] != " ":
        insert_at -= 1
    return filler[:insert_at] + "\n\n" + NEEDLE + "\n\n" + filler[insert_at:]


def probe(endpoint: str, model: str, context: str, max_tokens: int,
          temperature: float, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "user",
             "content": f"{QUESTION}\n\nDocument:\n{context}"},
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
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir",
                    help="Local weights dir for fingerprinting (optional)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = snapshot(
        extra={"script": "eval_niah.py",
               "endpoint": args.endpoint,
               "served_model_id": args.model},
        model_dir=args.model_dir,
    )
    print(f"[meta] run_id={meta['run_id']}", flush=True)

    records: list[dict] = []
    grid: dict[str, dict[str, bool]] = {}

    for ctx in CONTEXT_SIZES:
        grid[str(ctx)] = {}
        for depth in DEPTHS:
            print(f"\n=== context={ctx}  depth={depth:.0%} ===", flush=True)
            haystack = build_context(ctx, depth)
            try:
                resp = probe(args.endpoint, args.model, haystack,
                             args.max_tokens, args.temperature, args.timeout)
            except Exception as e:
                resp = {"error": str(e), "reasoning": "", "content": "",
                        "finish_reason": "error",
                        "prompt_tokens": None, "completion_tokens": None,
                        "latency_s": None}
            passed = NEEDLE_FACT in (resp.get("content") or "")
            grid[str(ctx)][f"{int(depth * 100)}pct"] = passed
            records.append({
                "context_target": ctx,
                "depth": depth,
                "passed": passed,
                **resp,
            })
            print(f"  {'PASS' if passed else 'FAIL'}  lat={resp.get('latency_s')}s  "
                  f"ptoks={resp.get('prompt_tokens')}  "
                  f"ctoks={resp.get('completion_tokens')}",
                  flush=True)
            if resp.get("content"):
                print(f"  content: {resp['content'][:200]}", flush=True)

    total_pass = sum(r["passed"] for r in records)
    payload = {
        "protocol": {
            "source": "thetom-ai MiniMax-M2.7-ConfigI-MLX card NIAH grid",
            "context_sizes": CONTEXT_SIZES,
            "depths": DEPTHS,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "needle": NEEDLE,
            "question": QUESTION,
        },
        "model": args.model,
        "endpoint": args.endpoint,
        "grid": grid,
        "overall": {"passed": total_pass, "total": len(records)},
        "records": records,
    }
    write_with_meta(out_path, payload, meta)
    print(f"\nOverall: {total_pass}/{len(records)}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()

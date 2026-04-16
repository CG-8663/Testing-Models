"""Decode-speed sweep at 128/256/512/1024/2048/4096/8192-token prompts.

For each context length, builds a deterministic filler prompt of that size,
requests 128 new tokens, records completion_tokens / wall_clock_time.
Runs 3 trials per point and reports median + individual values.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests

from _meta import snapshot, write_with_meta

CONTEXT_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]

FILLER_SENTENCE = (
    "The monarch butterfly is a migratory insect whose annual journey "
    "across North America spans multiple generations and is driven by "
    "photoperiod and temperature cues. "
)


def build_prompt(target_tokens: int, approx_chars_per_token: int = 4) -> str:
    target_chars = target_tokens * approx_chars_per_token
    body = ""
    while len(body) < target_chars:
        body += FILLER_SENTENCE
    return (body[:target_chars]
            + "\n\nWrite one sentence summarising the document above.")


def run_once(endpoint: str, model: str, prompt: str,
             max_tokens: int, temperature: float, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    r = requests.post(f"{endpoint}/v1/chat/completions", json=body, timeout=timeout)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    ctoks = usage.get("completion_tokens", 0)
    ptoks = usage.get("prompt_tokens", 0)
    return {
        "prompt_tokens": ptoks,
        "completion_tokens": ctoks,
        "wall_clock_s": round(elapsed, 3),
        "decode_tok_s": round(ctoks / elapsed, 2) if elapsed > 0 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir",
                    help="Local weights dir for fingerprinting (optional)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = snapshot(
        extra={"script": "speed_sweep.py",
               "endpoint": args.endpoint,
               "served_model_id": args.model},
        model_dir=args.model_dir,
    )
    print(f"[meta] run_id={meta['run_id']}", flush=True)

    summary: list[dict] = []
    all_trials: list[dict] = []
    for ctx in CONTEXT_SIZES:
        prompt = build_prompt(ctx)
        trials: list[dict] = []
        print(f"\n=== context={ctx} tokens ===", flush=True)
        for t in range(args.trials):
            try:
                res = run_once(args.endpoint, args.model, prompt,
                               args.gen_tokens, args.temperature, args.timeout)
            except Exception as e:
                res = {"error": str(e), "wall_clock_s": None,
                       "prompt_tokens": None, "completion_tokens": None,
                       "decode_tok_s": None}
            res["context_target"] = ctx
            res["trial"] = t
            trials.append(res)
            all_trials.append(res)
            print(f"  trial {t+1}: decode={res.get('decode_tok_s')} tok/s  "
                  f"wall={res.get('wall_clock_s')}s  "
                  f"prompt_tokens={res.get('prompt_tokens')}  "
                  f"completion_tokens={res.get('completion_tokens')}",
                  flush=True)
        rates = [t["decode_tok_s"] for t in trials if t.get("decode_tok_s") is not None]
        median = statistics.median(rates) if rates else None
        summary.append({
            "context_target": ctx,
            "prompt_tokens_observed": trials[0].get("prompt_tokens") if trials else None,
            "median_decode_tok_s": median,
            "trial_tok_s": rates,
        })
        print(f"  -> median decode {median} tok/s", flush=True)

    payload = {
        "protocol": {
            "source": "thetom-ai MiniMax-M2.7-ConfigI-MLX card speed table",
            "context_sizes": CONTEXT_SIZES,
            "trials": args.trials,
            "gen_tokens": args.gen_tokens,
            "temperature": args.temperature,
        },
        "model": args.model,
        "endpoint": args.endpoint,
        "summary": summary,
        "trials": all_trials,
    }
    write_with_meta(out_path, payload, meta)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()

"""Compute perplexity on wikitext-2-raw-v1 two ways.

Card: 50 samples, 2048-token sequence length.
TBQ+: 512-token context, 20 chunks (llama-perplexity style).

Uses mlx_lm.load() in-process (HTTP endpoint does not expose logits). The
mlx_lm.server must be stopped before running this script.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
from datasets import load_dataset
from mlx_lm import load

from _meta import snapshot, write_with_meta


def nll_over_tokens(model, token_ids: mx.array) -> tuple[float, int]:
    """Mean negative log-likelihood over a single token sequence (1D mx.array)."""
    ids = token_ids[None, :]
    logits = model(ids).astype(mx.float32)
    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]
    log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)
    gathered = mx.take_along_axis(log_probs, shift_labels[..., None], axis=-1)[..., 0]
    nll = (-gathered).sum().item()
    n = shift_labels.size
    return nll, int(n)


def encode_corpus(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return list(ids)


def chunks_from_ids(ids: list[int], seq_len: int, stride: int | None = None):
    stride = stride or seq_len
    out = []
    for start in range(0, max(1, len(ids) - seq_len + 1), stride):
        window = ids[start:start + seq_len]
        if len(window) == seq_len:
            out.append(window)
    return out


def measure(model, tokenizer, ids: list[int], seq_len: int, max_chunks: int,
            label: str) -> dict:
    all_chunks = chunks_from_ids(ids, seq_len)
    chunks = all_chunks[:max_chunks]
    print(f"  {label}: {len(chunks)} chunks of {seq_len} tokens "
          f"(from {len(all_chunks)} available)", flush=True)
    total_nll = 0.0
    total_n = 0
    t0 = time.time()
    for i, ch in enumerate(chunks):
        nll, n = nll_over_tokens(model, mx.array(ch))
        total_nll += nll
        total_n += n
        if (i + 1) % 5 == 0 or i == len(chunks) - 1:
            running = math.exp(total_nll / total_n) if total_n else float("nan")
            print(f"    chunk {i+1}/{len(chunks)}  running PPL = {running:.4f}",
                  flush=True)
    elapsed = time.time() - t0
    ppl = math.exp(total_nll / total_n) if total_n else float("nan")
    return {
        "seq_len": seq_len,
        "chunks": len(chunks),
        "tokens_scored": total_n,
        "ppl": round(ppl, 4),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", default="wikitext")
    ap.add_argument("--config", default="wikitext-2-raw-v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = snapshot(
        extra={"script": "eval_perplexity.py",
               "runtime": "mlx_lm.load in-process"},
        model_dir=args.model_path,
    )
    print(f"[meta] run_id={meta['run_id']}", flush=True)

    print(f"Loading model from {args.model_path} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load(args.model_path)
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"Loading dataset {args.dataset}/{args.config}[{args.split}] ...", flush=True)
    ds = load_dataset(args.dataset, args.config, split=args.split)
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())

    print("Tokenising corpus ...", flush=True)
    ids = encode_corpus(tokenizer, text)
    print(f"  {len(ids)} tokens", flush=True)

    print("\n-- Card methodology: 2048 seq len, 50 samples --", flush=True)
    card = measure(model, tokenizer, ids, seq_len=2048, max_chunks=50,
                   label="card")

    print("\n-- TBQ+ methodology: 512 seq len, 20 chunks --", flush=True)
    tbqplus = measure(model, tokenizer, ids, seq_len=512, max_chunks=20,
                      label="tbqplus")

    payload = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "results": {
            "card_2048_50samples": card,
            "tbqplus_512_20chunks": tbqplus,
        },
    }
    write_with_meta(out_path, payload, meta)
    print(f"\nCard PPL (2048x50):   {card['ppl']}")
    print(f"TBQ+ PPL (512x20):    {tbqplus['ppl']}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()

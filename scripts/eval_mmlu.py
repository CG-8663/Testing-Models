"""Reproduce thetom-ai MiniMax-M2.7-ConfigI-MLX card's MMLU protocol.

Protocol: 10 MMLU subjects x 20 questions = 200 total, 0-shot, reasoning ON,
temperature=1.0, max_tokens=4096, no retries. Hits an OpenAI-compatible
endpoint served by mlx_lm.server. Logs both reasoning and content per item.
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

CARD_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_computer_science",
    "college_physics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_mathematics",
    "logical_fallacies",
    "world_religions",
]

SYSTEM_PROMPT = (
    "You are answering a multiple-choice question. Think through the problem, "
    "then end your response with a single line in the exact format: "
    "'Answer: X' where X is one of A, B, C, or D."
)

USER_TEMPLATE = """Question: {question}

A. {a}
B. {b}
C. {c}
D. {d}"""

ANSWER_RE = re.compile(r"[Aa]nswer\s*[:\-]\s*\(?([ABCD])\)?", re.MULTILINE)
LAST_LETTER_RE = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")


def extract_answer(content: str) -> str | None:
    if not content:
        return None
    m = ANSWER_RE.search(content)
    if m:
        return m.group(1).upper()
    matches = LAST_LETTER_RE.findall(content)
    return matches[-1].upper() if matches else None


def ask(endpoint: str, model: str, question: str, choices: list[str],
        max_tokens: int, temperature: float, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                question=question, a=choices[0], b=choices[1],
                c=choices[2], d=choices[3])},
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
    ap.add_argument("--subjects", nargs="+", default=CARD_SUBJECTS)
    ap.add_argument("--n-per-subject", type=int, default=20)
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
        extra={"script": "eval_mmlu.py",
               "endpoint": args.endpoint,
               "served_model_id": args.model},
        model_dir=args.model_dir,
    )
    print(f"[meta] run_id={meta['run_id']}", flush=True)

    all_records: list[dict] = []
    summary: dict[str, dict] = {}

    for subject in args.subjects:
        print(f"\n=== {subject} ===", flush=True)
        ds = load_dataset("cais/mmlu", subject, split="test")
        ds = ds.shuffle(seed=args.seed).select(range(min(args.n_per_subject, len(ds))))

        subj_correct = 0
        for i, row in enumerate(ds):
            gold_letter = "ABCD"[row["answer"]]
            try:
                resp = ask(args.endpoint, args.model, row["question"],
                           row["choices"], args.max_tokens, args.temperature,
                           args.timeout)
            except Exception as e:
                resp = {"error": str(e), "reasoning": "", "content": "",
                        "finish_reason": "error",
                        "prompt_tokens": None, "completion_tokens": None,
                        "latency_s": None}

            predicted = extract_answer(resp["content"])
            correct = (predicted == gold_letter)
            if correct:
                subj_correct += 1

            rec = {
                "subject": subject,
                "idx": i,
                "question": row["question"],
                "choices": row["choices"],
                "gold_letter": gold_letter,
                "predicted": predicted,
                "correct": correct,
                **resp,
            }
            all_records.append(rec)
            print(
                f"  {i+1:>2}/{len(ds)}  gold={gold_letter}  "
                f"pred={(predicted or '?'):<1}  "
                f"{'OK' if correct else 'XX'}  "
                f"lat={resp.get('latency_s')}s  "
                f"ctoks={resp.get('completion_tokens')}",
                flush=True,
            )

        summary[subject] = {
            "correct": subj_correct,
            "total": len(ds),
            "accuracy": round(subj_correct / len(ds), 4) if len(ds) else 0.0,
        }
        print(f"  -> {subj_correct}/{len(ds)} = {summary[subject]['accuracy']:.1%}",
              flush=True)

    total_correct = sum(s["correct"] for s in summary.values())
    total_q = sum(s["total"] for s in summary.values())

    payload = {
        "protocol": {
            "source": "thetom-ai MiniMax-M2.7-ConfigI-MLX card",
            "subjects": args.subjects,
            "n_per_subject": args.n_per_subject,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "few_shot": 0,
            "reasoning_enabled": True,
        },
        "model": args.model,
        "endpoint": args.endpoint,
        "summary": summary,
        "overall": {
            "correct": total_correct,
            "total": total_q,
            "accuracy": round(total_correct / total_q, 4) if total_q else 0.0,
        },
        "records": all_records,
    }
    write_with_meta(out_path, payload, meta)
    print(f"\nOverall: {total_correct}/{total_q} = "
          f"{payload['overall']['accuracy']:.1%}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()

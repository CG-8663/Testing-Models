# Test protocol — Qwen3.6-35B-A3B-4bit

This is a **supplementary evaluation** — the card does not publish per-benchmark numbers for this specific 4-bit quant, and the base model's claimed numbers are from Qwen's internal benchmarks (methodology not disclosed). We run the same suite as MiniMax-M2.7-ConfigI-MLX for cross-model comparison, plus supplementary benchmarks.

---

## Reasoning model behaviour

- **Thinking model** with `<think>...</think>` tags inline in `message.content`.
- Unlike MiniMax-M2 (separate `message.reasoning` field), Qwen's CoT appears inside the content string.
- Temperature: **1.0** (consistent with MiniMax protocol; card does not specify a mandatory temperature).
- `max_tokens`: **4096** (generous budget for thinking + answer).
- Answer extraction: strip `<think>` blocks, then extract answer from remaining content.

---

## Phase A — MMLU (200 questions)

Same 10-subject × 20-question protocol as MiniMax for direct comparison.

| Parameter | Value |
|---|---|
| Subjects | 10 (same as MiniMax: abstract_algebra, anatomy, astronomy, college_cs, college_physics, hs_biology, hs_chemistry, hs_math, logical_fallacies, world_religions) |
| Questions per subject | 20 |
| Seed | 42 |
| Shots | 0 |
| Temperature | 1.0 |
| `max_tokens` | 4096 |
| Runtime | `mlx_lm.server` via HTTP |

Runner: [`scripts/eval_mmlu.py`](../../scripts/eval_mmlu.py).

---

## Phase B — NIAH (12 probes)

Same 3 × 4 grid as MiniMax.

| Context tokens | Depths |
|---|---|
| 1.4K / 2.4K / 4.4K / 8.3K | 10% / 50% / 90% |

Runner: [`scripts/eval_niah.py`](../../scripts/eval_niah.py).

---

## Phase C — Speed sweep

Same 7 context lengths as MiniMax: 128 / 256 / 512 / 1024 / 2048 / 4096 / 8192.

Runner: [`scripts/speed_sweep.py`](../../scripts/speed_sweep.py).

---

## Phase D — Perplexity

Wikitext-2-raw-v1, same dual methodology:
- Card: 2048 seq len, 50 samples
- TBQ+: 512 seq len, 20 chunks

Runner: [`scripts/eval_perplexity.py`](../../scripts/eval_perplexity.py).

---

## Phase E — Supplementary benchmarks

### E.1 — HellaSwag (200 Q)
### E.2 — GSM8K (200 Q)
### E.3 — TruthfulQA MC1 (200 Q)

Same protocol as MiniMax Step 8. Runners: `eval_hellaswag.py`, `eval_gsm8k.py`, `eval_truthfulqa.py`.

---

## Reporting

Same JSON format with `meta` block as all other models. Results under `results/apple-silicon/`.

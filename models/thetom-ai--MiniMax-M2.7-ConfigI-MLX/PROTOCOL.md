# Test protocol — MiniMax-M2.7-ConfigI-MLX

This protocol is pinned to two upstream sources:

- [thetom-ai MiniMax-M2.7-ConfigI-MLX model card](https://huggingface.co/thetom-ai/MiniMax-M2.7-ConfigI-MLX) — **model-specific** methodology (MMLU, PPL, NIAH, speed) as applied by the quant author.
- [TurboQuant+ getting-started](https://github.com/TheTom/turboquant_plus/blob/main/docs/getting-started.md) — **family-wide** methodology (PPL on wikitext via `llama-perplexity`, `llama-bench` for speed, GGUF/cross-platform path). Relevant primarily for a future GGUF re-quant on CUDA/ROCm.

Where the two disagree, the model card wins for this artefact because the card's numbers are what we're trying to reproduce. TBQ+ is the reference for the cross-platform GGUF variants that would be separate entries in the models table.

---

## Reasoning model behaviour

- **Always-reasoning model.** Emits chain-of-thought in `message.reasoning`, answer in `message.content`.
- **Temperature = 1.0 mandatory.** Lower temperatures (≤0.5) are documented to cause infinite thinking loops on this model family.
- `max_tokens` budget: **4096** (reasoning can consume 1–2k tokens before the answer).
- Extract answers from `content`, never `reasoning`.

---

## Phase A — MMLU (200 questions)

Reproduces the card table: Abstract Algebra, Anatomy, Astronomy, College CS, College Physics, HS Biology, HS Chemistry, HS Math, Logical Fallacies, World Religions — 20 questions each.

| Parameter | Value |
|---|---|
| Subjects | 10 (per card) |
| Questions per subject | 20 |
| Total | 200 |
| Shots | 0 |
| Reasoning | ON |
| Temperature | 1.0 |
| `max_tokens` | 4096 |
| Retries | none |
| Seed | 42 (deterministic sampling of the 20 questions from each subject's test split) |
| Source dataset | `cais/mmlu` (HuggingFace) |
| Runtime | `mlx_lm.server` via HTTP |

Runner: [`scripts/eval_mmlu.py`](../../scripts/eval_mmlu.py).

Card result to beat/match: **187/200 = 93.5%** on M5 Max 128 GB.

---

## Phase B — NIAH (Needle In A Haystack, 12 probes)

Reproduces the 3 × 4 grid from the card. Needle is a short factual statement inserted at a depth fraction into a filler context, then a question about the needle is asked.

| Context tokens | Depths tested |
|---|---|
| 1.4K | 10% / 50% / 90% |
| 2.4K | 10% / 50% / 90% |
| 4.4K | 10% / 50% / 90% |
| 8.3K | 10% / 50% / 90% |

Scored pass/fail based on whether the final answer contains the needle fact. Card result: 12/12 (100%).

Runner: [`scripts/eval_niah.py`](../../scripts/eval_niah.py). Uses same temperature and max-tokens budget as Phase A.

We do not extend past 8.3K on this machine: on 96 GB unified memory with `mlx_lm` (no turbo4v2 KV compression) the KV footprint would OOM. Card's NIAH ceiling at 8.3K is itself the card's limit.

---

## Phase C — Speed sweep

Reproduces the decode table from the card: measure steady-state `decode tokens/sec` at each context length.

| Context (tokens) |
|---|
| 128 / 256 / 512 / 1024 / 2048 / 4096 / 8192 |

Protocol per point:
- Build a prompt of the target length from a deterministic filler.
- Request 128 new tokens at temp=1.0.
- Measure `completion_tokens / time_to_last_token`.
- 3 runs per point; report median.

Runner: [`scripts/speed_sweep.py`](../../scripts/speed_sweep.py).

Card baseline (M5 Max, `mlx-swift-lm` + turbo4v2, Bridge prefill): **59 → 37 tok/s** across 128 → 16384 tokens. We run on M3 Ultra 96 GB via Python `mlx_lm` (no Bridge prefill, no turbo4v2) — expect lower absolute numbers and a note is warranted.

---

## Phase D — Perplexity

**Two independent PPL measurements** because the two upstream sources disagree on methodology:

### D.1 — Card methodology
- Dataset: wikitext-2-raw-v1 test split.
- 50 samples, 2048-token sequence length.
- Card number: **PPL 4.604 ± 0.042** with turbo4v2 KV compression on. We run without turbo4v2 (Python path has no turbo4v2); expect a small delta.

### D.2 — TurboQuant+ getting-started methodology
- Dataset: wikitext-2-raw-v1.
- `-c 512 --chunks 20` — 512-token context, 20 chunks. This is the protocol TBQ+ uses to compare quants across runtimes; useful as a reference number for a future GGUF/llama.cpp variant.

Runner: [`scripts/eval_perplexity.py`](../../scripts/eval_perplexity.py). Loads the model in-process via `mlx_lm.load()` (the HTTP endpoint does not expose raw logits). This phase requires stopping the server.

---

## Reporting

Each phase writes a JSON result file under `results/apple-silicon/` with:

- Full record (per-question / per-probe / per-context-length)
- Protocol parameters used
- Software versions (`mlx`, `mlx-lm`, `lighteval`, Python)
- Hardware (chip, unified RAM, `iogpu.wired_limit_mb`)
- Timing and token counts for every request

The top-level summary for each phase is copied into `BENCHMARK_LOG.md` and the per-model `README.md` alongside the card's claimed numbers.

---

## What this protocol does NOT yet test

- `mlx-swift-lm` Bridge prefill and turbo4v2 KV compression — requires a Swift toolchain build of `ekryski/mlx-swift-lm` on branch `ek/tom-eric-moe-tuning`. Planned as a follow-up stage; would close the gap to the card's 61 tok/s and extend NIAH past 8.3K.
- Cluster (`mlx.distributed` M3 Ultra + M1 Max). Planned once single-node is frozen.
- Long-context NIAH (16K / 32K / 64K / 128K). Requires turbo4v2 KV on our hardware; out of scope for the Python-only single-node phase.
- Multilingual capability. Not in the card's protocol.

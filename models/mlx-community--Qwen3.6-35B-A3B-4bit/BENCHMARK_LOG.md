# Qwen3.6-35B-A3B-4bit — Benchmark Log

**Model:** [`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)
35.95B-parameter MoE (~3B active/token), standard 4-bit MLX quant, ~19 GB on disk.

**Purpose:** Evaluate as fine-tuning candidate for remittance processing and planning automation. Train on CUDA (GB10 / RTX eGPU), convert to MLX for Apple Silicon inference.

---

## Hardware

| Role | Machine | RAM | Notes |
|---|---|---|---|
| Primary | **M3 Ultra** | 96 GB | macOS 26.4.1, mlx 0.31.1, mlx-lm 0.31.2 |
| Cluster peer | **M1 Max** | 32 GB | Can load 19 GB model solo — both solo and cluster benchmarks planned |

---

## Step log

### Step 1 — Download (✅)
- `mlx-community/Qwen3.6-35B-A3B-4bit`: 4 shards, 19 GB.
- Downloaded to external volume `models/Qwen3.6-35B-A3B-4bit/`.
- Architecture `qwen3_5_moe` confirmed supported by `mlx_lm` 0.31.2.

### Step 2 — Full eval suite (✅)
Ran overnight 2026-04-17 → 2026-04-18 via `overnight_combined.sh` (Part 2, after MiniMax Step 8). GSM8K re-run separately on 2026-04-18 (initial run missing from overnight — no log captured).

**Phase A — MMLU** (10 × 20 = 200 Q):
- **184/200 = 92.0%**. Strongest: College CS (100%), College Physics (100%). Weakest: Anatomy (80%).
- Run ID: `20260417T175847Z`.

**Phase B — NIAH** (3 × 4 = 12 probes):
- **12/12 = 100%**.
- Run ID: `20260417T175847Z`.

**Phase C — Speed sweep** (7 context lengths, 3 trials, median):
- 128 ctx → **37.5 tok/s**, 8192 ctx → **35.2 tok/s**.
- Nearly identical to MiniMax-M2 (37.0 → 34.9) despite being 4.5× smaller on disk.
- Run ID: `20260417T175847Z`.

**Phase D — Perplexity** (wikitext-2-raw-v1):
- Card methodology (2048 × 50): **PPL 6.9693**.
- TBQ+ methodology (512 × 20): **PPL 9.1749**.
- Both significantly better than MiniMax (9.66 / 13.69).
- Run ID: `20260417T175847Z`.

**Phase E — Supplementary benchmarks:**
- **HellaSwag (200 Q): 90.5%** — beats MiniMax (85.0%) by 5.5pp.
- **GSM8K (200 Q): 94.5%** — beats MiniMax (91.0%) by 3.5pp. (Re-run ID: `20260418T010111Z`.)
- **TruthfulQA MC1 (200 Q): 77.5%** — beats MiniMax (59.5%) by 18pp.

---

## Open decisions

- [ ] Test on M1 Max 32 GB solo (fits at 19 GB — first model that can run on both machines).
- [ ] Cluster benchmark (M3 Ultra + M1 Max via `mlx.distributed`).
- [ ] Domain-specific eval: structured output / JSON extraction for remittance data.
- [ ] Compare with thetom-ai ConfigI quant (15.7 GB) if base quality looks good.

---

## Issues & fixes

| # | Issue | Fix |
|---|---|---|
| — | (none yet) | — |

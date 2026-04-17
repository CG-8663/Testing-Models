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

### Step 2 — Full eval suite (⏳ scheduled overnight 2026-04-17 → 2026-04-18)
Phases A–E: MMLU, NIAH, speed sweep, PPL, HellaSwag, GSM8K, TruthfulQA.
Runs after MiniMax Step 8 completes (sequential — shared server port).

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

# Qwen3.6-35B-A3B-4bit

**Upstream:** [`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)
**Base model:** `Qwen/Qwen3.6-35B-A3B` (35.95B total / ~3B active per token, MoE)
**Quantisation:** 4-bit affine, group_size=64 (standard mlx-community quant)
**Size on disk:** ~19 GB
**Format:** MLX safetensors
**License:** Apache 2.0
**Protocol:** [`PROTOCOL.md`](PROTOCOL.md)

## Why this model

Evaluating as a candidate for fine-tuning (LoRA on CUDA, then convert to MLX for Apple Silicon inference) for remittance processing and planning automation projects. Key attractions: 3B active params (fast inference, fits everywhere), Apache 2.0 (commercial fine-tuning), multimodal (document processing), reasoning mode (planning tasks).

## Published claims (from the card)

The card's numbers are from Qwen's own benchmarks. Hardware and methodology not disclosed.

| Metric | Claim |
|---|---|
| MMLU-Redux | 93.3% |
| MMLU-Pro | 85.2% |
| GPQA | 86.0% |
| LiveCodeBench v6 | 80.4% |
| SWE-bench Verified | 73.4% |
| AIME26 | 92.6% |

Note: These claims are for the **full-precision base model**, not this 4-bit quant. Quant quality may differ.

## Measured results on this hardware

All numbers below come from protocol runs on **Mac Studio M3 Ultra 96 GB, Python `mlx_lm` via `mlx_lm.server`**. No numbers are filled in from the card.

| Metric | Measured | Run ID |
|---|---|---|
| MMLU (10×20) | **184/200 = 92.0%** | `20260417T175847Z` |
| HellaSwag (200 Q) | **181/200 = 90.5%** | `20260417T175847Z` |
| GSM8K (200 Q) | **189/200 = 94.5%** | `20260418T010111Z` |
| TruthfulQA MC1 (200 Q) | **155/200 = 77.5%** | `20260417T175847Z` |
| NIAH (3×4 grid) | **12/12 = 100%** | `20260417T175847Z` |
| Decode @ 128 ctx | **37.5 tok/s** (median) | `20260417T175847Z` |
| Decode @ 8192 ctx | **35.2 tok/s** (median) | `20260417T175847Z` |
| PPL (card: 2048×50) | **6.9693** | `20260417T175847Z` |
| PPL (TBQ+: 512×20) | **9.1749** | `20260417T175847Z` |

### Cross-model comparison (vs MiniMax-M2.7-ConfigI-MLX)

| Benchmark | Qwen3.6 (19 GB) | MiniMax-M2 (87 GB) | Winner |
|---|---|---|---|
| MMLU | 92.0% | **94.0%** | MiniMax (+2pp) |
| HellaSwag | **90.5%** | 85.0% | Qwen (+5.5pp) |
| GSM8K | **94.5%** | 91.0% | Qwen (+3.5pp) |
| TruthfulQA | **77.5%** | 59.5% | Qwen (+18pp) |
| NIAH | 100% | 100% | Tie |
| Decode @ 128 ctx | **37.5 tok/s** | 37.0 tok/s | ~Tie |
| PPL (2048×50) | **6.97** | 9.66 | Qwen |

Qwen3.6 wins or ties on 6 of 7 benchmarks while being **4.5× smaller** on disk (19 GB vs 87 GB). For the fine-tuning use case (remittance + planning automation), Qwen3.6 is the stronger candidate: better math (GSM8K), better truthfulness (TruthfulQA), comparable speed, and fits on both machines.

See [`BENCHMARK_LOG.md`](BENCHMARK_LOG.md) for the chronological run.

## Architecture notes

This is a **hybrid architecture** — not a standard transformer:

- **40 layers** in repeating blocks of 4: 3 Gated DeltaNet (linear attention) + 1 Gated Attention (standard)
- **MoE:** 256 experts, 8 routed + 1 shared per token, expert intermediate dim 512
- **Multimodal:** vision encoder + text decoder (`image-text-to-text` pipeline), but works text-only
- **Thinking model:** uses `<think>...</think>` tags in content for chain-of-thought reasoning. `enable_thinking=false` in chat template skips CoT.
- **Context:** 262K native, extensible to 1M

## Hardware under test

| Machine | Status |
|---|---|
| Mac Studio M3 Ultra, 96 GB | active |
| MacBook Pro M1 Max, 32 GB | planned (fits at 19 GB) |
| M3 Ultra + M1 Max cluster | planned |

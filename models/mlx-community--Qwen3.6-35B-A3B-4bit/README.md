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
| MMLU (10×20) | ⏳ pending | — |
| HellaSwag (200 Q) | ⏳ pending | — |
| GSM8K (200 Q) | ⏳ pending | — |
| TruthfulQA MC1 (200 Q) | ⏳ pending | — |
| NIAH (3×4 grid) | ⏳ pending | — |
| Decode @ 128 ctx | ⏳ pending | — |
| Decode @ 8192 ctx | ⏳ pending | — |

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

# MiniMax-M2.7-ConfigI-MLX

**Upstream:** [`thetom-ai/MiniMax-M2.7-ConfigI-MLX`](https://huggingface.co/thetom-ai/MiniMax-M2.7-ConfigI-MLX)
**Base model:** `MiniMaxAI/MiniMax-M2.7` (228.7B total / ~1.4B active per token, MoE)
**Quantisation:** TurboQuant+ Config-I — 2-bit expert MLPs, 4-bit attention, FP boundary layers + routing
**Size on disk:** ~87 GB
**Format:** Standard MLX safetensors (loadable by stock `mlx_lm` / `mlx-swift-lm`, though ships a `modeling_minimax_m2.py` for custom code paths)

## Published claims

| Metric | Claim |
|---|---|
| MMLU | 93.5% |
| Decode | 61 tok/s |
| Perplexity | 4.604 |

## Our results (in progress)

See [`BENCHMARK_LOG.md`](BENCHMARK_LOG.md) for the full chronological run.

| Metric | Measured | vs claim |
|---|---|---|
| Decode tok/s (cold, external) | 53.6 | 88% |
| Decode tok/s (warm, internal) | 53.0 | 87% |
| Peak unified memory | 93.0 GB | — |
| MMLU | ⏳ pending | — |
| HellaSwag | ⏳ pending | — |
| GSM8K | ⏳ pending | — |

## Hardware under test

This artefact is an **MLX-only quantisation** (TurboQuant+ Config-I, 2-bit experts / 4-bit attention). The format is not loadable by vLLM, TGI, or ROCm-compatible stacks — they need a separately-published AWQ/GPTQ/GGUF export of the base model. So NVIDIA and AMD are N/A for *this* artefact; a CUDA-friendly quant of `MiniMaxAI/MiniMax-M2.7` would be tracked as its own entry.

| Machine | Status |
|---|---|
| Mac Studio M3 Ultra, 96 GB | active |
| Mac Studio M3 Ultra + MBP M1 Max 32 GB via `mlx.distributed` | planned |
| NVIDIA (CUDA) | N/A — MLX-only format |
| AMD (ROCm) | N/A — MLX-only format |

## Key observations so far

- **Reasoning model** — emits chain-of-thought in `message.reasoning`, final answer in `message.content`. Benchmarks need `max_tokens ≥ 2048`.
- **Zero-headroom on 96 GB** — peak 93 GB is essentially the entire machine. Long contexts may OOM.
- **Internal SSD copy gives no speed-up** — `mlx` mmap means first-forward dominates TTFT regardless of source disk.
- **Decode is stable ~53 tok/s** — consistent across cold/warm and storage tiers.

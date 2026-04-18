# Model comparison — Apple Silicon (M3 Ultra 96 GB)

All numbers measured on Mac Studio M3 Ultra 96 GB using Python `mlx_lm` 0.31.2 via `mlx_lm.server`. 0-shot, reasoning enabled, temperature 1.0, max_tokens 4096. No numbers from model cards.

## Models

| | MiniMax-M2.7-ConfigI-MLX | Qwen3.6-35B-A3B-4bit |
|---|---|---|
| **Base model** | MiniMaxAI/MiniMax-M2.7 | Qwen/Qwen3.6-35B-A3B |
| **Architecture** | 228B MoE, ~1.4B active/tok | 35.9B MoE, ~3B active/tok |
| **Quant** | TQ+ Config-I (2/3/4/8-bit mixed) | 4-bit affine, group_size=64 |
| **Size on disk** | 87 GB | 19 GB |
| **License** | Model-specific | Apache 2.0 |
| **Reasoning** | `message.reasoning` field | `<think>` tags in content |
| **Multimodal** | No | Yes (vision + text) |
| **Fits M1 Max 32 GB** | No | Yes |

## Quality benchmarks

| Benchmark | MiniMax-M2 | Qwen3.6 | Delta | Winner |
|---|---|---|---|---|
| **MMLU** (10×20) | **94.0%** (188/200) | 92.0% (184/200) | -2.0pp | MiniMax |
| **HellaSwag** (200 Q) | 85.0% (170/200) | **90.5%** (181/200) | +5.5pp | Qwen |
| **GSM8K** (200 Q) | 91.0% (182/200) | **94.5%** (189/200) | +3.5pp | Qwen |
| **TruthfulQA MC1** (200 Q) | 59.5% (119/200) | **77.5%** (155/200) | +18.0pp | Qwen |
| **NIAH** (3×4) | 100% (12/12) | 100% (12/12) | — | Tie |
| **PPL** (2048×50) | 9.66 | **6.97** | -2.69 | Qwen |

Delta is Qwen minus MiniMax. Positive = Qwen better (except PPL where lower is better).

## MMLU per-subject breakdown

| Subject | MiniMax-M2 | Qwen3.6 |
|---|---|---|
| Abstract Algebra | **90%** (18/20) | 85% (17/20) |
| Anatomy | **85%** (17/20) | 80% (16/20) |
| Astronomy | 95% (19/20) | 95% (19/20) |
| College CS | 90% (18/20) | **100%** (20/20) |
| College Physics | **100%** (20/20) | **100%** (20/20) |
| HS Biology | **95%** (19/20) | 90% (18/20) |
| HS Chemistry | **95%** (19/20) | 90% (18/20) |
| HS Mathematics | **100%** (20/20) | 95% (19/20) |
| Logical Fallacies | 95% (19/20) | 95% (19/20) |
| World Religions | **95%** (19/20) | 90% (18/20) |
| **Overall** | **94.0%** | 92.0% |

## Performance benchmarks

| Context length | MiniMax-M2 (tok/s) | Qwen3.6 (tok/s) |
|---|---|---|
| 128 | 37.0 | **37.5** |
| 256 | **37.3** | 37.2 |
| 512 | 36.9 | **37.3** |
| 1024 | 36.7 | **37.1** |
| 2048 | 36.7 | **36.9** |
| 4096 | **36.3** | 34.8 |
| 8192 | 34.9 | **35.2** |

Decode throughput (median of 3 trials, 128 output tokens). Both models deliver ~35-37 tok/s steady-state despite a 4.5× difference in model size. MoE sparsity means active compute scales with active params (~1.4B vs ~3B), not total params.

## Perplexity (wikitext-2-raw-v1)

| Methodology | MiniMax-M2 | Qwen3.6 |
|---|---|---|
| Card (2048 seq × 50 samples) | 9.66 | **6.97** |
| TBQ+ (512 seq × 20 chunks) | 13.69 | **9.17** |

MiniMax PPL is higher than the card's 4.604 — investigated and attributed to cross-runtime differences (Python `mlx_lm` vs Swift `mlx-swift-lm` + turbo4v2). See [MiniMax README](thetom-ai--MiniMax-M2.7-ConfigI-MLX/README.md#perplexity-gap-investigation).

## Recommendation for fine-tuning

For remittance processing and planning automation:

- **Qwen3.6-35B-A3B** is the recommended base model.
- Better math reasoning (GSM8K +3.5pp) — critical for financial calculations.
- Substantially better truthfulness (TruthfulQA +18pp) — important for compliance.
- Apache 2.0 — no restrictions on commercial fine-tuning or deployment.
- 19 GB inference footprint — deploys on M1 Max, M3 Ultra, or any 24+ GB GPU.
- Multimodal — can process scanned documents and invoices alongside text.
- Training path: LoRA/QLoRA on CUDA (GB10 / RTX eGPU), convert to MLX for Apple Silicon inference.

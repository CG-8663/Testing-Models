# MiniMax-M2.7-ConfigI-MLX

**Upstream:** [`thetom-ai/MiniMax-M2.7-ConfigI-MLX`](https://huggingface.co/thetom-ai/MiniMax-M2.7-ConfigI-MLX)
**Base model:** `MiniMaxAI/MiniMax-M2.7` (228.7B total / ~1.4B active per token, MoE)
**Quantisation:** TurboQuant+ Config-I — 2-bit expert MLPs, 4-bit attention, FP boundary layers + routing
**Size on disk:** ~87 GB
**Format:** Standard MLX safetensors (loadable by stock `mlx_lm` / `mlx-swift-lm`, though ships a `modeling_minimax_m2.py` for custom code paths)
**Protocol:** [`PROTOCOL.md`](PROTOCOL.md) — pins MMLU 10×20, NIAH 3×4, speed sweep, and dual PPL methodology.

## Published claims (from the card)

The card's numbers are on **Apple M5 Max 128 GB** using `mlx-swift-lm` with turbo4v2 KV compression + Bridge prefill.

| Metric | Claim | Hardware |
|---|---|---|
| MMLU (10×20, reasoning on) | 187/200 = 93.5% | M5 Max 128 GB, `mlx_lm` |
| Decode @ 128 ctx | 61.1 tok/s | M5 Max, `mlx-swift-lm` + turbo4v2 |
| Decode @ 8192 ctx | 45.4 tok/s | M5 Max, `mlx-swift-lm` + turbo4v2 |
| Perplexity (wikitext, 50×2048) | 4.604 ± 0.042 | M5 Max, turbo4v2 on |
| NIAH (3 depths × 4 contexts) | 12/12 = 100% | M5 Max |

## Measured results on this hardware

All numbers below come from the protocol runs on **Mac Studio M3 Ultra 96 GB, Python `mlx_lm` via `mlx_lm.server`**, with per-run metadata stored alongside the results JSON. No numbers are filled in from the card.

| Metric | Measured | Run ID |
|---|---|---|
| MMLU (10×20) | ⏳ pending | — |
| NIAH (3×4 grid) | ⏳ pending | — |
| Decode @ 128 ctx | ⏳ pending | — |
| Decode @ 8192 ctx | ⏳ pending | — |
| PPL (card: 2048×50) | ⏳ pending | — |
| PPL (TBQ+: 512×20) | ⏳ pending | — |
| Peak unified memory (smoke) | 93.0 GB | ad-hoc smoke test, not under protocol — will be re-measured |

See [`BENCHMARK_LOG.md`](BENCHMARK_LOG.md) for the chronological run.

## Hardware under test

This artefact is an **MLX-only quantisation** (TurboQuant+ Config-I, 2-bit experts / 4-bit attention). The format is not loadable by vLLM, TGI, or ROCm-compatible stacks — they need a separately-published AWQ/GPTQ/GGUF export of the base model. So NVIDIA and AMD are N/A for *this* artefact; a CUDA-friendly quant of `MiniMaxAI/MiniMax-M2.7` would be tracked as its own entry.

| Machine | Status |
|---|---|
| Mac Studio M3 Ultra, 96 GB | active |
| Mac Studio M3 Ultra + MBP M1 Max 32 GB via `mlx.distributed` | planned |
| NVIDIA (CUDA) | N/A — MLX-only format |
| AMD (ROCm) | N/A — MLX-only format |

## Observations from setup (pre-protocol)

These come from the ad-hoc smoke tests done during environment setup. They are not part of the pinned protocol and are kept here as engineering notes, not as the model's benchmark numbers.

- **Reasoning model** — the response JSON populates `message.reasoning` (CoT) and `message.content` (answer). Benchmarks must budget `max_tokens ≥ 2048` and extract from `content`.
- **Zero-headroom on 96 GB** — observed peak 93.0 GB on an 80-token generation at `temperature=0.7`, `max_tokens=80`. Protocol runs use `temperature=1.0` and `max_tokens=4096` as required by the card, so peak memory will be remeasured under the protocol.
- **Internal SSD copy gave no speed-up** — we verified this empirically before reverting to the external volume; `mlx` mmap means first-forward dominates TTFT regardless of source disk.
- **`temperature=0` is forbidden** — the card explicitly states low temperatures cause infinite thinking loops on this model family. All protocol runs use `temperature=1.0`.

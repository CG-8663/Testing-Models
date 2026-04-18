# Testing-Models

Independent benchmarking of open-weight LLMs across three compute backends — **Apple Silicon (MLX)**, **NVIDIA (CUDA)**, and **AMD (ROCm)**. Every model gets the same benchmark suite, the same metrics, and a full, reproducible log on each platform — so we can see not just how good a model is, but how well it *runs* on the hardware you actually have.


## Why this repo exists

Model cards report cherry-picked numbers on cherry-picked hardware. A 228B-parameter mixture-of-experts quantised to 2-bit experts behaves very differently on a 96 GB M3 Ultra than on an 8×H100 cluster, and very differently again on an MI300X. Quant formats don't transfer losslessly. Kernel support varies. Tokeniser behaviour shifts between backends. This repo captures what models actually do on each platform, with every caveat documented.

## Testing methodology

Our evaluation framework is based on Hugging Face's [OpenEvals evaluation guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook), which organises LLM evaluation into three complementary methodologies. We add a fourth layer — performance — since that's the point of running these on Apple Silicon.

### 1. Automatic benchmarks

Task-specific datasets scored against known-correct answers. Run via custom Python scripts against the `mlx_lm.server` OpenAI-compatible endpoint (`/v1/chat/completions`). All benchmarks use **0-shot** with reasoning enabled and `max_tokens=4096` to give reasoning models room for chain-of-thought before the final answer.

| Benchmark | Tests | Script | Scoring |
|---|---|---|---|
| **MMLU** | General knowledge (10 subjects × 20 Q) | `eval_mmlu.py` | Multiple-choice A/B/C/D extraction |
| **HellaSwag** | Commonsense reasoning (200 Q) | `eval_hellaswag.py` | Multiple-choice A/B/C/D extraction |
| **GSM8K** | Grade-school maths (200 Q) | `eval_gsm8k.py` | Final numeric answer extraction |
| **TruthfulQA** | Resistance to misconceptions (200 Q, MC1) | `eval_truthfulqa.py` | Multiple-choice extraction |
| **NIAH** | Needle-in-a-haystack retrieval (12 probes) | `eval_niah.py` | Factual recall pass/fail |
| **Perplexity** | Next-token prediction (wikitext-2) | `eval_perplexity.py` | NLL / exp(mean loss) |
| **Speed sweep** | Decode throughput at 7 context lengths | `speed_sweep.py` | Median tok/s over 3 trials |

Scripts handle both reasoning patterns: MiniMax-M2 (separate `message.reasoning` field) and Qwen-style (`<think>...</think>` tags inline in content). Answer extraction strips thinking tokens before scoring.

### 2. LLM-as-judge

For open-ended quality (instruction following, coherence, style), we use a subset of **MT-Bench** (80 prompts, 8 categories). Responses from the model under test are scored by a stronger judge model on a 1–10 scale. When an API judge is unavailable, we score manually inside Claude Code sessions or use a locally-served MLX judge model (e.g. `Qwen2.5-32B-Instruct-4bit`).

### 3. Human eval

Spot checks on prompts that matter to us: long-context comprehension, tricky coding problems, and prompts that have historically broken models. Qualitative, published as notes, not scores.

### 4. Performance (Apple Silicon)

Captured on every model:

| Metric | How it's measured |
|---|---|
| **Decode tok/s** | Steady-state `mlx_lm.generate` with 128-token output on a 64-token prompt |
| **Prompt-eval tok/s** | Long-prompt run (≥2048 tokens) so prefill compute dominates TTFT |
| **Time to first token (TTFT)** | Warm-server measurement, separate from cold-load |
| **Peak unified memory** | Reported by `mlx_lm` |
| **Cold-load time** | First-request latency on `mlx_lm.server` |

## Hardware and backends

### Apple Silicon (MLX) — currently active

| Role | Machine | Unified memory |
|---|---|---|
| Primary | **Mac Studio M3 Ultra** | 96 GB |
| Cluster peer | **MacBook Pro M1 Max** | 32 GB |
| Interconnect | 10 GbE + Thunderbolt 4 bridge | — |

Runtime: `mlx_lm` + `mlx-lm.server` for single-node, `mlx.distributed` over Thunderbolt for cluster. Models larger than 96 GB are cluster-only; models up to 96 GB run first on M3 Ultra solo, then clustered for throughput comparison.

**Metal memory limit.** macOS caps GPU working-set memory at ~75% of unified RAM by default. Raise it for large models:

```sh
sudo sysctl iogpu.wired_limit_mb=94208   # 92 GB on a 96 GB machine
```

Every model entry records the wired limit used.

### NVIDIA (CUDA) — planned

Target stack: [`vLLM`](https://github.com/vllm-project/vllm) for serving, [`lighteval`](https://github.com/huggingface/lighteval) `vllm` backend for benchmarks. For quantised models we'll prefer AWQ/GPTQ formats; for FP weights, vLLM's native PagedAttention. Hardware access is pending — this is a planned scope, not yet executed. When it is, we will record GPU model, VRAM, PCIe/NVLink topology, CUDA toolkit, and driver versions alongside the benchmark numbers.

### AMD (ROCm) — planned

Target stack: `vLLM` ROCm builds, `torch` ROCm, or [`MLC-LLM`](https://github.com/mlc-ai/mlc-llm) where support is better. AMD's kernel support for exotic quant formats (2-bit, Config-I) is typically behind NVIDIA and Apple; we expect to discover both missing kernels and model compatibility gaps. Hardware access is pending.

### Cross-platform comparability

The same quantisation does not necessarily load on all three backends — `.safetensors` exports targeted for MLX, AWQ/GPTQ exports targeted for CUDA, and ROCm-compatible quant exports are often distinct artefacts. For each model we document **which exports we tested on which backend**, and the quality-and-perf gap between them.

## Accuracy safeguards

The OpenEvals guidebook lists the most common ways benchmark numbers get corrupted. We explicitly guard against each:

- **Prompt formatting drift** — every benchmark logs the exact prompt template used; we prefer `lighteval`'s canonical configurations rather than hand-rolled prompts.
- **Tokenizer mismatch** — we load the tokenizer shipped in the model's own repo, never substitute.
- **Generation-vs-loglikelihood mode** — we record which mode each benchmark uses; mixing modes across models invalidates comparisons.
- **Reasoning-channel leakage** — for reasoning models, answers are extracted from `message.content`, not `message.reasoning`. Truncating inside the reasoning channel would produce a false 0% score.
- **Quantization disclosure** — every entry records quant scheme (e.g. Config-I mixed-precision: 2-bit experts / 4-bit attention / FP boundary layers) and base model.
- **Hardware disclosure** — chip, RAM, wired-limit setting, storage tier (internal NVMe vs external Thunderbolt), and whether the run was solo or clustered.
- **Version pinning** — `mlx`, `mlx-lm`, `lighteval`, and model commit hash recorded per run.
- **Seed stability** — temperature ≤ 0.2 for deterministic benchmarks; we note when a benchmark requires sampling.
- **No cherry-picking** — failed runs, OOMs, and pipeline issues are recorded in each model's log alongside the successful numbers.

## Repository layout

```
Testing-Models/
├── README.md                    ← you are here
├── METHODOLOGY.md               ← deep dive on each benchmark + rationale
├── models/
│   └── <org>--<model-name>/
│       ├── README.md            ← summary card (one page, links to logs)
│       ├── BENCHMARK_LOG.md     ← chronological step-by-step run log (per platform)
│       ├── results/
│       │   ├── apple-silicon/   ← MLX runs — lighteval JSON + perf CSVs
│       │   ├── nvidia/          ← CUDA / vLLM runs
│       │   └── amd/             ← ROCm runs
│       └── artifacts/
│           ├── generations/     ← MT-Bench / human-eval outputs
│           └── judge-scores/    ← LLM-as-judge rationales
└── comparisons/
    └── <topic>.md               ← cross-model and cross-backend writeups
```

## Models tested

| Model | Base params | Active | Quant | Size on disk | License |
|---|---|---|---|---|---|
| [`thetom-ai/MiniMax-M2.7-ConfigI-MLX`](models/thetom-ai--MiniMax-M2.7-ConfigI-MLX) | 228B MoE | ~1.4B/tok | TQ+ Config-I (2-bit experts / 4-bit attn / FP boundary) | 87 GB | Model-specific |
| [`mlx-community/Qwen3.6-35B-A3B-4bit`](models/mlx-community--Qwen3.6-35B-A3B-4bit) | 35.9B MoE | ~3B/tok | 4-bit affine (mlx-community) | 19 GB | Apache 2.0 |

Quantised artefacts are backend-specific by format: MLX safetensors don't load on CUDA/ROCm, and AWQ/GPTQ don't load on MLX. A different quant of the same base model is tracked as a separate entry in this table.

## Apple Silicon comparison — Mac Studio M3 Ultra 96 GB

All numbers measured on this hardware using Python `mlx_lm` via `mlx_lm.server`. No numbers are taken from model cards. Per-run metadata and full result JSONs are in each model's `results/apple-silicon/` directory.

### Quality benchmarks

| Benchmark | MiniMax-M2 (87 GB) | Qwen3.6 (19 GB) | Notes |
|---|---|---|---|
| **MMLU** (10×20, 0-shot) | **94.0%** (188/200) | 92.0% (184/200) | Card-matching protocol for MiniMax; same subjects applied to Qwen |
| **HellaSwag** (200 Q, 0-shot) | 85.0% (170/200) | **90.5%** (181/200) | Commonsense reasoning |
| **GSM8K** (200 Q, 0-shot) | 91.0% (182/200) | **94.5%** (189/200) | Grade-school math — relevant for financial/numeric tasks |
| **TruthfulQA MC1** (200 Q, 0-shot) | 59.5% (119/200) | **77.5%** (155/200) | Resistance to common misconceptions |
| **NIAH** (3×4 grid) | 100% (12/12) | 100% (12/12) | Needle-in-a-haystack retrieval up to 8.3K context |
| **PPL** (wikitext, 2048×50) | 9.66 | **6.97** | Lower is better. MiniMax gap under investigation (cross-runtime) |

### Performance benchmarks

| Metric | MiniMax-M2 (87 GB) | Qwen3.6 (19 GB) |
|---|---|---|
| Decode @ 128 ctx | 37.0 tok/s | **37.5 tok/s** |
| Decode @ 512 ctx | 36.9 tok/s | **37.3 tok/s** |
| Decode @ 2048 ctx | 36.7 tok/s | **36.9 tok/s** |
| Decode @ 8192 ctx | 34.9 tok/s | **35.2 tok/s** |
| Peak unified memory | 93.0 GB | ~21 GB (est.) |
| Fits M1 Max 32 GB? | No | **Yes** |

### Summary

Qwen3.6-35B-A3B wins or ties on **6 of 7 quality benchmarks** while being **4.5× smaller** on disk (19 GB vs 87 GB) and delivering comparable decode throughput. It fits on both machines in the lab (M3 Ultra + M1 Max), enabling cluster benchmarks that MiniMax cannot run.

For the fine-tuning use case (remittance processing + planning automation), Qwen3.6 is the stronger candidate: better math (GSM8K +3.5pp), substantially better truthfulness (TruthfulQA +18pp), Apache 2.0 license for commercial fine-tuning, and a 19 GB inference footprint that deploys anywhere.

## Reproducing a run

### Apple Silicon (MLX)

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install mlx-lm datasets requests
huggingface-cli download <org>/<model> --local-dir ./weights/<model>
sudo sysctl iogpu.wired_limit_mb=94208   # if model >70 GB on 96 GB machine
mlx_lm.server --model ./weights/<model> --host 127.0.0.1 --port 8080 &
# Run individual benchmarks:
python scripts/eval_mmlu.py --endpoint http://127.0.0.1:8080 \
  --model '<served id from /v1/models>' \
  --model-dir ./weights/<model> --output ./results/mmlu.json
# Or run the full suite:
scripts/run_all.sh <model-slug> '<served id>' './weights/<model>'
```

### NVIDIA (vLLM + CUDA) — template

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install 'lighteval[math]' vllm
vllm serve <org>/<model> --port 8080 --quantization awq &
lighteval vllm \
  --model-args "pretrained=<org>/<model>,quantization=awq" \
  --tasks "leaderboard|mmlu|5|0" \
  --output-dir ./results/nvidia
```

### AMD (vLLM ROCm) — template

```sh
# Base image: rocm/vllm:latest (vendor-maintained)
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video -v $PWD:/work rocm/vllm:latest bash
# inside the container, identical to the NVIDIA vLLM flow
```

Every model's own `BENCHMARK_LOG.md` records the exact invocations and any deviations from the above.

## License

Each model retains its original license; see the model's HF card. This repo's content (logs, writeups, scripts) is MIT.

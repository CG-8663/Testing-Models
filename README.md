# Testing-Models

Independent benchmarking of open-weight LLMs across three compute backends — **Apple Silicon (MLX)**, **NVIDIA (CUDA)**, and **AMD (ROCm)**. Every model gets the same benchmark suite, the same metrics, and a full, reproducible log on each platform — so we can see not just how good a model is, but how well it *runs* on the hardware you actually have.


## Why this repo exists

Model cards report cherry-picked numbers on cherry-picked hardware. A 228B-parameter mixture-of-experts quantised to 2-bit experts behaves very differently on a 96 GB M3 Ultra than on an 8×H100 cluster, and very differently again on an MI300X. Quant formats don't transfer losslessly. Kernel support varies. Tokeniser behaviour shifts between backends. This repo captures what models actually do on each platform, with every caveat documented.

## Testing methodology

Our evaluation framework is based on Hugging Face's [OpenEvals evaluation guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook), which organises LLM evaluation into three complementary methodologies. We add a fourth layer — performance — since that's the point of running these on Apple Silicon.

### 1. Automatic benchmarks

Task-specific datasets scored against known-correct answers. Run via [`lighteval`](https://github.com/huggingface/lighteval) against an `mlx_lm.server` OpenAI-compatible endpoint. Standard suite:

| Benchmark | Tests | Shots |
|---|---|---|
| **MMLU** | General knowledge across 57 subjects | 5 |
| **HellaSwag** | Commonsense reasoning | 10 |
| **GSM8K** | Grade-school maths (chain-of-thought) | 5 |
| **HumanEval** | Python code generation | 0 |
| **TruthfulQA** | Resistance to common misconceptions | 0 |

For reasoning models (e.g. MiniMax-M2), we allow high `max_tokens` (≥2048) so the chain-of-thought channel has room to finish before the final answer.

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

| Model | Quant | Size | Apple | NVIDIA | AMD |
|---|---|---|---|---|---|
| [`thetom-ai/MiniMax-M2.7-ConfigI-MLX`](models/thetom-ai--MiniMax-M2.7-ConfigI-MLX) | TurboQuant+ Config-I (2-bit experts / 4-bit attn / FP boundary) | 87 GB | **94% MMLU**, 37 tok/s @ 128 ctx | N/A — MLX-only | N/A — MLX-only |

Quantised artefacts are backend-specific by format: MLX safetensors don't load on CUDA/ROCm, and AWQ/GPTQ don't load on MLX. A different quant of the same base model is tracked as a separate entry in this table.

## Reproducing a run

### Apple Silicon (MLX)

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install 'lighteval[math]' mlx-lm
hf download <org>/<model> --local-dir ./weights/<model>
sudo sysctl iogpu.wired_limit_mb=94208   # if model >70 GB on 96 GB machine
mlx_lm.server --model ./weights/<model> --host 127.0.0.1 --port 8080 &
lighteval endpoint litellm \
  --model-args "model=openai/<model>,base_url=http://127.0.0.1:8080/v1,api_key=sk-noop" \
  --tasks "leaderboard|mmlu|5|0" \
  --output-dir ./results/apple-silicon
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

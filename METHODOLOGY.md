# Methodology

This document goes deeper than the README on *why* we run each benchmark the way we do, and what each number does and does not tell you.

---

## 1. Automatic benchmarks

Automatic benchmarks are cheap, reproducible, and the only way to compare dozens of models on the same yardstick. They are also the easiest to corrupt — small prompt or tokenisation changes can move a score by several points for reasons unrelated to model quality. Our rule: run via a published harness (`lighteval`) using the canonical configuration, and document any deviation.

### The suite

| Benchmark | What it measures | Why we run it | Known weaknesses |
|---|---|---|---|
| **MMLU** | Factual/academic knowledge across 57 subjects | Industry standard; model cards report it | Contaminated (training data likely contains test questions); rewards memorisation |
| **HellaSwag** | Commonsense continuation | Cheap, fast, multi-choice | Saturated at the top of the leaderboard — useful mostly below 90% |
| **GSM8K** | Grade-school maths requiring chain-of-thought | Sensitive to reasoning ability, tokenisation, and decoding mode | Small (~1.3k test), numerical-answer format is brittle |
| **HumanEval** | Python code generation | Functional correctness via unit tests | Small (164 problems); weak on real-world code |
| **TruthfulQA** | Resistance to popular misconceptions | Catches models that mimic internet consensus | Adversarially written; some "correct" answers are debatable |

We favour **5-shot** MMLU and **10-shot** HellaSwag (the OpenLLM-Leaderboard defaults) so numbers are comparable to published results. GSM8K runs 5-shot with chain-of-thought; HumanEval and TruthfulQA run 0-shot.

### Running mode: generation vs log-likelihood

For multi-choice benchmarks (MMLU, HellaSwag, TruthfulQA), `lighteval` can either:

- **Log-likelihood**: score each answer choice by the model's assigned probability, pick the highest. Cheap, deterministic, no sampling artefacts.
- **Generation**: have the model write a letter (A/B/C/D) and string-match. Slower, sampling-dependent, closer to real use.

We run **both** where feasible and report them separately — the gap between them is itself a quality signal. For GSM8K and HumanEval the mode is generation by definition.

### Reasoning models

Chain-of-thought models (MiniMax-M2, DeepSeek-R1, etc.) emit their reasoning in a separate channel before the final answer. If the harness cuts generation short inside the reasoning channel, the answer is empty and the benchmark returns 0% — even though the model would have got it right.

Our rules for reasoning models:

- `max_tokens` ≥ 2048 (or higher if the model's own chat template recommends).
- Extract the final answer from `message.content` when using `mlx_lm.server`'s OpenAI-compatible response, not `message.reasoning`.
- Report **two** tok/s figures where relevant: end-to-end decode (including reasoning) and answer-only decode (reasoning excluded).

---

## 2. LLM-as-judge

Automatic benchmarks miss what a real user cares about: does the model write well, follow instructions, and reason coherently on open-ended prompts? We use a reduced **MT-Bench** protocol: 80 prompts across 8 categories (writing, roleplay, reasoning, math, coding, extraction, STEM, humanities), each answered by the model under test and scored 1–10 by a judge.

### Judge selection

- **Preferred**: a strong API model (Claude Opus / GPT-4 class) via a published prompt. Gives consistent, well-calibrated scores.
- **Fallback 1**: manual scoring inside a Claude Code session by pasting generations in batches. Slower, but free and transparent.
- **Fallback 2**: a local MLX judge (e.g. `Qwen2.5-32B-Instruct-4bit`). Cheap, reproducible, but known to be harsher on style and more forgiving on correctness than frontier models.

The judge used is recorded in every model's log.

### Position bias and rubric stability

LLM judges are systematically biased towards the first answer in a pairwise comparison and towards longer answers in single-answer scoring. We mitigate by:

- Using **single-answer grading with a fixed rubric** rather than pairwise ranking where possible.
- Randomising presentation order when we do run pairwise.
- Keeping the judge prompt identical across models.

---

## 3. Human evaluation

The third leg of the OpenEvals framework is humans actually reading generations. We do this narrowly but deliberately:

- A **standing set of ~20 prompts** covering long-context summarisation, tricky coding tasks, multi-step reasoning, and prompts that have historically broken models.
- Generations are saved to `artifacts/generations/` and reviewed by a human reviewer with notes on what worked, what didn't, and any behaviour that felt qualitatively different from other models.
- No numeric score — qualitative notes only. The goal is to catch issues invisible to automatic benchmarks (verbose filler, refusal patterns, formatting weirdness, hallucinated citations).

---

## 4. Performance metrics

These are orthogonal to quality — a dumb-but-fast model can dominate on tok/s. We treat them as a separate dimension, measured per backend.

### Backend-specific notes

- **Apple Silicon (MLX)**: single-stream, `mlx_lm.server`. Unified memory means no host⇄device copy cost. Dominant bottleneck for MoE is memory bandwidth. Metal kernel compile is paid once per server start; first request pays it on top of load.
- **NVIDIA (vLLM/CUDA)**: PagedAttention + continuous batching means single-stream numbers understate real throughput. We report both single-stream (comparable to Apple) and batch-8 (representative of production serving). FlashAttention-3 variants matter on Hopper/Blackwell.
- **AMD (ROCm)**: expect kernel-coverage gaps on non-mainstream quant formats. We document any quant we could not run rather than silently falling back. Triton/Composable-Kernel versions are recorded.

### Cross-backend comparability

The three backends usually run **different quantised artefacts** of the same base model (MLX-specific safetensors, AWQ/GPTQ for CUDA, ROCm-compatible quants). A single-number comparison across backends is misleading. We report:

1. The backend's best-available export of the model.
2. The quant scheme and expected quality gap vs the FP16 base.
3. Both speed and quality scores so cross-backend tables show the tradeoff, not just one axis.

### What we measure

| Metric | Method | Why it matters |
|---|---|---|
| **Decode tok/s** | 128-token completion on a 64-token prompt, warm server, 5 runs averaged | Real-world generation speed |
| **Prompt-eval tok/s** | 2048-token prompt, measure time to first output token, subtract out fixed overhead | Batch processing, RAG throughput |
| **TTFT** | Warm server, 64-token prompt | Interactive UX |
| **Cold-load time** | First request to `mlx_lm.server` after start | Ops cost |
| **Peak unified memory** | Reported by `mlx_lm` | Fit on a given machine |

### Why not just trust model-card numbers

Published tok/s figures on model cards depend on:

- Hardware tier (M1 Max vs M3 Ultra can be 2× apart)
- Storage tier (mmap'd from NVMe vs external SSD differs for long prompts)
- Whether measurement includes load time, compilation, and prompt prefill
- Single-stream vs batched
- Whether the KV cache is cold or warm

Our numbers are single-stream, warm-server, post-compile, and we always disclose hardware and storage path.

---

## Reproducibility checklist

Every model log must record:

- `mlx`, `mlx-lm`, `lighteval`, Python versions
- Model HF commit hash
- Tokenizer source (shipped-with-model unless documented otherwise)
- Quant scheme and base model
- Hardware chip + unified RAM
- `iogpu.wired_limit_mb` setting
- Storage path (internal NVMe / external TB)
- Solo vs clustered; if clustered, interconnect
- `max_tokens`, `temperature`, `top_p`, seed for each benchmark
- Exact `lighteval` command line
- Any deviations from the canonical harness config, with justification

Without all of these, a number isn't comparable. With them, another researcher can reproduce or challenge it.

---

## Known limitations of this methodology

- **Benchmark contamination**: MMLU and HellaSwag questions are almost certainly in training sets. Scores conflate knowledge with memorisation.
- **English-first**: we do not evaluate multilingual capability beyond what these benchmarks test.
- **Static suite**: adversarial probes (prompt injection resistance, jailbreak resilience) are not yet part of the suite.
- **Single-run stochasticity**: sampling-based benchmarks (HumanEval) can vary run-to-run. Where possible we fix temperature and seed; otherwise we run ≥3 times and report variance.
- **Quant-vs-base comparison**: a quantised model is not the same model as its FP16 base. We log both the base model's published scores and our measured quantised scores, with the gap as an explicit quality-loss metric.

---

## How to propose a change

Open a PR editing this file with the rationale in the commit message. Methodology changes are versioned — every model log records the methodology commit hash in use at run time.

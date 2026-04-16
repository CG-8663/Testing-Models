# MiniMax-M2.7-ConfigI-MLX — Benchmark Log

**Model:** [`thetom-ai/MiniMax-M2.7-ConfigI-MLX`](https://huggingface.co/thetom-ai/MiniMax-M2.7-ConfigI-MLX)
228B-parameter MoE (~1.4B active/token), Config-I mixed-precision quant
(2-bit expert MLPs, 4-bit attention, FP boundary layers), ~87 GB on disk.

**Claimed:** 93.5% MMLU · 61 tok/s decode · PPL 4.604.

**Evaluation framework:** Hugging Face [OpenEvals guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook)
(automatic benchmarks · LLM-as-judge · human eval) + Apple-silicon perf metrics.

---

## Hardware

| Role | Machine | RAM | Notes |
|---|---|---|---|
| Primary | **M3 Ultra** | 96 GB | macOS 26.4.1, mlx 0.31.1, mlx-lm 0.31.2 |
| Cluster peer | **M1 Max** | 32 GB | Cannot load 87 GB solo — cluster only |
| Interconnect | 10 GbE + Thunderbolt | — | TB bridge preferred for distributed inference |

---

## Test plan

1. **M3 Ultra solo** — baseline perf + full benchmark suite.
2. **M3 Ultra + M1 Max cluster** (128 GB pooled via `mlx.distributed`) — throughput comparison.
3. ~~M1 Max solo~~ — skipped (does not fit).

**Benchmarks (per OpenEvals methodology):**
- **Automatic:** MMLU (verify 93.5% claim), HellaSwag, GSM8K, HumanEval, TruthfulQA — via `lighteval`.
- **LLM-as-judge:** MT-Bench subset, judged manually in Claude Code (no API key).
- **Human:** spot checks across coding / reasoning / long-context.
- **Perf:** decode tok/s · prompt-eval tok/s · time-to-first-token · peak unified memory.

---

## Step log

### Step 1 — Environment (✅)
- Confirmed: Python 3.12, `mlx 0.31.1`, `mlx-lm 0.31.2`, Metal available, 497 GB free on external volume.
- Created venv at `.venv/`, installed `lighteval 0.13.0` + `mlx-lm`.

### Step 2 — Download (✅)
- Destination: external volume at `models/MiniMax-M2.7-ConfigI-MLX/` (shared with project).
- 28 files, ~87 GB total (18 safetensor shards × ~4.9 GB, plus tokenizer + custom `modeling_minimax_m2.py`).
- Unauth HF download ~40 GB/hr; full pull took ~2 h.

### Step 3 — Metal memory limit (✅)
- Default `iogpu.wired_limit_mb = 0` caps GPU working set at ~72 GB — insufficient for an 87 GB model.
- First smoke test OOM'd with `Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`.
- Raised to **92 GB** via `sudo sysctl iogpu.wired_limit_mb=94208`.
- Not persistent across reboots — add to `/etc/sysctl.conf` if needed permanently.

### Step 4 — Smoke test, cold / external volume (✅)
- Prompt: "Write one sentence explaining what a transformer is."
- Max tokens 80, temp 0.7.

| Metric | Value |
|---|---|
| Prompt eval | 0.425 tok/s |
| Generation | **53.6 tok/s** |
| Peak memory | 93.0 GB |

Notes:
- 0.425 tok/s prompt eval is the **cold-load penalty** (first read of 87 GB from external volume mmap'd into unified memory).
- Decode 53.6 tok/s ≈ 88 % of the card's claimed 61 tok/s. Expected to match/exceed once weights are on internal SSD.
- 93 GB peak = zero headroom; long contexts will OOM or swap.

### Step 5 — Copy weights to internal SSD (✅)
- Reason: external volume sustained ~312 MB/s; expected 10× speedup on internal NVMe.
- Destination: `~/models/MiniMax-M2.7-ConfigI-MLX/` (18 shards, 87 GB).
- `rsync -a --progress` took ~30 min (throttled by external read speed).

### Step 6 — Smoke test, warm / internal SSD (✅)
Same prompt, same flags, weights from internal SSD.

| Metric | Cold (external) | Warm (internal) |
|---|---|---|
| Prompt tok/s | 0.425 | 0.464 |
| Generation tok/s | **53.614** | **53.024** |
| Peak memory | 93.0 GB | 93.0 GB |

**Observation — no speedup from internal SSD.** `mlx_lm.generate` measures prompt tok/s as `prompt_tokens / TTFT`, and TTFT is dominated by **model load + first-pass Metal kernel compilation**, not storage. MLX uses `mmap`, so page-ins happen inside the first forward pass regardless of source disk. Decode tok/s (~53) is the stable end-user metric — ~88 % of the card's 61 tok/s claim.

True prefill speed will only surface with (a) a long prompt where compute dominates, or (b) `mlx_lm.server` with a persistent model.

### Step 6.5 — Server setup + reasoning model discovery (✅)
Stood up `mlx_lm.server` (OpenAI-compatible HTTP API on :8080) so benchmarks can drive the model without per-run load cost.

**First request (load): 92.5 s.** Subsequent requests are warm.

**Important finding — MiniMax-M2 is a reasoning model.** The response JSON populates a `message.reasoning` field (chain-of-thought), not `message.content`. With `max_tokens=20` it got cut off mid-reasoning with no final answer. Implications:

- Benchmarks must set **much higher `max_tokens`** (likely 1024–4096) so the model has room to finish reasoning and produce a final answer.
- We need to extract the final answer from `message.content` (or the post-reasoning segment), not `message.reasoning`.
- Some benchmarks may need a system-prompt hack to skip reasoning mode (e.g. "answer directly").
- This also means our 53 tok/s decode benchmark produced mostly *reasoning* tokens, not final-answer tokens — still a fair throughput number, but a caveat for UX comparisons.

### Step 6.6 — Internal SSD cleanup (✅)
- Deleted `~/models/MiniMax-M2.7-ConfigI-MLX/` — confirmed no perf benefit over external volume (MLX mmap).
- Internal free space: 514 → 610 GB.
- Restarted server pointing to the external volume copy.

### Step 7 — MMLU (⏳ next)
Open decision: **full MMLU (14k Q, hours)** vs **stratified ~1k-Q subset (~20–40 min)**.
Default: subset first, then full overnight if the subset looks clean.

### Step 8 — Remaining automatic benchmarks (⏳)
HellaSwag · GSM8K · HumanEval · TruthfulQA.

### Step 9 — LLM-as-judge / MT-Bench (⏳)
Generate MiniMax answers to ~80 MT-Bench prompts → judge in this Claude Code session
(no API key available; local judge model is a later option).

### Step 10 — Cluster setup (⏳)
`mlx.distributed` over Thunderbolt bridge between M3 Ultra + M1 Max → repeat perf benchmarks.

---

## Open decisions

- [ ] MMLU scope (full vs ~1k subset).
- [ ] Thunderbolt bridge vs 10 GbE for cluster interconnect (benchmark both if time permits).
- [ ] Persist `iogpu.wired_limit_mb=94208` in `/etc/sysctl.conf`?

---

## Issues & fixes

| # | Issue | Fix |
|---|---|---|
| 1 | Metal OOM on model load | Raised `iogpu.wired_limit_mb` from default to 94208 (92 GB) |
| 2 | `rsync --info=progress2` unsupported on stock macOS rsync | Fell back to `rsync -a --progress` |

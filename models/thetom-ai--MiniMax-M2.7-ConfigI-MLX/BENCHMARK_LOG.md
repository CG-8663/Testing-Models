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

### Step 7 — Overnight protocol run (✅)

Ran the full pinned protocol overnight (2026-04-16 → 2026-04-17) via `overnight_run.sh` → `run_all.sh`. All apps closed, `iogpu.wired_limit_mb=94208`, Qwen :8082 server still running (~4 GB).

**Phase A — MMLU** (10 subjects × 20 Q = 200 total, 0-shot, temp 1.0, max_tokens 4096):
- **188/200 = 94.0%** — slightly exceeds the card's 187/200 = 93.5%.
- Strongest: College Physics (100%), HS Mathematics (100%).
- Weakest: Anatomy (85%).
- Run ID: `20260416T175847Z`, result: `mmlu_20260416T175847Z.json`.

**Phase B — NIAH** (3 depths × 4 context lengths = 12 probes):
- **12/12 = 100%** — matches the card.
- Run ID: `20260416T175847Z`, result: `niah_20260416T175847Z.json`.

**Phase C — Speed sweep** (7 context lengths, 3 trials each, median tok/s):
- 128 ctx → **37.0 tok/s**, 8192 ctx → **34.9 tok/s**.
- Card comparison: 61.1 → 45.4 tok/s (M5 Max, `mlx-swift-lm` + turbo4v2 + Bridge prefill).
- The ~40% gap is expected: Python `mlx_lm` on M3 Ultra vs Swift runtime with turbo4v2 on M5 Max.
- Run ID: `20260416T175847Z`, result: `speed_20260416T175847Z.json`.

**Phase D — Perplexity** (wikitext-2-raw-v1, in-process `mlx_lm.load`):
- First attempt hit `AttributeError: module 'mlx.core' has no attribute 'log_softmax'` — fixed (`mx.logsumexp` manual computation).
- Card methodology (2048 × 50): **PPL 9.6646** — card claims 4.604 ± 0.042.
- TBQ+ methodology (512 × 20): **PPL 13.6855**.
- The PPL gap vs the card is large (2× higher). Card ran with turbo4v2 KV compression on `mlx-swift-lm`; we ran without. Whether turbo4v2 genuinely halves perplexity or there is another methodological factor requires investigation with the Swift runtime.
- Run ID: `20260417T033810Z`, result: `ppl_20260417T033806Z.json`.

### Step 8 — Remaining automatic benchmarks (⏳)
HellaSwag · GSM8K · HumanEval · TruthfulQA.

### Step 9 — LLM-as-judge / MT-Bench (⏳)
Generate MiniMax answers to ~80 MT-Bench prompts → judge in this Claude Code session
(no API key available; local judge model is a later option).

### Step 10 — Cluster setup (⏳)
`mlx.distributed` over Thunderbolt bridge between M3 Ultra + M1 Max → repeat perf benchmarks.

---

## Open decisions

- [x] ~~MMLU scope~~ — ran the pinned 200-Q card-matching protocol (Step 7).
- [ ] Thunderbolt bridge vs 10 GbE for cluster interconnect (benchmark both if time permits).
- [x] ~~Persist `iogpu.wired_limit_mb=94208`~~ — written to `/etc/sysctl.conf` (2026-04-17).
- [ ] Investigate PPL gap (9.66 measured vs 4.60 card) — turbo4v2 vs Python `mlx_lm`?

---

## Issues & fixes

| # | Issue | Fix |
|---|---|---|
| 1 | Metal OOM on model load | Raised `iogpu.wired_limit_mb` from default to 94208 (92 GB) |
| 2 | `rsync --info=progress2` unsupported on stock macOS rsync | Fell back to `rsync -a --progress` |
| 3 | Wired limit reset after reboot causing Phase D OOM | Persisted to `/etc/sysctl.conf`; added pre-flight check in `overnight_run.sh` |
| 4 | `mx.log_softmax` does not exist in `mlx.core` | Replaced with `x - mx.logsumexp(x, axis=-1, keepdims=True)` in `eval_perplexity.py` |

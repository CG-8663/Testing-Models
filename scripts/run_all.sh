#!/bin/bash
# run_all.sh — chain the four evaluation phases for a given model.
#
# Usage:
#   scripts/run_all.sh <model-slug> <served-model-id> <model-dir>
#
# Example:
#   scripts/run_all.sh thetom-ai--MiniMax-M2.7-ConfigI-MLX \
#     '/Volumes/Chronara-Storage/Projects/test models/models/MiniMax-M2.7-ConfigI-MLX' \
#     '/Volumes/Chronara-Storage/Projects/test models/models/MiniMax-M2.7-ConfigI-MLX'
#
# Phases A, B, C require mlx_lm.server running on 127.0.0.1:8080.
# Phase D loads the model in-process (mlx_lm.load) and MUST run after
# the server is stopped because the 87 GB working set cannot fit twice.

set -u -o pipefail

MODEL_SLUG="${1:?model slug, e.g. thetom-ai--MiniMax-M2.7-ConfigI-MLX}"
SERVED_ID="${2:?served model id as listed by /v1/models}"
MODEL_DIR="${3:?local path to the weights directory}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO/models/$MODEL_SLUG/results/apple-silicon"
LOG_DIR="$REPO/models/$MODEL_SLUG/results/_logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
VENV_PY="$REPO/../.venv/bin/python"
SCRIPTS="$REPO/scripts"

PHASE_LOG="$LOG_DIR/run_all_$RUN_TS.log"
echo "run_all starting at $RUN_TS" | tee -a "$PHASE_LOG"
echo "model_slug=$MODEL_SLUG" | tee -a "$PHASE_LOG"
echo "served_id=$SERVED_ID" | tee -a "$PHASE_LOG"
echo "model_dir=$MODEL_DIR" | tee -a "$PHASE_LOG"
echo "results_dir=$RESULTS_DIR" | tee -a "$PHASE_LOG"

run_phase () {
  local name="$1"
  shift
  echo "" | tee -a "$PHASE_LOG"
  echo "=== PHASE $name starting ($(date -u +%H:%M:%SZ)) ===" | tee -a "$PHASE_LOG"
  local t0=$(date +%s)
  if "$@" 2>&1 | tee -a "$PHASE_LOG"; then
    local t1=$(date +%s)
    echo "=== PHASE $name COMPLETE in $((t1 - t0))s ===" | tee -a "$PHASE_LOG"
    return 0
  else
    echo "=== PHASE $name FAILED ===" | tee -a "$PHASE_LOG"
    return 1
  fi
}

# ---------- PHASE A: MMLU ----------
run_phase A_mmlu "$VENV_PY" "$SCRIPTS/eval_mmlu.py" \
  --endpoint "http://127.0.0.1:8080" \
  --model "$SERVED_ID" \
  --model-dir "$MODEL_DIR" \
  --output "$RESULTS_DIR/mmlu_${RUN_TS}.json" || echo "[warn] A failed, continuing"

# ---------- PHASE B: NIAH ----------
run_phase B_niah "$VENV_PY" "$SCRIPTS/eval_niah.py" \
  --endpoint "http://127.0.0.1:8080" \
  --model "$SERVED_ID" \
  --model-dir "$MODEL_DIR" \
  --output "$RESULTS_DIR/niah_${RUN_TS}.json" || echo "[warn] B failed, continuing"

# ---------- PHASE C: Speed sweep ----------
run_phase C_speed "$VENV_PY" "$SCRIPTS/speed_sweep.py" \
  --endpoint "http://127.0.0.1:8080" \
  --model "$SERVED_ID" \
  --model-dir "$MODEL_DIR" \
  --output "$RESULTS_DIR/speed_${RUN_TS}.json" || echo "[warn] C failed, continuing"

# ---------- PHASE D: Perplexity (requires the server stopped) ----------
echo "" | tee -a "$PHASE_LOG"
echo "Stopping mlx_lm.server for MiniMax before PPL (D) ..." | tee -a "$PHASE_LOG"
pkill -f "mlx_lm.server.*MiniMax-M2.7" 2>/dev/null || true
# Give it a few seconds to release the unified-memory working set.
sleep 6

run_phase D_ppl "$VENV_PY" "$SCRIPTS/eval_perplexity.py" \
  --model-path "$MODEL_DIR" \
  --output "$RESULTS_DIR/ppl_${RUN_TS}.json" || echo "[warn] D failed, continuing"

echo "" | tee -a "$PHASE_LOG"
echo "=== run_all finished at $(date -u +%Y%m%dT%H%M%SZ) ===" | tee -a "$PHASE_LOG"
echo "results in $RESULTS_DIR" | tee -a "$PHASE_LOG"
echo "full log: $PHASE_LOG" | tee -a "$PHASE_LOG"

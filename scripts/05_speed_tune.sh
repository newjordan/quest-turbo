#!/usr/bin/env bash
# Phase 5 — runtime speed tuning on the chosen quant. Sweeps batch/ubatch, flash-attention, decode
# at depth, and A/Bs the MAIN build vs the MoE-reorder build (Q6_K mul_mat_id win). QUEST is an
# A3B MoE (48 layers) — a same-family A3B already runs ~85-94 t/s on the B70; that's the target band.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

QUANT="${QUANT:-Q5_K_M}"
M="$ART/quest-30b-rl-${QUANT}.gguf"
[[ -f "$M" ]] || { echo "missing $M — run 03 (or set QUANT)"; exit 1; }
QNGL="$(ngl_for "$M")"   # full offload if it fits 30GiB, else partial (Q8_0/BF16)
echo ">> Tuning model: $M  (-ngl $QNGL)"

echo "== batch / ubatch sweep — MoE expert-GEMM un-starve (prefill >= max ub so the knob is visible; b fixed >= max ub so n_ubatch isn't silently clamped) =="
"$RUN_BIN/llama-bench" -m "$M" -ngl "$QNGL" -p 4096 -n 128 -r 3 \
  -b 8192 -ub 1024,2048,4096 | tee "$RES/05_batch.txt"

echo "== flash-attention on/off =="
"$BIN/llama-bench" -m "$M" -ngl "$QNGL" -p 512 -n 128 -r 3 -fa 0,1 \
  | tee "$RES/05_flashattn.txt" || echo "   (check -fa flag spelling for this build)"

echo "== decode at depth (QUEST runs long agentic contexts; ctx up to 262144) =="
"$BIN/llama-bench" -m "$M" -ngl "$QNGL" -n 128 -d 0,4096,16384,32768,65536 -fa 1 -r 2 \
  | tee "$RES/05_depth.txt" || true

if [[ -x "$MOE_BIN/llama-bench" ]]; then
  echo "== A/B: MoE-reorder build vs main (Q6_K mul_mat_id + router/expert-combine fusions) =="
  "$MOE_BIN/llama-bench" -m "$M" -ngl "$QNGL" -p 512 -n 128 -r 3 -fa 1 \
    | tee "$RES/05_moe_reorder.txt"
  echo "   compare tg128 vs 05_flashattn.txt; reorder/fusions should help most at Q6_K."
else
  echo "== MoE-reorder build not built; skipping A/B =="
  echo "   build: cd $MOE_LCPP && cmake -B build -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
fi

echo "Phase 5 done -> $RES/05_*.txt"
echo "NOTE: QUEST uses standard GQA attn, NOT gated-delta-net — the prior campaign's gated-delta decode fusion does"
echo "      not apply; expect the router/expert-combine fusions + -ub prefill to carry the win here."

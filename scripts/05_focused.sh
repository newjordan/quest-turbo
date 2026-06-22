#!/usr/bin/env bash
# Phase 5 (focused) — Q5_K_M B70 speed-tune, deadline-trimmed per results/_prep/speed_tune_plan.md.
# NO `set -e`: one unsupported flag must not abort the whole sweep. RUN_BIN = MoE-reorder tree.
set -uo pipefail
source "$(dirname "$0")/../env.sh"
M="$ART/quest-30b-rl-Q5_K_M.gguf"
[[ -f "$M" ]] || { echo "missing $M"; exit 1; }
echo ">> Q5_K_M speed-tune (ngl 99 full offload). Baseline ceiling: pp512 1154.6 / tg128 79.9 (MoE tree)."

echo "== B5 build A/B @ ship flags (does the MoE-reorder win transfer to QUEST?) =="
echo "-- main build --"; "$BIN/llama-bench"     -m "$M" -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3 2>&1 | tee "$RES/05_build_main.txt" || true
echo "-- MoE build  --"; "$RUN_BIN/llama-bench" -m "$M" -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3 2>&1 | tee "$RES/05_build_moe.txt" || true

echo "== B1 -ub prefill sweep (the MoE expert-GEMM un-starve lever; -p 4096 so the knob is visible) =="
"$RUN_BIN/llama-bench" -m "$M" -ngl 99 -fa 1 -b 8192 -ub 512,2048,4096 -p 4096 -n 0 -r 3 2>&1 | tee "$RES/05_ub.txt" || true

echo "== B2 flash-attn on/off =="
"$RUN_BIN/llama-bench" -m "$M" -ngl 99 -ub 4096 -b 8192 -fa 0,1 -p 512 -n 128 -r 3 2>&1 | tee "$RES/05_fa.txt" || true

echo "== B4(trim) decode at depth, KV q8_0 (the agentic long-ctx payoff) =="
"$RUN_BIN/llama-bench" -m "$M" -ngl 99 -fa 1 -ub 4096 -b 8192 -ctk q8_0 -ctv q8_0 -n 128 -d 0,16384,32768 -r 2 2>&1 | tee "$RES/05_depth.txt" || true

echo ">> Phase 5 focused done -> $RES/05_*.txt"

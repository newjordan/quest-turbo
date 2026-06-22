#!/usr/bin/env bash
# Phase 0 — establish the B70 target band BEFORE QUEST is built. Benches a same-FAMILY A3B MoE
# already on disk (BASE_REF) on both the main and MoE-reorder builds, so we know the speed ceiling
# QUEST's quants should land near. Also records QUEST's published accuracy floor from the card.
# NOTE: BASE_REF (Qwen3.6-35B-A3B, 256 experts) is a *rough* proxy — QUEST is 30B/128-expert, so it
# will likely be a touch faster (fewer experts, smaller). Use this as a target band, not a guarantee.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

[[ -f "$BASE_REF" ]] || { echo "missing BASE_REF=$BASE_REF (a same-family A3B on disk)"; exit 1; }
NGLB="$(ngl_for "$BASE_REF")"
echo ">> Baseline speed reference: $(basename "$BASE_REF")  (-ngl $NGLB)"

echo "== main build: pp512 / tg128 ==" | tee "$RES/00_baseline.txt"
"$BIN/llama-bench" -m "$BASE_REF" -ngl "$NGLB" -p 512 -n 128 -r 3 -fa 1 \
  | tee -a "$RES/00_baseline.txt"

if [[ -x "$MOE_BIN/llama-bench" ]]; then
  echo "== MoE-reorder build: pp512 / tg128 (the speed tree) ==" | tee -a "$RES/00_baseline.txt"
  "$MOE_BIN/llama-bench" -m "$BASE_REF" -ngl "$NGLB" -p 512 -n 128 -r 3 -fa 1 \
    | tee -a "$RES/00_baseline.txt"
fi

cat >> "$RES/00_baseline.txt" <<'EOF'

# QUEST-30B-RL published floor (osunlp card, avg@3): GAIA 69.0 | LiveResearchBench 74.1 | DeepResearch 45.3
# These are AGENTIC benchmarks (tool-use + multi-step research) — our quant must preserve the BEHAVIOR
# behind them, which is what scripts/04b_agentic_gate.sh measures (PPL can't see it).
EOF
echo ">> Phase 0 done -> $RES/00_baseline.txt"

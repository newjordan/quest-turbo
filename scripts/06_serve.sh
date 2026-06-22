#!/usr/bin/env bash
# Phase 6 — serve the chosen Turbo quant on :8093 with the reasoning+tools template + FA + KV q8_0.
# Binds 0.0.0.0 (LAN-accessible) with no api-key by default — add --api-key for untrusted networks.
# Prefers the MoE-reorder build when built. --jinja enables both
# QUEST's <think> reasoning AND its tool-calling template.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

QUANT="${QUANT:-Q5_K_M}"        # default ship candidate; set QUANT=Q6_K for max-fidelity profile
M="$ART/quest-30b-rl-${QUANT}.gguf"
[[ -f "$M" ]] || { echo "missing $M — run 03 (or set QUANT)"; exit 1; }

CTX="${CTX:-32768}"             # model declares 262144; KV grows with ctx, raise as VRAM allows
KV="${KV:-q8_0}"               # q8_0 protects the <think> trace + tool-call JSON; q4_0 = long-ctx profile

SNGL="${SERVE_NGL:-$(ngl_for "$M")}"
echo ">> Serving $QUANT on :$PORT  (build=$RUN_BIN, -ngl $SNGL, ctx=$CTX, FA on, KV=$KV, jinja=think+tools)"
exec "$RUN_BIN/llama-server" \
  -m "$M" \
  -ngl "$SNGL" \
  --host 0.0.0.0 --port "$PORT" \
  -c "$CTX" \
  -fa on \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  -ub "$UBATCH" -b "$BBATCH" \
  --jinja \
  --alias quest-30b-rl-turbo

#!/usr/bin/env bash
# Phase 6 — serve the chosen Turbo quant on :8093 with the reasoning+tools template + FA + KV q8_0.
# Binds 127.0.0.1 by default. Set HOST=0.0.0.0 only on a trusted network, and pass
# --api-key (via SERVE_EXTRA) before exposing the port.
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
HOST="${HOST:-127.0.0.1}"
echo ">> Serving $QUANT on $HOST:$PORT  (build=$RUN_BIN, -ngl $SNGL, ctx=$CTX, FA on, KV=$KV, jinja=think+tools)"
# shellcheck disable=SC2086
exec "$RUN_BIN/llama-server" \
  -m "$M" \
  -ngl "$SNGL" \
  --host "$HOST" --port "$PORT" \
  -c "$CTX" \
  -fa on \
  --cache-type-k "$KV" --cache-type-v "$KV" \
  -ub "$UBATCH" -b "$BBATCH" \
  --jinja \
  --alias quest-30b-rl-turbo \
  ${SERVE_EXTRA:-}

#!/usr/bin/env bash
# Phase 2 — importance matrix on the BF16 source, reusing a shared calibration corpus.
#
# MoE CAVEAT: QUEST has 128 experts (8 active/token). imatrix only calibrates experts the corpus
# actually activates; a narrow corpus leaves long-tail experts on default scaling — exactly where
# low-bit damage hides. 128 experts is easier to cover than the prior campaign's 256, but still verify: if
# llama-imatrix warns about experts with few/zero activations, widen the corpus before trusting it.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

[[ -f "$BF16" ]]  || { echo "missing $BF16 — run 01_fetch_source.sh first"; exit 1; }
[[ -f "$CALIB" ]] || { echo "missing calibration corpus $CALIB"; exit 1; }

IMNGL="${IMATRIX_NGL:-$(ngl_for "$BF16")}"   # BF16 ~61GB > 30GiB VRAM -> partial offload
echo ">> Computing imatrix from $(basename "$CALIB") -> $IMATRIX  (-ngl $IMNGL, partial; rest on CPU)"
"$RUN_BIN/llama-imatrix" \
  -m "$BF16" \
  -f "$CALIB" \
  -o "$IMATRIX" \
  -ngl "$IMNGL" \
  --chunks "${IMATRIX_CHUNKS:-256}" \
  2>&1 | tee "$RES/02_imatrix.log"

echo ">> Expert-coverage check (grep the log for under-activated experts)"
grep -iE "expert|activation|warn|nan" "$RES/02_imatrix.log" | tail -20 || true

echo "Phase 2 done -> $IMATRIX"

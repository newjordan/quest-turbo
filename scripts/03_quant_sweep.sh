#!/usr/bin/env bash
# Phase 3 — imatrix-guided quant sweep. Produces the candidate Turbo quants.
# QUEST has NO published quant cliff, so the sweep IS the measurement: Q4_K_M is the suspected
# failure point (reproduce/measure it), Q5_K_M / Q6_K are the likely ship frontier, Q8_0 the
# near-lossless reference the agentic gate grades against.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

[[ -f "$BF16" ]]    || { echo "missing $BF16 — run 01"; exit 1; }
[[ -f "$IMATRIX" ]] || { echo "missing $IMATRIX — run 02"; exit 1; }

if [[ -n "${QUANTS:-}" ]]; then
  # shellcheck disable=SC2206
  QUANTS=($QUANTS)
else
  QUANTS=(Q4_K_M Q5_K_M Q6_K Q8_0)   # add IQ4_XS for a second low-bit data point if wanted
fi

: > "$RES/03_quant_sizes.txt"
for q in "${QUANTS[@]}"; do
  out="$ART/quest-30b-rl-${q}.gguf"
  echo ">> Quantizing $q -> $out"
  "$RUN_BIN/llama-quantize" --imatrix "$IMATRIX" "$BF16" "$out" "$q" \
    2>&1 | tee "$RES/03_quant_${q}.log"
  sz=$(stat -c '%s' "$out")
  gib=$(awk -v s="$sz" 'BEGIN { printf "%.2f", s/1073741824 }')
  printf '%-10s %15d bytes  %6s GiB\n' "$q" "$sz" "$gib" \
    | tee -a "$RES/03_quant_sizes.txt"
done

echo "Phase 3 done. Quants in $ART/, sizes -> $RES/03_quant_sizes.txt"
echo "NEXT: 04_accuracy.sh (PPL+parity) then the agentic gate (tool-call + research) — the real decision."

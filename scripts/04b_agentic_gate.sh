#!/usr/bin/env bash
# Phase 4b — the agentic ship-decision gate. Serves each model with the real chat template
# (--jinja => <think> reasoning AND tool-calling), runs the tool-integrity + reasoning probe set
# (scripts/04b_probes.py), and grades every quant vs a verified key AND vs the BF16 reference.
#
# THE collapse detector for QUEST: if a quant's TOOL_key (correct tool routing + parseable JSON args)
# or REAS_key drops here while its PPL barely moves, that's the agentic/reasoning cliff — ship the
# smallest quant that holds both within ~1-2 of BF16.
#
# Usage: MODELS="BF16 Q8_0 Q6_K Q5_K_M Q4_K_M" bash scripts/04b_agentic_gate.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"

MODELS=(${MODELS:-BF16 Q8_0 Q6_K Q5_K_M Q4_K_M})
REF_LABEL="${REF_LABEL:-BF16}"
GATE_PORT="${GATE_PORT:-8099}"
OUTDIR="$RES/04b_answers"
mkdir -p "$OUTDIR"
export MAXTOK="${MAXTOK:-4096}"

path_for() { [[ "$1" == "BF16" ]] && echo "$BF16" || echo "$ART/quest-30b-rl-$1.gguf"; }

for label in "${MODELS[@]}"; do
  M="$(path_for "$label")"
  if [[ ! -f "$M" ]]; then echo ">> SKIP $label (missing $M)"; continue; fi
  echo "============================================================"
  echo ">> $label  ($(basename "$M"))  -ngl $(ngl_for "$M")"
  "$RUN_BIN/llama-server" -m "$M" -ngl "$(ngl_for "$M")" --host 127.0.0.1 --port "$GATE_PORT" \
    -c "${CTX:-32768}" -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --alias "quest-4b-$label" >"$RES/04b_server_${label}.log" 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  echo -n "   waiting for health"
  ok=0
  for _ in $(seq 1 150); do
    if curl -sf "http://127.0.0.1:$GATE_PORT/health" >/dev/null 2>&1; then ok=1; echo " ok"; break; fi
    echo -n "."; sleep 2
  done
  if [[ "$ok" == 0 ]]; then
    echo " TIMEOUT — see $RES/04b_server_${label}.log"
    kill $SRV 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
    kill -9 $SRV 2>/dev/null || true
    wait $SRV 2>/dev/null || true
    continue
  fi
  "$CONV_PY" "$(dirname "$0")/04b_probes.py" collect "$GATE_PORT" "$label" "$OUTDIR" \
    2>&1 | tee "$RES/04b_run_${label}.txt"
  kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true
  trap - EXIT
done

echo "============================================================"
echo ">> Grading (reference = $REF_LABEL)"
"$CONV_PY" "$(dirname "$0")/04b_probes.py" grade "$OUTDIR" "$REF_LABEL" "${MODELS[@]}" \
  | tee "$RES/04b_summary.txt"
echo ">> Phase 4b done -> $RES/04b_summary.txt"

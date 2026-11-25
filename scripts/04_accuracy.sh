#!/usr/bin/env bash
# Phase 4 — offline accuracy: perplexity per quant + greedy parity vs BF16.
# NECESSARY BUT NOT SUFFICIENT: for an RL'd agentic model, behavioral damage (tool-call JSON,
# stopping, format) is ~invisible in PPL. This catches gross breakage / KV regressions cheaply;
# the ship decision comes from the agentic gate (04b_agentic_gate.sh).
set -euo pipefail
source "$(dirname "$0")/../env.sh"

if [[ -n "${QUANTS:-}" ]]; then
  # shellcheck disable=SC2206
  QUANTS=($QUANTS)
else
  QUANTS=(Q4_K_M Q5_K_M Q6_K Q8_0)
fi
PARITY_PROMPTS="${PARITY_PROMPTS:-$RES/parity_prompts.txt}"
PPL_CORPUS="${PPL_CORPUS:-$HOME/calib-data/wikitext-2-raw/wiki.test.raw}"
PPL_CHUNKS="${PPL_CHUNKS:-60}"

# Seed a reasoning-flavored parity set if absent (greedy, so divergence = quant sensitivity).
if [[ ! -f "$PARITY_PROMPTS" ]]; then
  cat > "$PARITY_PROMPTS" <<'EOF'
Solve step by step: if 3 machines make 3 widgets in 3 minutes, how long for 100 machines to make 100 widgets?
Write a Python function that returns the nth Fibonacci number, with an O(n) and an O(log n) version.
List the exact steps a research agent should take to answer: "What year did the author of the most-cited 2017 deep-learning paper join their current employer?"
A farmer has chickens and rabbits, 35 heads and 94 legs. How many of each? Show the algebra.
EOF
fi

echo ">> Perplexity per quant (corpus: $PPL_CORPUS)"
: > "$RES/04_ppl.txt"
if [[ -n "${PPL_CORPUS:-}" && -f "${PPL_CORPUS}" ]]; then
  for m_label in "BF16:$BF16" $(for q in "${QUANTS[@]}"; do echo "$q:$ART/quest-30b-rl-${q}.gguf"; done); do
    q="${m_label%%:*}"; m="${m_label#*:}"; [[ -f "$m" ]] || continue
    echo "-- $q"; "$RUN_BIN/llama-perplexity" -m "$m" -ngl "$(ngl_for "$m")" -f "$PPL_CORPUS" \
      ${PPL_CHUNKS:+--chunks "$PPL_CHUNKS"} \
      2>&1 | grep -iE "Final estimate|PPL =" | sed "s/^/$q: /" | tee -a "$RES/04_ppl.txt" || true
  done
else
  echo "   (PPL_CORPUS unset/missing; skipping ppl)" | tee "$RES/04_ppl.txt"
fi

echo ">> Greedy parity vs BF16 (reference = $BF16)"
: > "$RES/04_parity.txt"
GEN_BIN="$(llama_completion_bin || true)"
if [[ -z "$GEN_BIN" ]]; then
  echo "ERROR: no llama-cli or llama-completion under RUN_BIN ($RUN_BIN) or BIN ($BIN)" | tee -a "$RES/04_parity.txt"
  exit 1
fi
echo "   completion binary: $GEN_BIN" | tee -a "$RES/04_parity.txt"
[[ -f "$BF16" ]] || { echo "no BF16 reference; run 01 first" | tee -a "$RES/04_parity.txt"; exit 1; }

# Empty stdout used to compare equal and print EXACT when the binary was missing
# or had written the error only to stderr (which was discarded).
gen() {
  local model="$1" prompt="$2" err out status=0
  err="$(mktemp)"
  out="$("$GEN_BIN" -m "$model" -ngl "$(ngl_for "$model")" -n "$GEN_N" --temp 0 -p "$prompt" 2>"$err")" && status=0 || status=$?
  if [[ "$status" != 0 || -z "${out//[[:space:]]/}" ]]; then
    echo "   gen failed (status=$status) model=$(basename "$model")" >> "$RES/04_parity.txt"
    if [[ -s "$err" ]]; then tail -n 5 "$err" >> "$RES/04_parity.txt"; fi
    rm -f "$err"
    return 1
  fi
  rm -f "$err"
  printf '%s' "$out"
}

while IFS= read -r p; do
  [[ -z "$p" ]] && continue
  ref_ok=1
  ref="$(gen "$BF16" "$p")" || ref_ok=0
  for q in "${QUANTS[@]}"; do
    m="$ART/quest-30b-rl-${q}.gguf"; [[ -f "$m" ]] || continue
    cand_ok=1
    cand="$(gen "$m" "$p")" || cand_ok=0
    if [[ "$ref_ok" != 1 || "$cand_ok" != 1 ]]; then
      verdict="ERROR"
    elif [[ "$cand" == "$ref" ]]; then
      verdict="EXACT"
    else
      verdict="DIVERGES"
    fi
    echo "[$verdict] $q :: ${p:0:64}" | tee -a "$RES/04_parity.txt"
  done
done < "$PARITY_PROMPTS"

echo "Phase 4 done -> $RES/04_ppl.txt , $RES/04_parity.txt"
echo "REMINDER: parity DIVERGES is expected; trust 04b_agentic_gate.sh for the real verdict."

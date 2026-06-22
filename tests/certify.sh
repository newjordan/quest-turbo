#!/usr/bin/env bash
# tests/certify.sh — one command to certify a quant. The ONLY component that touches the B70.
#
# Reuses the proven 04b/06 server lifecycle and the FIX-B teardown (SIGTERM -> grace -> SIGKILL ->
# reap in a trap) so it NEVER leaves a server holding the B70. Honors the live-GPU constraint by
# running at most ONE GPU job at a time: any needed llama-perplexity / llama-bench runs are done
# as their own short-lived processes BEFORE the cert server starts, so there is never an OOM-causing
# overlap (prefer passing pre-computed --ppl-file / --bench-file from phase-04/05 to skip them).
#
# USAGE:
#   bash tests/certify.sh [--quant Q5_K_M] [--model PATH] [--ref-answers PATH] [--port 8099] [--ctx 32768]
#                         [--bench-file PATH] [--ppl-file PATH]
#                         [--skip-perf] [--skip-ppl] [--skip-metadata] [--skip-behavior]
#                         [--perf-tolerance 0.05] [--pytest-args "..."]
#
# EXIT CODES: 0 = all gates pass = SHIP. 1 = a gate failed (pytest). 2 = bad args / missing model.
#             3 = server health timeout (GPU busy / OOM).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 1. Source the campaign env (oneAPI setvars; binaries link libsvml.so). Resolves RUN_BIN/ART/RES/BF16/...
# shellcheck disable=SC1091
source "$HERE/../env.sh"

WIKITEXT="${WIKITEXT:-$HOME/calib-data/wikitext-2-raw/wiki.test.raw}"

# ---- arg defaults --------------------------------------------------------
QUANT="Q5_K_M"
MODEL=""
REF_ANSWERS=""
PORT=8099
CTX=32768
BENCH_FILE=""
PPL_FILE=""
PERF_TOL="0.05"
SKIP_PERF=0
SKIP_PPL=0
SKIP_METADATA=0
SKIP_BEHAVIOR=0
EXTRA_PYTEST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quant)          QUANT="$2"; shift 2;;
    --model)          MODEL="$2"; shift 2;;
    --ref-answers)    REF_ANSWERS="$2"; shift 2;;
    --port)           PORT="$2"; shift 2;;
    --ctx)            CTX="$2"; shift 2;;
    --bench-file)     BENCH_FILE="$2"; shift 2;;
    --ppl-file)       PPL_FILE="$2"; shift 2;;
    --skip-perf)      SKIP_PERF=1; shift;;
    --skip-ppl)       SKIP_PPL=1; shift;;
    --skip-metadata)  SKIP_METADATA=1; shift;;
    --skip-behavior)  SKIP_BEHAVIOR=1; shift;;
    --perf-tolerance) PERF_TOL="$2"; shift 2;;
    --pytest-args)    EXTRA_PYTEST="$2"; shift 2;;
    -h|--help)        sed -n '2,20p' "$0"; exit 0;;
    *) echo "certify.sh: unknown arg '$1'" >&2; exit 2;;
  esac
done

# ---- resolve model + reference -------------------------------------------
if [[ -z "$MODEL" ]]; then
  if [[ "$QUANT" == "BF16" ]]; then MODEL="$BF16"; else MODEL="$ART/quest-30b-rl-${QUANT}.gguf"; fi
fi
[[ -z "$REF_ANSWERS" ]] && REF_ANSWERS="$RES/04b_answers/BF16.json"

# 2. Preflight (NO GPU)
if [[ ! -f "$MODEL" ]]; then echo "certify.sh FATAL: model not found: $MODEL" >&2; exit 2; fi
if [[ "$SKIP_BEHAVIOR" == 0 && ! -f "$REF_ANSWERS" ]]; then
  echo "certify.sh WARN: ref-answers missing ($REF_ANSWERS) -> --skip-behavior (no BF16 ref to agree against)" >&2
  SKIP_BEHAVIOR=1
fi

NGL="$(ngl_for "$MODEL")"
SRV_LOG="$RES/certify_server_${QUANT}.log"
SRV_PID=""

stop_srv() {  # FIX-B: SIGTERM -> 10x1s grace -> SIGKILL -> reap. Idempotent.
  [[ -z "$SRV_PID" ]] && return 0
  kill "$SRV_PID" 2>/dev/null || true
  for _ in $(seq 1 10); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SRV_PID" 2>/dev/null || true
  wait "$SRV_PID" 2>/dev/null || true
  SRV_PID=""
}
trap 'stop_srv' EXIT INT TERM

# ---- serialized quality jobs (own short-lived processes, BEFORE the cert server) ----
# 4. PPL: use --ppl-file if given, else run llama-perplexity over wikitext (40 chunks).
if [[ "$SKIP_PPL" == 0 && -z "$PPL_FILE" ]]; then
  PPL_FILE="$RES/certify_ppl_${QUANT}.log"
  echo ">> [ppl] llama-perplexity -m $(basename "$MODEL") -ngl $NGL -> $PPL_FILE"
  "$RUN_BIN/llama-perplexity" -m "$MODEL" -ngl "$NGL" -f "$WIKITEXT" --chunks 40 \
    2>&1 | tee "$PPL_FILE" || true
fi

# 5. BENCH: use --bench-file if given, else run llama-bench (headline flags).
if [[ "$SKIP_PERF" == 0 && -z "$BENCH_FILE" ]]; then
  BENCH_FILE="$RES/certify_bench_${QUANT}.txt"
  echo ">> [bench] llama-bench -m $(basename "$MODEL") -ngl 99 -> $BENCH_FILE"
  "$RUN_BIN/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3 \
    2>&1 | tee "$BENCH_FILE" || true
fi

# 3. Start the ONE cert server (only when behavioral cert will run; it is the sole B70 owner now).
if [[ "$SKIP_BEHAVIOR" == 0 ]]; then
  echo ">> [server] llama-server -m $(basename "$MODEL") -ngl $NGL --port $PORT -> $SRV_LOG"
  GGML_SYCL_DISABLE_DNN="${GGML_SYCL_DISABLE_DNN:-1}" \
  "$RUN_BIN/llama-server" -m "$MODEL" -ngl "$NGL" --host 127.0.0.1 --port "$PORT" \
    -c "$CTX" -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
    -ub "$UBATCH" -b "$BBATCH" --jinja --alias "quest-cert-${QUANT}" \
    >"$SRV_LOG" 2>&1 &
  SRV_PID=$!
  echo -n ">> [server] waiting for health"
  ok=0
  for _ in $(seq 1 150); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; echo " ok"; break; fi
    echo -n "."; sleep 2
  done
  if [[ "$ok" == 0 ]]; then
    echo " TIMEOUT"
    tail -n 40 "$SRV_LOG" 2>/dev/null || true
    stop_srv
    exit 3
  fi
fi

# 6. Run pytest (the verdict engine). Omitting an --option makes the matching test skip.
PYTEST_ARGS=( "$HERE" -v --spec "$HERE/spec.yaml" --perf-tolerance "$PERF_TOL"
              --out-dir "$RES" --junitxml="$RES/certify_${QUANT}.xml" )
if [[ "$SKIP_BEHAVIOR" == 0 ]]; then
  PYTEST_ARGS+=( --server-url "http://127.0.0.1:$PORT" --ref-answers "$REF_ANSWERS" )
fi
[[ "$SKIP_METADATA" == 0 ]]                 && PYTEST_ARGS+=( --model-path "$MODEL" )
[[ -n "$BENCH_FILE" && -f "$BENCH_FILE" ]]  && PYTEST_ARGS+=( --bench-file "$BENCH_FILE" )
[[ -n "$PPL_FILE"   && -f "$PPL_FILE" ]]    && PYTEST_ARGS+=( --ppl-file "$PPL_FILE" )
# shellcheck disable=SC2206
[[ -n "$EXTRA_PYTEST" ]] && PYTEST_ARGS+=( $EXTRA_PYTEST )

echo ">> [pytest] $CONV_PY -m pytest ${PYTEST_ARGS[*]}"
"$CONV_PY" -m pytest "${PYTEST_ARGS[@]}"
RC=$?

# 7-8. Capture RC, stop server unconditionally (trap also guarantees this).
stop_srv

# 9. Banner.
if [[ "$RC" == 0 ]]; then
  echo "CERTIFY $QUANT: SHIP"
else
  echo "CERTIFY $QUANT: NO-SHIP (see report: $RES/certify_${QUANT}.xml, $RES/certify_${QUANT}.json)"
fi

# 10. Exit with pytest's exit code (0 = SHIP).
exit $RC

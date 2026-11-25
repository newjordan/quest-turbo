#!/usr/bin/env bash
# Phase 6 (smoke) — bring up Q5_K_M on :PORT, verify tool-call + <think> end-to-end, capture load VRAM,
# then STOP the server (validates the ship serve-config without permanently holding the B70).
set -uo pipefail
source "$(dirname "$0")/../env.sh"
M="$ART/quest-30b-rl-Q5_K_M.gguf"
[[ -f "$M" ]] || { echo "missing $M"; exit 1; }
CTX="${CTX:-65536}"; KV="${KV:-q8_0}"; SNGL="$(ngl_for "$M")"
echo ">> Serving Q5_K_M on 127.0.0.1:$PORT  (build=$RUN_BIN, ngl $SNGL, ctx $CTX, KV $KV, -ub $UBATCH -b $BBATCH, jinja)"
"$RUN_BIN/llama-server" -m "$M" -ngl "$SNGL" --host 127.0.0.1 --port "$PORT" -c "$CTX" -fa on \
  --cache-type-k "$KV" --cache-type-v "$KV" -ub "$UBATCH" -b "$BBATCH" --jinja --alias quest-30b-rl-turbo \
  > "$RES/06_server.log" 2>&1 &
SRV=$!
stop_srv() {
  [[ -z "${SRV:-}" ]] && return 0
  local pid="$SRV"
  SRV=""
  kill "$pid" 2>/dev/null || true
  local _
  for _ in $(seq 1 10); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}
trap 'stop_srv' EXIT
trap 'stop_srv; exit 130' INT
trap 'stop_srv; exit 143' TERM
echo -n "   waiting for health"; ok=0
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; echo " ok"; break; }
  echo -n "."; sleep 2
done
[ "$ok" = 1 ] || { echo " TIMEOUT"; tail -25 "$RES/06_server.log"; exit 1; }

echo "== load-time VRAM / buffers / MoE reorder (from server log) =="
grep -iE "reorder|buffer size|KV self|compute buffer|model size|n_ctx" "$RES/06_server.log" | tail -25

echo "== smoke tests (S1 tool-route, S2 tool round-trip, S3 think) =="
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# pipefail is set, but without an explicit exit the last `echo` used to hide a
# failing smoke run (tee's status became the script's status).
"$CONV_PY" - "$PORT" "$ROOT" <<'PY' | tee "$RES/06_smoke.txt"
import json, sys, urllib.request
sys.path.insert(0, sys.argv[2] + "/tests/lib")
import harness
port = sys.argv[1]
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))
WEATHER = {"type":"function","function":{"name":"get_weather","description":"Get the current weather for a location.",
    "parameters":{"type":"object","properties":{"location":{"type":"string"},
    "unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","unit"]}}}
CALC = {"type":"function","function":{"name":"calculator","description":"Evaluate an arithmetic expression.",
    "parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}

# S1 — tool routing canary
r = post({"model":"quest-30b-rl-turbo","messages":[{"role":"user",
    "content":"What is the weather in Paris right now, in celsius? Use the weather tool."}],
    "tools":[WEATHER],"tool_choice":"auto","temperature":0,"max_tokens":4096})
msg = r["choices"][0]["message"]; tcs = msg.get("tool_calls") or []
s1 = harness.smoke_tool_route(msg)
shown = tcs[0]["function"]["arguments"] if tcs else None
print(f"S1 tool-route : {'PASS' if s1 else 'FAIL'}  fn={tcs[0]['function']['name'] if tcs else None} args={shown}")

# S2 — tool round-trip (feed back a tool result, expect NL answer stating 391)
r = post({"model":"quest-30b-rl-turbo","temperature":0,"max_tokens":4096,"messages":[
    {"role":"user","content":"Use the calculator to compute 17 times 23."},
    {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function",
        "function":{"name":"calculator","arguments":"{\"expression\": \"17*23\"}"}}]},
    {"role":"tool","tool_call_id":"c1","content":"391"}]})
m2 = r["choices"][0]["message"]; txt2 = (m2.get("content") or "") + (m2.get("reasoning_content") or "")
s2 = harness.smoke_round_trip(m2)
print(f"S2 round-trip : {'PASS' if s2 else 'FAIL'}  reply={txt2[:80]!r}")

# S3 — <think> reasoning produces a real answer
r = post({"model":"quest-30b-rl-turbo","temperature":0,"max_tokens":4096,"messages":[{"role":"user",
    "content":"A store had 240 apples. It sold 1/3 on Monday, then 1/4 of the remaining on Tuesday. "
    "How many are left? Think step by step, then end with a line:\nANSWER: <number>"}]})
m3 = r["choices"][0]["message"]; txt3 = harness.message_text(m3)
s3 = harness.smoke_think(m3)
print(f"S3 think      : {'PASS' if s3 else 'FAIL'}  has120={'120' in txt3}  len={len(txt3)}")
print(f"\nSMOKE: S1={s1} S2={s2} S3={s3}  ->  {'ALL PASS' if (s1 and s2 and s3) else 'CHECK FAILURES'}")
sys.exit(0 if (s1 and s2 and s3) else 1)
PY
rc=${PIPESTATUS[0]}
echo ">> Smoke done (exit $rc). Serve for real with: QUANT=Q5_K_M CTX=$CTX bash scripts/06_serve.sh"
exit "$rc"

# Phase 6 — SERVE config: QUEST-30B-RL Q5_K_M on the B70 (:8093)

Design/research only (GPU busy with a live ppl run — nothing here was executed). Values resolved
from `env.sh` + `scripts/06_serve.sh` + verified `config.json`/`chat_template.jinja`.

## Resolved facts
- Ship quant: `artifacts/quest-30b-rl-Q5_K_M.gguf` = **20.23 GiB** (`03_quant_sizes.txt`); fits 30 GiB fully.
- `ngl_for(Q5)` → **99** (20719 MiB ≤ (30−3)·1024 = 27648 MiB), so `-ngl 99` offloads all 48 layers.
- `RUN_BIN` = the MoE-reorder speed tree (built & present):
  `$HOME/llama.cpp-sycl-moe-ready/build/bin/llama-server`
  (mul_mat_id Q6_K SYCL reorder + topk-moe router/expert-combine fusions + `-ub` prefill).
  These transfer to QUEST (qwen3_moe routing); the prior campaign's gated-delta-net decode fusion does NOT (QUEST = std GQA).
- Ship knobs (env.sh): `GGML_SYCL_DISABLE_DNN=1` (deterministic decode/native GEMM), `-ub 4096`, `-b 8192`.
- Template (`chat_template.jinja`): `--jinja` turns on BOTH `<think>` reasoning (the gen-prompt opens
  `<|im_start|>assistant\n<think>\n`) AND the Qwen `<tool_call>{"name":..,"arguments":..}</tool_call>` tool loop.
- EOS = `<|im_end|>` (151645). Card sampling: temp 0.7 agentic / temp 0 for verifiable. Generation needs a
  real `max_tokens` (model burns tokens inside `<think>`) — use ≥2048–4096, NOT env GEN_N=512.

## VRAM math (30 GiB B70, weights 20.23 GiB, KV q8_0, +~3 GiB compute graph @ -ub 4096, +~0.3 misc)
KV q8_0 = 51.0 KiB/token (48L × 4 KV-heads × 128 head_dim × 2(K+V) × 1.0625 B; q8_0 = 34 B / 32-block).

| ctx     | KV q8_0 | est. total | headroom |
|---------|---------|------------|----------|
| 32768   | 1.59 G  | ~25.1 G    | ~4.9 G   |
| 49152   | 2.39 G  | ~25.9 G    | ~4.1 G   |
| **65536** | **3.19 G** | **~26.7 G** | **~3.3 G** |
| 98304   | 4.78 G  | ~28.3 G    | ~1.7 G (tight) |
| 131072  | 6.38 G  | ~29.9 G    | ~0.1 G (OOM risk) |
| 262144  | 12.75 G | ~36.3 G    | over by 6.3 G (impossible w/ full weights) |

**Recommended `-c 65536`**: doubles the script default (32768), gives multi-step deep-research /
tool-response room, and keeps a real ~3.3 GiB cushion for compute-graph growth + fragmentation.
The compute graph (~3 GiB) is an estimate for `-ub 4096`; if a load probe shows a larger buffer,
65536 still has slack while 131072 would not.

### Context upgrade ladder (only if a longer agentic session is required)
- **131072** ctx: keep KV q8_0 only if you first drop `-ub 2048` (smaller compute buffer) AND confirm
  the load fits, OR switch KV to `q4_0` (long-ctx profile, env comment) → KV halves to ~3.2 G.
- **262144** (full train ctx): requires KV `q4_0` (~6.4 G) **and** `-ub 2048` and trimmed headroom;
  do not use as the default production serve.

## Serve command (Q5_K_M, :8093, production agentic)
Prereq: oneAPI runtime on the linker path (binaries link `libsvml.so`). Either `source env.sh`
(does setvars + exports DISABLE_DNN/UBATCH/BBATCH/PORT) or `source /opt/intel/oneapi/setvars.sh` first.

```bash
GGML_SYCL_DISABLE_DNN=1 $HOME/llama.cpp-sycl-moe-ready/build/bin/llama-server \
  -m $HOME/quest-turbo/artifacts/quest-30b-rl-Q5_K_M.gguf \
  -ngl 99 \
  --host 0.0.0.0 --port 8093 \
  -c 65536 \
  -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -ub 4096 -b 8192 \
  --jinja \
  --alias quest-30b-rl-turbo
```

Convenient equivalent via the committed wrapper (sets env + ngl_for automatically):
```bash
cd $HOME/quest-turbo && QUANT=Q5_K_M CTX=65536 KV=q8_0 bash scripts/06_serve.sh
```

Notes:
- `-fa on` + KV `q8_0/q8_0` is a supported FA combo (K and V types match). q8_0 protects `<think>`
  traces + tool-call JSON vs q4_0.
- The build's default `--reasoning-format auto` splits the `<think>` trace into
  `message.reasoning_content` and leaves the final answer in `message.content`. To keep the raw
  `<think>…</think>` inline in `content`, add `--reasoning-format none`. Either way `content` is non-empty.
- For agentic defaults you may pass per-request `temperature`/`top_p` (card: ~0.7); the smoke tests
  below use `temperature:0` for deterministic routing/answers.

## Smoke tests (curl → /v1/chat/completions on :8093)
Pre-flight: `curl -sf http://127.0.0.1:8093/health` → `{"status":"ok"}`.

### 1) Tool-call routing + enum + parseable JSON (the canary)
```bash
curl -s http://127.0.0.1:8093/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"quest-30b-rl-turbo",
  "messages":[{"role":"user","content":"What is the weather in Paris right now, in celsius? Use the weather tool."}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a location.","parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","unit"]}}}],
  "tool_choice":"auto","temperature":0,"max_tokens":4096
}' | jq '.choices[0].message.tool_calls'
```
PASS: `tool_calls` is non-null; `[0].function.name == "get_weather"`; `[0].function.arguments`
parses as JSON with `location` ~ "Paris" and `unit == "celsius"` (right tool, enum-constrained arg, valid JSON).

### 2) Full tool round-trip end-to-end (assistant → tool → final answer)
```bash
curl -s http://127.0.0.1:8093/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"quest-30b-rl-turbo",
  "messages":[
    {"role":"user","content":"What is 17 multiplied by 23? Use the calculator tool."},
    {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"calculator","arguments":"{\"expression\":\"17*23\"}"}}]},
    {"role":"tool","tool_call_id":"call_1","content":"391"}
  ],
  "tools":[{"type":"function","function":{"name":"calculator","description":"Evaluate an arithmetic expression.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}],
  "temperature":0,"max_tokens":4096
}' | jq -r '.choices[0].message.content'
```
PASS: non-empty natural-language reply that states **391** — proves the template ingests the prior
`tool_calls` + `tool` response and closes the loop with a final answer (not another tool call, not empty).

### 3) `<think>` reasoning yields a real, non-empty answer
```bash
curl -s http://127.0.0.1:8093/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"quest-30b-rl-turbo",
  "messages":[{"role":"user","content":"A store had 240 apples. It sold 1/3 on Monday, then 1/4 of the remaining on Tuesday. How many are left?\n\nThink step by step, then end with a line in exactly this form:\nANSWER: <number>"}],
  "temperature":0,"max_tokens":4096
}' | jq -r '.choices[0].message | "content=\(.content)\n---\nreasoning=\((.reasoning_content // "n/a"))"'
```
PASS: `content` is non-empty and contains **120** (and ends `ANSWER: 120`). With `--reasoning-format auto`
the `<think>` trace shows up under `reasoning_content`; with `--reasoning-format none` it is inline in
`content`. Either way: real reasoning + non-empty final answer (no empty/`""` content, no truncation —
hence the 4096-token budget, since QUEST spends tokens inside `<think>`).

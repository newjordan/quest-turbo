# Q5_K_M SHIP ACTION PLAN — QUEST-30B-RL Turbo (B70)

Apply-order plan for the engineer to run BETWEEN GPU phases (04 PPL [live now] -> 04b gate -> 05 speed-tune -> 06 serve).
All steps below are READ/EDIT only — no GPU job runs while authoring. Sources verified on disk: scripts/04_accuracy.sh,
04b_agentic_gate.sh, 05_speed_tune.sh, 04b_probes.py, 06_serve.sh, env.sh.

================================================================================
## (1) SCRIPT FIXES — apply BEFORE the phase that runs the script (verified real bugs)
================================================================================

### FIX-A  [HIGH]  scripts/04_accuracy.sh lines 29-31 — PPL pipeline aborts after BF16, writes nothing
PPL "Final estimate" is on **stderr** (perplexity.cpp:654 LOG_INF), but the loop does `2>/dev/null` then
`grep -iE "final|perplexity"` -> matches nothing -> exits 1. Under `set -euo pipefail` (line 6) pipefail
makes the whole script abort right after the FIRST (slowest, partial-offload) BF16 run. 04_ppl.txt ends empty,
no quant PPL computed, parity section never runs. This is the phase running on the GPU RIGHT NOW — it will
produce nothing. Fix and RE-RUN (or salvage: 04_ppl_Q5.txt already exists with Q5=7.19).

Replace lines 29-31 with:
```bash
    echo "-- $q"; "$RUN_BIN/llama-perplexity" -m "$m" -ngl "$(ngl_for "$m")" -f "$PPL_CORPUS" \
      ${PPL_CHUNKS:+--chunks "$PPL_CHUNKS"} \
      2>&1 | grep -iE "Final estimate|PPL =" | sed "s/^/$q: /" | tee -a "$RES/04_ppl.txt" || true
```
Three changes ALL required: `2>/dev/null`->`2>&1` (brings the result line into the pipe), tighten pattern to
`Final estimate|PPL =` (since 2>&1 now also pulls the LOG_INF "calculating perplexity over N chunks" noise that
the loose `perplexity` pattern would falsely match), and `|| true` (neutralize set -e if grep still misses).
DEADLINE MITIGATION: BF16 is listed first and runs at forced partial offload over 60 chunks = the slowest item
yet is only a gate reference. Drop BF16 from the PPL list or lower PPL_CHUNKS so the shipped Q5_K_M PPL lands
first inside the 2h budget.

### FIX-B  [MED]  scripts/04b_agentic_gate.sh line 39 — timeout branch leaks VRAM, cascades the gate
Timeout branch does `kill $SRV 2>/dev/null||true; continue` with NO reap. The SIGTERM'd server is still
releasing the 30 GiB B70 when `continue` immediately launches the next model -> OOM -> health fail -> cascade.
Worst case the ship quant Q5_K_M (4th in MODELS, after slow BF16) is spuriously OOM-failed = false negative on
the exact gate that picks it. Replace line 39 with reap + grace + hard-kill (bare `wait` alone can hang forever
on a wedged SYCL load, blowing the deadline — escalation is mandatory, not optional):
```bash
  if [[ "$ok" == 0 ]]; then
    echo " TIMEOUT — see $RES/04b_server_${label}.log"
    kill $SRV 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
    kill -9 $SRV 2>/dev/null || true
    wait $SRV 2>/dev/null || true
    continue
  fi
```
(Optional: apply the same SIGTERM->grace->SIGKILL pattern to the normal-exit branch line 42.)

### FIX-C  [MED]  scripts/05_speed_tune.sh lines 15-16 — sweep can't see the knob it exists for
`-p 512` with `-ub 512,2048,4096`: a 512-token prefill is one ubatch regardless of -ub, so the "-ub 4096
un-starves MoE expert GEMMs" prefill thesis (the whole point) is structurally invisible. Worse, n_ubatch is
silently clamped to n_batch (llama-context.cpp:196), so `-b 2048 -ub 4096` actually runs ub=2048 but llama-bench
REPORTS 4096 -> false rows. Replace lines 15-16 with (prefill length >= max ub, b fixed >= max ub):
```bash
"$BIN/llama-bench" -m "$M" -ngl "$QNGL" -p 4096 -n 128 -r 3 \
  -b 8192 -ub 1024,2048,4096 | tee "$RES/05_batch.txt"
```
With -p 4096, ub=1024/2048/4096 -> 4/2/1 ubatches (knob now visible); -b 8192 >= max ub (no clamp, no mislabels).
Do NOT keep -p 512 or -b 2048.

================================================================================
## (2) 04b_probes.py PROBE ADDITIONS — DESIGN/PASTE (apply before 04b gate)
================================================================================
Gate axes today: 5 TOOL + 10 REASON, scored vs key AND vs BF16 agreement. PPL is FLAT (Q5 7.19 vs BF16 calib
7.44) and imatrix left 42/144 expert tensors at 95-99% coverage (possible tool/<think> specialists) -> PPL is
blind to the agentic cliff; the gate is the decisive instrument. Each new probe costs ~1 BF16 + ~1 Q5 gen.

INCLUDE (under deadline): nested_enum, abstain, search_then_calc, par_weather.  DROP: crowded_route (overlaps
existing calc_route + calc_vs_search2; the only new stressor is the crowded namespace — lowest value/gen).
Minimal strong core if time collapses = nested_enum + abstain + search_then_calc.  ALSO include the think-integrity
check (ZERO extra gen cost — reuses the 10 existing REASON generations).

Pastes are pure-value (nested_enum, search_then_calc, crowded_route) EXCEPT abstain + par_weather, which need the
collect-hook below; without the hook abstain CANNOT pass (harness forces fail on bad=='no-tool-call') and par_
cannot require >=2 calls (arg_check only sees tool_calls[0]).

### 2a. New tool schema — paste after T_EVENT (after line 45):
```python
T_ORDER = {"type": "function", "function": {
    "name": "create_order", "description": "Place a purchase order.",
    "parameters": {"type": "object", "properties": {
        "priority": {"type": "string", "enum": ["low", "normal", "high"]},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "sku": {"type": "string"}, "qty": {"type": "integer"}},
            "required": ["sku", "qty"]}},
        "shipping": {"type": "object", "properties": {
            "method": {"type": "string", "enum": ["standard", "express"]},
            "address": {"type": "string"}},
            "required": ["method", "address"]}},
        "required": ["priority", "items", "shipping"]}}}
```

### 2b. New probe tuples — paste into TOOL_PROBES before the closing `]` (line 75):
```python
    ("nested_enum", [T_ORDER],
     "Place a HIGH priority order: 2 units of SKU A100 and 3 units of SKU B200, "
     "shipped express to '5 Main St'. Use the tool.",
     "create_order", lambda a: a.get("priority") == "high"
        and isinstance(a.get("items"), list) and len(a.get("items", [])) >= 2
        and all(isinstance(it, dict) and "sku" in it and "qty" in it for it in a.get("items", []))
        and isinstance(a.get("shipping"), dict)
        and a.get("shipping", {}).get("method") == "express"
        and _has(str(a.get("shipping", {}).get("address", "")), "main"),
     "deeply-nested JSON (array-of-objects + nested object + 2 enums) without corruption"),
    ("abstain", [T_CALC, T_SEARCH, T_WEATHER, T_EVENT],
     "Translate the phrase 'good morning' into French.",
     "__none__", lambda a: True,
     "no offered tool fits -> answers directly with NO tool call (catches over-calling)"),
    ("search_then_calc", [T_SEARCH, T_CALC],
     "What is the current population of Tokyo divided by 1000? Use tools.",
     "web_search", lambda a: _has(a.get("query", ""), "tokyo") and _has(a.get("query", ""), "pop"),
     "research before computing -> first call is web_search, not calculator on a guessed number"),
    ("par_weather", [T_WEATHER],
     "Compare the weather in Paris and Tokyo right now, both in celsius. Use the weather tool.",
     "get_weather", lambda a: (_has(a.get("location", ""), "paris") or _has(a.get("location", ""), "tokyo"))
        and a.get("unit") == "celsius",
     "TWO well-formed parallel get_weather calls (pid prefix par_ => >=2 calls required)"),
```
(crowded_route — DROPPED under deadline; keep this tuple on hand to add if time remains:)
```python
    # ("crowded_route", [T_SEARCH, T_CALC, T_WEATHER, T_EVENT, T_ORDER],
    #  "Across our 5 regional offices we each have 38 desks. What is the total number of desks? Use a tool.",
    #  "calculator", lambda a: "38" in str(a.get("expression", "")) and "5" in str(a.get("expression", "")),
    #  "routes prose-wrapped arithmetic to calculator (not web_search) among 5 tools"),
```

### 2c. Collect hook — REPLACE the TOOL-probe loop body in cmd_collect (lines 153-173) with:
```python
    for pid, tools, msg, want_fn, chk, why in TOOL_PROBES:
        try:
            m = ask_tool(port, tools, msg, maxtok)
            name, args, bad = extract_tool_call(m)
            n_calls = len(m.get("tool_calls") or [])
            content = m.get("content") or ""
            routed = (name == want_fn)
            args_ok = False
            try:
                args_ok = bool(chk(args)) if not bad else False
            except Exception:
                args_ok = False
            if want_fn == "__none__":                       # abstain: correct = NO call + real text answer
                ok = (bad == "no-tool-call") and bool(content.strip())
            elif pid.startswith("par_"):                    # parallel: >=2 calls AND first call well-formed
                ok = (n_calls >= 2) and routed and args_ok and not bad
            else:
                ok = routed and args_ok and not bad
            rec = {"want_fn": want_fn, "got_fn": name, "args": args, "malformed": bad,
                   "routed": routed, "args_ok": args_ok, "n_calls": n_calls, "pass": ok}
        except Exception as e:
            rec = {"want_fn": want_fn, "got_fn": "", "args": {}, "malformed": f"__ERROR__ {e}",
                   "routed": False, "args_ok": False, "n_calls": 0, "pass": False}
        out["tool"][pid] = rec
        tag = "ok  " if rec["pass"] else "FAIL"
        print(f"    [{tag}] {pid:16s} want={want_fn:13s} got={rec['got_fn'] or '-':13s} "
              f"{'' if rec['pass'] else '<' + (rec['malformed'] or ('wrong-fn' if not rec['routed'] else 'bad-args')) + '>'}",
              flush=True)
```
Note: `bad == "no-tool-call"` already implies n_calls==0 and no inline blob (extract_tool_call). A well-formed
inline call sets name and bad=="" so abstain still fails (correct). A garbled inline blob sets
bad=="unparseable-inline" so abstain fails (correct — it attempted a malformed call).

### 2d. <think> integrity — ZERO extra gen cost (reuses the 10 REASON gens):
Add helper near extract_ans:
```python
def think_struct_ok(reply):
    r = reply or ""
    o, c = r.count("<think>"), r.count("</think>")
    if o == 1 and c == 1 and r.index("<think>") < r.index("</think>"):
        tail = r.rsplit("</think>", 1)[-1]
        return bool(tail.strip()) and "ANSWER:" in tail.upper()
    if o == 0 and c == 0:                                   # stripping --reasoning-format: tags must NOT leak, answer must still follow
        return "ANSWER:" in (r or "").upper()
    return False                                            # unbalanced / duplicated / leaked / unclosed
```
In cmd_collect REASON loop (lines 175-182) capture the raw reply and store think_ok:
```python
    for pid, q, want in REASON_PROBES:
        try:
            reply = ask_text(port, q, maxtok)
            ans, err = extract_ans(reply), ""
        except Exception as e:
            reply, ans, err = "", "", f"__ERROR__ {e}"
        rec = {"want": want, "ans": ans, "norm": norm(ans) if ans else "",
               "raw": reply, "think_ok": think_struct_ok(reply), "err": err}
        out["reason"][pid] = rec
```
In cmd_grade add a THINK_ok column (header + per-row):
```python
    print(f"\n{'model':10s} {'TOOL_key':>9s} {'TOOLagree':>10s} {'REAS_key':>9s} {'REASagree':>10s} {'THINK_ok':>9s}")
    ...
        th = sum(1 for v in r.values() if v.get("think_ok"))
        print(f"{lab:10s} {tk:>4d}/{nt:<3d} {ta:>5d}/{nt:<3d} {rk:>4d}/{nr:<3d} {ra:>5d}/{nr:<3d} {th:>4d}/{nr:<3d}")
```
CAVEAT: the gate server runs `--jinja` with no explicit `--reasoning-format`. With `auto` the <think> trace lands
in reasoning_content (not content) -> o==c==0 branch applies (answer must still contain ANSWER:). With `none` it is
inline -> single-pair branch. Both covered. If THINK_ok is 0/10 on BF16 the rig is mis-templated, not the model.

After additions: TOOL total nt=9 (5 existing + 4 new), REASON 10, THINK over the 10 REASON gens. If a paste fails
to land, substitute the ACTUAL nt into the thresholds in section 5.

================================================================================
## (3) PHASE 05 SPEED-TUNE — bench commands in run-order (after FIX-C); ~35 min total GPU
================================================================================
$M = artifacts/quest-30b-rl-Q5_K_M.gguf ; $MOE_BIN == $RUN_BIN (MoE-reorder tree). f16 KV in B0/B1 to match
the 1154.6/79.9 baseline; switch to q8_0 KV only where noted. Order = max value-if-interrupted under 2h clock.

B0  anchor (reproduce/beat target; locks the reference):
    $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3
B1  -ub sweep, prefill only (biggest pp512 lever; QUEST=128E may saturate before the ref's 256E):
    $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -b 8192 -ub 512,2048,4096 -p 512,2048 -n 0 -r 3
B2  flash-attn on/off (keep fa1: required for q8_0 KV):
    $RUN_BIN/llama-bench -m $M -ngl 99 -ub 4096 -b 8192 -fa 0,1 -p 512 -n 128 -r 3
B3  KV q8_0 vs f16 @ d=0 (VEC vs TILE decode kernel; is q8_0 free?):
    $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -r 3
B5  build A/B (confirm MoE-tree +16%pp/+6.8%tg transfers to QUEST; if not, every later row is on wrong binary):
    $BIN/llama-bench     -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3
    $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3
B4  decode at depth + KV type (the real agentic payoff; most expensive -> after cheap flags locked):
    $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -ctk q8_0 -ctv q8_0 -n 128 -d 0,16384,32768,65536 -r 2
    $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -ctk f16  -ctv f16  -n 128 -d 0,16384,32768,65536 -r 2
    (f16 caps at 65536; q8_0 may add -d 131072.)
Speculative, only if time remains (B6 oneDNN prefill, B7 topk-moe A/B, B8 SYCL graph, B9 -t, B10 -b):
B6  GGML_SYCL_DISABLE_DNN=0 $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512,2048 -n 128 -r 3
B7  GGML_SYCL_DISABLE_TOPK_MOE=1 $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3
B8  GGML_SYCL_DISABLE_GRAPH=0 $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -n 128 -d 0,16384 -r 2
B9  $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -t 8,16,32 -p 512 -n 128 -r 2
B10 $RUN_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 4096,8192 -p 2048 -n 0 -r 3

STEPS B0,B1,B2,B3,B5 alone fully determine the ship flags.
RECOMMENDED FINAL FLAGS:
  Headline/bench:  $MOE_BIN/llama-bench -m $M -ngl 99 -fa 1 -ub 4096 -b 8192 -p 512 -n 128 -r 3
  Ship binary = MoE-reorder tree ($MOE_BIN == $RUN_BIN); GGML_SYCL_DISABLE_DNN=1; topk-moe router fusion
  default-ON; mul_mat_id Q5_K reorder auto-on (confirm 'reorder: Y' in load log); -t 16 (irrelevant at full
  offload). Revisit -ub only if B1 shows 2048 wins on 128E; revisit DNN=0 only if B6 shows a clean pp win with
  tg unharmed.

================================================================================
## (4) PHASE 06 SERVE + smoke tests
================================================================================
PREREQ: `source $HOME/quest-turbo/env.sh` first (oneAPI setvars; binaries link libsvml.so).
SERVE (long agentic ctx):
```
GGML_SYCL_DISABLE_DNN=1 $HOME/llama.cpp-sycl-moe-ready/build/bin/llama-server \
  -m $HOME/quest-turbo/artifacts/quest-30b-rl-Q5_K_M.gguf \
  -ngl 99 --host 0.0.0.0 --port 8093 -c 65536 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 -ub 4096 -b 8192 --jinja --alias quest-30b-rl-turbo
```
Wrapper equivalent: `cd $HOME/quest-turbo && QUANT=Q5_K_M CTX=65536 KV=q8_0 bash scripts/06_serve.sh`
VRAM @ -c 65536: 20.23 (weights) + 3.19 (KV q8_0) + ~3 (compute graph @ -ub4096) + ~0.3 = ~26.7 GiB / 30 -> ~3.3
GiB headroom. 131072 is OOM-risk (~29.9 GiB); needs -ub 2048 or KV q4_0. 262144 needs KV q4_0 + -ub 2048.
CONFIRM the actual SYCL compute buffer at first load before going past 65536 (the ~3 GiB is an estimate).

SMOKE TESTS (temp 0 for determinism; live agentic default is ~0.7; max_tokens 4096 — model burns tokens in <think>):
S1 TOOL ROUTING canary -> get_weather, location~Paris, unit=="celsius", arguments parse as JSON:
   curl -s :8093/v1/chat/completions -d '{"model":"quest-30b-rl-turbo","messages":[{"role":"user","content":"What is the weather in Paris right now, in celsius? Use the weather tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a location.","parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","unit"]}}}],"tool_choice":"auto","temperature":0,"max_tokens":4096}' -H 'Content-Type: application/json' | jq '.choices[0].message.tool_calls'
S2 TOOL ROUND-TRIP -> non-empty NL reply stating 391 (proves jinja ingests prior tool_calls + tool response):
   feed back assistant tool_call calculator{"expression":"17*23"} + tool role content "391"; PASS if reply states 391 and is not another tool call/empty.
S3 THINK non-empty -> content/reasoning_content non-empty, ends "ANSWER: 120":
   "A store had 240 apples. It sold 1/3 Monday, then 1/4 of the remaining Tuesday. How many left? ... ANSWER: <number>", temp 0, max_tokens 4096.
   (auto -> trace in reasoning_content; none -> inline in content.)

================================================================================
## (5) OPERATIONAL PASS THRESHOLDS — Q5 ship decision (2-model gate: Q5_K_M vs BF16)
================================================================================
After the 4 additions: TOOL nt=9, REASON=10, THINK over the 10 REASON gens. K=key-accuracy, A=BF16 mutual-pass
agreement. SHIP Q5 iff ALL aggregate tolerances hold AND no hard-fail fires.

SANITY GUARD (run FIRST): if BF16 scores <8/9 TOOL or <10/10 THINK, the rig (template / --reasoning-format /
tool schema) is broken — fix the rig before trusting ANY Q5 number. Expected BF16 ~ TOOL 9/9, REASON 9-10/10,
THINK 10/10.

AGGREGATE (TOOL strict, REASON looser):
  TOOL_key(Q5)   >= TOOL_key(BF16) - 1
  TOOL_agree(Q5) >= nt - 1                 (disagree with BF16 on <=1 tool probe)
  REAS_key(Q5)   >= REAS_key(BF16) - 2
  REAS_agree(Q5) >= 10 - 2
  THINK_ok(BF16) == 10/10  AND  THINK_ok(Q5) == THINK_ok(BF16)

HARD FAILS (any one BLOCKS ship regardless of counts):
  H1  Any NEW malformed/unparseable tool-call JSON on Q5 for a probe BF16 got clean
      (rec['malformed'] nonempty where BF16's was empty) — JSON corruption is the exact cliff.
  H2  Regression on a critical probe BF16 passed in {nested_enum, abstain, par_weather}.
  H3  Any <think> structural regression (Q5 think_ok < BF16 on any probe).

CONCRETE PASS LINE (with nt=9): SHIP Q5 iff TOOL_key(Q5) >= 8/9, TOOL_agree >= 8/9, REAS_key(Q5) >= 8/10,
REAS_agree >= 8/10, THINK_ok(Q5) == 10/10, and NO H1/H2/H3.

IF Q5 FAILS: widen the calib corpus with tool-call + <think> traces, re-run Phase 2 imatrix, re-quant Q5,
re-gate (per results/02_coverage_summary.txt remediation). Do NOT lock the serve config in section 4 as the ship
config until Q5_K_M clears this gate — PPL(Q5)=7.19 vs BF16 calib 7.44 is promising but PPL is blind to tool-call /
reasoning collapse.

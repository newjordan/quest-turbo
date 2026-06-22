# Gate hardening spec — 04b agentic ship-decision gate (Q5_K_M vs BF16)

Status: DESIGN ONLY (GPU busy with live PPL run; 2h deadline). No GPU/heavy jobs run.
Author target: the engineer who will paste probes into `scripts/04b_probes.py` and re-run `04b_agentic_gate.sh`.
Scope of the decision: we ship EXACTLY ONE quant (Q5_K_M, 20.23 GB, fits B70 at -ngl 99). BF16 (57 GB, -ngl ~20 partial) is the reference only. So the gate that matters is a **2-model gate: Q5_K_M vs BF16**.

---

## 1. Why this gate is decisive (evidence)

- Q5_K_M PPL = **7.19** (`results/04_ppl_Q5.txt`); BF16 calib PPL = **7.44** (`results/02_coverage_summary.txt`). PPL is essentially FLAT (Q5 even nominally lower on the eval corpus). **PPL is blind to tool/`<think>` damage** — confirmed, not hypothetical.
- imatrix coverage (`results/02_coverage_summary.txt`): NO dead experts, but **42/144 expert tensors across 14/48 layers have 1-6 uncovered long-tail experts (95.31-99.22%)** from a *generic English* corpus. The documented risk: these uncovered specialists may be the **tool-call / `<think>` format experts**. If so, damage appears ONLY as behavioral drift — exactly what this gate must catch.
- QUEST's value is agentic deep-research (GAIA 69.0 / LiveResearchBench 74.1 / DeepResearch 45.3). The behavior behind those benches = tool routing + well-formed `<tool_call>` JSON + multi-step planning + `<think>` discipline.

## 2. Current coverage assessment (what 04b already tests, and the gaps)

Current set = **5 TOOL probes + 10 REASON probes**, scored vs hand key AND vs BF16 agreement.

TOOL (5): `calc_route`, `search_route` (2-tool binary routing), `weather_enum` (1 enum field), `nested_json` (flat array-of-strings), `calc_vs_search2` (binary routing). 
REASON (10): verifiable single-answer math (`apples`, `xy`, `strings`, `shirt`, `taps`, `avgspeed`, `snail`, `div3or5`, `overtime`, `fatherson`).

**Strengths:** routing canary present; one enum field; one shallow nested arg; reasoning axis is well-covered (10 verifiable problems, one-slip-fails).

**Gaps vs QUEST's defining behaviors (what an RL deep-research agent does that PPL + current set miss):**
1. **Tool routing in a CROWDED namespace** — current routing is only 2-tool binary. Real agents juggle many tools; the cliff shows as a near-miss pick among 5+.
2. **Deep JSON structural integrity** — `nested_json` is only a flat string array. No array-of-objects, no nested object, no second enum. This is the most direct test of the suspect "tool-format experts."
3. **Abstention** — choosing NOT to call a tool when none fits. RL/quant degradation classically manifests as *over-calling* (forcing an irrelevant tool) or hallucinating. Completely untested today.
4. **Sequential / ordering judgment** — research-before-compute decomposition (deep-research planning). Untested.
5. **Parallel / multiple tool calls** — emitting >1 call in one turn; the harness today only inspects `tool_calls[0]`. Untested.
6. **`<think>` structural integrity** — `extract_ans` *consumes* `</think>` but never *verifies* it. A quant that fails to open/close `<think>` breaks downstream parsing silently. Untested as a graded axis.

**Harness facts that constrain probe design (verified by reading 04b_probes.py):**
- TOOL tuple = `(id, [tools], user_msg, expected_fn, arg_check(args_dict)->bool, why)`.
- `extract_tool_call` only parses **`tool_calls[0]`** (or the FIRST inline `<tool_call>`). `arg_check` receives only that first call's parsed args dict — it cannot see call count or message content.
- `pass = routed and args_ok and not bad`. There is **no way to score "no call" as a pass** today (`bad="no-tool-call"` forces fail), and **no way to require N>=2 calls**. So abstention and parallel need a small, surgical collect/grade hook (specified in section 5). All other proposed probes are **pure paste**.
- REASON collect stores only `ans`/`norm`, NOT the raw reply — so the `<think>` check needs collect to also store the raw reply (section 4).

## 3. New tool schemas to add (next to the existing `T_*` constants)

Two proposed probes need richer schemas. Paste these alongside `T_CALC/T_SEARCH/T_WEATHER/T_EVENT`:

```python
T_ORDER = {"type": "function", "function": {
    "name": "create_order", "description": "Place a purchase order.",
    "parameters": {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "sku": {"type": "string"}, "qty": {"type": "integer"}},
            "required": ["sku", "qty"]}},
        "shipping": {"type": "object", "properties": {
            "method": {"type": "string", "enum": ["standard", "express", "overnight"]},
            "address": {"type": "string"}}, "required": ["method", "address"]},
        "priority": {"type": "string", "enum": ["low", "normal", "high"]}},
        "required": ["items", "shipping", "priority"]}}}
T_TRANSLATE_ABSENT = None  # (no translate tool exists — that is the point of the abstain probe)
```

`crowded_route`, `abstain`, `search_then_calc`, `par_weather` reuse only existing tools (+ T_ORDER as a crowd distractor). Only `nested_enum` and `crowded_route` reference `T_ORDER`.

## 4. `<think>` structural-integrity check (separate axis — ZERO extra generation cost)

Reuses the existing 10 REASON generations (no new prompts). Two-line collect change + a grader column.

(a) In `cmd_collect`, store the raw reply on each reason rec:
```python
reply = ask_text(port, q, maxtok)          # was inlined into extract_ans
ans = extract_ans(reply)
rec = {"want": want, "ans": ans, "norm": norm(ans) if ans else "",
       "raw": reply, "think_ok": think_struct_ok(reply), "err": err}
```
(b) Add the checker (handles the inline-tags case the current `extract_ans` already assumes; falls back to `reasoning_content` if the server is run with a stripping `--reasoning-format`):
```python
def think_struct_ok(reply):
    r = reply or ""
    o, c = r.count("<think>"), r.count("</think>")
    if o == 1 and c == 1 and r.index("<think>") < r.index("</think>"):
        tail = r.rsplit("</think>", 1)[-1]
        return bool(tail.strip()) and "ANSWER:" in tail.upper()   # closed think, then a real answer
    if o == 0 and c == 0:        # server stripped tags into reasoning_content -> tags must NOT leak,
        return "ANSWER:" in (r or "").upper()                     # and an answer must still follow
    return False                 # unbalanced/duplicated/un-closed/leaked think = STRUCTURAL FAIL
```
(c) In `cmd_grade`, add a `THINK_ok` column: `sum(1 for v in r.values() if v.get("think_ok"))` per model.

Integrity rule: this is BINARY format discipline, so it is a **hard** sub-gate (section 6): BF16 must be 10/10 and Q5 must equal BF16 (no regressions). One dropped/un-closed `<think>` silently corrupts answer extraction and, in production, tool-call parsing.

## 5. Minimal collect/grade hook for abstention + parallel (the only non-paste change)

Replace the `ok`/`rec` computation inside the TOOL loop of `cmd_collect` with:
```python
n_calls = len(m.get("tool_calls") or [])
if want_fn == "__none__":                                  # ABSTENTION: correct = answer, no call
    ok = (n_calls == 0 and not (bad or "").startswith("unparseable")
          and bool((m.get("content") or "").strip()))
    routed = args_ok = ok
    name = name or "(none)"
else:
    routed = (name == want_fn)
    args_ok = bool(chk(args)) if not bad else False
    min_calls = 2 if pid.startswith("par_") else 1         # PARALLEL: pid prefix 'par_' requires >=2
    ok = routed and args_ok and (not bad) and n_calls >= min_calls
rec = {"want_fn": want_fn, "got_fn": name, "args": args, "malformed": bad,
       "n_calls": n_calls, "routed": routed, "args_ok": args_ok, "pass": ok}
```
This keeps `rec["pass"]` semantics intact, so `cmd_grade`'s key-accuracy AND BF16-agreement logic work unchanged. Convention: `want_fn == "__none__"` => abstention; `pid` starts with `par_` => parallel (>=2 calls). NOTE: parallel relies on the structured `tool_calls` array (the OpenAI endpoint + `--jinja` provides it for Qwen3); the inline `<tool_call>` fallback only yields the first call.

## 6. Proposed probe additions (ranked; add in order under deadline)

Order = signal x ease. Minimal strong core = #1-#3. Trim from the bottom (#5 first) if BF16 partial-offload time runs short. Each probe = ONE BF16 generation + ONE Q5 generation.

1. **nested_enum** (TOOL, PURE PASTE) — array-of-objects + nested object + two enums. Most direct test of the suspect tool-format experts; the cliff = corrupted/flattened nested JSON.
2. **abstain** (TOOL, needs sec-5 hook) — none of the offered tools fits; correct = answer directly, NO call. Catches RL/quant over-calling. Highest unique behavioral signal.
3. **search_then_calc** (TOOL, PURE PASTE) — must research BEFORE computing; correct FIRST action is `web_search`, not a calculator call on a hallucinated number. Tests deep-research ordering in one turn.
4. **par_weather** (TOOL, needs sec-5 hook) — two cities in one turn => >=2 well-formed `get_weather` calls. Tests parallel multi-call; cliff = collapsing to one call or corrupting the second.
5. **crowded_route** (TOOL, PURE PASTE) — correct tool is `calculator` but the prose-heavy prompt lures `web_search`, among 5 tools. Routing discrimination in a crowded namespace. (Most overlap with existing routing probes => trim first.)

Exact tuples (paste into `TOOL_PROBES`):

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
     "emits deeply-nested JSON (array-of-objects + nested object + 2 enums) without corruption"),
    ("abstain", [T_CALC, T_SEARCH, T_WEATHER, T_EVENT],
     "Translate the phrase 'good morning' into French.",
     "__none__", lambda a: True,
     "no offered tool fits -> answers directly with NO tool call (catches over-calling)"),
    ("search_then_calc", [T_SEARCH, T_CALC],
     "What is the current population of Tokyo divided by 1000? Use tools.",
     "web_search", lambda a: _has(a.get("query", ""), "tokyo") and _has(a.get("query", ""), "pop"),
     "must research before computing -> first call is web_search, not calculator on a guessed number"),
    ("par_weather", [T_WEATHER],
     "Compare the weather in Paris and Tokyo right now, both in celsius. Use the weather tool.",
     "get_weather", lambda a: (_has(a.get("location", ""), "paris") or _has(a.get("location", ""), "tokyo"))
        and a.get("unit") == "celsius",
     "emits TWO well-formed parallel get_weather calls (pid prefix par_ => >=2 calls required)"),
    ("crowded_route", [T_SEARCH, T_CALC, T_WEATHER, T_EVENT, T_ORDER],
     "Across our 5 regional offices we each have 38 desks. What is the total number of desks? Use a tool.",
     "calculator", lambda a: "38" in str(a.get("expression", "")) and "5" in str(a.get("expression", "")),
     "routes a prose-wrapped arithmetic task to calculator (not web_search) among 5 tools"),
```

## 7. Ship rule operationalized — concrete pass numbers (2-model gate)

After all additions: TOOL total = **10** (5 existing + 5 new), REASON total = **10**, THINK total = **10** (over the REASON probes). If fewer additions land, substitute the actual TOOL total `nt`.

Let `K(m)` = key-accuracy, `A(m)` = BF16-agreement (mutual pass) for axis on model `m`.

SHIP Q5_K_M iff ALL of the following hold:

Aggregate tolerances ("within ~1-2 of BF16"):
- **TOOL_key(Q5) >= TOOL_key(BF16) - 1**  (tool axis is strict: it is QUEST's value + the imatrix risk)
- **TOOL_agree(Q5) >= TOOL_total - 1**    (Q5 disagrees with BF16 on at most 1 tool probe)
- **REAS_key(Q5)  >= REAS_key(BF16) - 2** (reasoning is noisier + secondary to the mission)
- **REAS_agree(Q5) >= REAS_total - 2**
- **THINK_ok(BF16) == 10/10  AND  THINK_ok(Q5) == THINK_ok(BF16)** (binary discipline, hard)

HARD FAILS — any single one blocks ship regardless of the counts above:
- (H1) Any **new malformed/unparseable** tool-call JSON on Q5 for a probe BF16 emitted clean (`rec["malformed"]` non-empty where BF16's was empty). JSON corruption is the exact cliff this gate exists to catch.
- (H2) Regression on a **critical probe** BF16 passed, in {`nested_enum`, `abstain`, `par_weather`} (deep-JSON integrity, abstention, parallel — the behaviors with no slack).
- (H3) Any **`<think>` structural regression** (Q5 `think_ok` < BF16 on any probe).

Concrete instantiation (expected BF16 ~ TOOL 10/10, REASON 9-10/10, THINK 10/10):
- Ship iff: **TOOL_key(Q5) >= 9/10**, **TOOL_agree >= 9/10**, **REAS_key(Q5) >= 8/10**, **REAS_agree >= 8/10**, **THINK_ok(Q5) == 10/10**, and **no H1/H2/H3 trigger**.

If Q5 FAILS: per `02_coverage_summary.txt` remediation — widen the calib corpus with agentic/tool-call + `<think>` traces, re-run Phase 2 imatrix, re-quant Q5_K_M, re-gate. Do NOT ship a quant that fails any hard sub-gate.

Sanity guard: if BF16 itself scores below TOOL 8/10 or THINK 10/10, the harness/template/server config is suspect (e.g. wrong `--reasoning-format`, tool template) — fix the rig before trusting any Q5 number.

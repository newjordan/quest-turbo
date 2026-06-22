#!/usr/bin/env python3
"""Phase 4b — agentic + reasoning gate for QUEST-30B-RL (the ship-decision detector).

QUEST is an RL'd deep-research / tool-using model. Its quant damage shows up FIRST as *behavioral*
breakage — wrong tool routing, malformed tool-call JSON, lost format discipline — which perplexity
and even a pure-math gate miss entirely. So this gate grades two axes against a served model via the
OpenAI-compatible endpoint (--jinja enables both <think> and the tool-calling template):

  TOOL  — declare tools, ask a question that should trigger a SPECIFIC tool, and require the model to
          emit a well-formed call: right function name, JSON-parseable arguments, expected arg content.
          This is the canary: at the quant cliff, the model picks the wrong tool, emits unparseable
          JSON, or skips the call and hallucinates an answer.
  REASON— multi-step verifiable problems where one slip compounds to a wrong final answer.

Each axis is scored vs a hand-verified KEY and vs the BF16 reference's own answers (does the quant
still BEHAVE like the high-precision model). The ship quant is the smallest one that holds BOTH within
a couple points of BF16.

Usage:
  collect <port> <label> <outdir>          run all probes vs a running server -> <outdir>/<label>.json
  grade   <outdir> <ref_label> <labels...> print tool/reason key-accuracy + ref-agreement table
"""
import json, sys, re, urllib.request, os

# ---- tool schemas (OpenAI function format) --------------------------------
T_CALC = {"type": "function", "function": {
    "name": "calculator", "description": "Evaluate an arithmetic expression.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string", "description": "e.g. '17 * 23'"}},
        "required": ["expression"]}}}
T_SEARCH = {"type": "function", "function": {
    "name": "web_search", "description": "Search the web for current information.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}}, "required": ["query"]}}}
T_WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a location.",
    "parameters": {"type": "object", "properties": {
        "location": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["location", "unit"]}}}
T_EVENT = {"type": "function", "function": {
    "name": "create_event", "description": "Create a calendar event.",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
        "attendees": {"type": "array", "items": {"type": "string"}}},
        "required": ["title", "date", "attendees"]}}}
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

# Each tool probe: (id, tools, user_msg, expected_fn, arg_check(args_dict)->bool, why)
def _has(s, *subs):
    s = (s or "").lower()
    return all(x.lower() in s for x in subs)

TOOL_PROBES = [
    ("calc_route", [T_CALC, T_SEARCH],
     "Use a tool to compute 17 multiplied by 23.",
     "calculator", lambda a: _has(str(a.get("expression", "")), "17") and "23" in str(a.get("expression", "")),
     "picks calculator (not search) and passes both operands"),
    ("search_route", [T_CALC, T_SEARCH],
     "Find the current population of Tokyo using a tool.",
     "web_search", lambda a: _has(a.get("query", ""), "tokyo") and _has(a.get("query", ""), "pop"),
     "picks web_search and includes the subject in the query"),
    ("weather_enum", [T_WEATHER],
     "What's the weather in Paris right now, in celsius? Use the weather tool.",
     "get_weather", lambda a: _has(a.get("location", ""), "paris") and a.get("unit") == "celsius",
     "fills location AND the enum-constrained unit correctly"),
    ("nested_json", [T_EVENT],
     "Schedule a calendar event titled 'Q3 Review' on 2026-07-01 with attendees alice and bob.",
     "create_event", lambda a: _has(a.get("title", ""), "q3 review")
        and "2026-07-01" in str(a.get("date", ""))
        and isinstance(a.get("attendees"), list) and len(a.get("attendees", [])) >= 2,
     "emits nested/array JSON args without corruption"),
    ("calc_vs_search2", [T_CALC, T_SEARCH],
     "I need the value of 144 divided by 12. Use the appropriate tool.",
     "calculator", lambda a: "144" in str(a.get("expression", "")) and "12" in str(a.get("expression", "")),
     "routes a pure-math request to calculator, not search"),
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
]

# ---- reasoning probes (verifiable; one slip -> wrong final answer) ---------
PROMPT_SUFFIX = "\n\nThink step by step, then end your reply with a line in exactly this form:\nANSWER: <your final answer>"
REASON_PROBES = [
    ("apples",   "A store had 240 apples. It sold 1/3 on Monday, then 1/4 of the remaining on Tuesday. How many are left?", "120"),
    ("xy",       "Given x + y = 10 and x - y = 4, what is x times y?", "21"),
    ("strings",  "How many distinct 3-letter strings can be formed from {A,B,C,D} with no repeated letter?", "24"),
    ("shirt",    "A shirt costs $80, increased by 25%, then decreased by 20%. Final price in dollars?", "80"),
    ("taps",     "One tap fills a tank in 6 hours, another in 12 hours. Together, how many hours to fill it?", "4"),
    ("avgspeed", "A car goes 150 km at 50 km/h, then 150 km at 75 km/h. Average speed over 300 km, in km/h?", "60"),
    ("snail",    "A snail climbs 3 m up a well by day and slips 2 m back each night. The well is 10 m deep. On which day does it first reach the top?", "8"),
    ("div3or5",  "How many integers from 1 to 100 inclusive are divisible by 3 or by 5?", "47"),
    ("overtime", "A worker earns $15/h for the first 40 hours and 1.5x for overtime. They worked 46 hours. Total pay in dollars?", "735"),
    ("fatherson","A father is 3x as old as his son. In 12 years he'll be twice as old as the son then. How old is the son now?", "12"),
]

# ---- server I/O -----------------------------------------------------------
def _post(port, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def ask_tool(port, tools, msg, maxtok):
    r = _post(port, {"messages": [{"role": "user", "content": msg}],
                     "tools": tools, "tool_choice": "auto",
                     "temperature": 0, "max_tokens": maxtok})
    return r["choices"][0]["message"]

def ask_text(port, q, maxtok):
    r = _post(port, {"messages": [{"role": "user", "content": q + PROMPT_SUFFIX}],
                     "temperature": 0, "max_tokens": maxtok})
    return r["choices"][0]["message"]["content"]

# ---- tool-call extraction (OpenAI tool_calls OR raw <tool_call> fallback) --
def extract_tool_call(message):
    """Return (fn_name, args_dict, malformed_reason). Handles both the structured
    tool_calls field and a raw <tool_call>{...}</tool_call> blob in content."""
    tcs = message.get("tool_calls")
    if tcs:
        fn = tcs[0].get("function", {})
        name = fn.get("name", "")
        raw = fn.get("arguments", "")
        if isinstance(raw, dict):
            return name, raw, ""
        try:
            return name, json.loads(raw), ""
        except Exception as e:
            return name, {}, f"unparseable-args:{e}"
    # fallback: model emitted the call inline in content
    content = message.get("content") or ""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj.get("name", ""), obj.get("arguments", obj.get("parameters", {})), ""
        except Exception as e:
            return "", {}, f"unparseable-inline:{e}"
    return "", {}, "no-tool-call"

# ---- answer extraction / normalization (reasoning) ------------------------
def extract_ans(reply):
    seg = (reply or "").rsplit("</think>", 1)[-1]
    m = re.findall(r"ANSWER:\s*(.+)", seg, flags=re.IGNORECASE)
    raw = m[-1] if m else (seg.strip().splitlines()[-1] if seg.strip() else "")
    return raw.strip()

def norm(s):
    s = (s or "").strip().lower().replace("$", "").replace(",", "").replace("\\", "")
    s = re.sub(r"[.\s]+$", "", s).strip().strip(".").strip()
    mnum = re.search(r"-?\d+\s*/\s*\d+|-?\d+(?:\.\d+)?", s)
    return mnum.group(0).replace(" ", "") if mnum else s

def think_struct_ok(reply):
    r = reply or ""
    o, c = r.count("<think>"), r.count("</think>")
    if o == 1 and c == 1 and r.index("<think>") < r.index("</think>"):
        tail = r.rsplit("</think>", 1)[-1]
        return bool(tail.strip()) and "ANSWER:" in tail.upper()
    if o == 0 and c == 0:   # --reasoning-format strips tags: must NOT leak, answer must still follow
        return "ANSWER:" in (r or "").upper()
    return False            # unbalanced / duplicated / leaked / unclosed

# ---- collect / grade ------------------------------------------------------
def cmd_collect(port, label, outdir):
    maxtok = int(os.environ.get("MAXTOK", "4096"))
    out = {"tool": {}, "reason": {}}
    print(f"  -- TOOL probes ({label})", flush=True)
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
    print(f"  -- REASON probes ({label})", flush=True)
    for pid, q, want in REASON_PROBES:
        try:
            reply = ask_text(port, q, maxtok)
            ans, err = extract_ans(reply), ""
        except Exception as e:
            reply, ans, err = "", "", f"__ERROR__ {e}"
        rec = {"want": want, "ans": ans, "norm": norm(ans) if ans else "",
               "raw": reply, "think_ok": think_struct_ok(reply), "err": err}
        out["reason"][pid] = rec
        tag = "ok  " if rec["norm"] == norm(want) else "MISS"
        print(f"    [{tag}] {pid:16s} want={want:6s} got={ans[:36]!r}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{label}.json"), "w") as f:
        json.dump(out, f, indent=2)
    ts = sum(1 for v in out["tool"].values() if v["pass"])
    rs = sum(1 for v in out["reason"].values() if v["norm"] == norm(v["want"]))
    print(f"  == {label}: TOOL {ts}/{len(TOOL_PROBES)} | REASON {rs}/{len(REASON_PROBES)} (vs key)", flush=True)

def cmd_grade(outdir, ref_label, labels):
    data = {}
    for lab in [ref_label] + [l for l in labels if l != ref_label]:
        p = os.path.join(outdir, f"{lab}.json")
        if os.path.exists(p):
            data[lab] = json.load(open(p))
    ref = data.get(ref_label, {"tool": {}, "reason": {}})
    nt, nr = len(TOOL_PROBES), len(REASON_PROBES)
    print(f"\n{'model':10s} {'TOOL_key':>9s} {'TOOLagree':>10s} {'REAS_key':>9s} {'REASagree':>10s} {'THINK_ok':>9s}")
    print("-" * 62)
    for lab, d in data.items():
        t, r = d.get("tool", {}), d.get("reason", {})
        tk = sum(1 for v in t.values() if v.get("pass"))
        ta = sum(1 for pid, v in t.items()
                 if ref["tool"].get(pid) and v.get("pass") == ref["tool"][pid].get("pass") and v.get("pass"))
        rk = sum(1 for v in r.values() if v.get("norm") == norm(v.get("want")))
        ra = sum(1 for pid, v in r.items()
                 if ref["reason"].get(pid) and v.get("norm") == ref["reason"][pid].get("norm") and v.get("norm"))
        th = sum(1 for v in r.values() if v.get("think_ok"))
        print(f"{lab:10s} {tk:>4d}/{nt:<3d} {ta:>5d}/{nt:<3d} {rk:>4d}/{nr:<3d} {ra:>5d}/{nr:<3d} {th:>4d}/{nr:<3d}")
    print("\nSHIP RULE: pick the smallest quant whose TOOL_key and REAS_key both stay within ~1-2 of the")
    print("BF16 reference. A TOOL drop with flat PPL = the agentic cliff this gate exists to catch.")

if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "collect":
        cmd_collect(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) >= 4 and sys.argv[1] == "grade":
        cmd_grade(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        print(__doc__); sys.exit(1)

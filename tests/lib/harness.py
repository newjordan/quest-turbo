#!/usr/bin/env python3
"""QUEST-30B-RL certification harness — the ONE reusable module.

This generalizes scripts/04b_probes.py into a unit-testable, spec-driven library.

Three import-purity tiers (so `import harness` succeeds on a machine with only
stdlib + pytest — no network, no subprocess, no gguf, no yaml runs at import):

  (A) PURE scoring / aggregation / parsers — stdlib + re only.
      extract_tool_call, extract_answer, normalize_answer, think_ok,
      check_tool_args, score_tool, score_reason, summarize, agreement,
      hard_fails, certify, check_metadata, parse_llama_bench, select_bench,
      check_performance, parse_ppl, check_ppl, resolve_thresholds.
  (B) spec / config — lazy-imports yaml inside load_spec.
  (C) GPU / IO — LlamaServer (subprocess + urllib), chat / ask_* (urllib),
      read_gguf_metadata (lazy-imports gguf).

The arg-checks that 04b embedded as Python lambdas are migrated 1:1 into a tiny
declarative DSL (check_tool_args) that lives in spec.yaml as DATA, so the scorer
is PURE and trivially unit-testable with synthetic dicts. Everything else
(extract_tool_call, extract_ans->extract_answer, norm->normalize_answer,
think_struct_ok->think_ok, the abstain/parallel/min_calls pass logic, and the
mutual-pass agreement counting) is migrated VERBATIM, so existing
results/04b_answers/*.json remain valid inputs to certify().
"""
import os
import re
import json
import time
import subprocess
import urllib.request
import urllib.error
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------- #
# Default spec location, resolved relative to tests/lib/  ->  tests/spec.yaml
SPEC_PATH = os.path.join(os.path.dirname(__file__), "..", "spec.yaml")

# Literal fallback identical to 04b PROMPT_SUFFIX (== spec.meta.reasoning_suffix).
REASONING_SUFFIX = (
    "\n\nThink step by step, then end your reply with a line in exactly this form:\n"
    "ANSWER: <your final answer>"
)

_REQUIRED_SPEC_KEYS = ("meta", "metadata", "tools", "tool_probes", "reason_probes", "thresholds")


# --------------------------------------------------------------------------- #
# (B) spec / config — lazy yaml
# --------------------------------------------------------------------------- #
def load_spec(path=SPEC_PATH):
    """Lazy-import yaml, load the spec, validate required top-level keys."""
    import yaml  # lazy: tier-A import purity

    with open(path, "r") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"spec at {path} did not parse to a mapping")
    missing = [k for k in _REQUIRED_SPEC_KEYS if k not in spec]
    if missing:
        raise ValueError(f"spec at {path} missing required keys: {missing}")
    return spec


def resolve_thresholds(spec, *, perf_tolerance=None):
    """Flatten spec['thresholds'] and COMPUTE the perf minimums. PURE."""
    t = spec["thresholds"]
    perf = t["perf"]
    tol = perf_tolerance if perf_tolerance is not None else perf["tolerance"]
    critical = t.get("hard_fails", {}).get("critical_probes", [])
    return {
        "sanity": t["sanity"],
        "tool": t["tool"],
        "reason": t["reason"],
        "think": t["think"],
        "ppl": t["ppl"],
        "perf": perf,
        "critical_probes": critical,
        "pp512_min": round(perf["pp512_baseline"] * (1 - tol), 1),
        "tg128_min": round(perf["tg128_baseline"] * (1 - tol), 1),
    }


# --------------------------------------------------------------------------- #
# (A) PURE — tool-call extraction (VERBATIM from 04b_probes.py)
# --------------------------------------------------------------------------- #
def extract_tool_call(message):
    """Return (fn_name, args_dict, malformed_reason).

    Handles both the structured tool_calls field and a raw
    <tool_call>{...}</tool_call> blob in content. PURE.
    malformed in {'', 'unparseable-args:<e>', 'unparseable-inline:<e>', 'no-tool-call'}.
    """
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


# --------------------------------------------------------------------------- #
# (A) PURE — answer extraction / normalization (VERBATIM from 04b)
# --------------------------------------------------------------------------- #
def extract_answer(reply):
    """04b extract_ans: segment after last </think>, last ANSWER: line (ci)
    else last non-blank line, stripped. PURE."""
    seg = (reply or "").rsplit("</think>", 1)[-1]
    m = re.findall(r"ANSWER:\s*(.+)", seg, flags=re.IGNORECASE)
    raw = m[-1] if m else (seg.strip().splitlines()[-1] if seg.strip() else "")
    return raw.strip()


def normalize_answer(s):
    """04b norm: lower / strip $,\\ / strip trailing punct, then return the first
    number-or-fraction token (spaces removed) else the cleaned string. PURE."""
    s = (s or "").strip().lower().replace("$", "").replace(",", "").replace("\\", "")
    s = re.sub(r"[.\s]+$", "", s).strip().strip(".").strip()
    mnum = re.search(r"-?\d+\s*/\s*\d+|-?\d+(?:\.\d+)?", s)
    return mnum.group(0).replace(" ", "") if mnum else s


def think_ok(reply):
    """04b think_struct_ok: exactly one balanced <think>..</think> whose tail is
    non-empty AND contains 'ANSWER:', OR zero think-tags but 'ANSWER:' present.
    Else False. PURE."""
    r = reply or ""
    o, c = r.count("<think>"), r.count("</think>")
    if o == 1 and c == 1 and r.index("<think>") < r.index("</think>"):
        tail = r.rsplit("</think>", 1)[-1]
        return bool(tail.strip()) and "ANSWER:" in tail.upper()
    if o == 0 and c == 0:  # --reasoning-format strips tags: must NOT leak, ANSWER must follow
        return "ANSWER:" in (r or "").upper()
    return False  # unbalanced / duplicated / leaked / unclosed


# --------------------------------------------------------------------------- #
# (A) PURE — declarative arg-check DSL (the 1:1 migration of the 04b lambdas)
# --------------------------------------------------------------------------- #
_MISSING = object()


def _value_at(args, path):
    """Dot-nested lookup. Returns _MISSING if any segment is absent."""
    cur = args
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _apply_check(args, check):
    """Evaluate one declarative check. Any internal exception / missing path -> False."""
    try:
        field = check.get("field")
        if field is None:
            return False
        val = _value_at(args, field)

        if "equals" in check:
            if val is _MISSING:
                return False
            return str(val).strip().lower() == str(check["equals"]).lower()

        if "contains_all" in check:
            if val is _MISSING:
                return False
            s = str(val).lower()
            return all(str(x).lower() in s for x in check["contains_all"])

        if "contains_any" in check:
            if val is _MISSING:
                return False
            s = str(val).lower()
            return any(str(x).lower() in s for x in check["contains_any"])

        if "type" in check:
            if val is _MISSING:
                return False
            want = check["type"]
            return (
                (want == "string" and isinstance(val, str))
                or (want == "array" and isinstance(val, list))
                or (want == "object" and isinstance(val, dict))
            )

        if "min_len" in check:
            if val is _MISSING:
                return False
            return len(val) >= int(check["min_len"])

        if "items_have_keys" in check:
            if val is _MISSING or not isinstance(val, list):
                return False
            keys = check["items_have_keys"]
            return all(isinstance(it, dict) and all(k in it for k in keys) for it in val)

        return False  # unknown operator
    except Exception:
        return False


def check_tool_args(args, checks):
    """PURE declarative interpreter (replaces the 04b lambdas).

    Empty/absent checks -> True. ALL checks must hold (AND). Per-check field path
    supports dot-nesting. Missing path or any internal exception -> that check False.
    """
    if not checks:
        return True
    try:
        return all(_apply_check(args, c) for c in checks)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# (A) PURE — per-probe scorers (record shape identical to 04b_answers/*.json)
# --------------------------------------------------------------------------- #
def score_tool(probe, message):
    """PURE. Score one tool probe against a server message dict.

    pass logic EXACTLY as 04b cmd_collect:
      expect_fn == '__none__'  -> pass = (bad=='no-tool-call') and content non-empty
      min_calls >= 2           -> pass = n_calls>=min_calls and routed and args_ok and not bad
      else                     -> pass = routed and args_ok and not bad
    """
    name, args, bad = extract_tool_call(message)
    n_calls = len(message.get("tool_calls") or [])
    content = message.get("content") or ""
    expect_fn = probe["expect_fn"]
    routed = (name == expect_fn)
    args_ok = (not bad) and check_tool_args(args, probe.get("arg_checks", []))

    if expect_fn == "__none__":
        ok = (bad == "no-tool-call") and bool(content.strip())
    elif probe.get("min_calls", 1) >= 2:
        ok = (n_calls >= probe["min_calls"]) and routed and args_ok and not bad
    else:
        ok = routed and args_ok and not bad

    return {
        "want_fn": expect_fn,
        "got_fn": name,
        "args": args,
        "malformed": bad,
        "routed": routed,
        "args_ok": args_ok,
        "n_calls": n_calls,
        "pass": ok,
    }


def score_reason(probe, reply):
    """PURE. Score one reasoning probe against a text reply."""
    ans = extract_answer(reply)
    return {
        "want": probe["expect"],
        "ans": ans,
        "norm": normalize_answer(ans) if ans else "",
        "raw": reply,
        "think_ok": think_ok(reply),
        "err": "",
    }


# --------------------------------------------------------------------------- #
# (A) PURE — aggregation
# --------------------------------------------------------------------------- #
def summarize(answers):
    """PURE single-model counts. Returns 5 ints:
    tool_key, reason_key, think_ok, nt, nr."""
    tool = answers.get("tool", {})
    reason = answers.get("reason", {})
    tool_key = sum(1 for v in tool.values() if v.get("pass"))
    reason_key = sum(
        1 for v in reason.values() if v.get("norm") == normalize_answer(v.get("want", ""))
    )
    think = sum(1 for v in reason.values() if v.get("think_ok"))
    return {
        "tool_key": tool_key,
        "reason_key": reason_key,
        "think_ok": think,
        "nt": len(tool),
        "nr": len(reason),
    }


def agreement(cand, ref):
    """PURE, VERBATIM 04b cmd_grade semantics (MUTUAL pass only).

    Abstain (failed by both) contributes 0 either way, so it is never a regression.
    """
    ct = cand.get("tool", {})
    cr = cand.get("reason", {})
    reft = ref.get("tool", {})
    refr = ref.get("reason", {})
    tool_agree = sum(
        1
        for pid, v in ct.items()
        if reft.get(pid) and v.get("pass") and v.get("pass") == reft[pid].get("pass")
    )
    reason_agree = sum(
        1
        for pid, v in cr.items()
        if refr.get(pid) and v.get("norm") and v.get("norm") == refr[pid].get("norm")
    )
    return {"tool_agree": tool_agree, "reason_agree": reason_agree}


def hard_fails(cand, ref, critical):
    """PURE. Return fired hard-fails as dicts {'code','probe','detail'}.

    H1: ref tool clean (malformed=='') but cand malformed nonempty (NEW JSON corruption).
    H2: critical probe ref.pass and not cand.pass (guarded on ref-pass).
    H3: reason ref.think_ok and not cand.think_ok (think structural regression).
    """
    fails = []
    ct, reft = cand.get("tool", {}), ref.get("tool", {})
    for pid, cv in ct.items():
        rv = reft.get(pid)
        if rv is None:
            continue
        if rv.get("malformed", "") == "" and (cv.get("malformed") or "") != "":
            fails.append(
                {"code": "H1", "probe": pid,
                 "detail": f"new malformed tool-call {cv.get('malformed')!r} (ref clean)"}
            )
    for pid in (critical or []):
        rv, cv = reft.get(pid), ct.get(pid)
        if rv and cv and rv.get("pass") and not cv.get("pass"):
            fails.append(
                {"code": "H2", "probe": pid,
                 "detail": "critical probe regressed (ref pass -> cand fail)"}
            )
    cr, refr = cand.get("reason", {}), ref.get("reason", {})
    for pid, cv in cr.items():
        rv = refr.get(pid)
        if rv is None:
            continue
        if rv.get("think_ok") and not cv.get("think_ok"):
            fails.append(
                {"code": "H3", "probe": pid,
                 "detail": "think structure regressed (ref ok -> cand not)"}
            )
    return fails


# --------------------------------------------------------------------------- #
# (A) PURE — cert report dataclasses + the ship decision
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str  # 'sanity' | 'aggregate' | 'hard'
    detail: str


@dataclass
class CertReport:
    passed: bool
    sanity_ok: bool
    checks: list
    hard_fails: list
    summary: dict

    def render(self):
        """SHIP/NO-SHIP table (mirrors 04b_summary plus PASS/FAIL per check)."""
        verdict = "SHIP" if self.passed else "NO-SHIP"
        cs = self.summary.get("cand", {})
        rs = self.summary.get("ref", {})
        ag = self.summary.get("agreement", {})
        lines = [f"==== QUEST CERT: {verdict} ===="]
        if not self.sanity_ok:
            lines.append(
                "!! SANITY GUARD FAILED — RIG BROKEN (BF16 reference below floor); "
                "fix the rig (template/--reasoning-format/tool schema), not the model."
            )
        lines.append(
            f"  cand: TOOL_key {cs.get('tool_key')}/{cs.get('nt')}  "
            f"REAS_key {cs.get('reason_key')}/{cs.get('nr')}  "
            f"THINK {cs.get('think_ok')}/{cs.get('nr')}"
        )
        lines.append(
            f"  ref : TOOL_key {rs.get('tool_key')}/{rs.get('nt')}  "
            f"REAS_key {rs.get('reason_key')}/{rs.get('nr')}  "
            f"THINK {rs.get('think_ok')}/{rs.get('nr')}"
        )
        lines.append(
            f"  agree: TOOL {ag.get('tool_agree')}  REASON {ag.get('reason_agree')}"
        )
        lines.append("  checks:")
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"    [{mark}] ({c.severity}) {c.name}: {c.detail}")
        if self.hard_fails:
            lines.append(f"  HARD FAILS FIRED: {len(self.hard_fails)}")
        lines.append(f"==== {verdict} ====")
        return "\n".join(lines)

    def as_dict(self):
        """JSON-able dict for $RES/certify_<quant>.json."""
        return {
            "passed": self.passed,
            "sanity_ok": self.sanity_ok,
            "checks": [
                {"name": c.name, "passed": c.passed, "severity": c.severity, "detail": c.detail}
                for c in self.checks
            ],
            "hard_fails": self.hard_fails,
            "summary": self.summary,
        }


def certify(cand, ref, thresholds):
    """PURE — THE ship decision. Returns a CertReport.

    Order:
      (1) SANITY GUARD on ref: ref tool_key >= sanity.bf16_tool_min AND
          ref think_ok >= sanity.bf16_think_min. If it fails -> sanity_ok=False,
          passed=False (rig broken, not the model).
      (2) AGGREGATE checks (each a CheckResult).
      (3) HARD FAILS via hard_fails(...) -> appended as severity='hard'.
      passed = sanity_ok and all aggregate checks pass and no hard fails.
    """
    cand_sum = summarize(cand)
    ref_sum = summarize(ref)
    agr = agreement(cand, ref)

    sanity = thresholds.get("sanity", {})
    tool_t = thresholds.get("tool", {})
    reason_t = thresholds.get("reason", {})
    think_t = thresholds.get("think", {})
    critical = thresholds.get("critical_probes") or thresholds.get("hard_fails", {}).get(
        "critical_probes", []
    )

    checks = []

    # (1) SANITY GUARD on the reference
    sanity_tool_ok = ref_sum["tool_key"] >= sanity.get("bf16_tool_min", 0)
    sanity_think_ok = ref_sum["think_ok"] >= sanity.get("bf16_think_min", 0)
    sanity_ok = sanity_tool_ok and sanity_think_ok
    checks.append(
        CheckResult(
            "sanity_ref_tool", sanity_tool_ok, "sanity",
            f"ref tool_key {ref_sum['tool_key']}>={sanity.get('bf16_tool_min')}",
        )
    )
    checks.append(
        CheckResult(
            "sanity_ref_think", sanity_think_ok, "sanity",
            f"ref think_ok {ref_sum['think_ok']}>={sanity.get('bf16_think_min')}",
        )
    )

    # (2) AGGREGATE
    tk_c, tk_r = cand_sum["tool_key"], ref_sum["tool_key"]
    rk_c, rk_r = cand_sum["reason_key"], ref_sum["reason_key"]
    th_c, th_r = cand_sum["think_ok"], ref_sum["think_ok"]
    tool_drop = tool_t.get("key_drop_max", 0)
    reason_drop = reason_t.get("key_drop_max", 0)
    ref_required = think_t.get("ref_required")

    aggregate = [
        CheckResult(
            "tool_key", tk_c >= tk_r - tool_drop, "aggregate",
            f"tool_key {tk_c}>={tk_r - tool_drop} (ref {tk_r} - {tool_drop})",
        ),
        CheckResult(
            "tool_agree", agr["tool_agree"] >= tool_t.get("agree_min", 0), "aggregate",
            f"tool_agree {agr['tool_agree']}>={tool_t.get('agree_min')}",
        ),
        CheckResult(
            "reason_key", rk_c >= rk_r - reason_drop, "aggregate",
            f"reason_key {rk_c}>={rk_r - reason_drop} (ref {rk_r} - {reason_drop})",
        ),
        CheckResult(
            "reason_agree", agr["reason_agree"] >= reason_t.get("agree_min", 0), "aggregate",
            f"reason_agree {agr['reason_agree']}>={reason_t.get('agree_min')}",
        ),
        CheckResult(
            "think_ok", (th_r == ref_required) and (th_c == th_r), "aggregate",
            f"think_ok cand {th_c}==ref {th_r}==required {ref_required}",
        ),
    ]
    checks.extend(aggregate)

    # (3) HARD FAILS
    hfs = hard_fails(cand, ref, critical)
    for hf in hfs:
        checks.append(
            CheckResult(f"{hf['code']}:{hf['probe']}", False, "hard", hf["detail"])
        )

    passed = sanity_ok and all(c.passed for c in aggregate) and not hfs

    summary = {
        "cand": cand_sum,
        "ref": ref_sum,
        "agreement": agr,
        "pp512_min": thresholds.get("pp512_min"),
        "tg128_min": thresholds.get("tg128_min"),
    }
    return CertReport(passed=passed, sanity_ok=sanity_ok, checks=checks, hard_fails=hfs, summary=summary)


# --------------------------------------------------------------------------- #
# (A) PURE — metadata checks
# --------------------------------------------------------------------------- #
def check_metadata(meta, expectations):
    """PURE. For each {key, expected, type, rel_tol?} -> CheckResult."""
    results = []
    for exp in expectations:
        key = exp["key"]
        expected = exp["expected"]
        typ = exp.get("type", "str")
        if key not in meta:
            results.append(CheckResult(key, False, "aggregate", f"missing key {key}"))
            continue
        actual = meta[key]
        if typ == "int":
            try:
                ok = int(actual) == int(expected)
            except Exception:
                ok = False
            detail = f"{key}={actual} expected {expected}"
        elif typ == "float":
            rel_tol = exp.get("rel_tol", 1e-6)
            try:
                ok = abs(float(actual) - float(expected)) <= rel_tol * abs(float(expected))
            except Exception:
                ok = False
            detail = f"{key}={actual} expected {expected} (rel_tol {rel_tol})"
        else:  # str
            ok = str(actual) == str(expected)
            detail = f"{key}={actual!r} expected {expected!r}"
        results.append(CheckResult(key, ok, "aggregate", detail))
    return results


# --------------------------------------------------------------------------- #
# (A) PURE — llama-bench / llama-perplexity log parsers
# --------------------------------------------------------------------------- #
def parse_llama_bench(text):
    """PURE. Parse a llama-bench markdown table.

    Header-aware so it handles BOTH the narrow layout (results/00_baseline.txt:
    model|size|params|backend|ngl|fa|test|t/s) AND the wide layout
    (results/05_depth.txt: ...|ngl|n_batch|n_ubatch|type_k|type_v|fa|test|t/s).
    Columns are located by header name (re-mapped per table, so multiple tables in
    one file are fine); 'test' and 't/s' are otherwise the last two columns.

    Returns one dict per data row:
      {'model','size','params','backend','ngl':int,'fa':int,'test':str,'t_s':float,'depth':int}
    'tg128 @ d 16384' (or '@ d16384') normalizes to test='tg128', depth=16384 (else depth=0).
    Skips header / separator / 'build:' lines.
    """

    def _int(v, default=0):
        try:
            return int(v)
        except (ValueError, TypeError):
            return default

    rows = []
    header = None  # {lower_name: column_index}
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("|"):
            continue
        if "build:" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells:
            continue
        # separator row: first cell only dashes / colons / spaces
        if cells[0] and set(cells[0]) <= set("-: "):
            continue
        lc = [c.lower() for c in cells]
        # header row -> remap columns and move on
        if "model" in lc and "test" in lc:
            header = {name: i for i, name in enumerate(lc)}
            continue
        if len(cells) < 2:
            continue

        def by_name(name, pos):
            if header is not None and name in header and header[name] < len(cells):
                return cells[header[name]]
            return cells[pos] if pos is not None and -len(cells) <= pos < len(cells) else ""

        if header is not None and "test" in header and "t/s" in header:
            test = cells[header["test"]] if header["test"] < len(cells) else ""
            ts = cells[header["t/s"]] if header["t/s"] < len(cells) else ""
        else:
            if len(cells) < 8:
                continue
            test, ts = cells[-2], cells[-1]  # test + t/s are always the last two columns

        depth = 0
        tnorm = test
        m = re.search(r"@\s*d\s*(\d+)", test)
        if m:
            depth = int(m.group(1))
            tnorm = test.split("@")[0].strip()
        tval = re.split(r"±|\+/-", ts)[0].strip()
        try:
            t_s = float(tval)
        except ValueError:
            continue
        rows.append(
            {
                "model": by_name("model", 0),
                "size": by_name("size", 1),
                "params": by_name("params", 2),
                "backend": by_name("backend", 3),
                "ngl": _int(by_name("ngl", 4)),
                "fa": _int(by_name("fa", 5)),
                "test": tnorm,
                "t_s": t_s,
                "depth": depth,
            }
        )
    return rows


def select_bench(rows, test, depth=0):
    """PURE. t_s of the first row matching (test, depth); else None."""
    for row in rows:
        if row.get("test") == test and row.get("depth", 0) == depth:
            return row.get("t_s")
    return None


def check_performance(rows, thresholds):
    """PURE. pp512/tg128 vs computed minimums."""
    pp = select_bench(rows, "pp512")
    tg = select_bench(rows, "tg128")
    return [
        CheckResult(
            "pp512", pp is not None and pp >= thresholds["pp512_min"], "aggregate",
            f"{pp}>={thresholds['pp512_min']}",
        ),
        CheckResult(
            "tg128", tg is not None and tg >= thresholds["tg128_min"], "aggregate",
            f"{tg}>={thresholds['tg128_min']}",
        ),
    ]


def parse_ppl(text):
    """PURE. Extract the final perplexity from a llama-perplexity log, else None."""
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def check_ppl(ppl, thresholds):
    """PURE. ppl <= thresholds.ppl.max."""
    return CheckResult(
        "ppl", ppl is not None and ppl <= thresholds["ppl"]["max"], "aggregate",
        f"{ppl}<={thresholds['ppl']['max']}",
    )


# --------------------------------------------------------------------------- #
# (C) GPU / IO — server lifecycle (subprocess) + chat helpers (urllib)
# --------------------------------------------------------------------------- #
def wait_for_health(base_url, timeout_s=300, poll_interval_s=2.0):
    """Poll GET {base_url}/health. True on first HTTP 200, False on overall timeout.
    Never raises on connection-refused (server still loading)."""
    url = base_url.rstrip("/") + "/health"
    deadline = time.monotonic() + timeout_s
    per_req = max(1.0, min(5.0, poll_interval_s + 3.0))
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=per_req) as r:
                if getattr(r, "status", r.getcode()) == 200:
                    return True
        except Exception:
            pass
        time.sleep(poll_interval_s)
    return False


def stop_server(proc, grace_s=10.0):
    """FIX-B teardown: SIGTERM -> grace poll -> SIGKILL -> reap. Idempotent, never raises."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
        proc.terminate()
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.25)
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=grace_s)
        except Exception:
            pass
    except Exception:
        pass


class LlamaServer:
    """Context manager: in-Python twin of the 04b/06 server lifecycle.

    __enter__ launches one llama-server, polls /health, and on timeout runs the
    stop sequence then raises TimeoutError. __exit__ always stops the server.
    """

    def __init__(
        self,
        model_path,
        *,
        bin_dir=None,
        port=8099,
        ngl=99,
        ctx=32768,
        kv_type="q8_0",
        ubatch=4096,
        bbatch=8192,
        jinja=True,
        alias="quest-cert",
        extra_args=None,
        health_timeout_s=300,
        poll_interval_s=2.0,
        log_path=None,
    ):
        self.model_path = str(model_path)
        self.bin_dir = bin_dir if bin_dir is not None else os.environ.get("RUN_BIN")
        self.port = port
        self.ngl = ngl
        self.ctx = ctx
        self.kv_type = kv_type
        self.ubatch = ubatch
        self.bbatch = bbatch
        self.jinja = jinja
        self.alias = alias
        self.extra_args = list(extra_args) if extra_args else []
        self.health_timeout_s = health_timeout_s
        self.poll_interval_s = poll_interval_s
        self.log_path = log_path
        self.proc = None
        self.base_url = f"http://127.0.0.1:{port}"
        self._logf = None

    def _cmd(self):
        bd = self.bin_dir or ""
        cmd = [
            os.path.join(bd, "llama-server"),
            "-m", self.model_path,
            "-ngl", str(self.ngl),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.ctx),
            "-fa", "on",
            "--cache-type-k", self.kv_type,
            "--cache-type-v", self.kv_type,
            "-ub", str(self.ubatch),
            "-b", str(self.bbatch),
            ("--jinja" if self.jinja else None),
            "--alias", self.alias,
            *self.extra_args,
        ]
        return [c for c in cmd if c is not None]

    def __enter__(self):
        if self.log_path:
            self._logf = open(self.log_path, "w")
            out = self._logf
        else:
            out = subprocess.DEVNULL
        self.proc = subprocess.Popen(self._cmd(), stdout=out, stderr=subprocess.STDOUT)
        if not wait_for_health(self.base_url, self.health_timeout_s, self.poll_interval_s):
            stop_server(self.proc)
            if self._logf:
                try:
                    self._logf.close()
                except Exception:
                    pass
            raise TimeoutError(
                f"llama-server health timeout after {self.health_timeout_s}s (port {self.port})"
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        stop_server(self.proc)
        if self._logf:
            try:
                self._logf.close()
            except Exception:
                pass
        return False


def chat(base_url, messages, *, tools=None, tool_choice="auto", temperature=0.0, max_tokens=4096, timeout_s=900):
    """POST {base_url}/v1/chat/completions; return choices[0].message. urllib only."""
    body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    url = base_url.rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        resp = json.load(r)
    return resp["choices"][0]["message"]


def ask_tool(base_url, tools, user_msg, max_tokens=4096):
    """04b ask_tool (base_url form). Returns the message dict."""
    return chat(
        base_url,
        [{"role": "user", "content": user_msg}],
        tools=tools,
        tool_choice="auto",
        temperature=0,
        max_tokens=max_tokens,
    )


def ask_text(base_url, question, max_tokens=4096, *, suffix=REASONING_SUFFIX):
    """04b ask_text. Returns message['content'] or ''."""
    msg = chat(
        base_url,
        [{"role": "user", "content": question + suffix}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return msg.get("content") or ""


def collect(base_url, spec, max_tokens=4096, *, log=print):
    """NETWORK (gated). Run all probes against a live server.

    Returns {'tool':{id:rec}, 'reason':{id:rec}} — SAME shape as 04b output
    (back-compatible with existing results/04b_answers/*.json).
    """
    out = {"tool": {}, "reason": {}}
    for probe in spec["tool_probes"]:
        pid = probe["id"]
        try:
            tools = [spec["tools"][n] for n in probe["tools"]]
            msg = ask_tool(base_url, tools, probe["prompt"], max_tokens)
            rec = score_tool(probe, msg)
        except Exception as e:
            rec = {
                "want_fn": probe.get("expect_fn", ""),
                "got_fn": "",
                "args": {},
                "malformed": f"__ERROR__ {e}",
                "routed": False,
                "args_ok": False,
                "n_calls": 0,
                "pass": False,
            }
        out["tool"][pid] = rec
        if log:
            tag = "ok  " if rec["pass"] else "FAIL"
            log(f"  [{tag}] tool {pid:16s} want={rec['want_fn']:13s} got={rec['got_fn'] or '-'}")
    for probe in spec["reason_probes"]:
        pid = probe["id"]
        try:
            reply = ask_text(base_url, probe["prompt"], max_tokens)
            rec = score_reason(probe, reply)
        except Exception as e:
            rec = {
                "want": probe["expect"],
                "ans": "",
                "norm": "",
                "raw": "",
                "think_ok": False,
                "err": f"__ERROR__ {e}",
            }
        out["reason"][pid] = rec
        if log:
            tag = "ok  " if rec.get("norm") == normalize_answer(rec.get("want", "")) else "MISS"
            log(f"  [{tag}] reas {pid:16s} want={rec.get('want'):6s} got={rec.get('ans', '')[:36]!r}")
    return out


# --------------------------------------------------------------------------- #
# (C) GPU / IO — GGUF header read (lazy gguf; header-only, does NOT touch GPU)
# --------------------------------------------------------------------------- #
def _coerce(v):
    """Map numpy int->int, numpy float->float, bytes/np-str->str; passthrough else."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", "replace")
        except Exception:
            return str(v)
    item = getattr(v, "item", None)
    if callable(item):
        try:
            v2 = v.item()
            if isinstance(v2, bytes):
                return v2.decode("utf-8", "replace")
            return v2
        except Exception:
            pass
    return v


def read_gguf_metadata(model_path):
    """GATED file read (header-only KV store; does NOT load tensors or touch GPU).
    Lazy-imports gguf. Returns {name: coerced scalar}."""
    from gguf import GGUFReader  # lazy: tier-A import purity

    r = GGUFReader(str(model_path), mode="r")
    return {name: _coerce(field.contents()) for name, field in r.fields.items()}

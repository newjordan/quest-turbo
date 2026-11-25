"""PURE unit tests for the QUEST cert scorer (harness).

NO server, NO model, NO GPU, NO yaml/gguf — feeds SYNTHETIC in-memory message
dicts and reply strings into the pure scoring/aggregation API and asserts the
exact pass/fail behavior. This is the ONLY module that fully RUNS under a bare
`pytest` invocation (everything else is gated behind a live server / model file).

Covers, per the design contract:
  - extract_tool_call  : structured / inline / malformed / none
  - check_tool_args    : every DSL operator + dot-paths + AND semantics
  - score_tool         : routed / wrong-fn / malformed / abstain / parallel logic
  - normalize_answer / extract_answer / think_ok branches
  - summarize / agreement / hard_fails (H1/H2/H3) / certify ship-decision,
    including the sanity-guard trip, each aggregate tolerance boundary, and the
    rule that `abstain` (failed by BOTH models) is NEVER counted as a regression.
"""
import json

import pytest

import harness


# ===========================================================================
# synthetic builders — server "message" dicts and saved answer records
# ===========================================================================
def tool_msg(name, arguments, content=""):
    """One structured OpenAI tool_calls message. `arguments` may be dict OR str."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def multi_tool_msg(calls, content=""):
    """A parallel-tool message. `calls` = list of (name, arguments) pairs."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"type": "function", "function": {"name": n, "arguments": a}} for (n, a) in calls
        ],
    }


def inline_msg(blob):
    """A message that emitted the call inline in content (no tool_calls field)."""
    return {"role": "assistant", "content": f"<tool_call>{blob}</tool_call>"}


def text_msg(content):
    """A plain text answer with no tool call at all."""
    return {"role": "assistant", "content": content}


# Synthetic probes (independent of spec.yaml — keeps this module pure stdlib).
CALC_PROBE = {
    "id": "calc",
    "expect_fn": "calculator",
    "min_calls": 1,
    "arg_checks": [{"field": "expression", "contains_all": ["17", "23"]}],
}
ABSTAIN_PROBE = {"id": "abstain", "expect_fn": "__none__", "min_calls": 1, "arg_checks": []}
PAR_PROBE = {
    "id": "par_weather",
    "expect_fn": "get_weather",
    "min_calls": 2,
    "arg_checks": [{"field": "unit", "equals": "celsius"}],
}


# ===========================================================================
# extract_tool_call
# ===========================================================================
class TestExtractToolCall:
    def test_structured_string_args(self):
        name, args, bad = harness.extract_tool_call(
            tool_msg("calculator", json.dumps({"expression": "17 * 23"}))
        )
        assert name == "calculator"
        assert args == {"expression": "17 * 23"}
        assert bad == ""

    def test_structured_dict_args(self):
        name, args, bad = harness.extract_tool_call(
            tool_msg("get_weather", {"location": "Paris", "unit": "celsius"})
        )
        assert name == "get_weather"
        assert args["unit"] == "celsius"
        assert bad == ""

    def test_structured_malformed_args(self):
        name, args, bad = harness.extract_tool_call(tool_msg("calculator", "{not valid json"))
        assert name == "calculator"
        assert args == {}
        assert bad.startswith("unparseable-args:")

    def test_inline_tool_call(self):
        name, args, bad = harness.extract_tool_call(
            inline_msg(json.dumps({"name": "web_search", "arguments": {"query": "tokyo population"}}))
        )
        assert name == "web_search"
        assert args == {"query": "tokyo population"}
        assert bad == ""

    def test_inline_parameters_alias(self):
        # the fallback accepts `parameters` as an alias for `arguments`
        name, args, bad = harness.extract_tool_call(
            inline_msg(json.dumps({"name": "web_search", "parameters": {"query": "x"}}))
        )
        assert name == "web_search"
        assert args == {"query": "x"}
        assert bad == ""

    def test_inline_malformed(self):
        name, args, bad = harness.extract_tool_call(
            {"role": "assistant", "content": "<tool_call>{bad json here}</tool_call>"}
        )
        assert name == ""
        assert args == {}
        assert bad.startswith("unparseable-inline:")

    def test_no_tool_call(self):
        name, args, bad = harness.extract_tool_call(text_msg("Bonjour le monde."))
        assert name == ""
        assert args == {}
        assert bad == "no-tool-call"

    def test_none_content_is_no_tool_call(self):
        # server can send content: null and no tool_calls
        name, args, bad = harness.extract_tool_call({"role": "assistant", "content": None})
        assert bad == "no-tool-call"


# ===========================================================================
# check_tool_args — the declarative DSL (1:1 migration of the 04b lambdas)
# ===========================================================================
class TestCheckToolArgs:
    def test_empty_or_absent_checks_true(self):
        assert harness.check_tool_args({"x": 1}, []) is True
        assert harness.check_tool_args({}, None) is True

    def test_equals_case_insensitive(self):
        assert harness.check_tool_args({"unit": "Celsius"}, [{"field": "unit", "equals": "celsius"}])
        assert harness.check_tool_args({"priority": "HIGH"}, [{"field": "priority", "equals": "high"}])
        assert not harness.check_tool_args(
            {"unit": "fahrenheit"}, [{"field": "unit", "equals": "celsius"}]
        )

    def test_contains_all(self):
        chk = [{"field": "expression", "contains_all": ["17", "23"]}]
        assert harness.check_tool_args({"expression": "17 * 23"}, chk)
        assert not harness.check_tool_args({"expression": "17 * 5"}, chk)

    def test_contains_any(self):
        chk = [{"field": "location", "contains_any": ["paris", "tokyo"]}]
        assert harness.check_tool_args({"location": "Tokyo"}, chk)
        assert harness.check_tool_args({"location": "Paris, France"}, chk)
        assert not harness.check_tool_args({"location": "Berlin"}, chk)

    def test_type(self):
        assert harness.check_tool_args({"a": ["x", "y"]}, [{"field": "a", "type": "array"}])
        assert not harness.check_tool_args({"a": "x,y"}, [{"field": "a", "type": "array"}])
        assert harness.check_tool_args({"a": {"k": 1}}, [{"field": "a", "type": "object"}])
        assert harness.check_tool_args({"a": "str"}, [{"field": "a", "type": "string"}])
        assert not harness.check_tool_args({"a": {"k": 1}}, [{"field": "a", "type": "array"}])

    def test_min_len(self):
        assert harness.check_tool_args({"a": ["x", "y"]}, [{"field": "a", "min_len": 2}])
        assert not harness.check_tool_args({"a": ["x"]}, [{"field": "a", "min_len": 2}])

    def test_items_have_keys(self):
        good = {"items": [{"sku": "A100", "qty": 2}, {"sku": "B200", "qty": 3}]}
        bad = {"items": [{"sku": "A100"}, {"sku": "B200", "qty": 3}]}
        chk = [{"field": "items", "items_have_keys": ["sku", "qty"]}]
        assert harness.check_tool_args(good, chk)
        assert not harness.check_tool_args(bad, chk)

    def test_dot_nested_path(self):
        args = {"shipping": {"method": "express", "address": "5 Main St"}}
        assert harness.check_tool_args(args, [{"field": "shipping.method", "equals": "express"}])
        assert harness.check_tool_args(args, [{"field": "shipping.address", "contains_all": ["main"]}])
        assert not harness.check_tool_args(args, [{"field": "shipping.method", "equals": "standard"}])

    def test_missing_path_is_false(self):
        assert not harness.check_tool_args({"a": 1}, [{"field": "nope", "equals": "x"}])
        assert not harness.check_tool_args({"a": {"b": 1}}, [{"field": "a.c", "type": "object"}])

    def test_all_checks_and_together(self):
        args = {"location": "Paris", "unit": "celsius"}
        ok = [
            {"field": "location", "contains_all": ["paris"]},
            {"field": "unit", "equals": "celsius"},
        ]
        bad = [
            {"field": "location", "contains_all": ["paris"]},
            {"field": "unit", "equals": "fahrenheit"},
        ]
        assert harness.check_tool_args(args, ok)
        assert not harness.check_tool_args(args, bad)

    def test_nested_enum_full_probe(self):
        # the deepest real probe (array-of-objects + nested object + 2 enums)
        args = {
            "priority": "high",
            "items": [{"sku": "A100", "qty": 2}, {"sku": "B200", "qty": 3}],
            "shipping": {"method": "express", "address": "5 Main St"},
        }
        checks = [
            {"field": "priority", "equals": "high"},
            {"field": "items", "type": "array"},
            {"field": "items", "min_len": 2},
            {"field": "items", "items_have_keys": ["sku", "qty"]},
            {"field": "shipping", "type": "object"},
            {"field": "shipping.method", "equals": "express"},
            {"field": "shipping.address", "contains_all": ["main"]},
        ]
        assert harness.check_tool_args(args, checks)


# ===========================================================================
# score_tool — pass logic incl. abstain (__none__) and parallel (min_calls>=2)
# ===========================================================================
class TestScoreTool:
    def test_routed_pass(self):
        r = harness.score_tool(CALC_PROBE, tool_msg("calculator", json.dumps({"expression": "17 * 23"})))
        assert r["routed"] and r["args_ok"] and r["pass"]
        assert r["malformed"] == "" and r["got_fn"] == "calculator" and r["n_calls"] == 1

    def test_wrong_fn_fail(self):
        r = harness.score_tool(CALC_PROBE, tool_msg("web_search", json.dumps({"query": "17 23"})))
        assert not r["routed"]
        assert not r["pass"]

    def test_malformed_args_fail(self):
        r = harness.score_tool(CALC_PROBE, tool_msg("calculator", "{bad json"))
        assert r["got_fn"] == "calculator"
        assert r["malformed"].startswith("unparseable-args:")
        assert not r["args_ok"]
        assert not r["pass"]

    def test_bad_arg_content_fail(self):
        # right fn, parseable JSON, but operands missing -> args_ok False
        r = harness.score_tool(CALC_PROBE, tool_msg("calculator", json.dumps({"expression": "5 + 5"})))
        assert r["routed"] and r["malformed"] == ""
        assert not r["args_ok"]
        assert not r["pass"]

    def test_abstain_pass(self):
        r = harness.score_tool(ABSTAIN_PROBE, text_msg("Bonjour."))
        assert r["malformed"] == "no-tool-call"
        assert r["pass"]

    def test_abstain_fail_on_overcall(self):
        # made a (well-formed) call when none was warranted -> over-calling
        r = harness.score_tool(ABSTAIN_PROBE, tool_msg("calculator", json.dumps({"expression": "1+1"})))
        assert r["malformed"] == ""
        assert not r["pass"]

    def test_abstain_fail_on_empty_content(self):
        r = harness.score_tool(ABSTAIN_PROBE, {"role": "assistant", "content": ""})
        assert r["malformed"] == "no-tool-call"
        assert not r["pass"]

    def test_parallel_pass(self):
        msg = multi_tool_msg(
            [
                ("get_weather", json.dumps({"location": "Paris", "unit": "celsius"})),
                ("get_weather", json.dumps({"location": "Tokyo", "unit": "celsius"})),
            ]
        )
        r = harness.score_tool(PAR_PROBE, msg)
        assert r["n_calls"] == 2 and r["routed"] and r["args_ok"] and r["pass"]

    def test_parallel_fail_single_call(self):
        r = harness.score_tool(
            PAR_PROBE, tool_msg("get_weather", json.dumps({"location": "Paris", "unit": "celsius"}))
        )
        assert r["n_calls"] == 1
        assert not r["pass"]

    def test_record_shape(self):
        r = harness.score_tool(CALC_PROBE, tool_msg("calculator", json.dumps({"expression": "17 * 23"})))
        assert set(r) == {"want_fn", "got_fn", "args", "malformed", "routed", "args_ok", "n_calls", "pass"}


# ===========================================================================
# answer parsing — normalize_answer / extract_answer / think_ok
# ===========================================================================
class TestAnswerParsing:
    def test_normalize_currency(self):
        assert harness.normalize_answer("$80") == "80"
        assert harness.normalize_answer("80 dollars") == "80"
        assert harness.normalize_answer("$1,200.50") == "1200.50"

    def test_normalize_fraction(self):
        assert harness.normalize_answer("1/3") == "1/3"
        assert harness.normalize_answer("1 / 3") == "1/3"

    def test_normalize_embedded_number(self):
        assert harness.normalize_answer("There are 120 apples left.") == "120"

    def test_normalize_non_numeric(self):
        assert harness.normalize_answer("bonjour") == "bonjour"

    def test_extract_after_think(self):
        assert harness.extract_answer("<think>x=7,y=3</think>\nIt works.\nANSWER: 21") == "21"

    def test_extract_last_answer_line(self):
        assert harness.extract_answer("ANSWER: 5\nmore text\nANSWER: 47") == "47"

    def test_extract_fallback_last_line(self):
        assert harness.extract_answer("no marker here\n  120  ") == "120"

    def test_extract_empty(self):
        assert harness.extract_answer("") == ""
        assert harness.extract_answer("<think>only reasoning</think>") == ""


class TestThinkOk:
    def test_single_pair_with_answer(self):
        assert harness.think_ok("<think>reasoning</think>\nANSWER: 120")

    def test_single_pair_empty_tail(self):
        assert not harness.think_ok("<think>reasoning</think>")

    def test_single_pair_no_answer(self):
        assert not harness.think_ok("<think>reasoning</think>\nThe result is 5.")

    def test_no_tags_with_answer(self):
        # --reasoning-format auto: trace lands in reasoning_content; content must still hold ANSWER:
        assert harness.think_ok("Plain text reasoning.\nANSWER: 42")

    def test_no_tags_no_answer(self):
        assert not harness.think_ok("Just a plain reply with no marker.")

    def test_unclosed_think(self):
        assert not harness.think_ok("<think>reasoning never closed\nANSWER: 5")

    def test_duplicated_tags(self):
        assert not harness.think_ok("<think>a</think><think>b</think>\nANSWER: 5")


# ===========================================================================
# certify pipeline — summarize / agreement / hard_fails / certify
# ===========================================================================
TOOL_IDS = [
    "calc_route", "search_route", "weather_enum", "nested_json", "calc_vs_search2",
    "nested_enum", "abstain", "search_then_calc", "par_weather",
]
REASON_IDS = ["apples", "xy", "strings", "shirt", "taps", "avgspeed", "snail", "div3or5", "overtime", "fatherson"]
REASON_WANT = {
    "apples": "120", "xy": "21", "strings": "24", "shirt": "80", "taps": "4",
    "avgspeed": "60", "snail": "8", "div3or5": "47", "overtime": "735", "fatherson": "12",
}
CRITICAL = ["nested_enum", "abstain", "par_weather"]


def thresholds():
    """Resolved-shape thresholds dict (what resolve_thresholds() returns)."""
    return {
        "pp512_min": 1096.9,
        "tg128_min": 75.9,
        "tool": {"nt": 9, "key_drop_max": 1, "agree_min": 8},
        "reason": {"nr": 10, "key_drop_max": 2, "agree_min": 8},
        "think": {"ref_required": 10, "cand_must_equal_ref": True},
        "ppl": {"max": 8.5, "anchor": 7.19},
        "sanity": {"bf16_tool_min": 8, "bf16_think_min": 10},
        "critical_probes": list(CRITICAL),
    }


def _tool_rec(passed, malformed=""):
    return {
        "want_fn": "x", "got_fn": "x" if passed else "y", "args": {}, "malformed": malformed,
        "routed": passed, "args_ok": passed, "n_calls": 0 if malformed == "no-tool-call" else 1,
        "pass": passed,
    }


def _reason_rec(pid, correct=True, think=True):
    want = REASON_WANT[pid]
    norm = want if correct else "999"
    return {"want": want, "ans": norm, "norm": norm, "raw": "", "think_ok": think, "err": ""}


def make_answers(tool_pass=None, tool_malformed=None, reason_correct=None, reason_think=None):
    """Build a full {tool, reason} answer dict. Defaults model the real BF16/Q5
    profile: all tool probes pass EXCEPT `abstain` (base-model over-calling),
    every reason answer correct, all <think> structures OK."""
    tool_pass = tool_pass or {}
    tool_malformed = tool_malformed or {}
    reason_correct = reason_correct or {}
    reason_think = reason_think or {}
    tool = {
        pid: _tool_rec(tool_pass.get(pid, pid != "abstain"), tool_malformed.get(pid, ""))
        for pid in TOOL_IDS
    }
    reason = {
        pid: _reason_rec(pid, reason_correct.get(pid, True), reason_think.get(pid, True))
        for pid in REASON_IDS
    }
    return {"tool": tool, "reason": reason}


def full_pass_ref():
    """A (synthetic) reference where ALL 9 tool probes pass — used to isolate the
    aggregate tolerance boundaries from the abstain mutual-fail."""
    return make_answers(tool_pass={pid: True for pid in TOOL_IDS})


class TestSummarizeAgreement:
    def test_summarize_counts(self):
        s = harness.summarize(make_answers())
        assert s["tool_key"] == 8  # all but abstain
        assert s["reason_key"] == 10
        assert s["think_ok"] == 10
        assert s["nt"] == 9 and s["nr"] == 10

    def test_agreement_mutual_pass_only(self):
        agr = harness.agreement(make_answers(), make_answers())
        assert agr["tool_agree"] == 8  # abstain failed by both -> contributes 0
        assert agr["reason_agree"] == 10

    def test_agreement_abstain_zero_contribution(self):
        # if abstain passes in BOTH, the mutual count rises to 9 -> proves abstain
        # itself was simply not a mutual pass before, never a penalty.
        agr = harness.agreement(full_pass_ref(), full_pass_ref())
        assert agr["tool_agree"] == 9


class TestHardFails:
    def test_none_when_clean(self):
        assert harness.hard_fails(make_answers(), make_answers(), CRITICAL) == []

    def test_h1_new_malformed_json(self):
        ref = full_pass_ref()
        cand = make_answers(
            tool_pass={"nested_json": False},
            tool_malformed={"nested_json": "unparseable-args:Expecting value"},
        )
        hf = harness.hard_fails(cand, ref, CRITICAL)
        assert any(h["code"] == "H1" and h["probe"] == "nested_json" for h in hf)

    def test_h2_critical_regression(self):
        ref = full_pass_ref()  # nested_enum passes in ref
        cand = make_answers(tool_pass={"nested_enum": False})  # clean fail (no H1)
        hf = harness.hard_fails(cand, ref, CRITICAL)
        assert any(h["code"] == "H2" and h["probe"] == "nested_enum" for h in hf)

    def test_h2_not_fired_for_abstain(self):
        # abstain fails in BOTH ref and cand -> guarded on ref-pass -> H2 must NOT fire
        hf = harness.hard_fails(make_answers(), make_answers(), CRITICAL)
        assert not any(h["probe"] == "abstain" for h in hf)

    def test_h3_think_regression(self):
        ref = full_pass_ref()
        cand = make_answers(reason_think={"snail": False})
        hf = harness.hard_fails(cand, ref, CRITICAL)
        assert any(h["code"] == "H3" and h["probe"] == "snail" for h in hf)

    def test_hard_fail_record_shape(self):
        cand = make_answers(reason_think={"snail": False})
        hf = harness.hard_fails(cand, full_pass_ref(), CRITICAL)
        assert hf and set(hf[0]) == {"code", "probe", "detail"}


class TestCertify:
    def test_ship_identical(self):
        rep = harness.certify(make_answers(), make_answers(), thresholds())
        assert rep.sanity_ok
        assert rep.passed
        assert rep.hard_fails == []

    def test_abstain_not_a_regression(self):
        # the realistic case: both BF16 and Q5 fail abstain. tool_agree == 8 == agree_min.
        rep = harness.certify(make_answers(), make_answers(), thresholds())
        assert rep.passed
        assert not any(h["probe"] == "abstain" for h in rep.hard_fails)

    def test_sanity_trip_low_tool(self):
        # ref passes only 6/9 tool -> below bf16_tool_min (8) -> rig broken, not the model
        ref = make_answers(tool_pass={pid: False for pid in TOOL_IDS[:3]})
        rep = harness.certify(make_answers(), ref, thresholds())
        assert not rep.sanity_ok
        assert not rep.passed
        assert any(c.severity == "sanity" and not c.passed for c in rep.checks)

    def test_sanity_trip_low_think(self):
        ref = make_answers(reason_think={"apples": False})  # ref think 9/10 < 10
        rep = harness.certify(make_answers(), ref, thresholds())
        assert not rep.sanity_ok
        assert not rep.passed

    def test_tool_key_boundary_pass(self):
        # ref 9/9, cand 8/9 (drop one non-critical) -> tool_key 8>=9-1 and agree 8>=8 -> SHIP
        ref = full_pass_ref()
        cand = make_answers(tool_pass={pid: True for pid in TOOL_IDS} | {"search_then_calc": False})
        rep = harness.certify(cand, ref, thresholds())
        assert rep.passed

    def test_tool_key_boundary_fail(self):
        # cand 7/9 -> tool_key 7 < 9-1 -> NO-SHIP
        ref = full_pass_ref()
        cand = make_answers(
            tool_pass={pid: True for pid in TOOL_IDS} | {"search_then_calc": False, "calc_route": False}
        )
        rep = harness.certify(cand, ref, thresholds())
        assert not rep.passed

    def test_reason_key_boundary_pass(self):
        # ref 10/10 reason, cand 8/10 (2 wrong) -> reason_key 8>=10-2 and agree 8>=8 -> SHIP
        cand = make_answers(reason_correct={"snail": False, "div3or5": False})
        rep = harness.certify(cand, make_answers(), thresholds())
        assert rep.passed

    def test_reason_key_boundary_fail(self):
        # cand 7/10 reason -> reason_key 7 < 10-2 -> NO-SHIP
        cand = make_answers(reason_correct={"snail": False, "div3or5": False, "xy": False})
        rep = harness.certify(cand, make_answers(), thresholds())
        assert not rep.passed

    def test_h1_blocks_ship(self):
        ref = full_pass_ref()
        cand = make_answers(
            tool_pass={pid: True for pid in TOOL_IDS} | {"nested_json": False},
            tool_malformed={"nested_json": "unparseable-args:boom"},
        )
        rep = harness.certify(cand, ref, thresholds())
        assert not rep.passed
        assert any(h["code"] == "H1" for h in rep.hard_fails)

    def test_h2_blocks_ship(self):
        ref = full_pass_ref()
        cand = make_answers(tool_pass={pid: True for pid in TOOL_IDS} | {"par_weather": False})
        rep = harness.certify(cand, ref, thresholds())
        assert not rep.passed
        assert any(h["code"] == "H2" and h["probe"] == "par_weather" for h in rep.hard_fails)

    def test_h3_blocks_ship(self):
        ref = full_pass_ref()
        cand = make_answers(reason_think={"taps": False})
        rep = harness.certify(cand, ref, thresholds())
        assert not rep.passed
        assert any(h["code"] == "H3" for h in rep.hard_fails)

    def test_report_structure(self):
        rep = harness.certify(make_answers(), make_answers(), thresholds())
        severities = {c.severity for c in rep.checks}
        assert "aggregate" in severities
        assert "sanity" in severities
        assert isinstance(rep.render(), str) and rep.render()
        assert isinstance(rep.as_dict(), dict)
        for c in rep.checks:
            assert c.severity in {"sanity", "aggregate", "hard"}
            assert isinstance(c.name, str) and isinstance(c.passed, bool) and isinstance(c.detail, str)


class TestScoreReason:
    def test_correct_answer(self):
        r = harness.score_reason({"id": "xy", "expect": "21"}, "<think>x=7,y=3</think>\nANSWER: 21")
        assert r["norm"] == harness.normalize_answer("21")
        assert r["think_ok"]
        assert r["want"] == "21"

    def test_wrong_answer(self):
        r = harness.score_reason({"id": "xy", "expect": "21"}, "<think>...</think>\nANSWER: 99")
        assert r["norm"] != harness.normalize_answer("21")

    def test_empty_reply(self):
        r = harness.score_reason({"id": "xy", "expect": "21"}, "")
        assert r["norm"] == ""
        assert not r["think_ok"]


class TestNestedAndParallelCalls:
    def test_inline_nested_json_is_not_truncated(self):
        blob = json.dumps({
            "name": "create_event",
            "arguments": {
                "title": "Q3 Review",
                "date": "2026-07-01",
                "attendees": ["alice", "bob"],
            },
        })
        name, args, bad = harness.extract_tool_call(inline_msg(blob))
        assert bad == ""
        assert name == "create_event"
        assert args["attendees"] == ["alice", "bob"]
        assert args["date"] == "2026-07-01"

    def test_two_inline_calls_count(self):
        content = (
            '<tool_call>{"name":"get_weather","arguments":{"location":"Paris","unit":"celsius"}}</tool_call>'
            '<tool_call>{"name":"get_weather","arguments":{"location":"Tokyo","unit":"celsius"}}</tool_call>'
        )
        rec = harness.score_tool(PAR_PROBE, {"role": "assistant", "content": content})
        assert rec["n_calls"] == 2
        assert rec["pass"]

    def test_parallel_second_call_malformed_fails(self):
        msg = multi_tool_msg([
            ("get_weather", json.dumps({"location": "Paris", "unit": "celsius"})),
            ("get_weather", "{not json"),
        ])
        rec = harness.score_tool(PAR_PROBE, msg)
        assert rec["n_calls"] == 2
        assert rec["malformed"].startswith("unparseable-args:")
        assert not rec["pass"]

    def test_parallel_second_call_bad_enum_fails(self):
        msg = multi_tool_msg([
            ("get_weather", json.dumps({"location": "Paris", "unit": "celsius"})),
            ("get_weather", json.dumps({"location": "Tokyo", "unit": "fahrenheit"})),
        ])
        rec = harness.score_tool(PAR_PROBE, msg)
        assert rec["malformed"] == ""
        assert not rec["args_ok"]
        assert not rec["pass"]


class TestMessageTextAndSmoke:
    def test_reasoning_content_is_kept_when_content_has_no_answer(self):
        text = harness.message_text({
            "content": "",
            "reasoning_content": "sold a third, then a quarter of the rest\nANSWER: 120",
        })
        assert harness.think_ok(text)
        assert harness.extract_answer(text) == "120"

    def test_content_wins_when_it_already_has_the_answer(self):
        text = harness.message_text({
            "content": "ANSWER: 120",
            "reasoning_content": "ANSWER: 999",
        })
        assert text == "ANSWER: 120"

    def test_smoke_helpers(self):
        ok = tool_msg("get_weather", json.dumps({"location": "Paris", "unit": "celsius"}))
        assert harness.smoke_tool_route(ok)
        assert not harness.smoke_tool_route(tool_msg("get_weather", json.dumps({"location": "Paris", "unit": "fahrenheit"})))
        assert harness.smoke_round_trip({"content": "The product is 391.", "tool_calls": None})
        assert not harness.smoke_round_trip({"content": "391", "tool_calls": [{"id": "c1"}]})
        assert harness.smoke_think({"content": "<think>x</think>\nANSWER: 120"})
        assert not harness.smoke_think({"content": "ANSWER: 80"})


class TestParityAndThinkFlag:
    def test_empty_generations_are_not_exact(self):
        assert harness.parity_verdict("", "", True, True) == "ERROR"
        assert harness.parity_verdict("same", "same", False, True) == "ERROR"
        assert harness.parity_verdict("same", "same", True, True) == "EXACT"
        assert harness.parity_verdict("a", "b", True, True) == "DIVERGES"

    def test_think_equality_flag_is_honored(self):
        th = thresholds()
        th["think"] = {"ref_required": 10, "cand_must_equal_ref": False}
        cand = make_answers(reason_think={"taps": False})
        ref = make_answers()
        # Equality is off, but the candidate is still below the required 10.
        assert not harness.certify(cand, ref, th).passed
        th["sanity"] = {"bf16_tool_min": 8, "bf16_think_min": 9}
        th["think"] = {"ref_required": 9, "cand_must_equal_ref": False}
        ref_low = make_answers(reason_think={"taps": False})
        cand_ok = make_answers()
        # Both are at or above 9, so a one-probe think gap is allowed.
        assert harness.certify(cand_ok, ref_low, th).passed

#!/usr/bin/env python3
"""Phase 4b — agentic + reasoning gate for QUEST-30B-RL.

Probes, schemas, and the ship rule live in tests/spec.yaml. Scoring lives in
tests/lib/harness.py. This script is the server-facing collector and the
grade table. A previous copy of the probes in this file could drift from the
spec (case-sensitive enums, a regex that truncated nested tool-call JSON, and
parallel calls that only inspected tool_calls[0]).

Usage:
  collect <port> <label> <outdir>          run all probes vs a running server -> <outdir>/<label>.json
  grade   <outdir> <ref_label> <labels...> print tool/reason scores and a SHIP/NO-SHIP verdict
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "lib"))

import harness  # noqa: E402


def _base(port):
    return f"http://127.0.0.1:{port}"


def cmd_collect(port, label, outdir):
    spec = harness.load_spec()
    maxtok = int(os.environ.get("MAXTOK", spec.get("meta", {}).get("max_tokens", 4096)))
    print(f"  -- probes ({label}, max_tokens={maxtok})", flush=True)
    out = harness.collect(_base(port), spec, maxtok)
    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, f"{label}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    summary = harness.summarize(out)
    print(
        f"  == {label}: TOOL {summary['tool_key']}/{summary['nt']} | "
        f"REASON {summary['reason_key']}/{summary['nr']} | "
        f"THINK {summary['think_ok']}/{summary['nr']}",
        flush=True,
    )
    print(f"  wrote {dest}", flush=True)


def cmd_grade(outdir, ref_label, labels):
    spec = harness.load_spec()
    thresholds = harness.resolve_thresholds(spec)
    wanted = [ref_label] + [lab for lab in labels if lab != ref_label]
    data = {}
    missing = []
    for lab in wanted:
        path = os.path.join(outdir, f"{lab}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data[lab] = json.load(f)
        else:
            missing.append(lab)
    if ref_label not in data:
        print(f"grade: reference answers not found: {os.path.join(outdir, ref_label + '.json')}", file=sys.stderr)
        return 1
    ref = data[ref_label]
    print()
    print(f"{'model':<10} {'TOOL':>7} {'agree':>7} {'REASON':>8} {'agree':>7} {'THINK':>7}  verdict")
    print("-" * 64)
    exit_code = 0
    for lab, answers in data.items():
        summary = harness.summarize(answers)
        if lab == ref_label:
            agree = harness.agreement(answers, ref)
            verdict = "REF"
        else:
            agree = harness.agreement(answers, ref)
            report = harness.certify(answers, ref, thresholds)
            verdict = "SHIP" if report.passed else "NO-SHIP"
            if not report.passed:
                exit_code = 1
        print(harness.format_grade_row(lab, summary, agree, verdict))
    if missing:
        print(f"\nmissing answer files: {', '.join(missing)}")
    print()
    print("SHIP requires the sanity guard, the aggregate tolerances in tests/spec.yaml,")
    print("and no H1/H2/H3 hard fail. PPL is not this decision.")
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "collect":
        cmd_collect(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) >= 4 and sys.argv[1] == "grade":
        sys.exit(cmd_grade(sys.argv[2], sys.argv[3], sys.argv[4:]))
    else:
        print(__doc__)
        sys.exit(1)

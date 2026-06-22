"""Behavioral certification for QUEST-30B-RL — the ship-decision detector.

QUEST's quant damage shows up FIRST as behavioral breakage (wrong tool routing,
malformed tool-call JSON, lost <think> discipline) that perplexity is blind to.
This module grades a SERVED candidate quant against the BF16 reference on the two
axes migrated verbatim from scripts/04b_probes.py: TOOL routing/arg-fidelity and
REASON answer + <think> structure.

GATED: all tests depend on the `cand_answers` / `ref_answers` (and `model_path`-free)
fixtures, which pytest.skip() unless --server-url and --ref-answers are supplied.
So `pytest` with no live server collects these tests but skips them entirely.

Parametrization loads the spec at collection time from the default path. Per-probe
assertions enforce the hard-fail rules (H1 JSON corruption, H2 critical-probe
regression, H3 <think> regression); the aggregate tolerances are enforced in
test_certify via harness.certify(). `abstain` is failed by BOTH BF16 and Q5
(base-model tool-eagerness) and is therefore NEVER counted as a regression.
"""
import pytest

import harness

# Loaded at collection time (default spec path) so every probe is its own test.
try:
    _SPEC = harness.load_spec()
    _TOOL_PROBES = _SPEC["tool_probes"]
    _REASON_PROBES = _SPEC["reason_probes"]
    _CRITICAL = set(_SPEC["thresholds"]["hard_fails"]["critical_probes"])
except Exception:  # spec.yaml not landed yet -> empty (skipped) param sets, no collection error
    _TOOL_PROBES, _REASON_PROBES, _CRITICAL = [], [], set()


@pytest.mark.parametrize("probe", _TOOL_PROBES, ids=[p["id"] for p in _TOOL_PROBES])
def test_tool_probe(probe, cand_answers, ref_answers):
    pid = probe["id"]
    rec = cand_answers["tool"].get(pid)
    assert rec is not None, f"candidate produced no record for tool probe {pid!r}"
    ref = ref_answers["tool"].get(pid)

    # H1: NEW malformed/unparseable tool-call JSON where the reference was clean.
    if ref is not None and ref.get("malformed", "") == "":
        assert rec.get("malformed", "") == "", (
            f"H1 JSON corruption on {pid!r}: candidate malformed={rec.get('malformed')!r} "
            f"(BF16 was clean)"
        )

    # Regression = reference PASSED this probe and the candidate did NOT. Because
    # the guard is on ref-pass, probes the reference itself fails (e.g. `abstain`,
    # which both BF16 and Q5 fail) can never count as a regression here.
    if ref is not None and ref.get("pass"):
        tag = "H2 critical-probe" if pid in _CRITICAL else "tool"
        assert rec.get("pass"), (
            f"{tag} regression on {pid!r}: BF16 passed but candidate failed "
            f"(got_fn={rec.get('got_fn')!r}, malformed={rec.get('malformed')!r}, "
            f"routed={rec.get('routed')}, args_ok={rec.get('args_ok')}, "
            f"n_calls={rec.get('n_calls')})"
        )


@pytest.mark.parametrize("probe", _REASON_PROBES, ids=[p["id"] for p in _REASON_PROBES])
def test_reason_probe(probe, cand_answers, ref_answers):
    pid = probe["id"]
    rec = cand_answers["reason"].get(pid)
    assert rec is not None, f"candidate produced no record for reason probe {pid!r}"
    ref = ref_answers["reason"].get(pid)

    # H3: <think> structural regression — any probe whose structure the reference
    # got right but the candidate broke is a hard fail.
    if ref is not None and ref.get("think_ok"):
        assert rec.get("think_ok"), (
            f"H3 <think> structural regression on {pid!r}: BF16 think_ok but candidate not "
            f"(raw tail={rec.get('raw', '')[-120:]!r})"
        )
    # Per-probe answer correctness is intentionally NOT asserted here: the REASON
    # key is allowed to drop within tolerance (reason.key_drop_max), which is
    # enforced in aggregate by test_certify.


def test_certify(cand_answers, ref_answers, spec):
    """The single source of the SHIP / NO-SHIP boolean: sanity guard on the BF16
    reference, then all aggregate tolerances, then the H1/H2/H3 hard fails."""
    thresholds = harness.resolve_thresholds(spec)
    report = harness.certify(cand_answers, ref_answers, thresholds)
    assert report.sanity_ok, (
        "BF16 reference failed the sanity guard — the rig (template / "
        "--reasoning-format / tool schema) is broken, not the model:\n" + report.render()
    )
    assert report.passed, "Candidate quant did NOT certify SHIP:\n" + report.render()
